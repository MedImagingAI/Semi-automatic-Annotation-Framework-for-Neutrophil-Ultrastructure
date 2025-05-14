import os
import cv2
import numpy as np
import torch
import pandas as pd
from models.experimental import attempt_load
from utils.general import non_max_suppression
import segmentation_models_pytorch as smp

# Class names mapping
CLASS_NAMES = {
    0: "Primary granules",
    1: "Secondary granules",
    2: "Empty vesicles",
    3: "Emptying vesicles"
}

# Segmentation Helper Functions
def run_segmentation_model(seg_model, image, device):
    original_shape = image.shape[:2]
    image_resized = cv2.resize(image, (512, 512))
    image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float().to(device) / 255.0
    image_tensor = image_tensor.unsqueeze(0)
    with torch.no_grad():
        output = seg_model(image_tensor)
        predicted_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
    upscaled_mask = cv2.resize(predicted_mask, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)
    return upscaled_mask

def mask_central_cell(image, mask):
    return cv2.bitwise_and(image, image, mask=mask.astype(np.uint8))

# YOLO Helper Functions
def run_yolo_inference(model, image, img_size, device):
    image_resized = cv2.resize(image, (img_size, img_size))
    image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float().to(device) / 255.0
    image_tensor = image_tensor.unsqueeze(0)
    pred = model(image_tensor)[0]
    result = non_max_suppression(pred, 0.25, 0.45)[0]
    detections = []
    if result is not None:
        for det in result:
            if len(det) >= 6:
                x1, y1, x2, y2, conf, cls = det[:6]
                x1, y1, x2, y2 = (x1.item() * (image.shape[1] / img_size),
                                  y1.item() * (image.shape[0] / img_size),
                                  x2.item() * (image.shape[1] / img_size),
                                  y2.item() * (image.shape[0] / img_size))
                detections.append([x1, y1, x2, y2, conf.item(), cls.item()])
    return detections

def divide_image_into_patches(image, patch_size, overlap):
    height, width = image.shape[:2]
    patches = []
    positions = []
    for y in range(0, height, patch_size - overlap):
        for x in range(0, width, patch_size - overlap):
            patch = image[y:y + patch_size, x:x + patch_size]
            patches.append(patch)
            positions.append((x, y))
    return patches, positions

def run_yolo_inference_on_patches(model, patches, img_size, device):
    results = []
    for patch in patches:
        patch_resized = cv2.resize(patch, (img_size, img_size))
        patch_tensor = torch.from_numpy(patch_resized).permute(2, 0, 1).float().to(device) / 255.0
        patch_tensor = patch_tensor.unsqueeze(0)
        pred = model(patch_tensor)[0]
        results.append(non_max_suppression(pred, 0.25, 0.45))
    return results

def adjust_bboxes(patches, positions, results):
    all_detections = []
    for result, (x_offset, y_offset) in zip(results, positions):
        if result is not None and len(result) > 0:
            for det in result:
                if det is not None and det.shape[1] == 6:
                    for i in range(det.shape[0]):
                        x1, y1, x2, y2, conf, cls = det[i, :6]
                        x1, y1, x2, y2 = x1.item(), y1.item(), x2.item(), y2.item()
                        x1 += x_offset
                        y1 += y_offset
                        x2 += x_offset
                        y2 += y_offset
                        all_detections.append([x1, y1, x2, y2, conf.item(), cls.item()])
    return all_detections

def filter_detections_by_class(detections, included_classes):
    return [det for det in detections if int(det[5]) in included_classes]

def apply_nms_and_combine(all_detections, iou_threshold, score_thresholds):
    final_detections = []
    for cls_id, score_thresh in score_thresholds.items():
        class_detections = [det for det in all_detections if int(det[5]) == cls_id]
        if not class_detections:
            continue
        bboxes = [[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2, conf, cls in class_detections]
        confidences = [float(conf) for x1, y1, x2, y2, conf, cls in class_detections]
        nms_indices = cv2.dnn.NMSBoxes(bboxes, confidences, score_threshold=score_thresh, nms_threshold=iou_threshold)
        if isinstance(nms_indices, list) and len(nms_indices) > 0:
            final_detections.extend([class_detections[i[0]] for i in nms_indices])
        elif isinstance(nms_indices, np.ndarray) and nms_indices.size > 0:
            final_detections.extend([class_detections[i] for i in nms_indices.flatten()])
    return final_detections
    
def combine_detections(detections1, detections2):
    return detections1 + detections2
    
def draw_bboxes_on_image(image, final_detections):
    color_map = {
        0: (0, 0, 255),
        1: (0, 255, 0),
        2: (255, 0, 0),
        3: (255, 255, 0),
    }
    for x1, y1, x2, y2, conf, cls in final_detections:
        color = color_map.get(int(cls), (255, 255, 255))
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    return image

def count_instances_per_class(detections):
    class_counts = {}
    for _, _, _, _, _, cls in detections:
        cls = int(cls)
        if cls in class_counts:
            class_counts[cls] += 1
        else:
            class_counts[cls] = 1
    return class_counts

def save_predictions_to_excel(results, output_excel_path):
    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_excel_path), exist_ok=True)
    df.to_excel(output_excel_path, index=False)
    print(f"Excel file saved at: {output_excel_path}")


def process_single_image(image_path, excel_results):  # Added excel_results as the second argument
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    image = cv2.imread(image_path)
    segmentation_mask = run_segmentation_model(seg_model, image, device)
    masked_image = mask_central_cell(image, segmentation_mask)

    # Full-image inference for classes 2 and 3
    large_object_detections = run_yolo_inference(yolo_model, masked_image, img_size_full, device)
    large_object_detections_filtered = filter_detections_by_class(large_object_detections, [2, 3])

    # Patch-based inference for classes 0 and 1
    patches, positions = divide_image_into_patches(masked_image, patch_size, overlap)
    patches_inference_results = run_yolo_inference_on_patches(yolo_model, patches, img_size_patch, device)
    patches_all_detections = adjust_bboxes(patches, positions, patches_inference_results)
    patches_final_detections = filter_detections_by_class(patches_all_detections, [0, 1])
    patches_final_detections = apply_nms_and_combine(patches_final_detections, iou_threshold, score_thresholds)

    # Combine both detections
    combined_detections = combine_detections(patches_final_detections, large_object_detections_filtered)

    # Count instances per class
    class_counts = {CLASS_NAMES[cls]: 0 for cls in CLASS_NAMES.keys()}
    for _, _, _, _, _, cls in combined_detections:
        class_counts[CLASS_NAMES[int(cls)]] += 1

    # Append results to excel_results
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    row = {"Image Name": base_name}
    row.update(class_counts)
    excel_results.append(row)  # Append results to the list

    # Save predictions as an image
    output_folder = 'inferenceResultsUserVersion'
    os.makedirs(output_folder, exist_ok=True)
    output_path = f'{output_folder}/{base_name}_Combined_detections.jpg'
    cv2.imwrite(output_path, draw_bboxes_on_image(image.copy(), combined_detections))
    print(f"Prediction successful for {image_path}. Results saved at: {output_path}")


def process_folder(folder_path, excel_results):
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(('.jpg', '.png')):
            process_single_image(os.path.join(folder_path, file_name), excel_results)  # Pass excel_results
    print(f"Predictions for folder {folder_path} completed. Results saved in 'inferenceResultsUserVersion/'")


def main():
    segmentation_model_path = "unet_trained_model_fullCell.pth"
    model_path = 'best.pt'
    global device, seg_model, yolo_model, img_size_full, img_size_patch, patch_size, overlap, iou_threshold, score_thresholds
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seg_model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=2)
    seg_model.load_state_dict(torch.load(segmentation_model_path, map_location=device))
    seg_model.to(device).eval()
    yolo_model = attempt_load(model_path)
    yolo_model.to(device).eval()

    img_size_full = 1280
    img_size_patch = 256
    patch_size = 256
    overlap = 64
    iou_threshold = 0.2
    score_thresholds = {0: 0.6, 1: 0.55, 2: 0.65, 3: 0.65}

    excel_results = []  # Initialize the list to store results

    choice = input("Enter 1 for single image or 0 for folder: ").strip()
    if choice == "1":
        image_path = input("Enter image path: e.g foldername/imagename.jpg ").strip()
        process_single_image(image_path, excel_results)  # Pass excel_results
    elif choice == "0":
        folder_path = input("Enter folder path: e.g folder/imagesfolder ").strip()
        process_folder(folder_path, excel_results)  # Pass excel_results
    else:
        print("Invalid input.")

    # Save all results to Excel
    df = pd.DataFrame(excel_results)
    output_folder = 'inferenceResultsUserVersion'
    os.makedirs(output_folder, exist_ok=True)
    excel_path = f'{output_folder}/predicted_class_counts.xlsx'
    df.to_excel(excel_path, index=False)
    print(f"Excel file saved at: {excel_path}")

if __name__ == "__main__":
    main()