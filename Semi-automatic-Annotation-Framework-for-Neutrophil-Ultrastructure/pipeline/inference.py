import os
import sys
import cv2
import json
import torch
import argparse
import pandas as pd
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'yolov9_core', 'yolov9')))

from models.experimental import attempt_load
from utils.general import non_max_suppression

# Class mapping
CLASS_NAMES = {
    0: "Primary granules",
    1: "Secondary granules",
    2: "Empty vesicles",
    3: "Emptying vesicles"
}

def run_inference(model, image, img_size, device):
    """Run inference on a single image using YOLOv9."""
    resized = cv2.resize(image, (img_size, img_size))
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float().to(device) / 255.0
    tensor = tensor.unsqueeze(0)

    with torch.no_grad():
        pred = model(tensor)[0]
        detections = non_max_suppression(pred, 0.25, 0.45)[0]

    results = []
    if detections is not None:
        h, w = image.shape[:2]
        scale_x, scale_y = w / img_size, h / img_size
        for det in detections:
            if len(det) >= 6:
                x1, y1, x2, y2, conf, cls = det[:6]
                results.append([
                    x1.item() * scale_x,
                    y1.item() * scale_y,
                    x2.item() * scale_x,
                    y2.item() * scale_y,
                    conf.item(),
                    int(cls.item())
                ])
    return results

def save_coco_format(images, annotations, output_path):
    """Save predictions in COCO JSON format."""
    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": k, "name": v} for k, v in CLASS_NAMES.items()]
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(coco, f, indent=4)
    print(f"[✓] Predictions saved to {output_path}")

def process_folder(image_dir, weights_path, output_json, output_excel, img_size=1280):
    """Run inference over all images in a directory."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = attempt_load(weights_path)
    model.to(device).eval()

    images_data, annotations, summary = [], [], []
    image_id, annotation_id = 0, 0

    for fname in sorted(os.listdir(image_dir)):
        if not fname.lower().endswith(('.jpg', '.png')):
            continue

        image_path = os.path.join(image_dir, fname)
        image = cv2.imread(image_path)
        if image is None:
            continue

        h, w = image.shape[:2]
        detections = run_inference(model, image, img_size, device)

        images_data.append({
            "id": image_id,
            "file_name": fname,
            "width": w,
            "height": h
        })

        count_per_class = {v: 0 for v in CLASS_NAMES.values()}
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            bbox = [x1, y1, x2 - x1, y2 - y1]
            area = bbox[2] * bbox[3]
            annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": cls,
                "bbox": bbox,
                "score": conf,
                "area": area,
                "iscrowd": 0
            })
            count_per_class[CLASS_NAMES[cls]] += 1
            annotation_id += 1

        count_per_class["Image Name"] = Path(fname).stem
        summary.append(count_per_class)
        image_id += 1

    save_coco_format(images_data, annotations, output_json)

    df = pd.DataFrame(summary)
    os.makedirs(os.path.dirname(output_excel), exist_ok=True)
    df.to_excel(output_excel, index=False)
    print(f"[✓] Summary Excel saved to {output_excel}")

def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv9 inference on images.")
    parser.add_argument('--images', type=str, required=True, help="Path to input image folder")
    parser.add_argument('--weights', type=str, required=True, help="Path to YOLOv9 weights (.pt)")
    parser.add_argument('--json', type=str, required=True, help="Path to save COCO-format predictions")
    parser.add_argument('--excel', type=str, required=True, help="Path to save Excel summary")
    parser.add_argument('--imgsz', type=int, default=1280, help="Inference image size (default: 1280)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    process_folder(args.images, args.weights, args.json, args.excel, args.imgsz)
