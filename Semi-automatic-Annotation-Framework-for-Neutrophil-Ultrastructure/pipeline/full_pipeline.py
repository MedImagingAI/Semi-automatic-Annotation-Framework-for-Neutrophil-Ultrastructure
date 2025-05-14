import os
import json
import cv2
import torch
import argparse
import pandas as pd
from pathlib import Path
from yolov9_core.models.experimental import attempt_load
from yolov9_core.utils.general import non_max_suppression

# Class labels
CLASS_NAMES = {
    0: "Primary granules",
    1: "Secondary granules",
    2: "Empty vesicles",
    3: "Emptying vesicles"
}

def run_yolo_inference(model, image, img_size, device):
    image_resized = cv2.resize(image, (img_size, img_size))
    image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float().to(device) / 255.0
    image_tensor = image_tensor.unsqueeze(0)
    pred = model(image_tensor)[0]
    detections = non_max_suppression(pred, 0.25, 0.45)[0]

    results = []
    if detections is not None:
        for det in detections:
            if len(det) >= 6:
                x1, y1, x2, y2, conf, cls = det[:6]
                h_orig, w_orig = image.shape[:2]
                scale_x, scale_y = w_orig / img_size, h_orig / img_size
                x1 *= scale_x
                x2 *= scale_x
                y1 *= scale_y
                y2 *= scale_y
                results.append([x1.item(), y1.item(), x2.item(), y2.item(), conf.item(), int(cls.item())])
    return results

def save_predictions_to_coco(images_data, annotations_data, output_path):
    coco = {
        "images": images_data,
        "annotations": annotations_data,
        "categories": [{"id": k, "name": v} for k, v in CLASS_NAMES.items()]
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(coco, f, indent=4)
    print(f"[✓] COCO JSON saved to {output_path}")

def convert_bbox_to_ellipse(bbox):
    x, y, w, h = bbox
    cx = x + w / 2
    cy = y + h / 2
    rx = w / 2
    ry = h / 2
    return cx, cy, rx, ry

def convert_to_cvat_ellipse(json_path, output_xml_path):
    import xml.etree.ElementTree as ET

    with open(json_path, 'r') as f:
        data = json.load(f)

    category_map = {cat['id']: cat['name'] for cat in data['categories']}

    annotation = ET.Element("annotations")

    meta = ET.SubElement(annotation, "meta")
    task = ET.SubElement(meta, "task")
    labels = ET.SubElement(task, "labels")
    for _, name in category_map.items():
        label = ET.SubElement(labels, "label")
        name_el = ET.SubElement(label, "name")
        name_el.text = name

    for image in data['images']:
        image_el = ET.SubElement(annotation, "image", id=str(image['id']),
                                 name=image['file_name'],
                                 width=str(image['width']),
                                 height=str(image['height']))
        for ann in data['annotations']:
            if ann['image_id'] != image['id']:
                continue
            bbox = ann['bbox']
            label = category_map[ann['category_id']]
            cx, cy, rx, ry = convert_bbox_to_ellipse(bbox)
            ET.SubElement(image_el, "ellipse", label=label,
                          cx=f"{cx:.2f}", cy=f"{cy:.2f}",
                          rx=f"{rx:.2f}", ry=f"{ry:.2f}",
                          occluded="0", z_order="0")

    tree = ET.ElementTree(annotation)
    ET.indent(tree, space="  ", level=0)
    os.makedirs(os.path.dirname(output_xml_path), exist_ok=True)
    tree.write(output_xml_path, encoding="utf-8", xml_declaration=True)
    print(f"[✓] CVAT XML saved to {output_xml_path}")

def run_pipeline(image_dir, weights_path, coco_json_out, cvat_xml_out, img_size=1280):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = attempt_load(weights_path, map_location=device)
    model.to(device).eval()

    images_data, annotations_data, results_table = [], [], []
    image_id = 0
    annotation_id = 0

    for img_name in sorted(os.listdir(image_dir)):
        if not img_name.lower().endswith((".jpg", ".png")):
            continue

        img_path = os.path.join(image_dir, img_name)
        image = cv2.imread(img_path)
        if image is None:
            continue

        h, w = image.shape[:2]
        detections = run_yolo_inference(model, image, img_size, device)

        images_data.append({
            "id": image_id,
            "file_name": img_name,
            "width": w,
            "height": h
        })

        class_count = {v: 0 for v in CLASS_NAMES.values()}
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            bbox = [x1, y1, x2 - x1, y2 - y1]
            area = bbox[2] * bbox[3]
            annotations_data.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": cls,
                "bbox": bbox,
                "score": conf,
                "area": area,
                "iscrowd": 0
            })
            class_count[CLASS_NAMES[cls]] += 1
            annotation_id += 1

        class_count["Image Name"] = os.path.splitext(img_name)[0]
        results_table.append(class_count)
        image_id += 1

    save_predictions_to_coco(images_data, annotations_data, coco_json_out)
    convert_to_cvat_ellipse(coco_json_out, cvat_xml_out)

    df = pd.DataFrame(results_table)
    excel_out = Path(coco_json_out).with_suffix(".xlsx")
    df.to_excel(excel_out, index=False)
    print(f"[✓] Excel summary saved to {excel_out}")

def parse_args():
    parser = argparse.ArgumentParser(description="Full pipeline: inference -> COCO -> CVAT (ellipse).")
    parser.add_argument('--images', type=str, required=True, help="Path to folder with input images")
    parser.add_argument('--weights', type=str, required=True, help="Path to trained YOLOv9 weights (.pt)")
    parser.add_argument('--json', type=str, required=True, help="Path to save COCO-format prediction JSON")
    parser.add_argument('--xml', type=str, required=True, help="Path to save CVAT-compatible ellipse XML")
    parser.add_argument('--imgsz', type=int, default=1280, help="Inference size (default: 1280)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args.images, args.weights, args.json, args.xml, args.imgsz)
