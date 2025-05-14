import json
import os
import argparse
import xml.etree.ElementTree as ET

# Class ID to name mapping
CLASS_NAMES = {
    0: "Primary granules",
    1: "Secondary granules",
    2: "Empty vesicles",
    3: "Emptying vesicles"
}

def bbox_to_ellipse(bbox):
    """
    Convert a bounding box to ellipse parameters.
    
    Args:
        bbox (list): [x_min, y_min, width, height]
    
    Returns:
        tuple: (cx, cy, rx, ry)
    """
    x_min, y_min, width, height = bbox
    cx = x_min + width / 2
    cy = y_min + height / 2
    rx = width / 2
    ry = height / 2
    return cx, cy, rx, ry

def convert_coco_to_cvat_ellipse(coco_json_path, output_xml_path):
    """
    Convert COCO-format JSON annotations to CVAT XML format with ellipse shapes.
    
    Args:
        coco_json_path (str): Path to COCO JSON file.
        output_xml_path (str): Output path for CVAT-compatible XML file.
    """
    with open(coco_json_path, "r") as f:
        coco_data = json.load(f)

    category_id_to_name = {cat["id"]: cat["name"] for cat in coco_data["categories"]}
    annotations = ET.Element("annotations")

    # Metadata
    meta = ET.SubElement(annotations, "meta")
    task = ET.SubElement(meta, "task")
    labels = ET.SubElement(task, "labels")
    for class_id, class_name in CLASS_NAMES.items():
        label = ET.SubElement(labels, "label")
        name = ET.SubElement(label, "name")
        name.text = class_name

    # Images and Annotations
    for image in coco_data["images"]:
        image_elem = ET.SubElement(
            annotations,
            "image",
            id=str(image["id"]),
            name=image["file_name"],
            width=str(image["width"]),
            height=str(image["height"])
        )

        for ann in filter(lambda a: a["image_id"] == image["id"], coco_data["annotations"]):
            cx, cy, rx, ry = bbox_to_ellipse(ann["bbox"])
            class_name = category_id_to_name.get(ann["category_id"], "Unknown")

            ET.SubElement(
                image_elem,
                "ellipse",
                label=class_name,
                cx=f"{cx:.2f}",
                cy=f"{cy:.2f}",
                rx=f"{rx:.2f}",
                ry=f"{ry:.2f}",
                occluded="0",
                z_order="0"
            )

    tree = ET.ElementTree(annotations)
    ET.indent(tree, space="    ", level=0)
    os.makedirs(os.path.dirname(output_xml_path), exist_ok=True)
    tree.write(output_xml_path, encoding="utf-8", xml_declaration=True)
    print(f"[✓] CVAT XML saved to: {output_xml_path}")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert COCO-format predictions to CVAT-compatible XML with elliptical shapes."
    )
    parser.add_argument(
        "--json",
        type=str,
        default="inferenceResults/neutrophils_yolo_predictions.json",
        help="Path to input COCO-format JSON file with YOLO predictions."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="inferenceResults/neutrophils_yolo_predictions.xml",
        help="Path to output CVAT-compatible XML file with ellipses."
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    convert_coco_to_cvat_ellipse(args.json, args.output)
