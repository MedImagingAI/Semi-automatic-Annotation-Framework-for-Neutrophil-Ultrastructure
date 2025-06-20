# A Semi-automatic Annotation Framework for Neutrophil and Other Immune Cells Ultrastructure from TEM Images

This repository provides a complete pipeline to accelerate and streamline the annotation of ultrastructural components in Transmission Electron Microscopy (TEM) images, particularly focusing on neutrophils. It uses a YOLOv9-based object detector, formats output in both COCO and CVAT-compatible ellipse annotations, and is designed to reduce the manual annotation burden in biomedical imaging workflows.

---

## 🔬 Motivation

TEM images of neutrophils contain dense and complex ultrastructures such as granules and vesicles that are difficult and time-consuming to annotate manually. This framework enables:

- Deep learning-based detection of ultrastructures
- Automatic conversion of bounding boxes into CVAT-compatible ellipses
- Efficient upload and manual refinement using CVAT

---

## 📁 Repository Structure
```yaml
Semi-automatic-Annotation-Framework-for-Neutrophil-Ultrastructure/
│
├── README.md                      # Main documentation with instructions
├── figures/                       # CVAT annotation screenshots for refinement illustration
│   ├── *.png
│   └── --
│
├── sample_images/                # Folder containing test images
│   └── *.jpg
│
├── inferenceResults/             # Output directory for inference results
│   ├── neutrophils_yolo_predictions.json
│   ├── neutrophils_yolo_predictions.xml
│   └── neutrophils_yolo_predictions.xlsx
│
├── pipeline/
│   ├── weights/
│   │   └── best.pt               # Trained YOLOv9 model
│   │
│   ├── yolov9_core/
│   │   └── yolov9/               # YOLOv9 core files (models, utils, etc.)
│   │       ├── models/
│   │       ├── utils/
│   │       ├── detect.py
│   │       └── ...
│   │
│   ├── full_pipeline.py          # Inference + JSON export + CVAT ellipse conversion
│   ├── inference.py              # Only inference (JSON + Excel)
│   └── convert_to_cvat_ellipse.py # Convert COCO JSON → CVAT XML with ellipses
│
└── requirements.txt              # List of Python dependencies
```

## ✅ Features

- Detects 4 key ultrastructures:
  - Dark granules
  - Light granules
  - Empty vesicles
  - Emptying vesicles
- Converts YOLOv9 bounding boxes into CVAT-supported ellipses
- Provides COCO-format JSON + Excel summary
- Batch processing for folders of TEM images
- GPU-supported inference

---

## 🧪 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/MedImagingAI/Semi-automatic-Annotation-Framework-for-Neutrophil-Ultrastructure.git
cd Semi-automatic-Annotation-Framework-for-Neutrophil-Ultrastructure
```

### 2. Install Dependencies
We recommend using a virtual environment.
```bash
pip install -r requirements.txt
```

## Usage
### A. Run the Full Pipeline (Detection → Ellipse Conversion)

```bash
python pipeline/full_pipeline.py \
  --images sample_images \
  --weights pipeline/weights/best.pt \
  --json inferenceResults/neutrophils_yolo_predictions.json \
  --xml inferenceResults/neutrophils_yolo_predictions.xml
```

### B. Run Inference Only

```bash
python pipeline/inference.py \
  --images sample_images \
  --weights pipeline/weights/best.pt \
  --json inferenceResults/neutrophils_yolo_predictions.json \
  --excel inferenceResults/neutrophils_summary.xlsx
```

### C. Convert YOLO Predictions to CVAT-Compatible Ellipses

```bash
python pipeline/convert_to_cvat_ellipse.py \
  --json inferenceResults/neutrophils_yolo_predictions.json \
  --output inferenceResults/neutrophils_yolo_predictions.xml
```

## CVAT Annotation Interface Preview
Below is a screenshot showing how ellipse annotations appear after importing into CVAT:

<p align="center"> <img src="figures/framework_demonstration_cvat.png" alt="CVAT Interface" width="700"/> </p>

## How to Use with CVAT
1. Create a new task in CVAT.

2. Upload images from sample_images/.

3. Import the generated XML file from inferenceResults/.

4. Start refining ellipse annotations directly in the CVAT web interface.


## License
This project is licensed under the MIT License.

## Contact
For any questions or feedback, contact: medimagingai@gmail.com
