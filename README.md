# YOLO Object Detection Project

This project implements object detection using the YOLO (You Only Look Once) algorithm with transfer learning. The model was trained on the COCO128 dataset to detect and localize multiple objects in images.

## Dataset
The COCO128 dataset, a lightweight subset of the COCO dataset, was used for training and validation. The dataset contains labeled images of common objects such as persons, vehicles, and everyday items. The dataset was automatically downloaded using YOLO’s dataset handler.

## Methodology
- A pre trained YOLO model was used to apply transfer learning.
- The model was fine tuned on the COCO128 dataset.
- Training was performed for multiple epochs while monitoring loss and evaluation metrics.
- After training, the best-performing model weights were saved as `best.pt`.

## Inference
The trained model can be used to perform object detection on new images. The inference script loads the trained weights and generates bounding boxes with class labels and confidence scores.

Example command:
`bash
py Codebase/inference.py`

## Requirements:

The required Python libraries are listed in requirements.txt.

## Note:

The trained YOLO model file (best.pt) is included in this repository using Git LFS due to its large size.

