---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.18.1
kernelspec:
  display_name: geo
  language: python
  name: geo
---

# Object Detection

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/opengeos/tnview-website/blob/main/pages/geoai/05-object-detection.ipynb)

## Introduction

Image recognition assigns a single label to an entire image, and semantic segmentation labels every pixel, but neither can distinguish individual objects. Object detection fills this gap by producing bounding boxes, each with a class label and confidence score, to localize and identify individual objects within an image.

This tutorial covers the foundations of object detection for geospatial imagery. You will explore major architectural families, train a Faster R-CNN v2 detector on the NWPU-VHR-10 benchmark dataset using the geoai package, evaluate results with COCO-style metrics, run inference on new images, and publish trained models to Hugging Face Hub.

## Learning Objectives

By the end of this tutorial, you will be able to:

- Explain how object detection differs from image classification and semantic segmentation
- Describe bounding boxes, confidence scores, Non-Maximum Suppression, and anchor boxes
- Compare two-stage, single-stage, transformer-based, and zero-shot detection architectures
- Prepare detection datasets in COCO annotation format
- Train a multi-class object detection model using the `geoai` package
- Evaluate detection performance using COCO-style mean Average Precision (mAP)
- Run inference on new imagery and visualize detection results
- Publish trained models to Hugging Face Hub and run inference from hosted models

## Understanding Object Detection

### Classification vs. Detection

Image classification assigns a single label to an entire image but says nothing about where specific objects are or how many exist. Object detection produces a variable number of bounding boxes, each with a class label and confidence score, enabling counting, locating, and mapping of individual features.

Semantic segmentation provides pixel-level labels but does not separate individual objects. Object detection provides object-level localization but does not delineate exact boundaries. Instance segmentation combines both capabilities.

### Key Concepts

**Bounding boxes** are the fundamental output of a detection model, defined by four coordinates specifying a rectangle around a detected object. In geospatial applications, these pixel-space coordinates are transformed into geographic coordinates using the image's affine transform.

**Confidence scores** accompany each bounding box, ranging from 0 to 1, indicating the model's certainty about the prediction. A high threshold reduces false positives but may miss objects, while a low threshold captures more objects but introduces more false detections.

**Intersection over Union (IoU)** measures how well a predicted bounding box overlaps with a ground truth box, calculated as the area of overlap divided by the area of union. It is used during both training and evaluation.

**Non-Maximum Suppression (NMS)** is a post-processing step that removes redundant overlapping detections by keeping only the highest-confidence box for each object. This is essential for producing clean outputs in dense scenes.

**Anchor boxes** are pre-defined bounding box templates at various scales and aspect ratios that serve as starting points for prediction. The model learns small offsets relative to these anchors rather than predicting coordinates from scratch.

## Detection Architectures

### Two-Stage Detectors

Two-stage detectors split detection into proposal generation and classification. **Faster R-CNN** first uses a Region Proposal Network (RPN) to propose candidate regions, then classifies and refines each proposal through dedicated heads.

Two-stage detectors achieve high accuracy and handle multiple scales well, making them suitable for geospatial imagery. However, their sequential nature makes them slower than single-stage alternatives.

### Single-Stage Detectors

Single-stage detectors predict bounding boxes and class labels directly from the feature map in a single pass, making them significantly faster.

- **YOLO** (You Only Look Once) frames detection as a dense prediction problem and has evolved into a family of models that balance speed and accuracy effectively.
- **SSD** (Single Shot MultiBox Detector) predicts from multiple feature maps at different resolutions, detecting objects at various scales.
- **RetinaNet** uses a focal loss function to address class imbalance by focusing training on hard examples. Supported in `geoai` via `retinanet_resnet50_fpn_v2`.
- **FCOS** is an anchor-free detector that predicts bounding boxes directly at each spatial location. Supported in `geoai` via `fcos_resnet50_fpn`.

### Transformer-Based Detectors

[DETR](https://huggingface.co/docs/transformers/model_doc/detr) (DEtection TRansformer) frames object detection as a set prediction problem using a transformer encoder-decoder architecture, eliminating the need for anchors and NMS. Its main advantage is simplicity and global context reasoning, though it can be slower to converge and may struggle with very small objects.

### Zero-Shot Detection

Zero-shot detection models identify objects described by text prompts without task-specific training data. [OWL-ViT](https://huggingface.co/docs/transformers/en/model_doc/owlvit) combines a vision transformer with a text encoder, enabling detection via natural language descriptions like "solar panel" or "swimming pool."

[Grounding DINO](https://huggingface.co/docs/transformers/en/model_doc/grounding-dino) extends this paradigm by combining DINO-based detection with grounded language understanding, achieving strong zero-shot performance.

### Choosing an Architecture

The choice of architecture depends on your application's requirements:

- **Faster R-CNN** is a strong default when accuracy is the priority and inference speed is less critical. It handles multi-scale objects well and is the default architecture in the `geoai` package's detection pipeline.
- **YOLO** is preferred when processing speed matters, such as scanning large archives of satellite imagery or supporting near-real-time monitoring applications.
- **DETR and variants** are well-suited for scenes requiring global context reasoning and when you want to avoid tuning anchor box configurations.
- **Zero-shot detectors** (OWL-ViT, Grounding DINO) are ideal for exploratory analysis, rapid prototyping, or applications where labeled training data is unavailable.

The `geoai` package supports multiple detection architectures through the `model_name` parameter:

| Model Name                          | Type         | Notes                                  |
| ----------------------------------- | ------------ | -------------------------------------- |
| `fasterrcnn_resnet50_fpn_v2`        | Two-stage    | Default, good accuracy/speed trade-off |
| `fasterrcnn_mobilenet_v3_large_fpn` | Two-stage    | Fastest two-stage option               |
| `retinanet_resnet50_fpn_v2`         | Single-stage | Fast, handles class imbalance well     |
| `fcos_resnet50_fpn`                 | Single-stage | Anchor-free                            |
| `maskrcnn_resnet50_fpn`             | Two-stage    | Also produces instance masks (slowest) |

In practice, many projects start with a pre-trained or zero-shot model to assess feasibility, then move to a task-specific detector for production. Matching the detector's strengths to the scale of your target objects is often more important than choosing the newest architecture.

## Preparing Detection Datasets

### Annotation Formats

**COCO format** stores annotations in a single JSON file with image metadata, category definitions, and bounding boxes as (x, y, width, height) in absolute pixel coordinates. It is the format used by the `geoai` package.

**YOLO format** uses one text file per image, with each line describing one object as `class_id center_x center_y width height` in normalized coordinates.

### The NWPU-VHR-10 Dataset

The [NWPU-VHR-10](https://data.source.coop/opengeos/geoai/NWPU-VHR-10.zip) dataset is a widely used benchmark for object detection in very-high-resolution remote sensing imagery. It contains 800 images (650 positive, 150 background-only) covering 10 classes: airplane, ship, storage tank, baseball diamond, tennis court, basketball court, ground track field, harbor, bridge, and vehicle.

The `geoai` package provides the `prepare_nwpu_vhr10` function to automatically split the annotated images into training and validation sets with COCO-format annotations.

## Evaluating Detection Results

### Mean Average Precision (mAP)

A detection is a **true positive** if its IoU with a ground truth box exceeds a threshold and the class is correct; otherwise it is a **false positive**. An unmatched ground truth box is a **false negative**.

**Average Precision (AP)** for a single class is the area under the precision-recall curve. **Mean Average Precision (mAP)** averages AP across all classes.

### Precision-Recall Curves

Precision-recall curves plot precision against recall as the confidence threshold varies. A strong detector maintains high precision even at high recall. Examining per-class curves reveals which object types the model struggles with.

### IoU Thresholds

- **mAP@0.5** requires at least 50% overlap and is common in geospatial applications where approximate localization suffices.
- **mAP@0.75** requires 75% overlap, rewarding more precise localization.
- **mAP@0.5:0.95** averages across thresholds from 0.5 to 0.95 in steps of 0.05 and is the primary COCO benchmark metric.

## Installation

Uncomment the following line to install the required package.

```{code-cell} ipython3
# %pip install geoai-py
```

## Import Libraries

The `geoai` package provides functions for the full object detection pipeline, including dataset preparation, training, evaluation, inference, and model sharing.

```{code-cell} ipython3
import os
import json

import geoai
```

## Download the NWPU-VHR-10 Dataset

The NWPU-VHR-10 dataset is available as a zip archive hosted on Source Cooperative. The `download_file` utility downloads and extracts it automatically.

```{code-cell} ipython3
url = "https://data.source.coop/opengeos/geoai/NWPU-VHR-10.zip"
data_dir = geoai.download_file(url)
```

```{code-cell} ipython3
print(f"Dataset directory: {data_dir}")
print(f"Contents: {os.listdir(data_dir)}")
```

## Explore the Dataset

The dataset contains 10 object classes plus a background class at index 0.

```{code-cell} ipython3
print(f"\nNWPU-VHR-10 Classes:")
for i, name in enumerate(geoai.NWPU_VHR10_CLASSES):
    print(f"  {i}: {name}")
```

```text
NWPU-VHR-10 Classes:
  0: background
  1: airplane
  2: ship
  3: storage_tank
  4: baseball_diamond
  5: tennis_court
  6: basketball_court
  7: ground_track_field
  8: harbor
  9: bridge
  10: vehicle
```

+++

## Prepare the Dataset

The `prepare_nwpu_vhr10` function splits the positive images into training and validation sets and generates COCO-format annotation files for each split.

```{code-cell} ipython3
splits = geoai.prepare_nwpu_vhr10(data_dir, val_split=0.2, seed=42)
```

```{code-cell} ipython3
print(f"Images directory: {splits['images_dir']}")
print(f"Number of classes: {splits['num_classes']}")
print(f"Class names: {splits['class_names']}")
print(f"Training images: {len(splits['train_image_ids'])}")
print(f"Validation images: {len(splits['val_image_ids'])}")
```

```text
Images directory: ./NWPU-VHR-10/positive_image_set
Number of classes: 11
Class names: ['background', 'airplane', 'ship', 'storage_tank', 'baseball_diamond', 'tennis_court', 'basketball_court', 'ground_track_field', 'harbor', 'bridge', 'vehicle']
Training images: 509
Validation images: 128
```

+++

## Visualize Sample Annotations

Visualize sample images with their ground truth bounding boxes to verify annotations before training.

```{code-cell} ipython3
geoai.visualize_coco_annotations(
    annotations_path=splits["annotations_path"],
    images_dir=splits["images_dir"],
    num_samples=6,
    random=True,
    seed=1,
    cols=3,
    figsize=(12, 6),
)
```

+++

## Train a Multi-Class Detection Model

The `train_multiclass_detector` function handles the full training pipeline: loading data, constructing the model, running the training loop, and saving the best checkpoint. Key parameters include the model architecture, class names, number of epochs, batch size, and learning rate.

```{code-cell} ipython3
output_dir = "nwpu_output"

model_path = geoai.train_multiclass_detector(
    images_dir=splits["images_dir"],
    annotations_path=splits["train_annotations"],
    output_dir=output_dir,
    model_name="fasterrcnn_resnet50_fpn_v2",
    class_names=splits["class_names"],
    num_channels=3,
    batch_size=4,
    num_epochs=10,
    learning_rate=0.005,
    val_split=0.1,
    seed=42,
    pretrained=True,
    verbose=True,
)
```

## Plot Training Metrics

Plot training loss, validation IoU, and learning rate schedule over epochs to assess model convergence.

```{code-cell} ipython3
geoai.plot_detection_training_history(
    history_path=os.path.join(output_dir, "training_history.pth"),
)
```

+++

## Evaluate with COCO Metrics

The `evaluate_multiclass_detector` function computes COCO-style mAP metrics on the validation set, including per-class AP scores.

```{code-cell} ipython3
metrics = geoai.evaluate_multiclass_detector(
    model_path=model_path,
    images_dir=splits["images_dir"],
    annotations_path=splits["val_annotations"],
    num_classes=splits["num_classes"],
    class_names=splits["class_names"][1:],  # Exclude background
    batch_size=4,
)
```

```text
Evaluation Results:
  mAP@0.5:        0.7312
  mAP@0.75:       0.4936
  mAP@[0.5:0.95]: 0.4428

  Per-class AP@0.5:
    AP@0.5/airplane: 0.7106
    AP@0.5/baseball_diamond: 0.7885
    AP@0.5/basketball_court: 0.8957
    AP@0.5/bridge: 0.9052
    AP@0.5/ground_track_field: 0.7081
    AP@0.5/harbor: 0.5322
    AP@0.5/ship: 0.6349
    AP@0.5/storage_tank: 0.5624
    AP@0.5/tennis_court: 0.8967
    AP@0.5/vehicle: 0.6781
```

Large, distinctive objects like basketball courts and tennis courts achieve higher AP than smaller or more ambiguous objects like vehicles and storage tanks.

## Run Inference on Sample Images

The `multiclass_detection` function handles tiling, model inference, NMS across tile boundaries, and result assembly.

```{code-cell} ipython3
# Load validation data to pick a test image
with open(splits["val_annotations"], "r") as f:
    val_data = json.load(f)

test_img_info = val_data["images"][0]
test_img_path = os.path.join(splits["images_dir"], test_img_info["file_name"])
print(f"Test image: {test_img_path}")
```

```{code-cell} ipython3
output_raster = "nwpu_detection_output.tif"

result_path, inference_time, detections = geoai.multiclass_detection(
    input_path=test_img_path,
    output_path=output_raster,
    model_path=model_path,
    num_classes=splits["num_classes"],
    class_names=splits["class_names"],
    window_size=512,
    overlap=256,
    confidence_threshold=0.5,
    batch_size=4,
    num_channels=3,
)

print(f"\nInference time: {inference_time:.2f}s")
print(f"Total detections: {len(detections)}")
```

```text
Collected 27 detections before NMS
After NMS: 7 detections
Multi-class detection completed in 0.14 seconds
Final detections: 7
  harbor: 6 detections
  bridge: 1 detections
Saved multi-class detection to nwpu_detection_output.tif

Inference time: 0.14s
Total detections: 7
```

+++

## Visualize Detections

The `visualize_multiclass_detections` function overlays detected bounding boxes on the original image, colored by class and labeled with confidence scores.

```{code-cell} ipython3
geoai.visualize_multiclass_detections(
    image_path=test_img_path,
    detections=detections,
    class_names=splits["class_names"],
    confidence_threshold=0.5,
    figsize=(12, 10),
)
```

+++

## Batch Inference on Multiple Images

The `batch_multiclass_detection` function runs inference on multiple images and produces a visualization grid.

```{code-cell} ipython3
val_image_paths = [
    os.path.join(splits["images_dir"], img["file_name"])
    for img in val_data["images"][:4]
]

results = geoai.batch_multiclass_detection(
    image_paths=val_image_paths,
    output_dir="nwpu_batch_output",
    model_path=model_path,
    num_classes=splits["num_classes"],
    class_names=splits["class_names"],
    confidence_threshold=0.5,
    num_channels=3,
    figsize=(16, 12),
)
```

The predictions are not perfect. You may notice both false positives and false negatives across the results.

+++

## Publish and Reuse Models

### Push to Hugging Face Hub

Sharing a trained model through Hugging Face Hub lets collaborators run inference without the original training data or compute resources.

```{code-cell} ipython3
from huggingface_hub import notebook_login

notebook_login()
```

```{code-cell} ipython3
url = geoai.push_detector_to_hub(
    model_path=model_path,
    repo_id="your-username/nwpu-vhr10-fasterrcnn",
    model_name="fasterrcnn_resnet50_fpn_v2",
    num_classes=splits["num_classes"],
    class_names=splits["class_names"],
)
```

### Run Inference from Hub

The `predict_detector_from_hub` function downloads a model from the Hub and runs inference automatically, requiring no local checkpoint or class name list.

```{code-cell} ipython3
sample_img_path = os.path.join(splits["images_dir"], "608.jpg")

result_path, inference_time, detections = geoai.predict_detector_from_hub(
    input_path=sample_img_path,
    output_path="hub_detection.tif",
    repo_id="giswqs/nwpu-vhr10-fasterrcnn",
    confidence_threshold=0.5,
)

print(f"Inference time: {inference_time:.2f}s")
print(f"Total detections: {len(detections)}")

# Clean up
if os.path.exists("hub_detection.tif"):
    os.remove("hub_detection.tif")
```

```text
Collected 34 detections before NMS
After NMS: 8 detections
Multi-class detection completed in 0.13 seconds
Final detections: 8
  baseball_diamond: 1 detections
  tennis_court: 4 detections
  basketball_court: 3 detections
Saved multi-class detection to hub_detection.tif
Inference time: 0.13s
Total detections: 8
```

```{code-cell} ipython3
geoai.visualize_multiclass_detections(
    image_path=sample_img_path,
    detections=detections,
    class_names=geoai.NWPU_VHR10_CLASSES,
    confidence_threshold=0.5,
    figsize=(12, 10),
)
```

+++

## Key Takeaways

1. Object detection localizes and classifies individual objects using bounding boxes, filling the gap between image classification and pixel-level segmentation.

2. Bounding boxes, IoU, NMS, and anchor boxes are the foundational concepts underlying how detection models generate, refine, and filter predictions.

3. Two-stage detectors (Faster R-CNN) prioritize accuracy, single-stage detectors (YOLO, RetinaNet, FCOS) prioritize speed, transformer-based detectors (DETR) simplify the pipeline, and zero-shot detectors (OWL-ViT, Grounding DINO) eliminate the need for task-specific training data.

4. The NWPU-VHR-10 dataset provides a standard 10-class benchmark for multi-class object detection in remote sensing.

5. The `geoai` package streamlines the full detection workflow from dataset preparation through training, evaluation, inference, and model sharing.

6. COCO-style mAP metrics at multiple IoU thresholds provide comprehensive evaluation, with per-class AP revealing strengths and weaknesses.

7. Confidence threshold tuning balances precision and recall, with the optimal threshold depending on application-specific costs of false positives versus false negatives.

8. Publishing models to Hugging Face Hub enables sharing and reuse without requiring local training infrastructure.
