---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  display_name: Python 3
  language: python
  name: python3
title: "Instance Segmentation"
abstract: "This tutorial introduces instance segmentation with Mask R-CNN for delineating individual agricultural field boundaries from Sentinel-2 satellite imagery. You will learn to prepare training data from the Fields of The World benchmark dataset, train a model with the geoai package, clean instance masks, vectorize predictions, and extract geometric properties for each detected field."
authors:
  - name: Qiusheng Wu
    affiliations:
      - Department of Geography & Sustainability, University of Tennessee, Knoxville
    orcid: 0000-0001-5437-4073
    email: qwu18@utk.edu
exports:
  - format: typst
    template: lapreprint-typst
    output: _build/exports/typst/
---

# Instance Segmentation

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/opengeos/tnview-website/blob/main/pages/geoai/07-instance-segmentation.ipynb)

## Introduction

Semantic segmentation assigns a class label to every pixel but cannot distinguish between individual objects of the same class. Instance segmentation solves this by detecting each object individually and producing a separate mask for every instance.

This distinction matters for many geospatial applications. Agricultural analysts need individual field parcels for crop monitoring and yield estimation, urban planners need to count buildings and measure footprints, and conservation organizations use tree crown delineation to monitor forest health.

Automated instance segmentation has become increasingly important as satellite imagery availability has grown. Programs like the EU's Common Agricultural Policy require accurate per-field information for millions of parcels, making manual digitization impractical.

This tutorial introduces Mask R-CNN for instance segmentation in remote sensing. You will download the Fields of The World benchmark dataset, prepare Sentinel-2 imagery, train a Mask R-CNN model using geoai, run inference, clean and vectorize predictions, and extract geometric properties for each detected field.

## Learning Objectives

By the end of this tutorial, you will be able to:

- Explain how instance segmentation differs from semantic segmentation and when each approach is appropriate
- Describe the Mask R-CNN architecture and its key components (backbone, RPN, detection head, mask head)
- Download and explore the Fields of The World benchmark dataset for agricultural field boundary detection
- Prepare Sentinel-2 multispectral imagery and instance segmentation masks for model training
- Train a Mask R-CNN model for field boundary delineation using the `geoai` package
- Run sliding window inference on test imagery and interpret raw prediction masks
- Clean instance masks by removing small spurious detections and filling holes
- Vectorize raster predictions into polygon features and extract geometric properties
- Process multiple images in batch for efficient large-scale inference

## Instance Segmentation vs. Semantic Segmentation

**Semantic segmentation** produces a single label map where every pixel belongs to one class. Adjacent fields all receive the same "field" label with no boundaries between them, which is sufficient for mapping total class extent but not for identifying individual parcels.

**Instance segmentation** produces a separate mask for each detected object with its own confidence score and unique identifier. This enables counting, per-object measurement, and tracking across time at the cost of greater model complexity.

For example, a semantic segmentation model applied to a farming region produces a single connected "cropland" region with no way to determine the number of individual fields. An instance segmentation model produces separate masks for each field that can be converted to polygons with geometric properties like area, perimeter, and elongation.

Instance segmentation and semantic segmentation are complementary. Some architectures combine both into panoptic segmentation, but for most geospatial workflows, instance segmentation of specific features (fields, buildings, trees) is the primary use case.

## Mask R-CNN Architecture

Mask R-CNN extends the Faster R-CNN object detection framework by adding a mask prediction branch. The architecture has four main components.

### Backbone Encoder

The backbone is a CNN (typically ResNet-50 or ResNet-101) combined with a Feature Pyramid Network (FPN) that extracts hierarchical feature maps at multiple scales. For Sentinel-2 imagery with four bands (R, G, B, NIR), the first convolutional layer is adapted to accept four input channels.

### Region Proposal Network

The RPN scans feature maps and proposes rectangular regions likely to contain objects. It produces objectness scores and bounding box adjustments, then filters proposals through non-maximum suppression to remove redundant overlapping boxes.

### Detection Head

Each proposed region is sampled from the feature map using RoI Align. The detection head classifies each region into an object class (or background) and refines the bounding box coordinates.

### Mask Head

For each detected object, a small fully convolutional network predicts a binary mask within the bounding box region. The mask head operates through a separate branch so mask prediction does not interfere with class or box predictions. The output is a fixed-size mask (typically 28 x 28 pixels) resized to match the detected bounding box.

### RoI Align: Preserving Spatial Precision

RoI Align uses bilinear interpolation to compute feature values at precisely the required locations without rounding, eliminating the quantization errors of older RoI Pooling. This precision is essential for small objects like agricultural fields that may occupy only a few dozen pixels in a training chip.

### The Complete Pipeline

The backbone extracts multi-scale features, the RPN proposes candidate regions, the detection head predicts class labels and bounding boxes, and the mask head predicts a binary mask for each instance. After non-maximum suppression, the output is a set of instances each with a class label, confidence score, bounding box, and pixel-level mask.

## Installation

Uncomment the following line to install the required package.

```{code-cell} ipython3
# %pip install geoai-py
```

## Downloading the FTW Dataset

The [Fields of The World](https://fieldsofthe.world) dataset is a large-scale benchmark for agricultural field boundary detection. It contains 70,462 image chips from 24 countries, pairing Sentinel-2 imagery (four bands at 10 m resolution) with instance segmentation masks. The dataset is available on [Source Cooperative](https://source.coop/kerner-lab/fields-of-the-world).

We use the [Luxembourg subset](https://source.coop/kerner-lab/fields-of-the-world/luxembourg), one of the smallest country subsets, making it ideal for a tutorial.

```{code-cell} ipython3
from pathlib import Path

import geopandas as gpd
import geoai
```

```{code-cell} ipython3
geoai.download_ftw(countries=["luxembourg"], output_dir="ftw_data")
```

### Exploring the Dataset

The FTW dataset includes a GeoParquet file with metadata and geometry for each chip, including the official training, validation, and test split.

```{code-cell} ipython3
country_dir = Path("ftw_data") / "luxembourg"
chips_gdf = gpd.read_parquet(country_dir / "chips_luxembourg.parquet")

print(f"Total chips: {len(chips_gdf)}")
print(f"\nSplit distribution:")
print(chips_gdf["split"].value_counts())
```

Visualize the spatial distribution of training, validation, and test chips across Luxembourg.

```{code-cell} ipython3
geoai.view_vector_interactive(chips_gdf, column="split")
```

Display sample image-mask pairs. Each mask uses unique integer IDs to distinguish individual field instances.

```{code-cell} ipython3
geoai.display_ftw_samples("ftw_data", country="luxembourg", num_samples=4)
```

## Preparing Training Data

The `geoai` Mask R-CNN pipeline expects `images/` and `labels/` directories containing uint8 GeoTIFFs. The `prepare_ftw` function rescales raw Sentinel-2 reflectance values (0-10,000) to 0-255, organizes files into the expected structure, and sets aside test chips for inference.

```{code-cell} ipython3
data = geoai.prepare_ftw("ftw_data", country="luxembourg")
data
```

```text
FTW luxembourg: 808 total chips
  Train: 643, Val: 81, Test: 84
  Using 724 chips for training
Preparing training data...
Prepared 724 training chips (skipped 0)
Prepared 5 test chips
```

The function combines the 643 training and 81 validation chips into 724 chips for model development. During training, a fresh validation split is created with `val_split=0.2`.

Verify the prepared tiles by displaying a few image-label pairs.

```{code-cell} ipython3
geoai.display_training_tiles(
    output_dir="field_boundaries",
    num_tiles=4,
    figsize=(12, 6),
    cmap="tab20",
)
```

## Training a Mask R-CNN Model

We train a Mask R-CNN model with a ResNet-50 + FPN backbone. Key parameters:

- **`num_classes=2`**: Background (0) and field (1). All fields belong to a single class; instances are distinguished through separate masks.
- **`num_channels=4`**: Sentinel-2 bands (Red, Green, Blue, NIR). The near-infrared band helps distinguish vegetation boundaries not visible in RGB.
- **`instance_labels=True`**: The FTW masks already encode unique instance IDs per field, so `geoai` uses them directly.
- **`num_epochs=20`**: Sufficient for a tutorial; increase to 50-100 for production.
- **`val_split=0.2`**: Creates an internal validation split for monitoring performance during training.

```{code-cell} ipython3
geoai.train_instance_segmentation_model(
    images_dir=data["images_dir"],
    labels_dir=data["labels_dir"],
    output_dir="field_boundaries/models",
    num_classes=2,
    num_channels=4,
    batch_size=4,
    num_epochs=20,
    learning_rate=0.005,
    val_split=0.2,
    instance_labels=True,
    visualize=True,
    verbose=True,
)
```

The total loss is a weighted sum of RPN objectness loss, classification loss, bounding box regression loss, and mask loss. A steadily decreasing loss indicates healthy training. If you encounter out-of-memory errors, reduce the batch size to 2 or use a lighter backbone.

Plot the training metrics to assess convergence.

```{code-cell} ipython3
geoai.plot_performance_metrics(
    history_path="field_boundaries/models/training_history.pth",
    figsize=(15, 5),
    verbose=True,
)
```

## Running Inference

Apply the trained model to a test image using sliding window inference. The overlap of 128 pixels prevents objects near tile boundaries from being split. With `vectorize=True`, the function returns instance masks, class labels, confidence scores, and vectorized polygons.

```{code-cell} ipython3
test_images = sorted(Path(data["test_dir"]).glob("*.tif"))
test_image_path = str(test_images[0])
masks_path = "field_boundary_prediction.tif"
model_path = "field_boundaries/models/best_model.pth"

result = geoai.instance_segmentation(
    input_path=test_image_path,
    output_path=masks_path,
    model_path=model_path,
    num_classes=2,
    num_channels=4,
    window_size=256,
    overlap=128,
    confidence_threshold=0.5,
    batch_size=4,
    vectorize=True,
    class_names=["background", "building"],
)
result
```

### Visualizing Raw Predictions

The result dictionary contains four keys: `"instance"` (unique integer ID per field), `"class_label"` (predicted class), `"score"` (confidence), and `"vector"` (GeoDataFrame with one polygon per field).

Visualize the instance mask, where each color represents a distinct detected field.

```{code-cell} ipython3
geoai.view_raster(
    result["instance"],
    nodata=0,
    cmap="tab20",
    basemap=test_image_path,
    backend="ipyleaflet",
)
```

Visualize the class label raster and confidence score raster.

```{code-cell} ipython3
geoai.view_raster(
    result["class_label"],
    nodata=0,
    cmap="binary",
    basemap=test_image_path,
    backend="ipyleaflet",
)
```

```{code-cell} ipython3
geoai.view_raster(
    result["score"], nodata=0, basemap=test_image_path, backend="ipyleaflet"
)
```

Display the vectorized predictions colored by confidence score.

```{code-cell} ipython3
geoai.view_vector_interactive(result["vector"], tiles=test_image_path, column="score")
```

## Post-Processing Predictions

Raw instance masks often contain small spurious detections or holes. The `clean_instance_mask` function removes connected components smaller than a minimum area and fills holes up to a specified size.

```{code-cell} ipython3
cleaned_masks_path = "field_boundary_prediction_cleaned.tif"
geoai.clean_instance_mask(
    result["instance"], cleaned_masks_path, min_area=100, max_hole_area=100
)
```

```{code-cell} ipython3
geoai.view_raster(
    cleaned_masks_path,
    nodata=0,
    cmap="tab20",
    basemap=test_image_path,
    backend="ipyleaflet",
)
```

### Vectorizing Predictions

Convert the cleaned raster mask to vector polygons. Each unique pixel value becomes a separate polygon feature in a georeferenced GeoDataFrame.

```{code-cell} ipython3
output_vector_path = "field_boundary_prediction.geojson"
gdf = geoai.raster_to_vector(cleaned_masks_path, output_vector_path)
```

### Comparing Predictions with Imagery

Use a split map to visually compare detected field boundaries against the original Sentinel-2 imagery.

```{code-cell} ipython3
geoai.create_split_map(
    left_layer=gdf,
    right_layer=test_image_path,
    left_args={"style": {"color": "red", "fillOpacity": 0.2}},
    basemap=test_image_path,
)
```

## Extracting Geometric Properties

Computing geometric properties transforms raw predictions into a rich attribute table for spatial analysis. Reproject to a projected CRS first if working with latitude/longitude coordinates.

| Property       | Description                                                                |
| -------------- | -------------------------------------------------------------------------- |
| **Area**       | Field size in hectares, critical for yield estimation and subsidy programs |
| **Perimeter**  | Boundary length in meters, useful for fencing cost estimation              |
| **Elongation** | Major/minor axis ratio, distinguishes strip fields from compact parcels    |
| **Solidity**   | Area/convex hull area ratio, measures boundary irregularity                |
| **Extent**     | Area/bounding box area ratio, indicates how rectangular a field is         |

```{code-cell} ipython3
gdf_props = geoai.add_geometric_properties(gdf, area_unit="ha", length_unit="m")
gdf_props.head()
```

```{code-cell} ipython3
gdf_props.describe()
```

### Visualizing Fields by Property

Interactive maps colored by geometric properties reveal spatial patterns in field characteristics.

```{code-cell} ipython3
geoai.view_vector_interactive(gdf_props, column="area_ha", tiles=test_image_path)
```

Elongation highlights strip-shaped fields versus compact parcels.

```{code-cell} ipython3
geoai.view_vector_interactive(gdf_props, column="elongation", tiles=test_image_path)
```

## Batch Processing

Batch processing applies the same model to all images in a directory, avoiding repeated model loading and making better use of GPU resources.

```{code-cell} ipython3
geoai.instance_segmentation_batch(
    input_dir=data["test_dir"],
    output_dir="field_boundaries/predictions",
    model_path=model_path,
    num_classes=2,
    num_channels=4,
    window_size=256,
    overlap=128,
    confidence_threshold=0.5,
    batch_size=4,
)
```

## Key Takeaways

1. Instance segmentation detects individual objects with separate masks, enabling counting and per-object measurement unlike semantic segmentation.

2. Mask R-CNN combines a backbone encoder, Region Proposal Network, detection head, and mask head to classify, localize, and segment each object instance.

3. The Fields of The World dataset pairs Sentinel-2 imagery with instance masks across 24 countries for reproducible field boundary detection.

4. The near-infrared band captures vegetation differences invisible in RGB, improving boundary detection between adjacent fields.

5. The `instance_labels=True` flag tells the pipeline to use pre-existing unique instance IDs from the masks directly.

6. Post-processing with `clean_instance_mask` removes small spurious detections and fills holes for cleaner output.

7. The `raster_to_vector` function converts each unique instance ID into a separate GIS-ready polygon feature.

8. Geometric properties (area, perimeter, elongation, solidity, extent) enable statistical analysis of field characteristics across a study area.

9. Batch inference processes an entire directory of images in one call, avoiding repeated model loading.
