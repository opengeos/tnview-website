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
  name: python3
---

# Creating Training Data

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/opengeos/tnview-website/blob/main/pages/geoai/03-training-data.ipynb)

## Introduction

The performance of any GeoAI model depends on the quality and quantity of its training data. Data preparation often consumes 60-80% of total project time, and the effort invested here directly determines the quality of results.

This tutorial covers the end-to-end workflow for creating training datasets for geospatial deep learning. You will learn to convert vector labels to raster masks, generate image chips with the geoai package, work with both single-image and batch workflows, and visualize training tiles.

## Learning Objectives

By the end of this tutorial, you will be able to:

- Describe the full training data pipeline from raw imagery to model-ready datasets
- Convert vector annotations to raster masks using `geoai`
- Generate image chips from large satellite scenes using tiling and overlap strategies
- Create training tiles from single images and from batch folders of images and masks
- Use three different input modes for pairing images with vector annotations
- Visualize training data to verify label quality and alignment
- Organize datasets into training, validation, and test splits with spatial separation

## The Training Data Pipeline

Creating training data for GeoAI transforms raw geospatial data into the structured format that deep learning frameworks expect. The pipeline follows a consistent pattern regardless of the specific task:

1. **Acquire imagery and labels**: Obtain satellite or aerial imagery along with corresponding annotations such as building footprints, land cover polygons, or object bounding boxes.
2. **Convert vector labels to raster masks**: Rasterize vector annotations onto a grid that matches the imagery resolution and coordinate reference system.
3. **Tile the imagery**: Split large raster scenes into fixed-size image chips that fit into GPU memory and match the input dimensions expected by neural networks.
4. **Generate paired labels**: For each image chip, produce the corresponding label tile (raster mask for segmentation or bounding box file for detection).
5. **Visualize and validate**: Inspect the generated tiles to verify label alignment and quality before training.
6. **Split into subsets**: Divide the dataset into training, validation, and test sets, taking care to maintain spatial separation.

## Installation

Uncomment the following line to install the required package.

```{code-cell} ipython3
# %pip install geoai-py
```

## Generating Image Chips from a Single Image

Satellite and aerial imagery scenes are far too large to feed directly into a neural network. Tiling (or "chipping") slices these large scenes into manageable pieces, typically 256 x 256 or 512 x 512 pixels.

### Download Sample Data

We start by downloading a sample NAIP image and its corresponding building footprint annotations.

```{code-cell} ipython3
import geoai
```

```{code-cell} ipython3
raster_url = "https://data.source.coop/opengeos/geoai/naip-train.tif"
vector_url = "https://data.source.coop/opengeos/geoai/naip-train-buildings.geojson"
raster_path = geoai.download_file(raster_url)
vector_path = geoai.download_file(vector_url)
```

### Preview Data

Before generating chips, inspect the source imagery and annotations visually to catch problems like misaligned labels or image artifacts.

```{code-cell} ipython3
geoai.view_image(raster_path, figsize=(18, 10))
```

```{code-cell} ipython3
geoai.view_vector(vector_path, raster_path=raster_path, figsize=(18, 10))
```

```{code-cell} ipython3
geoai.view_vector_interactive(vector_path, tiles=raster_path)
```

### Convert Vector to Raster

Semantic segmentation models require pixel-level labels. The `vector_to_raster()` function rasterizes vector polygons onto a grid matching the reference raster's resolution, extent, and CRS, producing a binary mask (1 for buildings, 0 for background).

```{code-cell} ipython3
output_path = vector_path.replace(".geojson", ".tif")
geoai.vector_to_raster(vector_path, output_path, reference_raster=raster_path)
```

```{code-cell} ipython3
geoai.view_image(output_path, figsize=(18, 10))
```

### Tiling Parameters

The key parameters for chip generation are:

- **Tile size**: The width and height of each chip in pixels (e.g., 256 x 256 or 512 x 512). This should match the input size your model expects.
- **Stride**: The step size between consecutive chips. When stride equals tile size, chips are non-overlapping. When stride is smaller, chips overlap.

Overlapping tiles ensure that objects at chip boundaries appear fully in at least one chip. Common overlap ratios:

| Overlap | Stride (for 256px tiles) | Stride (for 512px tiles) | Use Case                                    |
| ------- | ------------------------ | ------------------------ | ------------------------------------------- |
| 0%      | 256                      | 512                      | Simple classification, large uniform areas  |
| 25%     | 192                      | 384                      | General purpose, moderate boundary objects  |
| 50%     | 128                      | 256                      | Dense object detection, building footprints |
| 75%     | 64                       | 128                      | Maximum coverage, small object detection    |

Higher overlap produces more chips and increases storage requirements. Choose the overlap based on your object sizes relative to tile dimensions and your computational budget.

### Generate Tiles

The `export_geotiff_tiles()` function handles the complete chip generation workflow: tiling imagery, rasterizing vector labels into masks, and saving paired image-mask tiles.

```{code-cell} ipython3
tiles = geoai.export_geotiff_tiles(
    in_raster=raster_path,
    out_folder="output",
    in_class_data=vector_path,
    tile_size=512,
    stride=384,
    buffer_radius=0,
    create_overview=True,
    quiet=True,
)
```

### Preview Image Chips

After generating tiles, inspect the overview and individual image-mask pairs to verify that labels are correctly aligned.

```{code-cell} ipython3
geoai.view_image("output/overview.png", figsize=(18, 10))
```

Visualize the generated tiles using the `display_training_tiles()` function.

```{code-cell} ipython3
fig = geoai.display_training_tiles(output_dir="output", num_tiles=4, figsize=(18, 10))
```

## Batch Processing Multiple Images

In real-world projects, training data often spans multiple satellite scenes. The `export_geotiff_tiles_batch()` function handles batch processing and supports three pairing modes:

1. **Single vector file covering all images** - Most efficient when you have one large annotation file
2. **Multiple vector files matched by sorted order** - Good for sequential datasets
3. **Multiple vector files matched by filename** - Good for paired datasets with matching names

### Download Batch Sample Data

The sample dataset contains two NAIP image tiles along with building annotations in two formats: a single GeoJSON file covering both tiles, and separate GeoJSON files for each tile.

```{code-cell} ipython3
import os
```

```{code-cell} ipython3
url = "https://data.source.coop/opengeos/geoai/naip-rgb-train-tiles.zip"
data_dir = geoai.download_file(url)
```

### Explore Sample Data

Let's list the files in the sample data directory to see what is included:

```{code-cell} ipython3
print("Images:")
for f in sorted(os.listdir(f"{data_dir}/images")):
    print(f"  - {f}")

print("\nAnnotations (single file):")
for f in sorted(os.listdir(f"{data_dir}/masks1")):
    print(f"  - {f}")

print("\nAnnotations (multiple files):")
for f in sorted(os.listdir(f"{data_dir}/masks2")):
    print(f"  - {f}")
```

```text
Images:
  - naip_rgb_train_tile1.tif
  - naip_rgb_train_tile2.tif

Annotations (single file):
  - naip_train_buildings.geojson

Annotations (multiple files):
  - naip_rgb_train_tile1.geojson
  - naip_rgb_train_tile2.geojson
```

### Visualize Image and Annotations

The `display_image_with_vector()` function overlays vector annotations on the source imagery for a quick visual check of label alignment.

```{code-cell} ipython3
image_path = f"{data_dir}/images/naip_rgb_train_tile1.tif"
mask_path = f"{data_dir}/masks2/naip_rgb_train_tile1.geojson"

fig, axes, info = geoai.display_image_with_vector(image_path, mask_path)
print(f"Number of buildings: {info['num_features']}")
```

### Method 1: Single Vector File Covering All Images

This method uses one annotation file covering multiple image tiles. The function spatially filters features for each image based on its bounds.

```{code-cell} ipython3
stats = geoai.export_geotiff_tiles_batch(
    images_folder=f"{data_dir}/images",
    masks_file=f"{data_dir}/masks1/naip_train_buildings.geojson",
    output_folder="output/method1_single_mask",
    tile_size=256,
    stride=128,
    class_value_field="class",
    skip_empty_tiles=True,
    quiet=False,
)

print(f"\n{'='*60}")
print("Results:")
print(f"  Images processed: {stats['processed_pairs']}")
print(f"  Total tiles generated: {stats['total_tiles']}")
print(f"  Tiles with features: {stats['tiles_with_features']}")
print(f"  Feature percentage: {stats['tiles_with_features']/stats['total_tiles']*100:.1f}%")
```

```text
============================================================
BATCH PROCESSING SUMMARY
============================================================
Total image pairs found: 2
Successfully processed: 2
Failed to process: 0
Total tiles generated: 141
Tiles with features: 141
Feature percentage: 100.0%
Output saved to: output/method1_single_mask
  Images: output/method1_single_mask/images
  Masks: output/method1_single_mask/masks
  Annotations: output/method1_single_mask/annotations

============================================================
Results:
  Images processed: 2
  Total tiles generated: 141
  Tiles with features: 141
  Feature percentage: 100.0%
```

### Method 2: Multiple Vector Files Matched by Sorted Order

This method pairs images and masks alphabetically by sorted order. Use this mode only when you are confident the sorted file lists line up correctly.

```{code-cell} ipython3
stats = geoai.export_geotiff_tiles_batch(
    images_folder=f"{data_dir}/images",
    masks_folder=f"{data_dir}/masks2",
    output_folder="output/method2_sorted_order",
    tile_size=256,
    stride=128,
    class_value_field="class",
    skip_empty_tiles=True,
    match_by_name=False,
)

print(f"\n{'='*60}")
print("Results:")
print(f"  Images processed: {stats['processed_pairs']}")
print(f"  Total tiles generated: {stats['total_tiles']}")
print(f"  Tiles with features: {stats['tiles_with_features']}")
```

### Method 3: Multiple Vector Files Matched by Filename

This method pairs images and masks by matching base filenames (e.g., `tile1.tif` with `tile1.geojson`). It is the safest option when files share the same base name.

```{code-cell} ipython3
stats = geoai.export_geotiff_tiles_batch(
    images_folder=f"{data_dir}/images",
    masks_folder=f"{data_dir}/masks2",
    output_folder="output/method3_matched_name",
    tile_size=256,
    stride=128,
    class_value_field="class",
    skip_empty_tiles=True,
    match_by_name=True,
)

print(f"\n{'='*60}")
print("Results:")
print(f"  Images processed: {stats['processed_pairs']}")
print(f"  Total tiles generated: {stats['total_tiles']}")
print(f"  Tiles with features: {stats['tiles_with_features']}")
```

### Visualize Generated Tiles

The `display_training_tiles()` function shows a grid of image-mask pairs from an output directory.

```{code-cell} ipython3
output_dir = "output/method1_single_mask"
fig = geoai.display_training_tiles(output_dir, num_tiles=4, figsize=(18, 10))
```

### Advanced Usage: Custom Parameters

The `export_geotiff_tiles_batch()` function supports many parameters for customization, including tile size, stride, buffer radius, and empty-tile filtering.

```{code-cell} ipython3
stats = geoai.export_geotiff_tiles_batch(
    images_folder=f"{data_dir}/images",
    masks_file=f"{data_dir}/masks1/naip_train_buildings.geojson",
    output_folder="output/advanced_example",
    tile_size=512,
    stride=256,
    class_value_field="class",
    buffer_radius=0.5,
    skip_empty_tiles=True,
    all_touched=True,
    max_tiles=10,
    quiet=False,
)

print(f"\nGenerated {stats['total_tiles']} tiles with 50% overlap")
print(f"Output structure:")
print(f"  - output/advanced_example/images/  (image tiles)")
print(f"  - output/advanced_example/masks/   (mask tiles)")
```

## Batch Processing with Raster Masks

When your annotations are already rasterized (e.g., land cover maps), you can pass a folder of raster masks directly instead of vector files.

```{code-cell} ipython3
url = "https://data.source.coop/opengeos/geoai/landcover-sample-data.zip"
data_dir2 = geoai.download_file(url)
```

```{code-cell} ipython3
images_dir = f"{data_dir2}/images"
masks_dir = f"{data_dir2}/masks"
tiles_dir = f"{data_dir2}/tiles"
```

```{code-cell} ipython3
result = geoai.export_geotiff_tiles_batch(
    images_folder=images_dir,
    masks_folder=masks_dir,
    output_folder=tiles_dir,
    tile_size=512,
    stride=384,
    quiet=True,
)
```

The generated tiles are stored in `images/` and `masks/` subfolders, with each image tile having a corresponding mask tile ready for training.

## Label Quality Considerations

High-quality labels are essential for model performance. Common issues to watch for include:

- **Incomplete annotations**: Missing objects in the training data teach the model to ignore those objects. Ensure all instances of your target classes are labeled within each chip.
- **Spatial misalignment**: Labels that do not align precisely with the imagery introduce noise. Always verify that vector annotations overlay correctly on the raster using `display_image_with_vector()`.
- **Class imbalance**: When one class dominates the dataset, the model may learn to predict the majority class exclusively. Consider oversampling, class weighting, or focal loss.
- **Edge effects**: Objects at chip boundaries may be partially labeled. The overlap strategy described earlier mitigates this issue. The `skip_empty_tiles` parameter also helps by excluding tiles with no features.
- **Buffer radius**: Adding a small buffer around vector features (using the `buffer_radius` parameter) can account for slight misalignment between imagery and annotations.

## Dataset Organization

### Train/Validation/Test Splits

Splitting data into training, validation, and test sets requires **spatial separation**. Nearby chips are highly correlated, so if training and validation chips come from adjacent locations, the model can achieve artificially high validation scores.

To avoid leakage, split data at the scene or region level:

- **Region-based splitting**: Assign entire geographic regions to different splits. For example, use imagery from one city for training, another for validation, and a third for testing.
- **Scene-based splitting**: If working with multiple satellite scenes, assign whole scenes to different splits.
- **Spatial buffer**: When splitting within a single scene, leave a buffer zone (several chip widths) between training and validation areas.

Common split ratios are 70/15/15 or 80/10/10 for training, validation, and test sets.

### Directory Structure

Most deep learning frameworks expect training data organized in a specific directory layout. The `geoai` package generates the following structure:

**Segmentation datasets** (image-mask pairs):

```
dataset/
    images/
        tile_000000.tif
        tile_000001.tif
        ...
    masks/
        tile_000000.tif
        tile_000001.tif
        ...
```

For projects that require separate train/validation/test folders, organize the tiles into subdirectories:

```
dataset/
    train/
        images/
        masks/
    val/
        images/
        masks/
    test/
        images/
        masks/
```

**Detection datasets** (YOLO format):

```
dataset/
    images/
        train/
            chip_0001.tif
        val/
            chip_0001.tif
    labels/
        train/
            chip_0001.txt
        val/
            chip_0001.txt
    data.yaml
```

Consistency in naming between image and label files is essential for data loaders to pair them correctly.

## Summary

This tutorial covered both single-image and batch workflows for turning raw geospatial data into model-ready training datasets. For batch processing, `export_geotiff_tiles_batch()` provides three flexible pairing methods:

| Method                    | Use Case                                | Parameter                                   |
| ------------------------- | --------------------------------------- | ------------------------------------------- |
| Single vector file        | One annotation file covering all images | `masks_file="path/to/file.geojson"`         |
| Multiple files (by order) | Paired files in sorted order            | `masks_folder="path/", match_by_name=False` |
| Multiple files (by name)  | Paired files with matching names        | `masks_folder="path/", match_by_name=True`  |

Across these workflows, the `geoai` training data pipeline provides:

- Supports both raster and vector masks
- Automatic CRS reprojection between imagery and annotations
- Spatial filtering for single mask files covering multiple images
- Configurable tile size, stride, and overlap
- Optional empty tile filtering and buffer support for vector annotations
- Built-in visualization for quality checking

## Key Takeaways

1. **Training data quality determines model quality** - invest time in careful annotation, label verification, and preprocessing before training.
2. **Tiling is essential** for processing large satellite scenes; use overlap to handle objects at chip boundaries.
3. **The `geoai` package** provides `export_geotiff_tiles()` for single-image workflows and `export_geotiff_tiles_batch()` for batch processing.
4. **Always visualize** your training data before training using `display_image_with_vector()` and `display_training_tiles()`.
5. **Vector-to-raster conversion** with `vector_to_raster()` creates pixel-level masks from polygon annotations with fine-grained control over rasterization.
6. **Spatial separation** in train/validation/test splits prevents data leakage from spatial autocorrelation.
7. **Document your preprocessing pipeline** thoroughly, including tiling parameters, stride settings, class definitions, and split ratios.
