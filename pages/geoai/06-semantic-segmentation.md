---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  display_name: geo
  language: python
  name: python3
title: "Semantic Segmentation"
abstract: "This tutorial introduces semantic segmentation for geospatial data, where the goal is to assign a class label to every pixel in a satellite or aerial image. You will learn to train segmentation models with the geoai package, evaluate model performance, generate prediction maps, detect clouds and cloud shadows, perform multi-class land cover classification, and share trained models through Hugging Face Hub."
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

# Semantic Segmentation

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/opengeos/tnview-website/blob/main/pages/geoai/06-semantic-segmentation.ipynb)

## Introduction

Semantic segmentation classifies every pixel in an image, producing a dense prediction map with the same spatial dimensions as the input. Unlike object detection with bounding boxes, it assigns a class label (e.g., building, road, vegetation, water) to each pixel.

In remote sensing, semantic segmentation transforms raw imagery into thematic maps for environmental monitoring, urban planning, disaster response, and climate science. This tutorial covers four progressively more complex applications:

1. **Building detection** from high-resolution aerial imagery, a binary segmentation task that illustrates the core concepts.
2. **Surface water mapping** across four workflows: standard RGB imagery, multispectral Sentinel-2 data, NAIP aerial photography, and a sensor-agnostic pre-trained model.
3. **Cloud and cloud shadow detection** with a pre-trained sensor-agnostic model, illustrating inference-only segmentation for a critical preprocessing step.
4. **Land cover classification**, a multi-class segmentation problem that assigns every pixel to one of 13 land cover categories.

Each application follows a consistent pattern: acquire data, create training tiles, train a U-Net model, evaluate performance, run inference, and visualize results.

## Learning Objectives

By the end of this tutorial, you will be able to:

- Explain how semantic segmentation differs from image classification and object detection
- Describe the encoder-decoder architecture and the role of skip connections in preserving spatial detail
- Compare common segmentation architectures (U-Net, DeepLabV3+, FPN, PSPNet) and encoders (ResNet, EfficientNet)
- Explain how transfer learning from ImageNet accelerates training on geospatial data
- Prepare geospatial training data by tiling rasters and rasterizing vector labels into binary or multi-class masks
- Train a semantic segmentation model using the `geoai` package with different input band configurations
- Evaluate segmentation performance using loss curves, IoU, and F1 score
- Run inference on new imagery using sliding window prediction and convert raster predictions to vector polygons
- Publish trained segmentation models to Hugging Face Hub and run inference from hosted models

## Foundations of Semantic Segmentation

A **deep learning architecture** defines the overall structure of a neural network -- how data flows through layers and how predictions are produced. An **encoder** compresses the input image into compact feature representations, while a **decoder** reconstructs these back to the original spatial resolution to produce the segmentation map.

In short:

- **Architecture** = the factory blueprint (overall design and data flow)
- **Encoder** = the preprocessing line (compresses inputs into learned features)
- **Decoder** = the finishing line (reconstructs features into a pixel-level output)

### Segmentation Architectures

Encoder-decoder architectures are the standard for semantic segmentation. **Skip connections** link corresponding encoder and decoder layers, forwarding high-resolution feature maps directly to the decoder so it can produce predictions that are both spatially precise and semantically accurate.

The [segmentation_models.pytorch](https://github.com/qubvel-org/segmentation_models.pytorch) library provides a modular framework separating architectures from encoders:

| Architecture    | Key Idea                                                       | Strengths                                               |
| --------------- | -------------------------------------------------------------- | ------------------------------------------------------- |
| `unet`          | Symmetric encoder-decoder with skip connections at every level | Fine boundary delineation; works well with limited data |
| `unetplusplus`  | Dense skip connections (nested U-Net)                          | Better feature fusion than standard U-Net               |
| `deeplabv3`     | Atrous convolutions with ASPP module                           | Multi-scale context capture                             |
| `deeplabv3plus` | Encoder-decoder with atrous separable convolution              | Multi-scale context with improved boundary detail       |
| `fpn`           | Feature pyramid aggregation from multiple encoder levels       | Handles objects at widely varying scales                |
| `pspnet`        | Pyramid pooling module for global scene context                | Strong for scenes requiring broad context               |
| `linknet`       | Lightweight decoder with sum-based skip fusion                 | Fast inference with good accuracy                       |
| `manet`         | Multi-scale attention-guided feature fusion                    | Attention mechanism improves boundary precision         |
| `pan`           | Pyramid attention across multiple scales                       | Hierarchical attention for multi-scale features         |
| `upernet`       | Unified perceptual parsing network                             | Comprehensive multi-scale feature integration           |
| `segformer`     | Vision transformer backbone for segmentation                   | Efficient transformer-based segmentation                |
| `dpt`           | Dense prediction transformer                                   | Fine-grained predictions from vision transformer tokens |

For most geospatial segmentation tasks, **U-Net with a pre-trained ResNet encoder** is an excellent starting point.

### Encoders and Transfer Learning

An **encoder** compresses an input image into a meaningful feature representation. Rather than training from scratch, **transfer learning** starts with an encoder pre-trained on ImageNet, which already recognizes edges, textures, and patterns that transfer well to remote sensing imagery.

Common encoder choices include `resnet34`, `resnet50`, `efficientnet-b0`, and `mobilenet_v2`.

### Practical Implementation

The `geoai` package builds on `segmentation_models.pytorch`, providing high-level functions for the common training and inference pipeline.

+++

## Installation

Uncomment the following line to install the required package.

```{code-cell} ipython3
# %pip install geoai-py
```

## Import Libraries

To get started, import the `geoai` package.

```{code-cell} ipython3
import geoai
```

## Building Detection from Aerial Imagery

Building detection is a binary segmentation problem (building vs. background) that provides a clear introduction to the segmentation workflow. We train a U-Net model on high-resolution NAIP aerial imagery with 3-band RGB input at 1-meter resolution.

### Download Sample Data

The training dataset consists of a NAIP image tile and corresponding building footprints from the Overture Maps project.

```{code-cell} ipython3
train_raster_url = (
    "https://data.source.coop/opengeos/geoai/naip_rgb_train.tif"
)
train_vector_url = "https://data.source.coop/opengeos/geoai/naip_train_buildings.geojson"
test_raster_url = (
    "https://data.source.coop/opengeos/geoai/naip_test.tif"
)
```

```{code-cell} ipython3
train_raster_path = geoai.download_file(train_raster_url)
train_vector_path = geoai.download_file(train_vector_url)
test_raster_path = geoai.download_file(test_raster_url)
```

### Visualize Sample Data

Inspect the data before training.

```{code-cell} ipython3
geoai.print_raster_info(train_raster_path, show_preview=False)
```

```{code-cell} ipython3
geoai.view_vector_interactive(train_vector_path, tiles=train_raster_path)
```

Display the held-out test image.

```{code-cell} ipython3
geoai.view_raster(test_raster_path)
```

### Create Training Tiles

The `export_geotiff_tiles` function slices the training raster and labels into 512x512 pixel tiles with 256-pixel stride, creating overlapping patches.

```{code-cell} ipython3
out_folder = "buildings"
tiles = geoai.export_geotiff_tiles(
    in_raster=train_raster_path,
    out_folder=out_folder,
    in_class_data=train_vector_path,
    tile_size=512,
    stride=256,
    buffer_radius=0,
)
```

### Train the Model

The model uses a U-Net architecture with a ResNet34 encoder pre-trained on ImageNet.

```{code-cell} ipython3
geoai.train_segmentation_model(
    images_dir=f"{out_folder}/images",
    labels_dir=f"{out_folder}/labels",
    output_dir=f"{out_folder}/unet_models",
    architecture="unet",
    encoder_name="resnet34",
    encoder_weights="imagenet",
    num_channels=3,
    num_classes=2,
    batch_size=8,
    num_epochs=20,
    learning_rate=0.001,
    val_split=0.2,
)
```

### Evaluate the Model

Training curves reveal whether the model is learning effectively. A growing gap between training and validation loss indicates overfitting.

```{code-cell} ipython3
geoai.plot_performance_metrics(
    history_path=f"{out_folder}/unet_models/training_history.pth",
    figsize=(15, 5),
)
```

```text
=== Performance Metrics Summary ===
IoU     - Best: 0.9012 | Final: 0.8999
F1      - Best: 0.9466 | Final: 0.9458
Precision - Best: 0.9650 | Final: 0.9605
Recall  - Best: 0.9490 | Final: 0.9326
Val Loss - Best: 0.0699 | Final: 0.0699
===================================
```

### Run Inference

The `semantic_segmentation` function slides a 512x512 window across the input raster with overlap, runs each patch through the model, and stitches predictions into a seamless output. We also generate a probability map storing per-class confidence at every pixel.

```{code-cell} ipython3
masks_path = "naip_buildings_prediction.tif"
probability_path = "naip_test_probability_map.tif"
model_path = f"{out_folder}/unet_models/best_model.pth"
```

```{code-cell} ipython3
geoai.semantic_segmentation(
    input_path=test_raster_path,
    output_path=masks_path,
    model_path=model_path,
    architecture="unet",
    encoder_name="resnet34",
    num_channels=3,
    num_classes=2,
    window_size=512,
    overlap=256,
    batch_size=4,
    probability_path=probability_path
)
```

### Visualize Raster Masks

Overlay the prediction on the original imagery for visual assessment.

```{code-cell} ipython3
geoai.view_raster(
    masks_path,
    nodata=0,
    colormap="binary",
    basemap=test_raster_path,
)
```

In this two-class example, band 2 of the probability map corresponds to the building class, where bright areas indicate high confidence.

```{code-cell} ipython3
geoai.view_raster(
    probability_path, indexes=[2], basemap=test_raster_path
)
```

### Vectorize Predictions

The `orthogonalize` function converts the raster prediction to polygons and regularizes them into clean, right-angled shapes.

```{code-cell} ipython3
output_vector_path = "naip_buildings_prediction.geojson"
gdf = geoai.orthogonalize(masks_path, output_vector_path, epsilon=2)
```

### Add Geometric Properties

Compute area, perimeter, and shape indices for filtering and analysis.

```{code-cell} ipython3
gdf_props = geoai.add_geometric_properties(gdf, area_unit="m2", length_unit="m")
```

### Visualize Results

Display detected buildings colored by area.

```{code-cell} ipython3
geoai.view_vector_interactive(gdf_props, column="area_m2", tiles=test_raster_path)
```

Filter out very small polygons (< 10 m2) to remove noise.

```{code-cell} ipython3
gdf_filtered = gdf_props[(gdf_props["area_m2"] > 10)]
geoai.view_vector_interactive(gdf_filtered, column="area_m2", tiles=test_raster_path)
```

Create a split map comparing predicted footprints (left) with NAIP imagery (right).

```{code-cell} ipython3
geoai.create_split_map(
    left_layer=gdf_filtered,
    right_layer=test_raster_path,
    left_args={"style": {"color": "red", "fillOpacity": 0.2}},
    basemap=test_raster_path,
)
```

Clear the GPU memory cache.

```{code-cell} ipython3
geoai.empty_cache()
```

## Surface Water Mapping

Surface water mapping is essential for ecosystem monitoring, flood risk assessment, and climate research. This section demonstrates four progressively more advanced workflows:

1. **Non-georeferenced satellite imagery** (RGB) introduces the basic workflow.
2. **Sentinel-2 multispectral imagery** (6 bands) demonstrates how additional spectral information improves discrimination.
3. **NAIP aerial imagery** (4 bands, 1-meter resolution) applies the workflow to the highest-resolution data commonly available.
4. **Sensor-agnostic water segmentation** using a pre-trained model that requires no custom training.

### Water Mapping with Non-Georeferenced Satellite Imagery

Working with standard image formats (JPG/PNG) without geographic coordinates is a natural starting point for learning the core computer vision workflow.

#### Download Sample Data

The [waterbody dataset](https://data.source.coop/opengeos/geoai/waterbody-dataset.zip) contains 2,841 image-mask pairs covering diverse geographic regions and water body types.

**Dataset characteristics:**
- **Total image pairs**: 2,841 training examples
- **Image format**: RGB satellite imagery (3 channels)
- **Mask format**: Binary masks (255 = water, 0 = background)
- **Variable image sizes**: 256x256 to 1024x1024+ pixels
- **Global coverage**: Samples from diverse geographic regions and water body types

```{code-cell} ipython3
url = "https://data.source.coop/opengeos/geoai/waterbody-dataset.zip"
```

```{code-cell} ipython3
out_folder = geoai.download_file(url)
print(f"Downloaded dataset to {out_folder}")
```

The unzipped dataset contains `images` and `masks` folders.

+++

#### Train the Model

We use a U-Net architecture with a ResNet34 encoder initialized from ImageNet weights.

Key training parameters:
- `num_channels=3`: RGB input
- `num_classes=2`: binary classification (background vs. water)
- `batch_size=16`: balances GPU memory usage and gradient stability
- `learning_rate=0.001`: standard starting point for Adam optimizer
- `val_split=0.2`: reserves 20% of the data for validation
- `target_size=(512, 512)`: standardizes variable-sized images to a uniform input size

For a complete list of available architectures and encoders, refer to the [segmentation_models.pytorch documentation](https://smp.readthedocs.io/en/latest/encoders.html).

```{code-cell} ipython3
geoai.train_segmentation_model(
    images_dir=f"{out_folder}/images",
    labels_dir=f"{out_folder}/masks",
    output_dir=f"{out_folder}/unet_models",
    architecture="unet",
    encoder_name="resnet34",
    encoder_weights="imagenet",
    num_channels=3,
    num_classes=2,
    batch_size=16,
    num_epochs=20,
    learning_rate=0.001,
    val_split=0.2,
    target_size=(512, 512),
    verbose=True,
)
```

Training produces several output files in the `unet_models` directory:

- `best_model.pth`: the checkpoint with the highest validation IoU
- `final_model.pth`: the checkpoint from the final epoch
- `training_history.pth`: complete training metrics for analysis and plotting
- `training_summary.txt`: a human-readable summary of configuration and results

+++

#### Evaluate the Model

**Loss** measures the discrepancy between predictions and ground truth; both training and validation loss should decrease together.

**IoU (Intersection over Union)** is the standard segmentation metric, defined as the overlap area divided by the union area, ranging from 0.0 to 1.0.

**F1 Score** is the harmonic mean of precision (fraction of predicted positives that are correct) and recall (fraction of actual positives that are found), ranging from 0.0 to 1.0.

```{code-cell} ipython3
geoai.plot_performance_metrics(
    history_path=f"{out_folder}/unet_models/training_history.pth",
    figsize=(15, 5),
    verbose=True,
)
```

```text
=== Performance Metrics Summary ===
IoU     - Best: 0.7079 | Final: 0.7079
F1      - Best: 0.8035 | Final: 0.8035
Precision - Best: 0.8555 | Final: 0.8309
Recall  - Best: 0.8246 | Final: 0.8246
Val Loss - Best: 0.3272 | Final: 0.3440
===================================
```

#### Run Inference on a Single Image

We select a single image for demonstration. Change the `index` variable to try different images.

```{code-cell} ipython3
index = 3
test_image_path = f"{out_folder}/images/water_body_{index}.jpg"
ground_truth_path = f"{out_folder}/masks/water_body_{index}.jpg"
prediction_path = f"{out_folder}/prediction/water_body_{index}.png"
model_path = f"{out_folder}/unet_models/best_model.pth"
```

```{code-cell} ipython3
geoai.semantic_segmentation(
    input_path=test_image_path,
    output_path=prediction_path,
    model_path=model_path,
    architecture="unet",
    encoder_name="resnet34",
    num_channels=3,
    num_classes=2,
    window_size=512,
    overlap=256,
    batch_size=32,
)
```

The side-by-side comparison shows the original image, the predicted water mask, and the ground truth mask.

```{code-cell} ipython3
fig = geoai.plot_prediction_comparison(
    original_image=test_image_path,
    prediction_image=prediction_path,
    ground_truth_image=ground_truth_path,
    titles=["Original", "Prediction", "Ground Truth"],
    figsize=(15, 5),
    save_path=f"{out_folder}/prediction/water_body_{index}_comparison.png",
    show_plot=True,
)
```

+++

#### Run Inference on Multiple Images

The `semantic_segmentation_batch` function processes an entire directory of images with consistent parameters.

```{code-cell} ipython3
url = "https://data.source.coop/opengeos/geoai/waterbody-dataset-sample.zip"
```

```{code-cell} ipython3
data_dir = geoai.download_file(url)
print(f"Downloaded dataset to {data_dir}")
```

Set up the input and output directories for batch processing.

```{code-cell} ipython3
images_dir = f"{data_dir}/images"
predictions_dir = f"{data_dir}/predictions"
```

```{code-cell} ipython3
geoai.semantic_segmentation_batch(
    input_dir=images_dir,
    output_dir=predictions_dir,
    model_path=model_path,
    architecture="unet",
    encoder_name="resnet34",
    num_channels=3,
    num_classes=2,
    window_size=512,
    overlap=256,
    batch_size=4,
    quiet=True,
)
```

Clear the GPU memory cache.

```{code-cell} ipython3
geoai.empty_cache()
```

### Water Mapping with Sentinel-2 Multispectral Imagery

**Sentinel-2** provides 13 spectral bands at 10-20 meter resolution. The additional spectral bands, particularly near-infrared and shortwave infrared, dramatically improve water discrimination because water absorbs strongly at these wavelengths.

The six spectral bands used in this analysis are:

| Band            | Wavelength         | Role in Water Detection                         |
| --------------- | ------------------ | ----------------------------------------------- |
| Blue (490 nm)   | Visible            | Water absorption, atmospheric correction        |
| Green (560 nm)  | Visible            | Water clarity, vegetation health                |
| Red (665 nm)    | Visible            | Land-water contrast                             |
| NIR (842 nm)    | Near-infrared      | Strong water absorption; critical discriminator |
| SWIR1 (1610 nm) | Shortwave infrared | Very low water reflectance; excellent separator |
| SWIR2 (2190 nm) | Shortwave infrared | Separates water from wet soil and vegetation    |

#### Download Sample Data

The Earth Surface Water Dataset contains Sentinel-2 Level 2A imagery with 6 spectral bands and expert-annotated water masks, with separate training and validation splits.

```{code-cell} ipython3
url = "https://data.source.coop/opengeos/geoai/dset-s2.zip"
data_dir = geoai.download_file(url, output_path="dset-s2.zip")
```

The dataset is organized into four directories:

- `tra_scene` / `tra_truth`: training images and corresponding water masks
- `val_scene` / `val_truth`: independent validation images and masks

```{code-cell} ipython3
images_dir = f"{data_dir}/tra_scene"
masks_dir = f"{data_dir}/tra_truth"
tiles_dir = f"{data_dir}/tiles"
```

#### Create Training Tiles

The `export_geotiff_tiles_batch` function processes all image-mask pairs in a directory at once.

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

#### Train the Model

The configuration is identical to the RGB model except for `num_channels=6` for 6-band input.

```{code-cell} ipython3
geoai.train_segmentation_model(
    images_dir=f"{tiles_dir}/images",
    labels_dir=f"{tiles_dir}/masks",
    output_dir=f"{tiles_dir}/unet_models",
    architecture="unet",
    encoder_name="resnet34",
    encoder_weights="imagenet",
    num_channels=6,
    num_classes=2,
    batch_size=32,
    num_epochs=20,
    learning_rate=0.001,
    val_split=0.2,
)
```

#### Evaluate the Model

The Sentinel-2 model achieves a best validation IoU of 0.90, markedly higher than the RGB-only model (0.71), thanks to the additional spectral bands.

```{code-cell} ipython3
geoai.plot_performance_metrics(
    history_path=f"{tiles_dir}/unet_models/training_history.pth",
    figsize=(15, 5),
)
```

```text
=== Performance Metrics Summary ===
IoU     - Best: 0.8989 | Final: 0.8989
F1      - Best: 0.9323 | Final: 0.9323
Precision - Best: 0.9500 | Final: 0.9491
Recall  - Best: 0.9453 | Final: 0.9387
Val Loss - Best: 0.0550 | Final: 0.0550
===================================
```

#### Run Inference on Validation Data

The validation set was held out entirely during training.

```{code-cell} ipython3
images_dir = f"{data_dir}/val_scene"
masks_dir = f"{data_dir}/val_truth"
predictions_dir = f"{data_dir}/predictions"
model_path = f"{tiles_dir}/unet_models/best_model.pth"
```

```{code-cell} ipython3
geoai.semantic_segmentation_batch(
    input_dir=images_dir,
    output_dir=predictions_dir,
    model_path=model_path,
    architecture="unet",
    encoder_name="resnet34",
    num_channels=6,
    num_classes=2,
    window_size=512,
    overlap=256,
    batch_size=32,
    quiet=True,
)
```

#### Visualize Results

A side-by-side comparison using a false-color composite (bands 5-4-3: SWIR1-NIR-Red), the model prediction, and the ground truth mask.

```{code-cell} ipython3
image_id = "S2A_L2A_20190318_N0211_R061"
test_image_path = f"{data_dir}/val_scene/{image_id}_6Bands_S2.tif"
ground_truth_path = f"{data_dir}/val_truth/{image_id}_S2_Truth.tif"
prediction_path = f"{data_dir}/predictions/{image_id}_6Bands_S2_mask.tif"
save_path = f"{data_dir}/{image_id}_6Bands_S2_comparison.png"

fig = geoai.plot_prediction_comparison(
    original_image=test_image_path,
    prediction_image=prediction_path,
    ground_truth_image=ground_truth_path,
    titles=["Original", "Prediction", "Ground Truth"],
    figsize=(15, 5),
    save_path=save_path,
    show_plot=True,
    indexes=[5, 4, 3],
    divider=5000,
)
```

#### Apply the Model to New Sentinel-2 Imagery

We download a Sentinel-2 scene over Minnesota and run the trained model on it as a generalization check.

```{code-cell} ipython3
s2_path = "s2.tif"
url = "https://data.source.coop/opengeos/geoai/s2-minnesota-2025-08-31-subset.tif"
geoai.download_file(url, output_path=s2_path)
```

```{code-cell} ipython3
geoai.view_raster(
    s2_path, indexes=[4, 3, 2], vmin=0, vmax=5000, layer_name="Sentinel-2"
)
```

Define the output path and run inference.

```{code-cell} ipython3
s2_mask = "s2_mask.tif"
model_path = f"{tiles_dir}/unet_models/best_model.pth"
```

```{code-cell} ipython3
geoai.semantic_segmentation(
    input_path=s2_path,
    output_path=s2_mask,
    model_path=model_path,
    architecture="unet",
    encoder_name="resnet34",
    num_channels=6,
    num_classes=2,
    window_size=512,
    overlap=256,
    batch_size=32,
)
```

```{code-cell} ipython3
geoai.view_raster(
    s2_mask,
    nodata=0,
    colormap="binary",
    layer_name="Water",
    basemap=s2_path,
    basemap_args={"indexes": [4, 3, 2], "vmin": 0, "vmax": 5000},
)
```

#### Vectorize Water Mask

Convert the predicted raster mask to vector polygons.

```{code-cell} ipython3
output_vector_path = "s2_water_polygons.geojson"
gdf = geoai.raster_to_vector(
    raster_path=s2_mask,
    output_path=output_vector_path,
    min_area=100,
    simplify_tolerance=None,
)
```

#### Smooth Water Body Polygons

Smooth the polygons for more natural-looking boundaries.

```{code-cell} ipython3
smoothed_path = "s2_water_smoothed.geojson"
gdf = geoai.smooth_vector(
    gdf,
    smooth_iterations=3,
    output_path=smoothed_path,
)
```

#### Add Geometric Properties

Calculate area and perimeter for each detected water body.

```{code-cell} ipython3
gdf_props = geoai.add_geometric_properties(gdf, area_unit="m2", length_unit="m")
gdf_props.head()
```

#### Filter Small Artifacts

Remove small detected regions unlikely to be actual water bodies.

```{code-cell} ipython3
gdf_filtered = gdf_props[gdf_props["area_m2"] > 100]
print(f"Water bodies detected: {len(gdf_filtered)}")
print(f"Removed {len(gdf_props) - len(gdf_filtered)} small artifacts")
```

#### Visualize Water Body Polygons

Display detected water body polygons colored by area.

```{code-cell} ipython3
geoai.view_vector_interactive(
    gdf_filtered,
    column="area_m2",
    opacity=0.5,
    tiles=s2_path,
    tiles_args={"indexes": [4, 3, 2], "vmin": 0, "vmax": 5000},
)
```

#### Split Map Comparison

Compare detected water bodies with the original imagery side by side.

```{code-cell} ipython3
geoai.create_split_map(
    left_layer=gdf_filtered,
    right_layer=s2_path,
    left_args={"style": {"color": "blue", "fillOpacity": 0.3}},
    right_args={"indexes": [4, 3, 2], "vmin": 0, "vmax": 5000},
    basemap=s2_path,
)
```

+++

#### Water Body Area Statistics

Analyze the distribution of water body sizes.

```{code-cell} ipython3
print(gdf_filtered["area_m2"].describe())
```

#### Save Results

Save the final water body polygons.

```{code-cell} ipython3
gdf_filtered.to_file("s2_water_bodies_final.geojson", driver="GeoJSON")
print(f"Saved {len(gdf_filtered)} water body polygons to s2_water_bodies_final.geojson")
```

### Water Mapping with NAIP Aerial Imagery

**NAIP** provides 1-meter resolution aerial photography with four bands (Red, Green, Blue, Near-Infrared). At this resolution, NAIP captures fine-scale features invisible at Sentinel-2's 10-20 meter resolution, such as narrow streams, small ponds, and detailed shoreline geometry.

#### Download Sample Data

The training and test imagery are pre-processed NAIP scenes with corresponding rasterized water masks.

```{code-cell} ipython3
train_raster_url = "https://data.source.coop/opengeos/geoai/naip_water_train.tif"
train_masks_url = "https://data.source.coop/opengeos/geoai/naip_water_masks.tif"
test_raster_url = "https://data.source.coop/opengeos/geoai/naip_water_test.tif"
```

```{code-cell} ipython3
train_raster_path = geoai.download_file(train_raster_url)
train_masks_path = geoai.download_file(train_masks_url)
test_raster_path = geoai.download_file(test_raster_url)
```

Inspect the training raster metadata.

```{code-cell} ipython3
geoai.print_raster_info(train_raster_path, show_preview=False)
```

#### Visualize Sample Data

Overlay the training water mask on the NAIP imagery, then display the test image.

```{code-cell} ipython3
geoai.view_raster(train_masks_path, nodata=0, opacity=0.5, basemap=train_raster_path)
```

```{code-cell} ipython3
geoai.view_raster(test_raster_path)
```

#### Create Training Tiles

Slice the training raster and water mask into overlapping 512x512 tiles.

```{code-cell} ipython3
out_folder = "naip"
```

```{code-cell} ipython3
tiles = geoai.export_geotiff_tiles(
    in_raster=train_raster_path,
    out_folder=out_folder,
    in_class_data=train_masks_path,
    tile_size=512,
    stride=256,
    buffer_radius=0,
)
```

#### Train the Model

The NAIP model uses `num_channels=4` for the four-band input (RGB + NIR).

```{code-cell} ipython3
geoai.train_segmentation_model(
    images_dir=f"{out_folder}/images",
    labels_dir=f"{out_folder}/labels",
    output_dir=f"{out_folder}/models",
    architecture="unet",
    encoder_name="resnet34",
    encoder_weights="imagenet",
    num_channels=4,
    num_classes=2,
    batch_size=8,
    num_epochs=20,
    learning_rate=0.005,
    val_split=0.2,
)
```

#### Evaluate the Model

The model achieves a best validation IoU of 0.97, reflecting the advantage of high-resolution NAIP imagery.

```{code-cell} ipython3
geoai.plot_performance_metrics(
    history_path=f"{out_folder}/models/training_history.pth",
    figsize=(15, 5),
    verbose=True,
)
```

```text
=== Performance Metrics Summary ===
IoU     - Best: 0.9695 | Final: 0.9479
F1      - Best: 0.9817 | Final: 0.9658
Precision - Best: 0.9934 | Final: 0.9648
Recall  - Best: 0.9822 | Final: 0.9811
Val Loss - Best: 0.0105 | Final: 0.0173
===================================
```

#### Run Inference

Apply the trained model to the held-out test image.

```{code-cell} ipython3
masks_path = "naip_water_prediction.tif"
model_path = f"{out_folder}/models/best_model.pth"
```

```{code-cell} ipython3
geoai.semantic_segmentation(
    input_path=test_raster_path,
    output_path=masks_path,
    model_path=model_path,
    architecture="unet",
    encoder_name="resnet34",
    num_channels=4,
    num_classes=2,
    window_size=512,
    overlap=128,
    batch_size=32,
)
```

```{code-cell} ipython3
geoai.view_raster(
    masks_path,
    nodata=0,
    layer_name="Water",
    basemap=test_raster_path,
)
```

#### Vectorize and Analyze Predictions

Convert raster masks to vector polygons and filter by geometric properties.

```{code-cell} ipython3
output_path = "naip_water_prediction.geojson"
gdf = geoai.raster_to_vector(
    masks_path, output_path, min_area=1000, simplify_tolerance=1
)
```

Compute geometric properties and display detected water bodies.

```{code-cell} ipython3
gdf = geoai.add_geometric_properties(gdf)
len(gdf)
```

```{code-cell} ipython3
geoai.view_vector_interactive(gdf, tiles=test_raster_path)
```

Filter out highly elongated features (elongation > 10) to remove false positives like road edges or shadows.

```{code-cell} ipython3
gdf["elongation"].hist()
```

```{code-cell} ipython3
gdf_filtered = gdf[gdf["elongation"] < 10]
```

```{code-cell} ipython3
len(gdf_filtered)
```

#### Visualize Results

Display filtered water body polygons and a split map comparison.

```{code-cell} ipython3
geoai.view_vector_interactive(gdf_filtered, tiles=test_raster_path)
```

```{code-cell} ipython3
geoai.create_split_map(
    left_layer=gdf_filtered,
    right_layer=test_raster_path,
    left_args={"style": {"color": "red", "fillOpacity": 0.2}},
    basemap=test_raster_path,
)
```

### Sensor-Agnostic Water Segmentation

[OmniWaterMask](https://github.com/DPIRD-DMA/OmniWaterMask) provides a pre-trained, sensor-agnostic model that combines deep learning, NDWI calculations, and OpenStreetMap reference data for robust water detection across multiple sensors (Sentinel-2, NAIP, Landsat, PlanetScope, Maxar).

The `geoai.segment_water()` function wraps OmniWaterMask with built-in support for vectorization and polygon smoothing via the [Smoothify](https://github.com/DPIRD-DMA/Smoothify) library.

#### Sentinel-2 Water Segmentation

##### Download Sentinel-2 Data

```{code-cell} ipython3
s2_url = "https://data.source.coop/opengeos/geoai/S2A-L2A-20190318-N0211-R061-6Bands-S2.tif"
s2_path = geoai.download_file(s2_url)
```

##### Visualize Input Data

View the Sentinel-2 scene using a false-color composite (NIR, Red, Green).

```{code-cell} ipython3
geoai.view_raster(s2_path, indexes=[4, 3, 2], vmax=3000)
```

##### Run Water Segmentation

Run `segment_water()` with the `"sentinel2"` band order preset.

```{code-cell} ipython3
s2_mask_path = geoai.segment_water(
    s2_path,
    band_order="sentinel2",
    output_raster="s2_owm_water_mask.tif",
)
```

##### Visualize Raster Mask

```{code-cell} ipython3
geoai.view_raster(
    s2_mask_path,
    nodata=0,
    basemap=s2_path,
    opacity=0.5,
    backend="ipyleaflet",
)
```

##### Vectorize and Smooth Water Bodies

```{code-cell} ipython3
s2_gdf = geoai.segment_water(
    s2_path,
    band_order="sentinel2",
    output_raster="s2_owm_water_mask.tif",
    output_vector="s2_owm_water_bodies.geojson",
    smooth=True,
    smooth_iterations=3,
    min_size=100,
)
```

##### Filter Small Artifacts

```{code-cell} ipython3
s2_filtered = s2_gdf[s2_gdf["area_m2"] > 100]
print(f"Water bodies detected: {len(s2_filtered)}")
print(f"Removed {len(s2_gdf) - len(s2_filtered)} small artifacts")
```

##### Visualize Water Body Polygons

```{code-cell} ipython3
geoai.view_vector_interactive(
    s2_filtered,
    column="area_m2",
    tiles=s2_path,
    tiles_args={"indexes": [4, 3, 2], "vmax": 3000},
)
```

##### Split Map Comparison

```{code-cell} ipython3
geoai.create_split_map(
    left_layer=s2_filtered,
    right_layer=s2_path,
    left_args={"style": {"color": "blue", "fillOpacity": 0.3}},
    right_args={"indexes": [4, 3, 2], "vmax": 3000},
    basemap=s2_path,
)
```

#### NAIP Water Segmentation

##### Download NAIP Data

```{code-cell} ipython3
naip_url = "https://data.source.coop/opengeos/geoai/naip_water_test_subset.tif"
naip_path = geoai.download_file(naip_url)
```

##### Visualize NAIP Imagery

View the NAIP scene using a false-color composite (NIR, Red, Green).

```{code-cell} ipython3
geoai.view_raster(naip_path, indexes=[4, 1, 2])
```

##### Run Water Segmentation

Use `segment_water()` with the `"naip"` band order preset, producing both a raster mask and smoothed vector polygons.

```{code-cell} ipython3
naip_gdf = geoai.segment_water(
    naip_path,
    band_order="naip",
    output_raster="naip_owm_water_mask.tif",
    output_vector="naip_owm_water_bodies.geojson",
    smooth=True,
    smooth_iterations=3,
    min_size=100,
)
```

##### Visualize Raster Mask

```{code-cell} ipython3
geoai.view_raster(
    "naip_owm_water_mask.tif",
    nodata=0,
    basemap=naip_path,
    opacity=0.5,
    backend="ipyleaflet",
)
```

##### Filter Small Artifacts

```{code-cell} ipython3
naip_filtered = naip_gdf[naip_gdf["area_m2"] > 100]
print(f"Water bodies detected: {len(naip_filtered)}")
print(f"Removed {len(naip_gdf) - len(naip_filtered)} small artifacts")
```

##### Visualize Water Body Polygons

```{code-cell} ipython3
geoai.view_vector_interactive(
    naip_filtered,
    column="area_m2",
    tiles=naip_path,
)
```

##### Split Map Comparison

```{code-cell} ipython3
geoai.create_split_map(
    left_layer=naip_filtered,
    right_layer=naip_path,
    left_args={"style": {"color": "blue", "fillOpacity": 0.3}},
    basemap=naip_path,
)
```

```{code-cell} ipython3
geoai.empty_cache()
```

## Cloud and Cloud Shadow Detection

Cloud contamination is one of the most persistent challenges in optical remote sensing. Each pixel is assigned to one of four classes:

| Value | Class        |
| ----- | ------------ |
| 0     | Clear        |
| 1     | Thick Cloud  |
| 2     | Thin Cloud   |
| 3     | Cloud Shadow |

This section uses [OmniCloudMask](https://github.com/DPIRD-DMA/OmniCloudMask) integration in GeoAI, a sensor-agnostic model requiring only Red, Green, and NIR bands.

### Download Sample Data

We use a Sentinel-2 L2A subset over Knoxville, Tennessee.

```{code-cell} ipython3
url = "https://data.source.coop/opengeos/geoai/S2C-MSIL2A-20250920T162001-subset.tif"
s2_path = geoai.download_file(url)
```

### Predict Cloud Mask

The function reads Red (band 1), Green (band 2), and NIR (band 4) and classifies each pixel.

```{code-cell} ipython3
pred_path = "cloud_mask.tif"
geoai.predict_cloud_mask_from_raster(
    input_path=s2_path,
    output_path=pred_path,
    red_band=1,
    green_band=2,
    nir_band=4,
    batch_size=4,
    inference_dtype="bf16",
)
```

### Cloud Statistics

Pixel-level statistics summarize cloud and shadow coverage.

```{code-cell} ipython3
import rasterio as rio
import numpy as np

with rio.open(pred_path) as src:
    pred_array = src.read(1)

stats = geoai.calculate_cloud_statistics(pred_array)
for key, value in stats.items():
    if "percent" in key:
        print(f"{key}: {value:.2f}%")
    else:
        print(f"{key}: {value:,}")
```

```text
total_pixels: 46,930,758
clear_pixels: 36,797,157
thick_cloud_pixels: 6,014,605
thin_cloud_pixels: 485
shadow_pixels: 4,118,511
clear_percent: 78.41%
cloud_percent: 12.82%
shadow_percent: 8.78%
```

+++

### Visualize Results

Display a false-color composite (NIR/Red/Green) alongside the predicted cloud mask.

```{code-cell} ipython3
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

with rio.open(s2_path) as src:
    scene_fc = src.read([4, 1, 2]).astype(np.float32)

# Percentile-based contrast stretch per band
for i in range(3):
    band = scene_fc[i]
    p2, p98 = np.percentile(band, (2, 98))
    scene_fc[i] = (band - p2) / (p98 - p2)
scene_fc = np.clip(scene_fc, 0, 1)

cmap = ListedColormap(["green", "white", "gray", "black"])
labels = ["Clear", "Thick Cloud", "Thin Cloud", "Cloud Shadow"]
patches = [
    Patch(facecolor=cmap(i), edgecolor="black", label=labels[i]) for i in range(4)
]

fig, ax = plt.subplots(1, 2, figsize=(20, 10))
ax[0].imshow(scene_fc.transpose(1, 2, 0))
ax[0].set_title("False Color (NIR/R/G)")
ax[0].set_axis_off()
ax[1].imshow(pred_array, cmap=cmap, vmin=0, vmax=3)
ax[1].set_title("Cloud Mask")
ax[1].legend(handles=patches, loc="upper left", fontsize=12)
ax[1].set_axis_off()
plt.tight_layout()
plt.show()
```

Zoom in on a region with cloud cover to inspect boundary detail.

```{code-cell} ipython3
from scipy import ndimage

cloud_binary = (pred_array == 1) | (pred_array == 2)
labeled, _ = ndimage.label(cloud_binary)
if labeled.max() > 0:
    sizes = ndimage.sum(cloud_binary, labeled, range(1, labeled.max() + 1))
    largest_label = np.argmax(sizes) + 1
    cy, cx = ndimage.center_of_mass(cloud_binary, labeled, largest_label)
    cy, cx = int(cy), int(cx)
else:
    cy, cx = pred_array.shape[0] // 2, pred_array.shape[1] // 2

# Crop a 1000x1000 window around the center
h, w = pred_array.shape
half = 500
r0, r1 = max(0, cy - half), min(h, cy + half)
c0, c1 = max(0, cx - half), min(w, cx + half)

fig, ax = plt.subplots(1, 2, figsize=(20, 10))
ax[0].imshow(scene_fc[:, r0:r1, c0:c1].transpose(1, 2, 0))
ax[0].contour(pred_array[r0:r1, c0:c1], levels=[0.5, 1.5, 2.5], colors="cyan")
ax[0].set_title("False Color with Cloud Contours")
ax[0].set_axis_off()
ax[1].imshow(pred_array[r0:r1, c0:c1], cmap=cmap, vmin=0, vmax=3)
ax[1].set_title("Cloud Mask (Zoomed)")
ax[1].legend(handles=patches, loc="upper left", fontsize=12)
ax[1].set_axis_off()
plt.tight_layout()
plt.show()
```

### Post-Processing

Clean the mask, convert to vector polygons, and smooth boundaries.

```{code-cell} ipython3
cleaned_mask_path = "cleaned_mask.tif"
geoai.clean_raster(pred_path, cleaned_mask_path)
```

```{code-cell} ipython3
cloud_vector = "cloud_vector.geojson"
cloud_gdf = geoai.raster_to_vector(cleaned_mask_path, cloud_vector)
```

```{code-cell} ipython3
smoothed_cloud_gdf = geoai.smooth_vector(
    cloud_gdf, smooth_iterations=3, output_path="smoothed_cloud_vector.geojson"
)
```

### Interactive Visualization

Overlay the smoothed cloud polygons on the satellite imagery.

```{code-cell} ipython3
geoai.view_vector_interactive(
    smoothed_cloud_gdf, tiles=s2_path, tiles_args={"indexes": [4, 1, 2], "vmax": 4000}
)
```

A split map compares the cloud mask overlay with the original imagery.

```{code-cell} ipython3
geoai.create_split_map(
    left_layer=smoothed_cloud_gdf,
    right_layer=s2_path,
    left_args={"style": {"color": "cyan", "fillOpacity": 0.2}},
    right_args={"indexes": [4, 1, 2], "vmax": 4000},
    basemap=s2_path,
)
```

### Geometric Properties

Calculate geometric properties for each cloud polygon.

```{code-cell} ipython3
props_gdf = geoai.add_geometric_properties(cloud_gdf)
props_gdf.describe()
```

### Cloud-Free Mask

Create a binary mask of usable (cloud-free) pixels.

```{code-cell} ipython3
cloud_free = geoai.create_cloud_free_mask(pred_array)
usable_pct = cloud_free.sum() / cloud_free.size * 100
print(f"Usable (cloud-free) pixels: {usable_pct:.2f}%")

fig, ax = plt.subplots(1, 2, figsize=(20, 10))
ax[0].imshow(scene_fc.transpose(1, 2, 0))
ax[0].set_title("False Color (NIR/R/G)")
ax[0].set_axis_off()
ax[1].imshow(cloud_free, cmap="RdYlGn", vmin=0, vmax=1)
ax[1].set_title("Cloud-Free Mask (green = usable)")
ax[1].set_axis_off()
plt.tight_layout()
plt.show()
```

Clear the GPU memory cache.

```{code-cell} ipython3
geoai.empty_cache()
```

## Land Cover Classification

**Land cover classification** extends binary segmentation to **multi-class segmentation**, where every pixel is assigned to one of many categories. The key differences are: `num_classes` increases (13 in this example), the loss function and metrics operate across all classes, and class imbalance may require weighted losses or oversampling.

The classification scheme follows the [Chesapeake Land Cover](https://planetarycomputer.microsoft.com/dataset/chesapeake-lc-13) project with 13 land cover classes. The training data consists of NAIP 4-band aerial imagery paired with rasterized land cover labels.

### Download Sample Data

The training dataset consists of a NAIP image tile and its corresponding 13-class land cover label raster.

```{code-cell} ipython3
train_raster_url = "https://data.source.coop/opengeos/geoai/m_3807511_ne_18_060_20181104.tif"
train_landcover_url = "https://data.source.coop/opengeos/geoai/m_3807511_ne_18_060_20181104_landcover.tif"
test_raster_url = "https://data.source.coop/opengeos/geoai/m_3807511_se_18_060_20181104.tif"
```

```{code-cell} ipython3
train_raster_path = geoai.download_file(train_raster_url)
train_landcover_path = geoai.download_file(train_landcover_url)
test_raster_path = geoai.download_file(test_raster_url)
```

### Visualize Sample Data

Visualize training labels overlaid on the NAIP imagery, then create a split map.

```{code-cell} ipython3
legend_args = {"builtin_legend": "Chesapeake", "title": "Land Cover Type"}
geoai.view_raster(train_landcover_path, basemap=train_raster_path, legend_args=legend_args)
```

```{code-cell} ipython3
geoai.create_split_map(
    left_layer=train_landcover_path,
    right_layer=train_raster_path,
)
```

Display the test image.

```{code-cell} ipython3
geoai.view_raster(test_raster_path)
```

### Create Training Tiles

Label images contain integer class values (0 through 12).

```{code-cell} ipython3
out_folder = "landcover"
```

```{code-cell} ipython3
tiles = geoai.export_geotiff_tiles(
    in_raster=train_raster_path,
    out_folder=out_folder,
    in_class_data=train_landcover_path,
    tile_size=512,
    stride=256,
    buffer_radius=0,
)
```

### Train the Model

The configuration uses `num_classes=13` for the full land cover scheme.

```{code-cell} ipython3
geoai.train_segmentation_model(
    images_dir=f"{out_folder}/images",
    labels_dir=f"{out_folder}/labels",
    output_dir=f"{out_folder}/unet_models",
    architecture="unet",
    encoder_name="resnet34",
    encoder_weights="imagenet",
    num_channels=4,
    num_classes=13,
    batch_size=8,
    num_epochs=20,
    learning_rate=0.001,
    val_split=0.2,
    verbose=True,
    plot_curves=True,
)
```

### Evaluate the Model

For multi-class segmentation, check whether validation IoU and loss continue improving and whether aggregate metrics hide weak performance on minority classes.

```{code-cell} ipython3
geoai.plot_performance_metrics(
    history_path=f"{out_folder}/unet_models/training_history.pth",
    figsize=(15, 5),
    verbose=True,
)
```

### Run Inference

Apply the model to a neighboring NAIP tile not in the training set.

```{code-cell} ipython3
masks_path = "landcover_prediction.tif"
model_path = f"{out_folder}/unet_models/best_model.pth"
```

```{code-cell} ipython3
geoai.semantic_segmentation(
    input_path=test_raster_path,
    output_path=masks_path,
    model_path=model_path,
    architecture="unet",
    encoder_name="resnet34",
    num_channels=4,
    num_classes=13,
    window_size=512,
    overlap=128,
    batch_size=4,
)
```

### Visualize Results

Colorize the predicted class map using the same scheme as the training labels.

```{code-cell} ipython3
geoai.write_colormap(masks_path, train_landcover_path, output=masks_path)
```

```{code-cell} ipython3
geoai.view_raster(masks_path, basemap=test_raster_path, legend_args=legend_args)
```

## Publish and Reuse Models

Sharing a trained model through [Hugging Face Hub](https://huggingface.co/docs/hub) lets collaborators run inference without access to the original training data or compute resources.

### Push to Hugging Face Hub

You need a free Hugging Face account and a write-access token.

```{code-cell} ipython3
from huggingface_hub import notebook_login

notebook_login()
```

Upload the trained model weights and configuration. Replace `"your-username"` with your Hugging Face username.

```{code-cell} ipython3
repo_url = geoai.push_timm_model_to_hub(
    model_path=model_path,
    repo_id="your-username/chesapeake-landcover-unet-resnet34",
    encoder_name="resnet34",
    architecture="unet",
    num_channels=4,
    num_classes=13,
)
print(repo_url)
```

### Run Inference from Hub

The `timm_segmentation_from_hub` function downloads and runs the model automatically -- no local checkpoint needed.

```{code-cell} ipython3
hub_output = "landcover_hub_prediction.tif"
geoai.timm_segmentation_from_hub(
    input_path=test_raster_path,
    output_path=hub_output,
    repo_id="your-username/chesapeake-landcover-unet-resnet34",
    window_size=512,
    overlap=128,
    batch_size=4,
)
```

Apply the same colormap and visualize to confirm results match.

```{code-cell} ipython3
geoai.write_colormap(hub_output, train_landcover_path, output=hub_output)
geoai.view_raster(hub_output, basemap=test_raster_path, legend_args=legend_args)
```

Clear the GPU memory cache.

```{code-cell} ipython3
geoai.empty_cache()
```

## Key Takeaways

1. Semantic segmentation assigns a class label to every pixel, producing dense prediction maps at input resolution.

2. The encoder-decoder architecture with skip connections is the foundation of modern segmentation models.

3. Transfer learning from ImageNet accelerates training by providing pre-trained feature extractors that transfer well to remote sensing.

4. The same workflow applies across sensors; changing `num_channels` is the primary adaptation.

5. Additional spectral bands (especially NIR and SWIR) significantly improve discrimination for targets like water.

6. IoU and F1 score are the standard segmentation evaluation metrics, and should always be computed on held-out data.

7. Post-processing (vectorization, regularization, geometric filtering) transforms raster masks into GIS-ready vector features.

8. Multi-class segmentation extends binary segmentation by increasing `num_classes` and may require class balancing strategies.

9. Publishing models to Hugging Face Hub enables sharing and reuse without local training or data access.
