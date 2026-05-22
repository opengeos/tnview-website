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
title: "Pixel-Level Regression"
abstract: "This tutorial covers pixel-level regression for geospatial data, where the goal is to predict continuous values for each pixel rather than discrete class labels. You will learn to train regression models for applications such as NDVI prediction, canopy height estimation, and biomass mapping using the geoai package."
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

# Pixel-Level Regression

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/opengeos/tnview-website/blob/main/pages/geoai/10-pixel-regression.ipynb)

## Introduction

Many important Earth science quantities, such as canopy height, biomass, soil moisture, and vegetation indices, are continuous variables rather than categories. Pixel-level regression predicts a single continuous value for every pixel, using encoder-decoder architectures similar to segmentation but with a different output layer, loss function, and evaluation metrics.

This tutorial walks through the complete regression workflow using the geoai package, from data preparation through training, evaluation, and inference. The case study uses Landsat imagery paired with NDVI data to predict vegetation index values.

While classification tells you what is present and segmentation tells you where it is, regression tells you how much of a measurable quantity exists at each location. This quantitative output is directly usable for scientific analysis and policy decisions without the information loss from discretizing continuous variables.

## Learning Objectives

By the end of this tutorial, you will be able to:

- Explain the difference between pixel-level classification and pixel-level regression
- Describe real-world applications of continuous-value prediction from remote sensing imagery
- Adapt encoder-decoder segmentation architectures for regression tasks
- Prepare paired image and target rasters using `geoai.create_regression_tiles`
- Train a pixel-level regression model using `geoai.train_pixel_regressor`
- Evaluate regression results using RMSE, MAE, R-squared, and residual analysis
- Run inference on large rasters with overlapping tiles and blending
- Apply a trained model to new imagery for temporal prediction

## Understanding Pixel Regression

### Classification vs. Regression

In classification, the model selects from a predefined set of discrete labels and uses cross-entropy loss. Regression predicts a continuous numeric value (such as NDVI of 0.65 or canopy height of 12.3 meters) and uses MSE or a variant, penalizing the magnitude of the difference between predicted and actual values.

This distinction has practical implications throughout the pipeline:

- **Output layer**: Classification uses a softmax activation over N classes. Regression uses a single output channel with no activation (or a ReLU to enforce non-negative values).
- **Loss function**: Cross-entropy becomes MSE, MAE, or Huber loss.
- **Evaluation metrics**: Accuracy and IoU become RMSE, MAE, and R-squared.
- **Label format**: Integer class masks become floating-point rasters with continuous values.

### Applications

Pixel-level regression has broad applicability across geospatial domains:

**Vegetation index prediction.** NDVI and other spectral indices can be predicted from multispectral imagery, enabling temporal gap-filling and cross-sensor harmonization.

**Canopy height estimation.** Models trained on LiDAR-derived height maps can generalize to regions where LiDAR is unavailable, providing wall-to-wall height estimates from optical imagery alone.

**Above-ground biomass mapping.** Regression models combine spectral bands, vegetation indices, and texture features to predict biomass density per pixel for carbon accounting and forest management.

**Building height estimation.** Regression models trained on LiDAR-derived heights can predict building heights from widely available aerial or satellite imagery for urban planning and disaster risk assessment.

**Soil property prediction.** Soil moisture, organic carbon, pH, and nutrient concentrations can be estimated from multispectral imagery combined with terrain and climate variables.

These applications share a common workflow: pair imagery with reference measurements from a more direct but geographically limited source, train a regression model, and apply it to produce wall-to-wall predictions where direct measurements are unavailable.

## Regression Architectures

### Adapting Segmentation Models for Regression

Encoder-decoder networks like U-Net, UNet++, DeepLabV3+, and FPN all work for regression with minimal modification. The key change is in the output head: replace the multi-class softmax with a single output channel and no activation (or ReLU for non-negative targets). The `geoai` package handles this automatically by setting `classes=1`.

### Loss Functions for Regression

Choosing the right loss function affects both training stability and prediction quality:

**Mean Squared Error (MSE)** penalizes large errors heavily due to squaring, making it sensitive to outliers but effective for minimizing worst-case errors.

**Mean Absolute Error (MAE, L1 loss)** is more robust to outliers but can slow convergence near the optimum due to constant gradients.

**Huber loss (Smooth L1)** behaves like MSE for small errors and MAE for large errors, providing a balanced compromise. It is a good default when target data may contain noise or extreme values.

The `geoai` package supports all three through the `loss_type` parameter: `"mse"`, `"l1"`, and `"huber"`.

### Output Activation and Scaling

For many geospatial tasks, the target variable has a known physical range. You can incorporate this domain knowledge in two ways:

1. **Target range filtering**: During tile creation, use `target_min` and `target_max` to clip values to the valid range, ensuring clean training data.
2. **Post-prediction clipping**: Clip output values to the valid range during inference using the `clip_range` parameter in `predict_raster`.

In practice, combining both approaches works well.

## Installation

Uncomment the following line to install the required package.

```{code-cell} ipython3
# %pip install geoai-py
```

## Case Study: NDVI Prediction from Landsat Imagery

This case study trains a model to predict NDVI at the pixel level from Landsat imagery. The model is trained on 2022 data from Knoxville, Tennessee, and then applied to 2023 imagery to demonstrate temporal generalization.

### Environment Setup

```{code-cell} ipython3
import geoai
from sklearn.model_selection import train_test_split
```

### Download Data

Download Landsat imagery and NDVI rasters for 2022 (training) and 2023 (testing).

```{code-cell} ipython3
train_raster = geoai.download_file(
    "https://data.source.coop/opengeos/geoai/tn_landsat_2022.tif"
)
train_target = geoai.download_file(
    "https://data.source.coop/opengeos/geoai/tn_ndvi_2022.tif"
)
test_raster = geoai.download_file(
    "https://data.source.coop/opengeos/geoai/tn_landsat_2023.tif"
)
```

### Inspect the Data

Examine the input and target rasters to understand the data dimensions and value ranges.

```{code-cell} ipython3
import rasterio

with rasterio.open(train_raster) as src:
    in_channels = src.count
    print(f"Input shape: {src.height} x {src.width}, {src.count} bands")
    print(f"Input CRS: {src.crs}")
    print(f"Input resolution: {src.res}")

with rasterio.open(train_target) as src:
    print(f"Target shape: {src.height} x {src.width}, {src.count} band(s)")
    target_data = src.read(1)
    print(f"Target value range: [{target_data.min():.2f}, {target_data.max():.2f}]")
```

### Create Training Tiles

Tile the large rasters into smaller patches for training. NDVI values are clipped to [-1, 1], tiles with too many nodata pixels are filtered out, and 50% overlap (`stride=128` with `tile_size=256`) increases training samples.

```{code-cell} ipython3
image_paths, target_paths = geoai.create_regression_tiles(
    input_raster=train_raster,
    target_raster=train_target,
    output_dir="ndvi_tiles",
    tile_size=256,
    stride=128,
    target_band=1,
    min_valid_ratio=0.9,
    target_min=-1.0,
    target_max=1.0,
)
print(f"Created {len(image_paths)} tiles")
```

### Split Data

Split tiles into 80% training and 20% validation sets.

```{code-cell} ipython3
train_imgs, val_imgs, train_tgts, val_tgts = train_test_split(
    image_paths, target_paths, test_size=0.2, random_state=42
)
print(f"Training: {len(train_imgs)}, Validation: {len(val_imgs)}")
```

### Train the Model

Train a U-Net model with a ResNet34 encoder for pixel-level NDVI regression. The function handles model creation, data loading, training with AdamW optimizer, learning rate scheduling, and early stopping.

```{code-cell} ipython3
model = geoai.train_pixel_regressor(
    train_image_paths=train_imgs,
    train_target_paths=train_tgts,
    val_image_paths=val_imgs,
    val_target_paths=val_tgts,
    encoder_name="resnet34",
    architecture="unet",
    in_channels=in_channels,
    output_dir="ndvi_model",
    batch_size=8,
    num_epochs=100,
    learning_rate=1e-3,
    num_workers=0,
    loss_type="mse",
    patience=20,
    devices=1,
    verbose=False,
)
```

### Monitor Training History

Plot training and validation loss and R-squared curves to diagnose model behavior. Both losses should decrease together, with R-squared increasing toward 1.0.

```{code-cell} ipython3
fig, history_df = geoai.plot_training_history(
    log_dir="ndvi_model",
    metrics=["loss", "r2"],
)
```

### Run Inference on Training Area

Run inference on the 2022 raster to evaluate against ground truth. The function uses a sliding window with overlap and Gaussian-weighted blending for seamless predictions.

```{code-cell} ipython3
geoai.predict_raster(
    model=model,
    input_raster=train_raster,
    output_raster="ndvi_model/predicted_ndvi_2022.tif",
    tile_size=256,
    overlap=64,
    batch_size=8,
    clip_range=(-1.0, 1.0),
)
```

### Evaluate Results

Compare predicted NDVI with ground truth. The function produces ground truth, prediction, and residual maps side by side.

```{code-cell} ipython3
fig, metrics = geoai.plot_regression_comparison(
    true_raster=train_target,
    pred_raster="ndvi_model/predicted_ndvi_2022.tif",
    title="NDVI Prediction Results",
    cmap="RdYlGn",
    vmin=-0.2,
    vmax=0.8,
    valid_range=(-1.0, 1.0),
)
```

Create a scatter plot of predicted versus actual values to reveal systematic biases. Points should cluster along the 1:1 line.

```{code-cell} ipython3
fig, metrics = geoai.plot_scatter(
    true_raster=train_target,
    pred_raster="ndvi_model/predicted_ndvi_2022.tif",
    sample_size=50000,
    valid_range=(-1.0, 1.0),
    fit_line=True,
)
```

### Predict on New Data (2023)

Apply the trained model to 2023 Landsat imagery to demonstrate temporal generalization.

```{code-cell} ipython3
geoai.predict_raster(
    model=model,
    input_raster=test_raster,
    output_raster="ndvi_model/predicted_ndvi_2023.tif",
    tile_size=256,
    overlap=64,
    batch_size=8,
    clip_range=(-1.0, 1.0),
)
```

Visualize the 2023 input imagery alongside the predicted NDVI map.

```{code-cell} ipython3
geoai.visualize_prediction(
    input_raster=test_raster,
    pred_raster="ndvi_model/predicted_ndvi_2023.tif",
    cmap="RdYlGn",
    vmin=-0.2,
    vmax=0.8,
)
```

## Evaluation Metrics

Regression models use different metrics than classification models:

**Root Mean Squared Error (RMSE)** measures prediction error standard deviation in the same units as the target variable.

**Mean Absolute Error (MAE)** is the average absolute prediction error, less sensitive to outliers than RMSE.

**R-squared** measures the proportion of variance explained by the model, with values near 1.0 indicating strong performance.

**Pearson correlation** measures the linear relationship between predictions and targets, capturing whether the model ranks pixels correctly even with systematic bias.

Examining residual maps can reveal geographic patterns in model error, such as higher errors near water bodies or in shadowed areas.

## Key Takeaways

1. **Pixel-level regression** predicts continuous values for every pixel, unlike classification and segmentation which assign discrete labels.

2. **Encoder-decoder architectures** for segmentation transfer directly to regression by changing the output to a single channel and switching the loss function.

3. **Loss function selection matters**: MSE penalizes large errors, MAE is robust to outliers, and Huber loss provides a balanced compromise.

4. **Data quality is critical**: use `target_min`/`target_max` during tiling and `clip_range` during inference to enforce valid value ranges.

5. **The `geoai` package** provides a streamlined workflow through `create_regression_tiles`, `train_pixel_regressor`, `predict_raster`, and evaluation functions.

6. **Evaluation uses regression-specific metrics**: RMSE, MAE, R-squared, and correlation, supplemented by residual maps and scatter plots.

7. **Overlapping tiles with blending** produce seamless prediction maps without visible tile boundary artifacts.

8. **Temporal generalization** is possible when the model learns robust feature-to-target mappings, enabling time-series analysis and gap-filling.
