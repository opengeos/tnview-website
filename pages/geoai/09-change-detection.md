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
title: "Change Detection"
abstract: "This tutorial introduces change detection in geospatial imagery, where the goal is to identify meaningful differences between images of the same area captured at different times. You will learn traditional techniques such as image differencing and change vector analysis, then apply the ChangeStar deep learning model through the geoai package to detect building changes in NAIP aerial imagery, compare model variants, adjust detection thresholds, and export georeferenced change maps and vector polygons."
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

# Change Detection

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/opengeos/tnview-website/blob/main/pages/geoai/09-change-detection.ipynb)

## Introduction

Change detection compares images of the same geographic area captured at different times and identifies where meaningful differences have occurred. It is widely used in urban planning, environmental protection, disaster response, and climate monitoring.

The practice is challenging because images from different dates differ not only due to real surface changes but also due to variations in sun angle, atmospheric conditions, and vegetation phenology. Traditional approaches like image differencing and change vector analysis detect changes through pixel-level arithmetic, while deep learning models learn spatial patterns to distinguish meaningful changes from confounding radiometric variations.

This tutorial covers traditional change detection methods on Landsat imagery, then deep learning-based change detection using the ChangeStar model through the geoai package. You will apply ChangeStar to NAIP aerial imagery, compare model variants, adjust detection thresholds, and export georeferenced results.

## Learning Objectives

By the end of this tutorial, you will be able to:

- Explain the principles of bitemporal change detection and distinguish it from multitemporal analysis
- Apply traditional change detection methods including image differencing and change vector analysis
- Describe how siamese networks detect changes by comparing learned feature representations from image pairs
- Use the `geoai` package to run ChangeStar change detection on aerial imagery
- Interpret ChangeStar's three outputs: change map, T1 building segmentation, and T2 building segmentation
- Compare different ChangeStar model variants and adjust probability thresholds to control detection sensitivity
- Export change detection results as georeferenced GeoTIFFs and vector polygons for GIS integration

## Understanding Change Detection

Change detection produces a map highlighting where the surface has changed. The output can be a binary mask (changed vs. unchanged), a categorical map (type of change), or a continuous surface (degree of change).

### Types of Change

**Abrupt changes** occur suddenly and produce dramatic spectral differences, such as building demolition, wildfire, or flooding.

**Gradual changes** unfold over months or years, such as urban sprawl, coastal erosion, or glacier retreat. Detecting them often requires long time series.

**Seasonal changes** are cyclical variations like leaf drop or crop harvesting. A robust system must distinguish these from real surface changes, typically by comparing images from the same season in different years.

**Permanent versus temporary changes** matter for interpretation. New construction is permanent; a flooded field may return to normal within weeks.

### Challenges

- **Geometric misalignment**: If two images are not precisely co-registered, every misaligned pixel appears as a false change.
- **Atmospheric and illumination effects**: Differences in haze, cloud shadows, and sun angle introduce radiometric variations unrelated to surface change.
- **Seasonal and phenological variation**: A forest looks very different in summer versus winter, creating spectral differences unrelated to the changes of interest.
- **Mixed pixels**: At coarser resolutions, small changes within a pixel may be too subtle to detect.

## Traditional Change Detection Methods

These unsupervised methods remain useful as baselines, for quick analysis, and when labeled data is unavailable.

### Image Differencing

Subtract pixel values of one date from another, then threshold the result. Pixels with large absolute differences are classified as changed.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import geoai

# Download Landsat imagery for two dates
url_2023 = "https://data.source.coop/opengeos/geoai/knoxville_landsat_2023.tif"
url_2024 = "https://data.source.coop/opengeos/geoai/knoxville_landsat_2024.tif"
path_2023 = geoai.download_file(url_2023)
path_2024 = geoai.download_file(url_2024)

# Read the NIR band (Band 5 in Landsat 8/9) from both dates
with rasterio.open(path_2023) as src:
    nir_2023 = src.read(5).astype(np.float32)

with rasterio.open(path_2024) as src:
    nir_2024 = src.read(5).astype(np.float32)

# Compute the difference
diff = nir_2024 - nir_2023

# Threshold to identify significant changes
threshold = 2 * np.std(diff)
change_mask = np.abs(diff) > threshold

print(f"Difference range: {diff.min():.2f} to {diff.max():.2f}")
print(f"Threshold: {threshold:.2f}")
print(f"Changed pixels: {change_mask.sum():,} ({100 * change_mask.mean():.1f}%)")
```

```text
Difference range: -0.23 to 0.48
Threshold: 0.02
Changed pixels: 16,286 (3.6%)
```

+++

The threshold choice is critical -- too low and noise floods the map; too high and real changes are missed. Common approaches include standard deviation multiples, Otsu's method, or empirical tuning.

```{code-cell} ipython3
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(diff, cmap="RdBu", vmin=-threshold * 2, vmax=threshold * 2)
axes[0].set_title("NIR Difference (2024 - 2023)")
axes[0].axis("off")
axes[1].hist(diff.ravel(), bins=100, color="steelblue", edgecolor="none")
axes[1].axvline(-threshold, color="red", linestyle="--", label=f"Threshold (±{threshold:.2f})")
axes[1].axvline(threshold, color="red", linestyle="--")
axes[1].set_title("Difference Histogram")
axes[1].legend()
axes[2].imshow(change_mask, cmap="Reds")
axes[2].set_title(f"Change Mask ({100 * change_mask.mean():.1f}% changed)")
axes[2].axis("off")
plt.tight_layout()
plt.show()
```

### Change Vector Analysis

Change Vector Analysis (CVA) extends image differencing to multiple bands. It treats each pixel as a vector in spectral space and computes the magnitude and direction of change between dates.

```{code-cell} ipython3
# Read Red and NIR bands from both dates
with rasterio.open(path_2023) as src:
    red_2023 = src.read(4).astype(np.float32)
    nir_2023 = src.read(5).astype(np.float32)
with rasterio.open(path_2024) as src:
    red_2024 = src.read(4).astype(np.float32)
    nir_2024 = src.read(5).astype(np.float32)

delta_red = red_2024 - red_2023
delta_nir = nir_2024 - nir_2023
magnitude = np.sqrt(delta_red**2 + delta_nir**2)
direction = np.degrees(np.arctan2(delta_nir, delta_red))

mag_threshold = np.percentile(magnitude, 95)
significant_change = magnitude > mag_threshold
print(f"CVA magnitude range: {magnitude.min():.2f} to {magnitude.max():.2f}")
print(f"95th percentile threshold: {mag_threshold:.2f}")
print(f"Significant change pixels: {significant_change.sum():,}")
```

```text
CVA magnitude range: 0.00 to 0.68
95th percentile threshold: 0.03
Significant change pixels: 22,910
```

+++

The direction component provides information about the nature of the change. For example, in Red-NIR space, a transition from vegetation to bare soil produces a change vector pointing in a characteristic direction.

## Deep Learning for Change Detection

Deep learning models learn spatial patterns and contextual information from labeled examples, enabling them to distinguish meaningful surface changes from irrelevant radiometric variations.

### Siamese Networks

Siamese networks process two images through parallel encoder branches with shared weights. Both branches extract features in the same representation space, and a comparison module identifies differences at corresponding spatial locations. Notable architectures include FC-Siam-diff, FC-Siam-conc, and BIT (Binary change detection with a Transformer).

### ChangeStar

ChangeStar jointly performs change detection and semantic segmentation. It produces three outputs: a change mask, and building segmentation maps for both time steps. The model uses Changen2 pre-trained weights, which are learned through a change generation strategy that generalizes well to new regions without fine-tuning. The `geoai` package integrates ChangeStar through the `ChangeStarDetection` class.

## Installation

Uncomment the following line to install the required package.

```{code-cell} ipython3
# %pip install geoai-py
```

## Import Libraries

```{code-cell} ipython3
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import geoai
from geoai.change_detection import (
    ChangeStarDetection,
    changestar_detect,
    list_changestar_models,
)
```

## List Available Models

ChangeStar offers several model variants. The naming convention reflects the pretraining stage and backbone architecture (`vitb` for ViT-Base, `vitl` for ViT-Large).

```{code-cell} ipython3
models = list_changestar_models()
for short_name, full_name in models.items():
    print(f"{short_name:30s} -> {full_name}")
```

```text
s0_s1c1_vitb                   -> s0_init_s1c1_changestar_vitb_1x256
s0_s1c5_vitb                   -> s0_init_s1c5_changestar_vitb_1x256
s0_s9c1_vitb                   -> s0_init_s9c1_changestar_vitb_1x256
s0_xview2_s1c5_vitb            -> s0_init_xView2_ft_s1c5_changestar_vitb_1x256
s1_s1c1_vitb                   -> s1_init_s1c1_changestar_vitb_1x256
s1_s1c1_vitl                   -> s1_init_s1c1_changestar_vitl_1x256
s9_s9c1_vitb                   -> s9_init_s9c1_changestar_vitb_1x256
```

+++

## Setup

Check the available compute device and create an output directory.

```{code-cell} ipython3
device = geoai.get_device()
print(f"Using device: {device}")

out_folder = "changestar_results"
Path(out_folder).mkdir(exist_ok=True)
print(f"Output directory: {out_folder}")
```

## Download Sample Data

We use NAIP aerial imagery of Las Vegas captured in 2019 and 2022. Las Vegas experienced rapid suburban development during this period, making it a good test case for building change detection.

```{code-cell} ipython3
naip_2019_url = "https://data.source.coop/opengeos/geoai/las_vegas_naip_2019_a.tif"
naip_2022_url = "https://data.source.coop/opengeos/geoai/las_vegas_naip_2022_a.tif"

naip_2019_path = geoai.download_file(naip_2019_url)
naip_2022_path = geoai.download_file(naip_2022_url)

print(f"Downloaded 2019 NAIP: {naip_2019_path}")
print(f"Downloaded 2022 NAIP: {naip_2022_path}")
```

## Visualize Input Imagery

Create a split map of the 2019 and 2022 NAIP images to inspect the study area.

```{code-cell} ipython3
geoai.create_split_map(
    left_layer=naip_2019_path,
    right_layer=naip_2022_path,
    left_label="NAIP 2019",
    right_label="NAIP 2022",
)
```

## Initialize the ChangeStar Model

Create a `ChangeStarDetection` instance with the `s1_s1c1_vitb` variant (ViT-Base backbone with stage-1 Changen2 pretraining). Weights are downloaded automatically on first use.

```{code-cell} ipython3
detector = ChangeStarDetection(model_name="s1_s1c1_vitb")
```

## Run Change Detection

The `predict` method takes two GeoTIFF paths and returns a dictionary with the binary change map, change probabilities, and segmentation results. Large images are automatically tiled with overlap averaging.

```{code-cell} ipython3
result = detector.predict(
    naip_2019_path,
    naip_2022_path,
    output_change=os.path.join(out_folder, "change_map.tif"),
    output_t1_semantic=os.path.join(out_folder, "t1_buildings.tif"),
    output_t2_semantic=os.path.join(out_folder, "t2_buildings.tif"),
    output_vector=os.path.join(out_folder, "changes.gpkg"),
)
```

Inspect the result dictionary.

```{code-cell} ipython3
print("Result keys:", list(result.keys()))
for key, value in result.items():
    if hasattr(value, "shape"):
        print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
    else:
        print(f"  {key}: {value}")
```

```text
Result keys: ['change_map', 'change_prob', 't1_semantic', 't2_semantic', 't1_semantic_prob', 't2_semantic_prob', 'num_semantic_classes']
  change_map: shape=(1419, 2721), dtype=uint8
  change_prob: shape=(1419, 2721), dtype=float32
  t1_semantic: shape=(1419, 2721), dtype=uint8
  t2_semantic: shape=(1419, 2721), dtype=uint8
  t1_semantic_prob: shape=(1419, 2721), dtype=float32
  t2_semantic_prob: shape=(1419, 2721), dtype=float32
  num_semantic_classes: 1
```

+++

## Visualize Results

The `visualize` method displays a five-panel figure showing the before image, after image, change map, and building segmentation for both time steps.

```{code-cell} ipython3
fig = detector.visualize(
    naip_2019_path,
    naip_2022_path,
    result=result,
    figsize=(25, 10),
    title1="NAIP 2019",
    title2="NAIP 2022",
)
plt.show()
```

### Overlay Visualization

The `visualize_overlay` method shows building segmentation (blue) and changes (red) overlaid on the original imagery.

```{code-cell} ipython3
fig = detector.visualize_overlay(
    naip_2019_path,
    naip_2022_path,
    result=result,
    figsize=(20, 6),
    title1="NAIP 2019",
    title2="NAIP 2022",
)
plt.show()
```

## Using the Convenience Function

For quick one-off analysis, `changestar_detect()` initializes the model, runs prediction, and returns results in a single call.

```{code-cell} ipython3
result2 = changestar_detect(
    naip_2019_path,
    naip_2022_path,
    model_name="s1_s1c1_vitb",
    output_change=os.path.join(out_folder, "change_map_v2.tif"),
)

print(f"Change pixels: {result2['change_map'].sum():,}")
print(
    f"Changed area: {result2['change_map'].sum() / result2['change_map'].size * 100:.2f}%"
)
```

## Comparing Model Variants

Different ChangeStar variants use different pretraining strategies. Here we compare the `s1` and `s9` variants.

```{code-cell} ipython3
result_s1 = result

detector_s9 = ChangeStarDetection(model_name="s9_s9c1_vitb")
result_s9 = detector_s9.predict(naip_2019_path, naip_2022_path)
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(result_s1["change_map"], cmap="gray")
axes[0].set_title(f"S1 Model\n(Changed pixels: {result_s1['change_map'].sum():,})")
axes[0].axis("off")

axes[1].imshow(result_s9["change_map"], cmap="gray")
axes[1].set_title(f"S9 Model\n(Changed pixels: {result_s9['change_map'].sum():,})")
axes[1].axis("off")

combined = np.zeros((*result_s1["change_map"].shape, 3), dtype=np.uint8)
combined[result_s1["change_map"] == 1, 0] = 255  # S1 in red
combined[result_s9["change_map"] == 1, 2] = 255  # S9 in blue
both = (result_s1["change_map"] == 1) & (result_s9["change_map"] == 1)
combined[both] = [255, 0, 255]

axes[2].imshow(combined)
axes[2].set_title("Comparison\n(Red=S1 only, Blue=S9 only, Magenta=Both)")
axes[2].axis("off")

plt.tight_layout()
plt.show()
```

## Adjusting the Threshold

The default probability threshold is 0.5. Lowering it increases sensitivity (more detections, more false alarms); raising it increases specificity (fewer detections, higher confidence).

```{code-cell} ipython3
thresholds = [0.3, 0.5, 0.7]
fig, axes = plt.subplots(1, len(thresholds), figsize=(18, 6))

for ax, thresh in zip(axes, thresholds):
    result_t = detector.predict(naip_2019_path, naip_2022_path, threshold=thresh)
    ax.imshow(result_t["change_map"], cmap="gray")
    ax.set_title(
        f"Threshold = {thresh}\n"
        f"(Changed pixels: {result_t['change_map'].sum():,})"
    )
    ax.axis("off")

plt.tight_layout()
plt.show()
```

## View Saved Outputs

The saved GeoTIFFs and GeoPackage can be loaded directly into any GIS software.

```{code-cell} ipython3
for f in sorted(os.listdir(out_folder)):
    fpath = os.path.join(out_folder, f)
    size_mb = os.path.getsize(fpath) / 1024 / 1024
    print(f"{f:40s} {size_mb:.2f} MB")
```

```text
change_map.tif                           0.11 MB
change_map_v2.tif                        0.11 MB
changes.gpkg                             0.61 MB
t1_buildings.tif                         0.10 MB
t2_buildings.tif                         0.13 MB
```

+++

Display the saved change map on an interactive map.

```{code-cell} ipython3
change_map_path = os.path.join(out_folder, "change_map.tif")
geoai.view_raster(
    change_map_path,
    nodata=0,
    cmap="Reds",
    opacity=0.8,
    basemap=naip_2019_path,
    backend="ipyleaflet",
)
```

## Key Takeaways

1. Change detection identifies meaningful differences between multi-temporal images of the same area.

2. Traditional methods like image differencing and CVA are unsupervised and interpretable but treat each pixel independently.

3. Deep learning siamese networks compare image pairs in feature space, distinguishing real changes from radiometric variations.

4. ChangeStar jointly performs change detection and semantic segmentation, producing a change map and building footprints for both time steps.

5. Multiple model variants and adjustable thresholds control the balance between sensitivity and specificity.

6. The `geoai` package provides a complete change detection workflow with georeferenced GeoTIFF and vector polygon output.
