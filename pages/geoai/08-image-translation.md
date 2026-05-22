---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.18.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
title: "Image Translation"
abstract: "This tutorial introduces image-to-image translation for geospatial data, with a focus on super-resolution. You will learn to enhance Sentinel-2 satellite imagery from 10-meter to 2.5-meter resolution using a latent diffusion model, compute per-pixel uncertainty maps, and process large scenes through tiled inference with overlap blending."
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

# Image Translation

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/opengeos/tnview-website/blob/main/pages/geoai/08-image-translation.ipynb)

## Introduction

Previous tutorials focused on extracting labels or masks from images. Image translation takes a different approach: instead of producing labels, it produces new images by learning a mapping from one image domain to another while preserving spatial structure.

Image-to-image translation encompasses many remote sensing tasks, including sensor translation (e.g., SAR to optical), temporal gap filling, colorization, and synthetic data generation. All share a common structure: a model maps an input image $x$ from domain $A$ to an output image $y$ in domain $B$, where the domains differ in resolution, spectral content, sensor type, or visual style.

Among these, super-resolution is one of the most practically important. Missions like Sentinel-2 provide free global coverage every five days at 10 meters, while commercial satellites achieve sub-meter resolution at much higher cost. Super-resolution uses deep learning to enhance the spatial resolution of freely available imagery, partially bridging this gap.

This tutorial focuses on enhancing Sentinel-2 imagery from 10-meter to 2.5-meter resolution using the LDSR-S2 latent diffusion model. You will download a Sentinel-2 scene, run super-resolution on individual patches and larger regions, visualize enhanced output, and compute uncertainty maps.

## Learning Objectives

By the end of this tutorial, you will be able to:

- Explain what image-to-image translation is and identify its key applications in remote sensing
- Describe the super-resolution problem and why it matters for satellite imagery analysis
- Explain how latent diffusion models perform super-resolution through encoding, denoising, and decoding
- Download and inspect Sentinel-2 multispectral imagery for super-resolution processing
- Run single-patch super-resolution to enhance a 128x128 pixel region by a factor of four in each spatial dimension
- Compare low-resolution inputs with super-resolution outputs to assess enhancement quality
- Compute and interpret per-pixel uncertainty maps using stochastic forward passes
- Process regions larger than a single patch using tiled inference with overlap blending

## Foundations of Image Translation

### What Is Image-to-Image Translation?

Image-to-image translation converts an image from one domain to another, producing a new image rather than categorical labels. The input and output share spatial structure but differ in some property such as resolution, spectral content, or sensor modality.

Paired translation frameworks like Pix2Pix use conditional GANs to learn mappings between aligned image pairs, while CycleGAN extends this to unpaired settings. In remote sensing, these architectures have been adapted for SAR-to-optical translation, cross-sensor harmonization, and map generation.

For geospatial applications, the most important categories of image translation include:

1. **Super-resolution**: Enhancing spatial resolution, such as converting 10-meter Sentinel-2 pixels to 2.5-meter pixels
2. **Sensor translation**: Converting between imaging modalities, such as generating optical imagery from SAR data
3. **Temporal synthesis**: Generating images for dates without direct observations, filling gaps caused by clouds or revisit schedules
4. **Spectral enhancement**: Adding spectral bands to imagery that lacks them, such as predicting multispectral content from RGB inputs
5. **Synthetic data generation**: Creating realistic training imagery for regions where labeled data is unavailable

### Super-Resolution for Remote Sensing

Super-resolution (SR) reconstructs a high-resolution image from one or more low-resolution inputs, producing finer spatial detail than the sensor natively provides. If a model can reliably enhance freely available 10-meter imagery to approach 2.5-meter quality, it unlocks detailed spatial analysis globally at no additional data cost.

Single-image super-resolution (SISR) is the most challenging variant because the model must infer fine-scale details not directly observed. The problem is ill-posed: many high-resolution images could produce the same low-resolution observation when downsampled. Early CNN-based approaches minimized pixel-wise error but produced overly smooth outputs. Generative models (GANs and more recently diffusion models) address this by learning to produce sharp, realistic outputs.

### Latent Diffusion Models for Super-Resolution

Diffusion models generate images by gradually adding noise to a training image (forward phase) and then learning to reverse this corruption step by step (reverse phase). Conditioning the reverse process on a low-resolution input lets the model generate consistent high-resolution outputs with plausible fine-scale detail.

Latent diffusion models (LDMs) reduce computational cost by working in a compressed representation space rather than directly on full-resolution pixels. An encoder maps the image to a lower-dimensional latent space, diffusion operates there, and a decoder maps the result back to pixel space.

The LDSR-S2 model applies latent diffusion to Sentinel-2 super-resolution. It operates on four spectral bands (Red, Green, Blue, and Near-Infrared) and performs 4x spatial upsampling, converting 10-meter pixels to 2.5-meter pixels. The workflow for each 128x128 patch is:

1. **Encode**: The low-resolution patch (128x128, 4 bands) is compressed into a latent representation
2. **Denoise**: The diffusion process runs for a specified number of sampling steps in latent space, conditioned on the low-resolution input
3. **Decode**: The denoised latent representation is decoded back to pixel space at 4x resolution (512x512, 4 bands)

A key advantage of diffusion models is uncertainty quantification. Running the model multiple times with different random seeds produces slightly different outputs, and the standard deviation across these variations provides a per-pixel uncertainty estimate.

## Installation

Uncomment the following line to install the required package.

```{code-cell} ipython3
# %pip install geoai-py
```

## Import Libraries

```{code-cell} ipython3
import geoai
import numpy as np
import rasterio as rio
from matplotlib import pyplot as plt
```

## Download Sample Data

We use a Sentinel-2 Level-2A subset over Knoxville, Tennessee. The image contains four 10-meter bands (Red, Green, Blue, and Near-Infrared) stored as `uint16` bottom-of-atmosphere (BOA) reflectance values in the range 0 to 10,000.

```{code-cell} ipython3
url = "https://data.source.coop/opengeos/geoai/S2C-MSIL2A-20250920T162001-Knoxville.tif"
s2_path = geoai.download_file(url)
```

### Inspecting the Input Data

Inspect the image dimensions, coordinate reference system, and pixel resolution before processing.

```{code-cell} ipython3
with rio.open(s2_path) as src:
    print(f"Bands: {src.count}")
    print(f"Size: {src.width} x {src.height}")
    print(f"CRS: {src.crs}")
    print(f"Resolution: {src.res[0]:.2f} m")
    print(f"Dtype: {src.dtypes[0]}")
```

```text
Bands: 4
Size: 2874 x 1623
CRS: EPSG:3857
Resolution: 10.00 m
Dtype: uint16
```

+++

The image has four bands at 10-meter resolution. The `uint16` data type and 0 to 10,000 value range are standard for Sentinel-2 L2A products, where 1,000 corresponds to a surface reflectance of 0.1 (10%).

## Visualize the Input RGB Composite

Display a true-color composite using the Red, Green, and Blue bands with a percentile-based contrast stretch that maps the 2nd and 98th percentile values to 0 and 1.

```{code-cell} ipython3
with rio.open(s2_path) as src:
    rgb = src.read([1, 2, 3]).astype(np.float32)

for i in range(3):
    band = rgb[i]
    p2, p98 = np.percentile(band, (2, 98))
    rgb[i] = (band - p2) / (p98 - p2)
rgb = np.clip(rgb, 0, 1)

fig, ax = plt.subplots(figsize=(12, 7))
ax.imshow(rgb.transpose(1, 2, 0))
ax.set_title("Sentinel-2 RGB Composite (10 m)")
ax.set_axis_off()
plt.tight_layout()
plt.show()
```

## Single-Patch Super-Resolution

The LDSR-S2 model processes 128x128 pixel patches. The `window` parameter specifies `(row_offset, col_offset, height, width)` in pixel coordinates. The model encodes the patch, runs denoising diffusion, and decodes the result at 4x resolution, producing a 512x512 output at 2.5 meters.

```{code-cell} ipython3
sr_output = "sr_output.tif"
sr_image, _ = geoai.super_resolution(
    input_lr_path=s2_path,
    output_sr_path=sr_output,
    rgb_nir_bands=[1, 2, 3, 4],
    window=(700, 1300, 128, 128),
    sampling_steps=100,
)
```

```{code-cell} ipython3
print(f"Input shape:  (4, 128, 128) at 10 m")
print(f"Output shape: {sr_image.shape} at 2.5 m")
```

```text
Input shape:  (4, 128, 128) at 10 m
Output shape: (4, 512, 512) at 2.5 m
```

+++

### Comparing Low-Resolution and Super-Resolution

Use `plot_sr_comparison` to display a side-by-side RGB comparison of the low-resolution input and super-resolution output.

```{code-cell} ipython3
geoai.plot_sr_comparison(s2_path, sr_output, bands=[1, 2, 3])
plt.show()
```

Verify that the output GeoTIFF has the correct spatial reference and 2.5-meter pixel size.

```{code-cell} ipython3
with rio.open(sr_output) as src:
    print(f"SR Bands: {src.count}")
    print(f"SR Size: {src.width} x {src.height}")
    print(f"SR CRS: {src.crs}")
    print(f"SR Resolution: {src.res[0]:.2f} m")
```

```text
SR Bands: 4
SR Size: 512 x 512
SR CRS: EPSG:3857
SR Resolution: 2.50 m
```

+++

## Uncertainty Estimation

Diffusion-based super-resolution can quantify per-pixel uncertainty. Running the model multiple times with different random seeds produces slightly different outputs, and the standard deviation across these variations yields an uncertainty map.

Set `compute_uncertainty=True` and use `n_variations` to control how many stochastic forward passes to run. More variations produce smoother estimates but increase computation time.

```{code-cell} ipython3
sr_unc_output = "sr_with_uncertainty.tif"
unc_output = "uncertainty.tif"
sr_image2, uncertainty = geoai.super_resolution(
    input_lr_path=s2_path,
    output_sr_path=sr_unc_output,
    output_uncertainty_path=unc_output,
    rgb_nir_bands=[1, 2, 3, 4],
    window=(700, 1300, 128, 128),
    compute_uncertainty=True,
    n_variations=5,
    sampling_steps=100,
)
```

### Visualizing the Uncertainty Map

Display the uncertainty map using pseudocolor visualization. Red/yellow regions indicate higher uncertainty (more variable outputs across seeds), while green regions indicate higher confidence.

High uncertainty often appears along sharp boundaries and in areas with complex fine-scale texture. Homogeneous areas like open water or bare soil typically show low uncertainty.

```{code-cell} ipython3
geoai.plot_sr_uncertainty(unc_output)
plt.show()
```

## Tiled Inference for Larger Regions

For regions larger than 128x128 pixels, the function automatically tiles the input into overlapping patches, processes each independently, and stitches results using linear blending to prevent visible seams.

The `patch_size` parameter sets the tile size and `overlap` controls how many pixels adjacent patches share. Larger overlap produces smoother transitions but increases computation time.

```{code-cell} ipython3
sr_large = "sr_large.tif"
sr_large_img, _ = geoai.super_resolution(
    input_lr_path=s2_path,
    output_sr_path=sr_large,
    rgb_nir_bands=[1, 2, 3, 4],
    window=(700, 1300, 256, 256),
    patch_size=128,
    overlap=16,
    sampling_steps=100,
)
```

```{code-cell} ipython3
print(f"Input shape:  (4, 256, 256) at 10 m")
print(f"Output shape: {sr_large_img.shape} at 2.5 m")
```

```text
Input shape:  (4, 256, 256) at 10 m
Output shape: (4, 1024, 1024) at 2.5 m
```

+++

### Comparing Results for the Larger Region

Compare the low-resolution input with the stitched super-resolution output for the 256x256 region.

```{code-cell} ipython3
geoai.plot_sr_comparison(s2_path, sr_large, bands=[1, 2, 3])
plt.show()
```

### Interactive Split Map Comparison

Use a split map to interactively compare the super-resolution output against high-resolution basemap imagery. Note that the basemap may come from a different acquisition date, so mismatches can reflect temporal differences as well as model errors.

```{code-cell} ipython3
geoai.create_split_map(
    left_layer=sr_large, right_layer="Esri.WorldImagery", left_args={"vmax": 0.3}
)
```

## Limitations and Cautions

Super-resolution models enhance visual detail, but they do not recover true ground information. The fine-scale features are statistical predictions, not direct observations. Building footprints may appear sharper without matching their true outlines, and road edges may be hallucinated where none exist.

This matters most for quantitative analysis. Measuring building areas or delineating field boundaries from super-resolved outputs can introduce systematic errors. Uncertainty maps help flag low-confidence areas, but low uncertainty does not guarantee correctness.

Super-resolution models also carry biases from their training data and may not preserve radiometric relationships needed for vegetation index computation or change detection. Super-resolution is best treated as a visual enhancement tool rather than a replacement for genuinely high-resolution imagery.

## Key Takeaways

1. **Image translation produces new images rather than labels**, learning a mapping between image domains while preserving spatial structure.

2. **Super-resolution enhances spatial detail beyond a sensor's native capability** by inferring plausible fine-scale structure from coarser inputs.

3. **Latent diffusion models perform super-resolution by encoding to latent space, denoising conditioned on the low-resolution input, and decoding back to pixel space.**

4. **Working in latent space reduces computational cost** while maintaining high-quality results on standard hardware.

5. **The LDSR-S2 model performs 4x super-resolution on Sentinel-2 imagery**, processing four bands in 128x128 patches to produce 512x512 outputs at 2.5 meters.

6. **Uncertainty maps quantify per-pixel model confidence** by computing standard deviation across multiple stochastic forward passes.

7. **Tiled inference with overlap blending handles arbitrarily large scenes** by processing overlapping patches and blending them seamlessly.

8. **Output GeoTIFFs preserve full georeferencing**, with the geotransform automatically adjusted for the finer pixel size.
