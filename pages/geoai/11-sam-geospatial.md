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
title: "Segment Anything for Geospatial"
abstract: "This tutorial introduces the Segment Anything Model 3 (SAM 3) and its application to geospatial data. You will learn to perform zero-shot, prompt-based segmentation of satellite imagery and video using the samgeo package, including interactive segmentation, batch processing, tiled segmentation for large images, and object tracking across video frames."
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

# Segment Anything for Geospatial

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/opengeos/tnview-website/blob/main/pages/geoai/11-sam-geospatial.ipynb)

## Introduction

Traditional image segmentation in remote sensing requires collecting labeled training data, training a model, and hoping it generalizes to new scenes. Every new geography, sensor, or season could mean starting the labeling process over again.

The Segment Anything Model (SAM), released by Meta AI in April 2023, changed this equation by enabling zero-shot segmentation -- segmenting virtually any object in any image without task-specific training. SAM 3, the latest generation, builds on this foundation with enhanced segmentation quality, multi-modal prompt support, and a streaming memory architecture for tracking objects across video frames.

A single model can delineate building footprints in Nairobi, map agricultural fields in Iowa, and trace river networks in the Amazon without seeing a single labeled example from any of those regions. While SAM 3 does not replace domain expertise, it dramatically accelerates the most labor-intensive phase of many geospatial workflows.

The segment-geospatial Python package (samgeo) bridges SAM 3 and the geospatial world through the SamGeo3 and SamGeo3Video classes. These classes handle georeferenced inputs, coordinate systems, and vector output formats so that predictions integrate directly into GIS workflows.

This tutorial walks through the practical workflows that samgeo provides for SAM 3, including text, point, and box prompts; tiled and batch segmentation; interactive segmentation; and video object tracking.

## Learning Objectives

By the end of this tutorial, you will be able to:

- Explain how SAM 3 works for zero-shot image and video segmentation
- Use `SamGeo3` to segment satellite imagery with text, point, and box prompts
- Save segmentation masks with confidence scores as georeferenced GeoTIFF files
- Process multiple images in batches using a shared workflow
- Segment large GeoTIFF images using the tiled approach
- Segment imagery interactively using the built-in map interface
- Track objects across video frames with `SamGeo3Video`

## How SAM 3 Works

SAM 3's architecture consists of three core image-segmentation components, plus a streaming memory module for video.

**Image encoder.** A Vision Transformer (ViT) processes the input image into a high-dimensional feature embedding. This expensive step runs only once per image and generalizes well to unseen imagery, including satellite and aerial photos.

**Prompt encoder.** The prompt encoder converts user-provided hints (text descriptions, point coordinates, or bounding boxes) into a format the model can use alongside the image embedding. It is lightweight, so different prompts can be tested rapidly against the same cached embedding.

**Mask decoder.** The mask decoder combines the image embedding and encoded prompts to produce final segmentation masks with confidence scores, using a modified Transformer decoder with cross-attention.

**Streaming memory for video.** SAM 3 maintains a memory bank of previously seen frames and their segmentation results. As new frames arrive, the memory bank informs the segmentation so objects can be tracked coherently across hundreds of frames.

This design means the expensive image encoding happens once, and then any number of prompts can be evaluated cheaply against the cached embedding.

## Setting Up the Environment

Install the required package and import the libraries used throughout this tutorial.

```{code-cell} ipython3
# %pip install geoai-py "segment-geospatial[samgeo3]"
```

```{code-cell} ipython3
import os
import geoai
import leafmap
from samgeo import SamGeo3, SamGeo3Video, download_file, show_image
from samgeo.common import raster_to_vector, regularize
```

SAM 3 requires access approval through Hugging Face before first use. Request access at the [SAM 3 model page](https://huggingface.co/facebook/sam3). Once access is granted, authenticate using the Hugging Face login:

```{code-cell} ipython3
# from huggingface_hub import login
# login()
```

## Image Segmentation

The core workflow follows three steps: load an image, generate masks with a prompt, and save the results.

Download a sample satellite image covering the University of California, Berkeley:

```{code-cell} ipython3
url = "https://data.source.coop/opengeos/geoai/uc-berkeley.tif"
image_path = download_file(url)
```

Display the satellite image on an interactive map:

```{code-cell} ipython3
m = leafmap.Map()
m.add_raster(image_path, layer_name="Satellite image")
m
```

Initialize `SamGeo3` and load the image. The `set_image()` method runs the image encoder once and caches the embedding:

```{code-cell} ipython3
sam3 = SamGeo3(backend="meta", device=None, checkpoint_path=None, load_from_HF=True)
sam3.set_image(image_path)
```

### Text-Prompted Segmentation

Generate masks using a natural language text prompt:

```{code-cell} ipython3
sam3.generate_masks(prompt="building")
```

Visualize the detected objects as colored overlays on the original image:

```{code-cell} ipython3
sam3.show_anns()
```

Use `show_masks()` to display only the segmentation masks on a blank background:

```{code-cell} ipython3
sam3.show_masks()
```

Save the masks as a GeoTIFF. Setting `unique=True` assigns a distinct integer value to each segmented object:

```{code-cell} ipython3
sam3.save_masks(output="building_masks.tif", unique=True)
```

Use the `save_scores` parameter to also capture prediction confidence:

```{code-cell} ipython3
sam3.save_masks(
    output="building_masks_with_scores.tif",
    save_scores="building_scores.tif",
    unique=True,
)
```

Display the masks colored by confidence score:

```{code-cell} ipython3
sam3.show_masks(cmap="coolwarm")
```

Visualize the confidence scores on an interactive map:

```{code-cell} ipython3
m.add_raster("building_masks.tif", layer_name="Building masks", visible=False)
m.add_raster(
    "building_scores.tif",
    layer_name="Building scores",
    cmap="coolwarm",
    opacity=0.8,
    nodata=0,
    vmin=0.5,
    vmax=1.0,
)
m.add_colormap(cmap="coolwarm", vmin=0.5, vmax=1.0, label="Confidence score")
m
```

### Box-Prompted Segmentation

A bounding box prompt tells SAM 3 to use the object inside the box as a reference and search for similar objects elsewhere in the image. Specify boxes in `[xmin, ymin, xmax, ymax]` format using geographic coordinates:

```{code-cell} ipython3
# Define boxes in [xmin, ymin, xmax, ymax] format
boxes = [[-122.2597, 37.8709, -122.2587, 37.8717]]
box_labels = [True]  # True=include, False=exclude

sam3.generate_masks_by_boxes(boxes, box_labels, box_crs="EPSG:4326")
```

Overlay the bounding box on the image:

```{code-cell} ipython3
sam3.show_boxes(boxes, box_labels, box_crs="EPSG:4326")
```

Visualize the segmentation masks:

```{code-cell} ipython3
sam3.show_anns()
```

Save the box-prompted segmentation mask as a georeferenced GeoTIFF:

```{code-cell} ipython3
building_mask_path = "building_masks.tif"
sam3.save_masks(output=building_mask_path, unique=True)
```

Overlay the mask on the original satellite image:

```{code-cell} ipython3
geoai.view_raster(
    building_mask_path, nodata=0, opacity=0.7, basemap=image_path
)
```

Free GPU memory before moving to the next section:

```{code-cell} ipython3
geoai.empty_cache()
```

## Point Prompts for Instance Segmentation

Point prompts segment specific objects using foreground points (label=1) and background points (label=0). Initialize `SamGeo3` with `enable_inst_interactivity=True` to enable this mode.

```{code-cell} ipython3
url = "https://data.source.coop/opengeos/geoai/truck-example.jpg"
image_path = download_file(url)
```

Display the image with axis labels for reading pixel coordinates:

```{code-cell} ipython3
show_image(image_path, axis="on")
```

Initialize SAM 3 with instance interactivity and load the image:

```{code-cell} ipython3
sam = SamGeo3(backend="meta", enable_inst_interactivity=True)
sam.set_image(image_path)
```

### Single Point Prompt

Specify a single foreground point in `[x, y]` pixel coordinates:

```{code-cell} ipython3
sam.generate_masks_by_points([[750, 370]])
```

Overlay the point marker on the image:

```{code-cell} ipython3
sam.show_points([[750, 370]], [1])
```

Visualize the resulting segmentation mask:

```{code-cell} ipython3
sam.show_anns()
```

Save the mask to a PNG file:

```{code-cell} ipython3
sam.save_masks("truck_mask.png", unique=True)
```

### Multiple Points

Multiple foreground points on the same object improve coverage for large or irregularly shaped objects:

```{code-cell} ipython3
sam.generate_masks_by_points([[500, 375], [1125, 625]], point_labels=[1, 1])
```

Overlay both point markers:

```{code-cell} ipython3
sam.show_points([[500, 375], [1125, 625]], [1, 1])
```

Visualize the resulting segmentation mask:

```{code-cell} ipython3
sam.show_anns()
```

### Background Points

A background point (label=0) excludes a region from the mask to refine ambiguous boundaries:

```{code-cell} ipython3
sam.generate_masks_by_points([[750, 370], [1125, 625]], point_labels=[1, 0])
```

Overlay the foreground (green) and background (red) point markers:

```{code-cell} ipython3
sam.show_points([[750, 370], [1125, 625]], [1, 0])
```

Visualize the refined segmentation mask:

```{code-cell} ipython3
sam.show_anns()
```

### Multiple Box Prompts

Multiple boxes can be processed simultaneously to segment several objects in one call:

```{code-cell} ipython3
boxes = [
    [75, 275, 1725, 850],  # Whole truck
    [425, 600, 700, 875],  # Rear wheel
    [1375, 550, 1650, 800],  # Front wheel on the passenger side
    [1240, 675, 1400, 750],  # Front wheel on the driver's side
]
sam.generate_masks_by_boxes_inst(boxes)
```

Overlay the bounding boxes on the image:

```{code-cell} ipython3
sam.show_boxes(boxes)
```

Visualize the segmentation masks:

```{code-cell} ipython3
sam.show_anns()
```

Save the multi-box segmentation masks:

```{code-cell} ipython3
sam.save_masks("truck_boxes_mask.png", unique=True)
```

Free GPU memory before proceeding:

```{code-cell} ipython3
geoai.empty_cache()
```

### Batch Point Prompts for Geospatial Data

For georeferenced imagery, supply geographic coordinates as point prompts. The `generate_masks_by_points_patch()` method segments the object at each point and saves georeferenced results.

Download a satellite image and building centroids GeoJSON:

```{code-cell} ipython3
image_url = "https://data.source.coop/opengeos/geoai/wa-building-image.tif"
geojson_url = "https://data.source.coop/opengeos/geoai/wa-building-centroids.geojson"
image_path = download_file(image_url)
geojson_path = download_file(geojson_url)
```

Display the satellite image:

```{code-cell} ipython3
m = leafmap.Map()
m.add_raster(image_path, layer_name="Satellite image")
m
```

Initialize SAM 3 and set the image:

```{code-cell} ipython3
sam = SamGeo3(backend="meta", enable_inst_interactivity=True)
sam.set_image(image_path)
```

Segment buildings at each point location:

```{code-cell} ipython3
point_coords_batch = [
    [-117.599896, 47.655345],
    [-117.59992, 47.655167],
    [-117.599928, 47.654974],
    [-117.599518, 47.655337],
]

sam.generate_masks_by_points_patch(
    point_coords_batch=point_coords_batch,
    point_crs="EPSG:4326",
    output="masks.tif",
    dtype="uint8",
)
```

Overlay the point markers on the image:

```{code-cell} ipython3
sam.show_points(point_coords_batch, point_crs="EPSG:4326")
```

Add the masks to the interactive map:

```{code-cell} ipython3
m.add_raster("masks.tif", cmap="viridis", nodata=0, opacity=0.7, layer_name="Mask")
m
```

You can also supply a GeoJSON file containing point geometries directly:

```{code-cell} ipython3
sam.generate_masks_by_points_patch(
    point_coords_batch=geojson_path,
    point_crs="EPSG:4326",
    output="building_masks.tif",
    dtype="uint16",
)
```

Add the building masks and centroid markers to the map:

```{code-cell} ipython3
m.add_raster(
    "building_masks.tif", cmap="jet", nodata=0, opacity=0.7, layer_name="Building masks"
)
m.add_circle_markers_from_xy(
    geojson_path, radius=3, color="red", fill_color="yellow", fill_opacity=0.8
)
m
```

Free GPU memory before proceeding:

```{code-cell} ipython3
geoai.empty_cache()
```

## Box Prompts for Building Extraction

Box prompts work well for building footprints because buildings have well-defined rectangular extents. This section demonstrates the complete workflow from segmentation through vector export and regularization.

Display the satellite image:

```{code-cell} ipython3
m = leafmap.Map()
m.add_raster(image_path, layer_name="Satellite image")
m
```

Initialize SAM 3:

```{code-cell} ipython3
sam = SamGeo3(backend="meta", enable_inst_interactivity=True)
sam.set_image(image_path)
```

Define bounding boxes around buildings in geographic coordinates:

```{code-cell} ipython3
if m.user_rois is not None:
    boxes = m.user_rois
else:
    boxes = [
        [-117.5995, 47.6518, -117.5988, 47.652],
        [-117.5987, 47.6518, -117.5979, 47.652],
    ]
```

Generate masks using the bounding boxes:

```{code-cell} ipython3
sam.generate_masks_by_boxes_inst(boxes=boxes, box_crs="EPSG:4326")
```

Save the building masks as a GeoTIFF:

```{code-cell} ipython3
sam.save_masks(output="mask.tif", dtype="uint8")
```

Overlay the masks on the map:

```{code-cell} ipython3
m.add_raster("mask.tif", cmap="viridis", nodata=0, opacity=0.5, layer_name="Mask")
m
```

### Using a Vector File as Box Prompts

Instead of specifying individual coordinates, pass a GeoJSON or Shapefile directly:

```{code-cell} ipython3
url = "https://data.source.coop/opengeos/geoai/wa-building-bboxes.geojson"
geojson_path = download_file(url)
```

Overlay the bounding boxes on the satellite image:

```{code-cell} ipython3
m = leafmap.Map()
m.add_raster(image_path, layer_name="Image")
style = {
    "color": "#ffff00",
    "weight": 2,
    "fillColor": "#7c4185",
    "fillOpacity": 0,
}
m.add_vector(geojson_path, style=style, zoom_to_layer=True, layer_name="Bboxes")
m
```

Generate masks for all buildings using the GeoJSON file as box prompts:

```{code-cell} ipython3
output_masks = "building_masks.tif"
sam.generate_masks_by_boxes_inst(
    boxes=geojson_path,
    box_crs="EPSG:4326",
    output=output_masks,
    dtype="uint16",
    multimask_output=False,
)
```

Display the building masks:

```{code-cell} ipython3
m.add_raster(
    output_masks, cmap="jet", nodata=0, opacity=0.5, layer_name="Building masks"
)
m
```

### Converting to Vector and Regularizing

Convert the raster mask to vector polygons:

```{code-cell} ipython3
output_vector = "building_vector.geojson"
raster_to_vector(output_masks, output_vector)
```

The `regularize()` function adjusts polygon geometries to produce cleaner, more regular footprints:

```{code-cell} ipython3
output_regularized = "building_regularized.geojson"
regularize(output_vector, output_regularized)
```

Add the regularized footprints to the map:

```{code-cell} ipython3
m.add_vector(
    output_regularized, style=style, layer_name="Building regularized", info_mode=None
)
m
```

Free GPU memory before proceeding:

```{code-cell} ipython3
geoai.empty_cache()
```

## Batch Segmentation

Batch segmentation processes multiple images with the same prompt in a single workflow.

Download four satellite image tiles covering the UC Berkeley campus:

```{code-cell} ipython3
image_paths = []
for i in range(1, 5):
    url = f"https://data.source.coop/opengeos/geoai/uc-berkeley-{i}.tif"
    image_path = download_file(url)
    image_paths.append(image_path)
```

Display all four tiles on an interactive map:

```{code-cell} ipython3
m = leafmap.Map()
for i, image_path in enumerate(image_paths):
    m.add_raster(image_path, layer_name=f"image_{i + 1}")
m
```

Initialize SAM 3 and load all images as a batch:

```{code-cell} ipython3
sam3 = SamGeo3(backend="meta", device=None, checkpoint_path=None, load_from_HF=True)
sam3.set_image_batch(image_paths)
```

Generate masks for all images using a single text prompt:

```{code-cell} ipython3
sam3.generate_masks_batch("building", min_size=100)
```

```text
Processed 4 image(s), found 174 total object(s).
```

Inspect the number of objects detected in each image:

```{code-cell} ipython3
for i, result in enumerate(sam3.batch_results):
    print(f"Image {i + 1}: Found {len(result['masks'])} objects")
```

```text
Image 1: Found 46 objects
Image 2: Found 50 objects
Image 3: Found 64 objects
Image 4: Found 14 objects
```

+++

Visualize all results in a grid layout:

```{code-cell} ipython3
sam3.show_anns_batch(ncols=2, show_bbox=True, show_score=True, figsize=(12, 8))
```

Save the annotation images to disk:

```{code-cell} ipython3
sam3.show_anns_batch(output_dir="output/annotations/", prefix="ann", dpi=300)
```

Export each image's masks as a separate georeferenced GeoTIFF:

```{code-cell} ipython3
saved_files = sam3.save_masks_batch(
    output_dir="output/", prefix="building_mask", unique=True
)
```

Free GPU memory before proceeding:

```{code-cell} ipython3
geoai.empty_cache()
```

## Tiled Segmentation for Large Images

Large satellite images often exceed GPU memory limits. The `generate_masks_tiled()` method divides the image into overlapping tiles, processes each independently, and merges the results.

Download a large NAIP image for water body detection:

```{code-cell} ipython3
url = "https://data.source.coop/opengeos/geoai/naip_water_train.tif"
image_path = download_file(url)
```

Check the image dimensions:

```{code-cell} ipython3
geoai.print_raster_info(image_path)
```

Initialize SAM 3 and run tiled segmentation:

```{code-cell} ipython3
sam = SamGeo3(backend="meta")

output_path = "segmentation_mask.tif"

sam.generate_masks_tiled(
    source=image_path,
    prompt="water",
    output=output_path,
    tile_size=1024,
    overlap=128,
    min_size=100,
    unique=False,
    dtype="int32",
    verbose=True,
)
```

Visualize the water segmentation results:

```{code-cell} ipython3
m = leafmap.Map()
m.add_raster(image_path, layer_name="Original Image")
m.add_raster(
    output_path, nodata=0, opacity=0.8, cmap="Blues", layer_name="Segmentation Mask"
)
m
```

Convert the raster mask to vector polygons:

```{code-cell} ipython3
vector_path = "segmentation_mask.gpkg"
geoai.raster_to_vector(output_path, vector_path)
```

Smooth the polygon boundaries to reduce pixelation artifacts:

```{code-cell} ipython3
smooth_vector_path = "segmentation_mask_smooth.gpkg"
gdf = geoai.smooth_vector(vector_path, smooth_vector_path)
```

Add the smoothed polygons to the map:

```{code-cell} ipython3
style = {
    "color": "#ff0000",
    "weight": 2,
    "fillOpacity": 0,
}
m.add_gdf(gdf, layer_name="Smoothed Vector", info_mode=None, style=style)
m
```

Free GPU memory before proceeding:

```{code-cell} ipython3
geoai.empty_cache()
```

When tuning tiled segmentation, consider these parameters:

- **tile_size**: Larger tiles capture more context but need more GPU memory. Start with 1024.
- **overlap**: Higher overlap (128-256 pixels) prevents boundary artifacts at the cost of processing time.
- **min_size / max_size**: Use these to filter out noise or irrelevant large regions.
- **dtype**: Use `int32` for many objects, `uint16` for up to 65,535 objects, or `uint8` for binary masks.

## Interactive Segmentation

The `show_map()` method launches an interactive Jupyter widget combining a map interface with SAM 3 segmentation.

```{code-cell} ipython3
url = "https://data.source.coop/opengeos/geoai/uc-berkeley.tif"
image_path = download_file(url)
```

Initialize SAM 3 using the Hugging Face Transformers backend:

```{code-cell} ipython3
sam3 = SamGeo3(
    backend="transformers", device=None, checkpoint_path=None, load_from_HF=True
)
sam3.set_image(image_path)
```

Generate an initial set of masks:

```{code-cell} ipython3
sam3.generate_masks(prompt="building")
sam3.save_masks("masks.tif")
```

Launch the interactive interface. Type a text prompt or draw a rectangle, then click **Segment**:

```{code-cell} ipython3
sam3.show_map(height="700px", min_size=10)
```

The interface supports text prompt mode and bounding box mode. Results update without re-running the image encoder.

Free GPU memory before proceeding:

```{code-cell} ipython3
geoai.empty_cache()
```

## Video Segmentation

`SamGeo3Video` extends SAM 3 to video, supporting MP4 files, directories of JPEG frames, and directories of GeoTIFF files.

### Text-Prompted Video Segmentation

Download a sample video of cars:

```{code-cell} ipython3
url = "https://data.source.coop/opengeos/geoai/cars.mp4"
video_path = download_file(url)
```

Initialize `SamGeo3Video`, load the video, and preview it:

```{code-cell} ipython3
sam = SamGeo3Video()
sam.set_video(video_path)
sam.show_video(video_path)
```

Generate masks using a text prompt:

```{code-cell} ipython3
sam.generate_masks("car")
```

Display the first frame to verify detections:

```{code-cell} ipython3
sam.show_frame(0, axis="on")
```

Display a grid of frames sampled every 20th frame:

```{code-cell} ipython3
sam.show_frames(frame_stride=20, ncols=3)
```

Remove an unwanted object by ID and re-propagate:

```{code-cell} ipython3
sam.remove_object(2)
sam.propagate()
sam.show_frame(0)
```

Save per-frame masks:

```{code-cell} ipython3
os.makedirs("output", exist_ok=True)
sam.save_masks("output/masks")
```

Render the annotated video:

```{code-cell} ipython3
sam.save_video("output/segmented.mp4", fps=25)
```

Close the video session:

```{code-cell} ipython3
sam.close()
```

### Point-Prompted Video Segmentation

Point prompts give precise control over which objects to track.

Initialize a new video session:

```{code-cell} ipython3
sam = SamGeo3Video()
sam.set_video(video_path)
sam.init_tracker()
sam.show_frame(0, axis="on")
```

Add point prompts for each object and propagate:

```{code-cell} ipython3
sam.add_point_prompts([[300, 200]], [1], obj_id=1, frame_idx=0)
sam.add_point_prompts([[420, 200]], [1], obj_id=2, frame_idx=0)
sam.propagate()
```

Display the first frame with tracked masks:

```{code-cell} ipython3
sam.show_frame(0, axis="on")
```

Visualize tracking across sampled frames:

```{code-cell} ipython3
sam.show_frames(frame_stride=20, ncols=3)
```

Use positive and negative points together to refine a mask:

```{code-cell} ipython3
# Positive point on windshield, negative point on car body
sam.add_point_prompts(
    points=[[335, 195], [335, 220]],
    labels=[1, 0],
    obj_id=1,
    frame_idx=0,
)
sam.propagate()
sam.show_frames(frame_stride=20, ncols=3)
```

Save the refined results and close the session:

```{code-cell} ipython3
sam.save_masks("output/masks")
sam.save_video("output/segmented.mp4", fps=25)
sam.close()
```

### Object Tracking

SAM 3 tracks multiple objects across video frames, useful for monitoring vehicles, players, and other moving targets.

Download a basketball video:

```{code-cell} ipython3
url = "https://data.source.coop/opengeos/geoai/basketball.mp4"
video_path = download_file(url)
```

Initialize and preview the video:

```{code-cell} ipython3
sam = SamGeo3Video()
sam.set_video(video_path)
sam.show_video(video_path)
```

Detect and track all players with a text prompt:

```{code-cell} ipython3
sam.generate_masks("player")
```

Create display name labels and visualize the first frame:

```{code-cell} ipython3
player_names = {}
for i in range(15):
    player_names[i] = f"Player {i}"
sam.show_frame(0, axis="on", show_ids=player_names)
```

Remove spurious detections:

```{code-cell} ipython3
sam.remove_object(obj_id=[5, 8, 12, 13])
sam.propagate()
sam.show_frame(0, show_ids=player_names)
```

Save masks, render the annotated video, and preview:

```{code-cell} ipython3
os.makedirs("output", exist_ok=True)
sam.save_masks("output/masks")
sam.save_video("output/players_segmented.mp4", fps=60, show_ids=player_names)
sam.show_video("output/players_segmented.mp4")
```

Close the session and release GPU memory:

```{code-cell} ipython3
sam.close()
sam.shutdown()
```

## Key Takeaways

1. SAM 3 enables zero-shot segmentation of geospatial imagery and video without task-specific training data.

2. Text, point, and box prompts serve different needs, from accessible natural language queries to precise spatial control.

3. Confidence scores let you filter low-quality predictions in downstream processing.

4. Batch and tiled segmentation extend SAM 3 to operational scales and large images that exceed GPU memory.

5. The `show_map()` interactive interface enables rapid exploratory analysis without writing code.

6. `SamGeo3Video` tracks objects coherently across frames using a streaming memory architecture.

7. Georeferencing is preserved throughout the pipeline, so outputs are immediately ready for GIS software.
