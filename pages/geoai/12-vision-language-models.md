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
title: "Vision-Language Models"
abstract: "This tutorial explores vision-language models for geospatial scene understanding. You will learn to use Moondream for image captioning, visual question answering, and object detection on aerial and satellite imagery, and build sliding-window workflows for large rasters."
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

# Vision-Language Models

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/opengeos/tnview-website/blob/main/pages/geoai/12-vision-language-models.ipynb)

## Introduction

Vision-language models (VLMs) bridge visual perception and natural language understanding. Unlike traditional computer vision models that output numeric labels or bounding boxes, VLMs can describe scenes in plain language, answer questions about image content, and locate objects from text descriptions.

VLM development builds on progress in both computer vision and natural language processing. The breakthrough came with large-scale contrastive pre-training on image-text pairs, as demonstrated by CLIP in 2021, which showed that a model trained to match images with captions could perform zero-shot classification on hundreds of visual tasks without task-specific training.

For geospatial work, VLMs enable tasks that were impractical with traditional approaches. Instead of training a dedicated classifier to detect swimming pools, you can simply ask a VLM "Are there any swimming pools in this image?" This zero-shot capability dramatically lowers the barrier to extracting information from remote sensing imagery.

VLMs benefit applications such as rapid disaster damage assessment, preliminary site surveys, environmental monitoring, and large-scale land use inventories. Analysts can begin extracting insights from imagery immediately using carefully crafted prompts rather than waiting for a custom model to be trained.

In this tutorial, you will work with Moondream, a lightweight open-source VLM optimized for edge deployment, through the MoondreamGeo class in the geoai library. You will also use CLIP for text-guided segmentation of satellite imagery.

## Learning Objectives

By the end of this tutorial, you will be able to:

- Understand how vision-language models combine visual and textual reasoning
- Initialize and configure `MoondreamGeo` for reproducible geospatial analysis
- Use Moondream for image captioning and visual question answering on satellite imagery
- Detect and locate objects in georeferenced imagery using text-guided prompts
- Use the interactive GUI for exploratory VLM analysis
- Implement a sliding-window approach for applying VLMs to large rasters with configurable combine strategies
- Perform CLIP-based zero-shot segmentation of satellite imagery
- Evaluate the strengths and limitations of VLMs for Earth observation

## How Vision-Language Models Work

Vision-language models learn to associate images with text by training on large datasets of image-text pairs, mapping both modalities into a shared embedding space where related concepts are close together. See this [VLM explainer](https://huggingface.co/blog/vlms) for more details.

**CLIP and contrastive learning.** CLIP (Contrastive Language-Image Pre-training) uses an image encoder and a text encoder trained to maximize similarity between matching image-text pairs while minimizing similarity for non-matching pairs. After training on hundreds of millions of web-sourced image-text pairs, CLIP develops remarkably general visual understanding.

**From CLIP to generative VLMs.** Generative VLMs like Moondream combine a visual encoder (often CLIP-based or SigLIP-based) with a language model decoder. The visual encoder converts an image into feature tokens, which are fed alongside text tokens into the language model to generate free-form text responses.

**Moondream architecture.** Moondream pairs a SigLIP visual encoder with a Phi language model, totaling roughly 2 billion parameters. Despite its small size, it supports captioning, visual question answering, object detection, and point localization.

**Why VLMs matter for geospatial analysis.** Traditional remote sensing pipelines require selecting a task, collecting labeled data, training a model, and deploying it. VLMs collapse this into a single flexible interface where the same model can caption, answer questions, detect objects, and describe spatial relationships through different prompts.

## Setting Up the Environment

Install the required packages and import the libraries used throughout this tutorial.

```{code-cell} ipython3
# %pip install geoai-py transformers==4.57.6
```

Moondream v2 is incompatible with Transformers 5.x, so install version 4.57.6 or another compatible 4.x release.

```{code-cell} ipython3
import geoai
import leafmap
from geoai import MoondreamGeo
```

## Sample Data

Download a georeferenced aerial image of a parking lot containing buildings, vehicles, and vegetation.

```{code-cell} ipython3
url = "https://data.source.coop/opengeos/geoai/parking-lot.tif"
image_path = geoai.download_file(url)
```

Visualize the image on an interactive map.

```{code-cell} ipython3
m = leafmap.Map()
m.add_raster(image_path, layer_name="Satellite Image")
m
```

## Initializing the Moondream Processor

The `MoondreamGeo` class wraps the Moondream model with geospatial support. Pinning the `revision` date locks the model weights to a specific checkpoint for reproducibility. The first run downloads approximately 3.8 GB from [Hugging Face](https://huggingface.co/vikhyatk/moondream2).

```{code-cell} ipython3
processor = MoondreamGeo(
    model_name="vikhyatk/moondream2",
    revision="2025-06-21",
)
```

When using GeoTIFF inputs, `MoondreamGeo` automatically transforms pixel-coordinate outputs into geographic coordinates, returning results as GeoDataFrames ready for GIS workflows.

## Image Captioning

Image captioning generates a natural language description of an image's content. The `caption` method accepts a `length` parameter: `"short"`, `"normal"`, or `"long"`.

```{code-cell} ipython3
result = processor.caption(image_path, length="short")
print(result["caption"])
```

```text
An aerial view of a city parking lot displays two buildings, vehicles, and trees, with a complex network of roads and parking spaces.
```

```{code-cell} ipython3
result = processor.caption(image_path, length="normal")
print(result["caption"])
```

```text
An aerial view showcases a complex of warehouse-like structures, mostly light gray or white, with several parking lots scattered throughout. A wide, paved road runs through the center, with multiple lanes of traffic. A single large tree is visible near the center-left of the image, providing a touch of green amidst the industrial landscape. The parking lots are filled with cars, indicating active human activity in the area. The aerial perspective highlights the intricate layout of the complex.
```

```{code-cell} ipython3
result = processor.caption(image_path, length="long")
print(result["caption"])
```

```text
This aerial image shows a spacious parking lot area surrounded by two large buildings. The parking lot is divided into multiple sections by clearly marked parking spaces, roads, and sidewalks. The parking lots are filled with numerous cars parked neatly in rows, predominantly white and red vehicles. The buildings have flat, gray roofs and appear modern in design. The area is well-maintained, with green trees and shrubs planted around the parking lots, providing a touch of greenery to the urban environment. The image captures a comprehensive view of the parking lot layout, parking conditions, and surrounding structures, offering a clear view of both commercial and residential areas.
```

+++

The `"long"` caption identifies specific features that shorter captions omit. As with any generative model, treat captions as descriptive summaries rather than ground truth.

## Visual Question Answering

Visual question answering (VQA) lets you ask specific questions about an image and receive text answers, which is useful for assessing land cover, counting features, or evaluating conditions.

```{code-cell} ipython3
result = processor.query("How many buildings are in the image?", image_path)
print(result["answer"])
```

```text
There are two buildings in the image.
```

```{code-cell} ipython3
result = processor.query("What are the building roof colors?", image_path)
print(result["answer"])
```

```text
The building roof colors are white and red.
```

```{code-cell} ipython3
result = processor.query("What types of vehicles are visible in the parking areas?", image_path)
print(result["answer"])
```

```text
The parking areas contain various types of vehicles, including cars, trucks, and buses.
```

+++

VQA is well suited to rapid assessment tasks such as querying aerial photos after a natural disaster. Specific, well-scoped questions produce more reliable answers than vague ones, and providing context about altitude or scale can help calibrate responses.

## Object Detection and Point Localization

Moondream can locate objects within images based on text prompts. The `detect` method returns bounding boxes, while `point` returns center-point coordinates. Both methods automatically georeference results from GeoTIFF inputs.

### Detect Buildings

```{code-cell} ipython3
result = processor.detect(image_path, "building", output_path="buildings.geojson")
print(f"Detected {len(result['objects'])} buildings")
```

```{code-cell} ipython3
result["gdf"]
```

Add the georeferenced detections to the map.

```{code-cell} ipython3
style = {"color": "red", "weight": 2}
m.add_gdf(result["gdf"], layer_name="Buildings", style=style)
m
```

### Locate Building Centroids

The `point` method finds the center point of each object, useful for counting features and analyzing spatial distributions.

```{code-cell} ipython3
result = processor.point(
    image_path, "building", output_path="building_centroids.geojson"
)
print(f"Found {len(result['points'])} building centroids")
```

```{code-cell} ipython3
m.add_gdf(result["gdf"], layer_name="Building Centroids")
m
```

### Detect Trees

Apply the same detection approach to locate trees.

```{code-cell} ipython3
result = processor.detect(image_path, "tree", output_path="trees.geojson")
print(f"Detected {len(result['objects'])} trees")
```

```{code-cell} ipython3
m.add_gdf(result["gdf"], layer_name="Trees", style={"color": "green", "weight": 2})
```

### Locate Tree Centroids

```{code-cell} ipython3
result = processor.point(image_path, "tree", output_path="tree_centroids.geojson")
print(f"Found {len(result['points'])} tree centroids")
```

Add the tree centroids to see all detection layers together.

```{code-cell} ipython3
m.add_gdf(result["gdf"], layer_name="Tree Centroids")
m
```

Detection is useful for counting features and identifying spatial distributions. For precise delineation, consider using VLM detections as initial prompts for a segmentation model like SAM.

## Interactive GUI

The `MoondreamGeo` class includes a built-in interactive GUI for exploratory analysis without writing code for each query.

```{code-cell} ipython3
moondream = MoondreamGeo(
    model_name="vikhyatk/moondream2",
    revision="2025-06-21",
)
moondream.load_image(image_path)
m_gui = moondream.show_gui()
m_gui
```

The GUI provides the following controls:

- **Mode**: Switch between Caption, Query, Detect, and Point analysis modes
- **Prompt**: Enter a text prompt for Query, Detect, or Point modes (for example, "building", "green tree", or "How many cars are parked?")
- **Length**: Select caption length (short, normal, long) when using Caption mode
- **Opacity**: Adjust the transparency of detection overlays
- **Color**: Choose the color for bounding boxes and point markers
- **Run**: Execute the selected operation and display results on the map
- **Save**: Export results to a GeoJSON file
- **Reset**: Clear all results and restore the original map view

Detection results appear as bounding boxes and point results appear as circle markers directly on the map. After running an operation, you can access the result as a GeoDataFrame.

```{code-cell} ipython3
gdf = m_gui.last_result_as_gdf
gdf
```

The GUI is particularly useful for exploring imagery, testing prompts, or demonstrating VLM capabilities to stakeholders.

## Sliding Window Analysis for Large Rasters

Satellite and aerial images are often too large to process as a single VLM input. The sliding window approach divides the raster into smaller overlapping tiles, processes each independently, and aggregates the results.

The key parameters are:

- `window_size`: Pixel dimensions of each square tile (default: 512)
- `overlap`: Number of pixels to overlap between adjacent tiles (default: 64)
- `iou_threshold`: Intersection-over-Union threshold for Non-Maximum Suppression when merging overlapping detections (default: 0.5)
- `combine_strategy`: How to merge per-tile text results for query and caption methods (`"concatenate"` or `"summarize"`)

### Object Detection with Sliding Window

Detect objects across all tiles with automatic Non-Maximum Suppression (NMS) to remove duplicates from overlapping regions.

```{code-cell} ipython3
result = processor.detect_sliding_window(
    image_path,
    "car",
    window_size=512,
    overlap=64,
    iou_threshold=0.5,
    output_path="cars_sliding_window.geojson",
)
print(f"Detected {len(result['objects'])} cars")
```

```{code-cell} ipython3
result["gdf"].head()
```

Visualize the sliding window car detections on an interactive map.

```{code-cell} ipython3
m2 = leafmap.Map()
m2.add_raster(image_path, layer_name="Satellite Image")
m2.add_gdf(
    result["gdf"],
    layer_name="Detected Cars",
    style={"color": "red", "fillOpacity": 0.3},
)
m2
```

### Point Detection with Sliding Window

Find specific object locations as points across large images.

```{code-cell} ipython3
trees = processor.point_sliding_window(
    image_path,
    "tree",
    window_size=512,
    overlap=64,
    output_path="trees_sliding_window.geojson",
)
print(f"Found {len(trees['points'])} tree locations")
```

Visualize the detected tree locations on an interactive map.

```{code-cell} ipython3
m3 = leafmap.Map()
m3.add_raster(image_path, layer_name="Satellite Image")
m3.add_gdf(trees["gdf"], layer_name="Trees", style={"color": "green", "radius": 3})
m3
```

### Visual Question Answering with Sliding Window

Query large images tile-by-tile and combine the per-tile answers. The `"concatenate"` strategy joins all tile answers into a single string, preserving all information but potentially reading as repetitive.

```{code-cell} ipython3
result = processor.query_sliding_window(
    "What types of vehicles are visible?",
    image_path,
    window_size=512,
    overlap=64,
    combine_strategy="concatenate",
)
print(result["answer"])
```

```text
Tile 0 (region (0, 0, 512, 512)): The vehicles visible in the image include cars and trucks.
Tile 1 (region (448, 0, 960, 512)): The vehicles visible in the image include cars and trucks.
Tile 2 (region (896, 0, 1408, 512)): The vehicles visible in the image include cars and trucks.
Tile 3 (region (1344, 0, 1659, 512)): Cars are visible in the image.
Tile 4 (region (0, 448, 512, 938)): The vehicles visible in the image include cars and trucks.
Tile 5 (region (448, 448, 960, 938)): The vehicles visible in the image include cars and trucks.
Tile 6 (region (896, 448, 1408, 938)): The vehicles visible in the image include cars and trucks.
Tile 7 (region (1344, 448, 1659, 938)): Cars are visible in the image.
```

+++

The `"summarize"` strategy makes an additional model call to distill per-tile answers into a single coherent response at the cost of extra computation.

```{code-cell} ipython3
result = processor.query_sliding_window(
    "Describe the land use and features in this area.",
    image_path,
    window_size=512,
    overlap=64,
    combine_strategy="summarize",
)
print(result["answer"])
```

```text
The area in the image showcases a diverse range of land use and features, highlighting the adaptability and functionality of different urban environments. The parking lot, situated next to a building, demonstrates a well-organized system for managing the parking needs of vehicles. The parking lot is divided into multiple sections, with some areas having more open space than others, ensuring that vehicles can park efficiently and comfortably. The presence of trees surrounding the parking lot adds a touch of greenery and enhances the overall aesthetic appeal of the urban landscape. The parking lot appears to be well-maintained and organized, offering ample parking space for vehicles and accommodating various car types and sizes.
```

+++

Individual tile answers are accessible for spatial analysis.

```{code-cell} ipython3
for tile in result["tile_answers"][:2]:  # Show first 2 tiles
    print(f"Tile {tile['tile_id']}: {tile['answer']}\n")
```

```text
Tile 0: The area features a large parking lot with numerous cars parked in rows and columns. The parking lot is situated next to a building, possibly a commercial or office building. The parking lot is surrounded by trees, providing a natural element to the urban setting. The parking lot appears to be well-organized, with designated spaces for different types of vehicles.

Tile 1: The area in the image appears to be a parking lot with multiple parking spaces, filled with various cars. The parking lot is surrounded by trees, giving the area a natural and pleasant atmosphere. The parking spaces are arranged in rows and columns, with some areas having more open space than others. The cars are parked at various angles and distances, occupying the entire parking lot. The overall layout and arrangement of the parking spaces suggest a well-organized and efficient parking system for vehicles.
```

### Image Captioning with Sliding Window

Generate comprehensive captions for large images by captioning each tile and combining the results.

```{code-cell} ipython3
result = processor.caption_sliding_window(
    image_path,
    window_size=512,
    overlap=64,
    length="normal",
    combine_strategy="concatenate",
)
print(result["caption"])
```

```{code-cell} ipython3
result = processor.caption_sliding_window(
    image_path,
    window_size=512,
    overlap=64,
    length="long",
    combine_strategy="summarize",
)
print(result["caption"])
```

```{code-cell} ipython3
geoai.empty_cache()
```

### Convenience Functions

For one-off operations without managing a `MoondreamGeo` instance, the geoai library exposes module-level convenience functions.

```{code-cell} ipython3
from geoai import moondream_detect_sliding_window

result = moondream_detect_sliding_window(
    image_path,
    "car",
    window_size=512,
    overlap=64,
    model_name="vikhyatk/moondream2",
    revision="2025-06-21",
)
print(f"Detected {len(result['objects'])} cars")
```

```{code-cell} ipython3
geoai.view_vector_interactive(result["gdf"], tiles=image_path)
```

### Comparing Regular vs. Sliding Window Detection

Processing the full image at once versus tile-by-tile often produces different detection counts, particularly for small or densely packed objects.

```{code-cell} ipython3
processor = MoondreamGeo(
    model_name="vikhyatk/moondream2",
    revision="2025-06-21",
)
```

```{code-cell} ipython3
regular_result = processor.detect(image_path, "car")
print(f"Regular detection: {len(regular_result['objects'])} cars")

sliding_result = processor.detect_sliding_window(
    image_path, "car", window_size=512, overlap=64
)
print(f"Sliding window detection: {len(sliding_result['objects'])} cars")
```

```text
Regular detection: 50 cars
Sliding window detection: 236 cars
```

The sliding window approach finds more cars because small vehicles occupy more pixels within each tile. Full-image inference is often sufficient for large objects like buildings, while sliding windows are more reliable for small features like vehicles.

```{code-cell} ipython3
geoai.empty_cache()
```

### Performance Tips

When tuning sliding window parameters, consider these trade-offs:

- **Window size**: Smaller tiles (256-512 px) improve detection of small objects but increase processing time. Larger tiles (512-1024 px) are faster but may miss fine details.
- **Overlap**: Larger overlaps (64-128 px) reduce missed objects near tile boundaries but increase tile count. Smaller overlaps (32-64 px) are faster but may miss boundary objects.
- **IoU threshold** (detection only): Higher thresholds (0.6-0.8) keep more detections at the cost of potential duplicates. Lower thresholds (0.3-0.5) merge aggressively and may discard genuine nearby objects.
- **Combine strategy**: Use `"concatenate"` for speed or per-tile inspection. Use `"summarize"` when a coherent summary is needed.

## CLIP-Based Segmentation

CLIPSeg generates segmentation masks from text prompts without requiring bounding boxes or point prompts, producing a soft probability map where higher values indicate stronger alignment with the text description.

```{code-cell} ipython3
clip_image_url = "https://data.source.coop/opengeos/geoai/uc-berkeley.tif"
clip_image_path = geoai.download_file(clip_image_url)
```

Visualize the satellite image on an interactive map.

```{code-cell} ipython3
geoai.view_raster(clip_image_path)
```

Initialize the CLIPSeg model with a tile size of 512 pixels and 32-pixel overlap.

```{code-cell} ipython3
segmenter = geoai.CLIPSegmentation(tile_size=512, overlap=32)
```

Define the output path and text prompt for the segmentation.

```{code-cell} ipython3
mask_output_path = "tree_masks.tif"
text_prompt = "trees"
```

Run the segmentation with a probability threshold of 0.5 and Gaussian smoothing.

```{code-cell} ipython3
segmenter.segment_image(
    clip_image_path,
    output_path=mask_output_path,
    text_prompt=text_prompt,
    threshold=0.5,
    smoothing_sigma=1.0,
)
```

Visualize the resulting tree mask overlaid on the original satellite image.

```{code-cell} ipython3
geoai.view_raster(
    mask_output_path,
    nodata=0,
    opacity=0.7,
    colormap="greens",
    layer_name="Trees",
    basemap=clip_image_path,
)
```

Create a split map to compare the segmentation result with the original image.

```{code-cell} ipython3
geoai.create_split_map(
    left_layer=mask_output_path,
    right_layer=clip_image_path,
    left_label="Trees",
    right_label="Satellite Image",
    left_args={"nodata": 0, "opacity": 0.8, "colormap": "greens"},
    basemap=clip_image_path,
)
```

CLIP-based segmentation is useful for rapid land cover mapping when training data is unavailable. The soft probability outputs can be thresholded to produce binary masks or combined across multiple prompts for multi-class mapping, though results are generally less accurate than supervised methods.

## Practical Applications in Earth Observation

VLMs are finding applications across a range of geospatial domains.

**Disaster damage assessment.** VLMs can process thousands of post-event aerial images with queries like "Is the roof of this building intact?" to prioritize areas for detailed assessment.

**Environmental monitoring.** VLMs can assist with identifying illegal mining, detecting deforestation, assessing wetland health, and monitoring coastal erosion through appropriate prompts.

**Urban analysis.** City planners can use VLMs to characterize urban environments at scale, answering questions about building density, road conditions, and green space availability across thousands of image tiles.

**Agricultural monitoring.** VLMs can describe crop conditions, identify field boundaries, and detect irrigation infrastructure without crop-specific training data.

## Limitations and Considerations

**Spatial resolution and scale.** VLMs may struggle with the bird's-eye perspective in satellite imagery, particularly at high altitudes where features occupy only a few pixels.

**Hallucination.** VLMs can produce confident but incorrect responses, describing nonexistent features or miscounting objects. Always verify outputs against reference data.

**Spatial precision.** Bounding boxes and point coordinates are approximate and should not be treated as survey-grade measurements.

**Consistency.** VLMs may produce different outputs for the same input across runs. Structured prompts with constrained answer formats yield more reliable results.

**Prompt sensitivity.** Small changes in wording can produce substantially different responses. Developing standardized prompt templates and validating against ground truth is important for operational workflows.

**Domain gap.** Most VLMs are trained on web imagery rather than remote sensing datasets, so performance can vary across regions, seasons, and sensor types.

## Key Takeaways

1. VLMs bridge visual perception and natural language, enabling flexible zero-shot analysis of geospatial imagery.
2. CLIP learns shared image-text representations through contrastive learning, forming the foundation for many VLM architectures.
3. `MoondreamGeo` wraps Moondream with geospatial support, automatically georeferencing outputs as GeoDataFrames.
4. Moondream provides captioning, VQA, object detection, and point localization through a consistent API.
5. The interactive GUI supports exploratory analysis without writing additional code.
6. The sliding window approach extends VLM capabilities to arbitrarily large rasters with configurable combine strategies.
7. CLIPSeg produces text-guided segmentation masks for rapid land cover mapping without training data.
8. VLM outputs should be validated due to potential hallucination, limited spatial precision, and scale sensitivity.
9. Specific, structured prompts produce more reliable answers than open-ended queries.
