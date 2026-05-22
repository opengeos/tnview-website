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
---

# Interactive Mapping & Visualization

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/opengeos/tnview-website/blob/main/pages/geoai/02-visualization.ipynb)

## Introduction

Visualization is essential at every stage of a GeoAI workflow, from inspecting imagery and verifying labels before training to evaluating predictions and communicating results afterward.

Static plots work for simple checks, but geospatial data demands interactivity so you can pan, zoom, toggle layers, and compare datasets side by side.

The leafmap library provides a Pythonic interface for building interactive maps in Jupyter notebooks, with convenient functions for loading raster, vector, and cloud-hosted geospatial data.

This tutorial covers core visualization techniques: creating interactive maps, adding raster and vector layers, building split-panel comparisons, and overlaying model predictions on source imagery. All examples use datasets from a Las Vegas building detection project, including NAIP aerial imagery, a LiDAR-derived height above ground (HAG) raster, building footprint annotations, and a rasterized building mask.

## Learning Objectives

By the end of this tutorial, you will be able to:

- Create interactive maps with leafmap and customize their appearance
- Add raster data (GeoTIFF, COG) to maps with appropriate colormaps and band combinations
- Visualize vector data (GeoJSON, GeoDataFrame) with custom styling
- Build split-panel maps to compare datasets or time periods
- Overlay model predictions on source imagery for visual evaluation
- Visualize cloud-hosted data from Microsoft Planetary Computer

## Installation

Uncomment the following line to install the required package.

```{code-cell} ipython3
# %pip install geoai-py
```

## Creating Interactive Maps with Leafmap

### Your First Map

Call `leafmap.Map()` to create an interactive map widget with pan, zoom, and layer controls. You can configure the map center, zoom level, and display dimensions.

```{code-cell} ipython3
import leafmap

m = leafmap.Map(center=[36.1617, -115.1524], zoom=11, height="600px")
m
```

The `center` parameter takes a `[latitude, longitude]` pair, `zoom` controls the initial zoom level, and `height` sets the widget height.

### Adding Basemaps

Basemaps provide geographic context that makes geospatial visualizations meaningful. `leafmap` provides access to hundreds of basemaps through the `add_basemap()` method.

```{code-cell} ipython3
m = leafmap.Map()
m.add_basemap("Esri.WorldImagery")
m
```

To see all available basemaps, inspect the `leafmap.basemaps` dictionary:

```{code-cell} ipython3
basemaps = list(leafmap.basemaps.keys())
print(f"Total basemaps: {len(basemaps)}")
print("First 10:", basemaps[:10])
```

You can also add multiple basemaps and switch between them using the layer control:

```{code-cell} ipython3
m = leafmap.Map()
m.add_basemap("Esri.WorldImagery")
m.add_basemap("OpenTopoMap")
m
```

## Working with Raster Data on Maps

Raster data forms the backbone of most GeoAI workflows. `leafmap` provides several methods for adding raster data to maps, each optimized for different data sources and access patterns.

### Adding GeoTIFF Layers

The `add_raster()` method loads a local or remote GeoTIFF and displays it on the map with a colormap of your choice. Throughout this tutorial, we use four datasets from a Las Vegas building detection project:

- **NAIP aerial imagery** (4-band GeoTIFF): Red, Green, Blue, and Near-Infrared bands at 60 cm resolution
- **Height Above Ground (HAG)** (single-band GeoTIFF): LiDAR-derived surface height in meters
- **Building footprints** (GeoJSON): Vector polygons outlining individual buildings
- **Building mask** (single-band GeoTIFF): A rasterized binary mask where 1 indicates building pixels and 0 indicates background

```{code-cell} ipython3
import geoai
```

We define the URLs for each dataset and use `geoai.download_file()` to fetch them locally:

```{code-cell} ipython3
naip_url = "https://data.source.coop/opengeos/geoai/las-vegas-train-naip.tif"
hag_url = "https://data.source.coop/opengeos/geoai/las-vegas-train-hag.tif"
buildings_url = (
    "https://data.source.coop/opengeos/geoai/las-vegas-buildings-train.geojson"
)
buildings_mask_url = (
    "https://data.source.coop/opengeos/geoai/las-vegas-buildings-mask.tif"
)
```

```{code-cell} ipython3
naip_path = geoai.download_file(naip_url)
hag_path = geoai.download_file(hag_url)
buildings_path = geoai.download_file(buildings_url)
mask_path = geoai.download_file(buildings_mask_url)
```

With the data downloaded, we add the NAIP imagery as a raster layer:

```{code-cell} ipython3
m = leafmap.Map()
m.add_raster(naip_path, layer_name="NAIP Image")
m
```

For single-band rasters, you can specify a colormap and value range using `vmin` and `vmax`. Here we add the height above ground raster, capping the display at 10 meters so building-scale structures stand out:

```{code-cell} ipython3
m.add_raster(
    hag_path, vmin=0, vmax=10, colormap="terrain", layer_name="Height Above Ground"
)
```

`add_raster()` accepts both local file paths and remote URLs. Use the layer control to toggle between layers.

### Cloud-Optimized GeoTIFF (COG)

Cloud-Optimized GeoTIFFs let you stream raster data directly from remote servers without downloading the entire file.

```{code-cell} ipython3
m = leafmap.Map()
m.add_cog_layer(naip_url, name="Las Vegas NAIP")
m
```

COGs are particularly useful when working with large archives hosted on cloud platforms like the Planetary Computer or AWS Open Data.

### Band Combinations

The `indexes` parameter in `add_raster()` lets you select which bands to display and in what order. Placing the NIR band in the red channel makes vegetation appear bright red, helping distinguish vegetated areas from impervious surfaces:

```{code-cell} ipython3
m = leafmap.Map()
m.add_raster(naip_path, indexes=[4, 1, 2], layer_name="False Color")
m
```

Common NAIP band combinations include:

- **True color** (bands 1, 2, 3): Natural appearance
- **False color** (bands 4, 1, 2): Vegetation appears bright red, useful for distinguishing buildings from vegetation

## Visualizing Data from Planetary Computer

Microsoft Planetary Computer hosts petabytes of geospatial data accessible through a STAC API. The `geoai` library provides convenience functions to search, visualize, and download data from this catalog.

### Browsing Available Collections

The `pc_collection_list()` function retrieves all available collections as a DataFrame.

```{code-cell} ipython3
collections = geoai.pc_collection_list()
print(f"Total collections: {len(collections)}")
collections.head(10)
```

### Searching for STAC Items

Use `pc_stac_search()` to find items matching your spatial and temporal criteria. Here we search for NAIP imagery covering Baltimore:

```{code-cell} ipython3
naip_items = geoai.pc_stac_search(
    collection="naip",
    bbox=[-76.6657, 39.2648, -76.6478, 39.2724],
    time_range="2013-01-01/2014-12-31",
)
naip_items
```

```text
Found 1 items matching search criteria
[<Item id=md_m_3907643_se_18_1_20130917_20131112>]
```

You can list the available assets for any item:

```{code-cell} ipython3
geoai.pc_item_asset_list(naip_items[0])
```

```text
['image', 'metadata', 'thumbnail', 'tilejson', 'rendered_preview']
```

### Visualizing NAIP Imagery

The `view_pc_item()` function renders a STAC item on an interactive map by streaming tiles, with no download required.

```{code-cell} ipython3
geoai.view_pc_item(item=naip_items[0])
```

### Visualizing Land Cover Data

Here we search for the Chesapeake Bay high-resolution land cover dataset and display it with a categorical colormap:

```{code-cell} ipython3
lc_items = geoai.pc_stac_search(
    collection="chesapeake-lc-13",
    bbox=[-76.6657, 39.2648, -76.6478, 39.2724],
    time_range="2013-01-01/2014-12-31",
    max_items=10,
)
lc_items
```

```{code-cell} ipython3
geoai.view_pc_item(item=lc_items[0], colormap_name="tab10", basemap="SATELLITE")
```

Categorical datasets like land cover work well with qualitative colormaps such as `"tab10"` or `"Set3"`.

### Visualizing Landsat Imagery

For multispectral data like Landsat, you can select specific bands or compute band math on the fly:

```{code-cell} ipython3
landsat_items = geoai.pc_stac_search(
    collection="landsat-c2-l2",
    bbox=[-76.6657, 39.2648, -76.6478, 39.2724],
    time_range="2024-10-27/2024-12-31",
    query={"eo:cloud_cover": {"lt": 1}},
    max_items=10,
)
landsat_items
```

```{code-cell} ipython3
geoai.pc_item_asset_list(landsat_items[0])
```

A true color composite using the red, green, and blue bands:

```{code-cell} ipython3
geoai.view_pc_item(item=landsat_items[0], assets=["red", "green", "blue"])
```

A false-color composite places the near-infrared band in the red channel, making vegetation appear bright red:

```{code-cell} ipython3
geoai.view_pc_item(item=landsat_items[0], assets=["nir08", "red", "green"])
```

You can also compute spectral indices using the `expression` parameter:

```{code-cell} ipython3
geoai.view_pc_item(
    item=landsat_items[0],
    expression="(nir08-red)/(nir08+red)",
    rescale="-1,1",
    colormap_name="greens",
    name="NDVI",
)
```

The `expression` parameter accepts band math using asset names as variables, and `rescale` controls the value range. Computation is handled server-side, so no local processing is needed.

### Downloading Data from Planetary Computer

Use `pc_stac_download()` to download specific assets to disk:

```{code-cell} ipython3
geoai.pc_stac_download(naip_items, output_dir="data", assets=["image", "thumbnail"])
```

You can also read an asset directly into an xarray DataArray without saving it:

```{code-cell} ipython3
ds = geoai.read_pc_item_asset(lc_items[0], asset="data")
ds
```

## Working with Vector Data on Maps

Vector data serves as training labels, reference boundaries, and model outputs in GeoAI. `leafmap` provides flexible methods for adding vector layers from local files, remote URLs, and in-memory GeoDataFrames.

### Adding GeoJSON and GeoDataFrame

You can visualize vector data from GeoJSON files, URLs, or GeoDataFrames at any stage of your workflow.

```{code-cell} ipython3
import geopandas as gpd

gdf = gpd.read_file(buildings_path)
print(f"Features: {len(gdf)}")
gdf.head()
```

```{code-cell} ipython3
m = leafmap.Map()
m.add_raster(naip_path)
m.add_gdf(gdf, layer_name="Building Footprints", zoom_to_layer=True)
m
```

You can also add GeoJSON directly from a URL without first loading it into a GeoDataFrame:

```{code-cell} ipython3
m = leafmap.Map()
m.add_raster(naip_path)
m.add_geojson(buildings_url, layer_name="Buildings", zoom_to_layer=True)
m
```

### Styling Vector Layers

Custom styling helps distinguish feature types and highlight important patterns. You can control fill color, outline color, line width, and opacity:

```{code-cell} ipython3
style = {
    "color": "red",
    "weight": 2,
    "fillColor": "yellow",
    "fillOpacity": 0.3,
}
m = leafmap.Map()
m.add_raster(naip_path)
m.add_gdf(gdf, layer_name="Styled Buildings", style=style, zoom_to_layer=True)
m
```

The `style` dictionary follows the Leaflet path options convention: `color` (outline), `weight` (outline width), `fillColor`, and `fillOpacity`.

### Adding Markers and Points

For point data, you can add markers to the map. For dense point layers, consider using marker clustering to keep things readable.

```{code-cell} ipython3
m = leafmap.Map(center=[36.1617, -115.1524], zoom=11)
m.add_marker(location=[36.1617, -115.1524])
m
```

## Split-Panel Maps for Comparison

Split-panel maps provide a structured way to compare datasets -- such as images from different dates or predictions versus ground truth -- without losing spatial context.

### Side-by-Side Comparison

The `split_map()` method creates a map with a draggable divider between two layers. Here we compare NAIP imagery as a false-color composite with the height above ground raster:

```{code-cell} ipython3
m = leafmap.Map()
m.split_map(
    left_layer=naip_path,
    right_layer=hag_path,
    left_args={"indexes": [4, 1, 2]},
    right_args={"vmin": 0, "vmax": 10, "cmap": "terrain"},
    left_label="NAIP",
    right_label="HAG",
)
m
```

Drag the slider left and right to reveal each layer and see how tall structures in the imagery correspond to high HAG values.

+++

## Visualizing Model Results

Visual inspection of model outputs reveals where and why a model fails, revealing spatial error patterns that summary statistics cannot capture.

### Overlaying Predictions

Overlay prediction rasters or vector outputs on source imagery with partial transparency to assess whether highlighted features correspond to real objects:

```{code-cell} ipython3
m = leafmap.Map()
m.add_raster(naip_path, layer_name="NAIP Imagery")
m.add_raster(mask_path, opacity=0.8, nodata=0, layer_name="Building Mask")
m
```

Setting `opacity` below 1 lets the imagery show through, and `nodata=0` makes non-building pixels transparent.

### Comparing Labels with Source Imagery

Split-panel maps also work for verifying that reference labels align with source imagery before comparing predictions:

```{code-cell} ipython3
import leafmap

m = leafmap.Map()
m.split_map(
    left_layer=buildings_path,
    right_layer=naip_path,
    left_args={"style": {"color": "red", "fillOpacity": 0.2}},
)
m
```

The `geoai` library also provides `create_split_map()` that adds a basemap beneath both panels:

```{code-cell} ipython3
geoai.create_split_map(
    left_layer=buildings_path,
    right_layer=naip_path,
    left_args={"style": {"color": "red", "fillOpacity": 0.2}},
    basemap=naip_path,
)
```

Drag the slider to check whether annotations accurately trace building boundaries, whether buildings were missed, or whether non-building structures were mislabeled.

+++

## Best Practices for GeoAI Visualization

**Choose appropriate colormaps for your data type.** Sequential colormaps like `"viridis"` or `"terrain"` work well for continuous data such as elevation or NDVI. Diverging colormaps like `"RdBu"` are better for data with a meaningful center point, such as temperature anomalies or change values. Categorical colormaps should use distinct, easily distinguishable colors for each class. Avoid rainbow colormaps, which can be misleading because they lack a natural perceptual ordering.

**Always include spatial context.** A prediction map without a basemap or reference features is difficult to interpret. Include satellite imagery, administrative boundaries, or other reference layers that help viewers orient themselves and understand the geographic setting of your results.

**Use consistent styling across comparisons.** When comparing model outputs from different experiments or time periods, use the same colormap, value range, and opacity settings for all layers. Inconsistent styling can create false visual differences that do not reflect actual data differences.

**Label your layers clearly.** Give each layer a descriptive name that appears in the layer control. Names like "Building Predictions (U-Net)" are far more useful than "Layer 1" when you have multiple layers active simultaneously.

**Consider your audience.** Technical colleagues may appreciate detailed multi-layer maps with full interactivity, while decision-makers may need simpler visualizations with clear legends and annotations. Tailor the complexity and format of your visualizations to the people who will use them.

## Key Takeaways

1. Interactive maps reveal spatial patterns that summary statistics cannot capture, making visualization essential throughout the GeoAI lifecycle.
2. `leafmap` provides a high-level Python interface for creating interactive geospatial maps in Jupyter notebooks with built-in support for basemaps, raster layers, and vector overlays.
3. Raster visualization supports local GeoTIFFs via `add_raster()` and remote COGs via `add_cog_layer()`, with customizable colormaps, band combinations, and value ranges.
4. Vector visualization handles GeoJSON files, GeoDataFrames, and point markers with flexible styling for overlaying annotations on imagery.
5. Split-panel maps enable side-by-side comparisons with a draggable slider for evaluating predictions, comparing imagery, and verifying annotations.
6. Model result overlays combine prediction rasters with source imagery using `nodata` for transparency and `opacity` for layer blending.
7. Planetary Computer integration through `geoai` lets you search, preview, and download cloud-hosted datasets with server-side rendering of band combinations and spectral indices.
8. Consistent styling and clear labeling improve the readability and effectiveness of your visualizations.
