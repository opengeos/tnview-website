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
title: "Satellite Embeddings"
abstract: "This tutorial explores satellite embedding datasets and workflows for Earth observation. You will learn to browse precomputed embeddings from multiple foundation models, download and analyze Clay Foundation Model embeddings from Hugging Face, visualize embedding spaces with PCA, perform similarity search and clustering, train lightweight classifiers, download and visualize TESSERA temporal embeddings, and explore AlphaEarth satellite embeddings in Google Earth Engine for multi-year change detection."
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

# Satellite Embeddings

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/opengeos/tnview-website/blob/main/pages/geoai/13-foundation-models.ipynb)

## Introduction

Traditional remote sensing workflows treat each analysis task as an isolated problem, requiring separate labeled data, model training, and validation for every new project. This per-task approach is expensive, slow, and difficult to scale.

Satellite embeddings offer a fundamentally different approach. A foundation model pre-trained on vast quantities of satellite imagery compresses each image patch or pixel into a compact numerical vector (an embedding) that captures semantic content like spectral signatures, spatial textures, and land cover characteristics. Once computed, embeddings serve as a universal representation reusable across many tasks without retraining.

The ecosystem of satellite embedding datasets has grown rapidly. Projects like Clay, Major TOM, TESSERA, and AlphaEarth have released precomputed embeddings covering the entire globe at resolutions from 10 meters to 5 kilometers. This tutorial introduces satellite embedding workflows using the geoai Python package, covering three systems: the Clay Foundation Model for patch-based embeddings, TESSERA for pixel-based temporal embeddings, and AlphaEarth for cloud-based analysis through Google Earth Engine.

## Learning Objectives

By the end of this tutorial, you will be able to:

- Explain what satellite embeddings are and distinguish between patch-based and pixel-based formats
- Browse and query the registry of available embedding datasets using `geoai`
- Download and load Clay Foundation Model embeddings from Hugging Face
- Visualize high-dimensional embedding spaces using Principal Component Analysis (PCA)
- Cluster embeddings to discover spatial patterns without labeled data
- Perform similarity search to find locations with matching characteristics
- Train lightweight classifiers (k-NN, Random Forest, logistic regression) on embedding features
- Compare embeddings across locations for change detection
- Download, visualize, and sample TESSERA temporal embeddings
- Explore AlphaEarth satellite embeddings in Google Earth Engine for multi-year comparison

## What Are Satellite Embeddings?

A satellite embedding is a fixed-length numerical vector that encodes the content of a satellite image patch or pixel. Foundation models produce these vectors by processing raw imagery through deep neural networks, creating a high-dimensional space where geometric proximity encodes semantic similarity.

Satellite embeddings come in two formats. **Patch-based embeddings** summarize an entire image tile into a single vector stored in GeoParquet files, suited for scene-level tasks like retrieval and classification. **Pixel-based embeddings** assign a vector to every pixel in a GeoTIFF raster, supporting fine-grained analysis like segmentation and per-pixel classification.

The `geoai` package provides a unified registry of embedding datasets from multiple foundation models, letting you list, filter, and retrieve detailed metadata.

## Setting Up the Environment

Install the required packages and import the libraries used throughout this tutorial.

```{code-cell} ipython3
# %pip install geoai-py
```

```{code-cell} ipython3
import geoai
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
```

## Browsing Available Embedding Datasets

The `geoai` package provides a registry of embedding datasets backed by [TorchGeo](https://torchgeo.readthedocs.io) dataset classes. Use `list_embedding_datasets()` to see what is available.

```{code-cell} ipython3
df = geoai.list_embedding_datasets(verbose=False)
df
```

| name              | class                    | kind  | spatial_extent | resolution   | temporal_extent | dimensions |
| ----------------- | ------------------------ | ----- | -------------- | ------------ | --------------- | ---------- |
| clay              | ClayEmbeddings           | patch | Global\*       | 5.12 km      | 2018-2023\*     | 768        |
| major_tom         | MajorTOMEmbeddings       | patch | Global         | 2.14-3.56 km | 2015-2024\*     | 2048       |
| earth_index       | EarthIndexEmbeddings     | patch | Global         | 320 m        | 2024            | 384        |
| earth_embeddings  | EarthEmbeddings          | patch | Global\*       | 2.24-3.84 km | 2015-2024\*     | 256-1152   |
| copernicus_embed  | CopernicusEmbed          | pixel | Global         | 0.25 deg     | 2021            | 768        |
| presto            | PrestoEmbeddings         | pixel | Togo           | 10 m         | 2019-2020       | 128        |
| tessera           | TesseraEmbeddings        | pixel | Global         | 10 m         | 2017-2024\*     | 128        |
| google_satellite  | GoogleSatelliteEmbedding | pixel | Global         | 10 m         | 2017-2024       | 64         |
| embedded_seamless | EmbeddedSeamlessData     | pixel | Global         | 30 m         | 2000-2024       | 12         |

+++

You can filter by type to see only patch-based or pixel-based datasets.

```{code-cell} ipython3
geoai.list_embedding_datasets(kind="patch", verbose=False)
```

```{code-cell} ipython3
geoai.list_embedding_datasets(kind="pixel", verbose=False)
```

Use `get_embedding_info()` to inspect a specific dataset in detail.

```{code-cell} ipython3
info = geoai.get_embedding_info("clay")
for key, value in info.items():
    print(f"{key}: {value}")
```

```text
class_name: ClayEmbeddings
kind: patch
base: NonGeoDataset
spatial_extent: Global*
spatial_resolution: 5.12 km
temporal_extent: 2018-2023*
dimensions: 768
dtype: float32
license: ODC-By-1.0
description: Clay v0 and v1.5 embeddings from the Clay Foundation Model. Stored as GeoParquet files.
paper: https://clay-foundation.github.io/model/
data_source: https://source.coop/clay/clay-model-v0-embeddings
```

+++

## Exploring Patch-Based Embeddings

Patch-based embeddings compress entire image tiles into single vectors. The Clay Foundation Model produces 768-dimensional embeddings from sources including Sentinel-2, Landsat, NAIP, Sentinel-1, and MODIS data.

We will use Clay embeddings for the San Francisco Bay Area from [Hugging Face](https://huggingface.co/datasets/made-with-clay/classify-embeddings-sf-baseball-marinas), containing 768-dimensional embeddings from NAIP imagery across 20 tiles, along with labeled locations for baseball fields (class 0) and marinas (class 1).

### Downloading Clay Embeddings

List the available GeoParquet files in the Hugging Face repository.

```{code-cell} ipython3
repo_id = "made-with-clay/classify-embeddings-sf-baseball-marinas"

api = HfApi()
embedding_files = [
    f.path
    for f in api.list_repo_tree(repo_id, repo_type="dataset")
    if f.path.endswith(".gpq")
]
print(f"Found {len(embedding_files)} embedding tiles")
```

```text
Found 20 embedding tiles
```

+++

Download all tiles and concatenate them into a single GeoDataFrame.

```{code-cell} ipython3
all_gdfs = []
for f in embedding_files:
    path = hf_hub_download(repo_id, f, repo_type="dataset")
    gdf = gpd.read_parquet(path)
    all_gdfs.append(gdf)

embeddings_gdf = pd.concat(all_gdfs, ignore_index=True)
embeddings_gdf = gpd.GeoDataFrame(
    embeddings_gdf, geometry="geometry", crs=all_gdfs[0].crs
)
print(f"Combined: {len(embeddings_gdf)} patches")
print(f"Bounds: {embeddings_gdf.total_bounds}")
print(f"Embedding dimension: {len(embeddings_gdf.iloc[0]['embeddings'])}")
```

```text
Combined: 36024 patches
Bounds: [-122.62763615   37.56033178 -122.3108981    37.87715656]
Embedding dimension: 768
```

+++

Download labeled point locations for two classes: baseball fields (class 0) and marinas (class 1).

```{code-cell} ipython3
labels_file = hf_hub_download(repo_id, "baseball.geojson", repo_type="dataset")
labels_gdf = gpd.read_file(labels_file)
print(f"Labeled locations: {len(labels_gdf)}")
print(f"Class distribution:")
print(labels_gdf["class"].value_counts())
```

```text
Labeled locations: 97
Class distribution:
class
0    74
1    23
Name: count, dtype: int64
```

+++

### Extracting Embedding Vectors

Stack embedding vectors into a NumPy matrix and extract centroid coordinates for geographic plotting.

```{code-cell} ipython3
embeddings = np.stack(embeddings_gdf["embeddings"].values)
centroids = embeddings_gdf.geometry.centroid
coords_x = centroids.x.values
coords_y = centroids.y.values

print(f"Embeddings shape: {embeddings.shape}")
print(f"X range: [{coords_x.min():.4f}, {coords_x.max():.4f}]")
print(f"Y range: [{coords_y.min():.4f}, {coords_y.max():.4f}]")
```

```text
Embeddings shape: (36024, 768)
X range: [-122.6268, -122.3118]
Y range: [37.5610, 37.8765]
```

+++

## Visualizing Embeddings

PCA (Principal Component Analysis) projects high-dimensional embeddings into two dimensions while preserving important structural relationships. Plot a few individual embedding vectors to see their raw patterns across the 768 dimensions.

```{code-cell} ipython3
fig, axes = plt.subplots(1, 3, figsize=(15, 3))
for i, ax in enumerate(axes):
    idx = i * (len(embeddings) // 3)
    ax.plot(embeddings[idx], linewidth=0.5)
    ax.set_title(f"Patch {idx}")
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Value")
plt.tight_layout()
plt.show()
```

Project all embeddings into a two-dimensional PCA scatter plot.

```{code-cell} ipython3
fig = geoai.visualize_embeddings(
    embeddings,
    method="pca",
    figsize=(10, 8),
    s=3,
    alpha=0.4,
    title="PCA of Clay Embeddings (SF Bay Area)",
)
plt.show()
```

Points close together represent tiles with similar visual and spectral characteristics, and distinct clusters often correspond to different land cover types.

## Clustering Embeddings

Clustering groups embeddings into discrete categories based on proximity in the embedding space, requiring no labeled data. The `cluster_embeddings()` function supports K-means clustering.

```{code-cell} ipython3
result = geoai.cluster_embeddings(embeddings, n_clusters=8, method="kmeans")
cluster_labels = result["labels"]
print(f"Number of clusters: {result['n_clusters']}")
print(f"Cluster sizes: {np.bincount(cluster_labels)}")
```

```text
Number of clusters: 8
Cluster sizes: [4284 5424 7097 5556 4136 3952 2936 2639]
```

+++

Visualize the clusters by coloring the PCA embedding plot with cluster assignments.

```{code-cell} ipython3
fig = geoai.visualize_embeddings(
    embeddings,
    labels=cluster_labels,
    method="pca",
    figsize=(10, 8),
    s=5,
    alpha=0.5,
    title="K-Means Clusters of Clay Embeddings",
)
plt.show()
```

Map the cluster assignments geographically to see how they correspond to spatial patterns.

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(10, 8))
n_clusters = len(set(cluster_labels))
scatter = ax.scatter(
    coords_x,
    coords_y,
    c=cluster_labels,
    cmap=plt.colormaps["tab10"].resampled(n_clusters),
    vmin=-0.5,
    vmax=n_clusters - 0.5,
    s=3,
    alpha=0.6,
)
plt.colorbar(scatter, ax=ax, label="Cluster", ticks=range(n_clusters))
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Geographic Distribution of Embedding Clusters")
ax.set_aspect("equal")
plt.tight_layout()
plt.show()
```

Examining the geographic distribution of clusters can reveal spatial patterns such as urban versus vegetated areas that emerge naturally without any human-defined categories.

+++

## Similarity Search

Similarity search finds locations whose embeddings are most similar to a query, useful for disaster response, ecological monitoring, and urban planning. The `embedding_similarity()` function computes pairwise cosine similarity scores between a query and all other embeddings.

```{code-cell} ipython3
query_idx = 0
query = embeddings[query_idx]
print(f"Query location (lat, lon): ({coords_y[query_idx]:.4f}, {coords_x[query_idx]:.4f})")

results = geoai.embedding_similarity(
    query=query, embeddings=embeddings, metric="cosine", top_k=10
)

print("\nTop 10 most similar locations:")
for rank, (idx, score) in enumerate(
    zip(results["indices"], results["scores"]), start=1
):
    print(
        f"  {rank}. Index {idx}: similarity={score:.4f}, "
        f"location=({coords_y[idx]:.4f}, {coords_x[idx]:.4f})"
    )
```

```text
Query location (lat, lon): (37.8764, -122.5639)

Top 10 most similar locations:
  1. Index 0: similarity=1.0000, location=(37.8764, -122.5639)
  2. Index 1822: similarity=0.9521, location=(37.8761, -122.5636)
  3. Index 1935: similarity=0.8796, location=(37.8720, -122.5653)
  4. Index 1860: similarity=0.8707, location=(37.8747, -122.5636)
  5. Index 38: similarity=0.8695, location=(37.8750, -122.5639)
  6. Index 1892: similarity=0.8582, location=(37.8734, -122.5741)
  7. Index 1820: similarity=0.8579, location=(37.8761, -122.5671)
  8. Index 450: similarity=0.8561, location=(37.8609, -122.5081)
  9. Index 606: similarity=0.8520, location=(37.8554, -122.5011)
  10. Index 1852: similarity=0.8385, location=(37.8748, -122.5775)
```

+++

Visualize the query point and its nearest neighbors on a geographic plot.

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(10, 8))

# Background: all embeddings in gray
ax.scatter(coords_x, coords_y, c="lightgray", s=1, alpha=0.3)

# Highlight nearest neighbors
nn_indices = results["indices"]
ax.scatter(
    coords_x[nn_indices],
    coords_y[nn_indices],
    c="blue",
    s=50,
    marker="o",
    label="Nearest Neighbors",
    edgecolors="black",
    linewidths=0.5,
)

# Highlight the query point
ax.scatter(
    coords_x[query_idx],
    coords_y[query_idx],
    c="red",
    s=100,
    marker="*",
    label="Query",
    edgecolors="black",
    linewidths=0.5,
    zorder=5,
)

ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Similarity Search: Query and Nearest Neighbors")
ax.legend()
ax.set_aspect("equal")
plt.tight_layout()
plt.show()
```

A score of 1.0 means identical embeddings, while lower scores indicate decreasing similarity. The geographic plot reveals whether similar embeddings cluster spatially.

## Training Classifiers on Embeddings

Because the foundation model has already extracted rich features, only a simple classifier is needed. The workflow involves matching labeled points to embedding patches via spatial join, then splitting into training and validation sets.

```{code-cell} ipython3
# Ensure both GeoDataFrames use the same CRS
if labels_gdf.crs != embeddings_gdf.crs:
    labels_gdf = labels_gdf.to_crs(embeddings_gdf.crs)

# Spatial join: find which embedding patch each labeled point falls within
joined = gpd.sjoin(labels_gdf, embeddings_gdf, how="inner", predicate="within")
print(f"Matched {len(joined)} labeled points to embedding patches")
print(f"Class distribution: {joined['class'].value_counts().to_dict()}")
```

```text
Matched 106 labeled points to embedding patches
Class distribution: {0: 79, 1: 27}
```

Extract the embedding vector for each matched point.

```{code-cell} ipython3
labeled_embeddings = np.stack(
    [embeddings_gdf.iloc[idx]["embeddings"] for idx in joined["index_right"]]
)
class_labels = joined["class"].values

print(f"Labeled embeddings shape: {labeled_embeddings.shape}")
print(f"Labels shape: {class_labels.shape}")
```

```text
Labeled embeddings shape: (106, 768)
Labels shape: (106,)
```

Split into 70% training and 30% validation with stratified sampling.

```{code-cell} ipython3
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    labeled_embeddings,
    class_labels,
    test_size=0.3,
    random_state=42,
    stratify=class_labels,
)
print(f"Train: {X_train.shape[0]} samples")
print(f"Val:   {X_val.shape[0]} samples")
```

```text
Train: 74 samples
Val:   32 samples
```

Train a k-NN classifier.

```{code-cell} ipython3
label_names = ["Baseball Field", "Marina"]

result = geoai.train_embedding_classifier(
    train_embeddings=X_train,
    train_labels=y_train,
    val_embeddings=X_val,
    val_labels=y_val,
    method="knn",
    n_neighbors=5,
    label_names=label_names,
)

print(f"\nTrain accuracy: {result['train_accuracy']:.2%}")
print(f"Val accuracy:   {result['val_accuracy']:.2%}")
```

```text
precision    recall  f1-score   support

Baseball Field      0.864     0.792     0.826        24
        Marina      0.500     0.625     0.556         8

      accuracy                          0.750        32
     macro avg      0.682     0.708     0.691        32
  weighted avg      0.773     0.750     0.758        32

Train accuracy: 86.49%
Val accuracy:   75.00%
```

+++

Compare three classifier methods.

```{code-cell} ipython3
methods = ["knn", "random_forest", "logistic_regression"]
results_summary = []

for method in methods:
    res = geoai.train_embedding_classifier(
        train_embeddings=X_train,
        train_labels=y_train,
        val_embeddings=X_val,
        val_labels=y_val,
        method=method,
        label_names=label_names,
        verbose=False,
    )
    results_summary.append(
        {
            "Method": method,
            "Train Acc": f"{res['train_accuracy']:.2%}",
            "Val Acc": f"{res['val_accuracy']:.2%}",
        }
    )

pd.DataFrame(results_summary)
```

Visualize the labeled embeddings in PCA space to see how well the two classes separate.

```{code-cell} ipython3
fig = geoai.visualize_embeddings(
    labeled_embeddings,
    labels=class_labels,
    label_names=label_names,
    method="pca",
    figsize=(8, 8),
    s=30,
    alpha=0.8,
    title="PCA of Labeled Embeddings (Baseball vs Marina)",
)
plt.show()
```

The foundation model does the heavy lifting of feature extraction, and the downstream classifier trains in seconds with very few labeled samples.

## Comparing Embeddings for Change Detection

Comparing embeddings from the same location at different times reveals landscape changes. The `compare_embeddings()` function quantifies similarity, where higher cosine values indicate stability and lower values indicate change.

Here we demonstrate by comparing embeddings from different spatial patches.

```{code-cell} ipython3
n = len(embeddings)
half = n // 2
emb_a = embeddings[:half]
emb_b = embeddings[half : half + half]

similarity = geoai.compare_embeddings(emb_a, emb_b, metric="cosine")

fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(similarity, bins=50, edgecolor="black", alpha=0.7)
ax.axvline(
    similarity.mean(),
    color="red",
    linestyle="--",
    label=f"Mean: {similarity.mean():.3f}",
)
ax.set_xlabel("Cosine Similarity")
ax.set_ylabel("Count")
ax.set_title("Embedding Similarity Distribution")
ax.legend()
plt.tight_layout()
plt.show()
```

In a true temporal comparison, you would compare embeddings from the same locations at different dates, and low-similarity outliers would indicate areas of significant change.

## Loading Datasets with TorchGeo

For advanced usage, load TorchGeo dataset classes directly through `geoai.load_embedding_dataset()` for access to transforms, sampling, and built-in plotting.

Note that the TorchGeo `ClayEmbeddings` class expects a `date` or `datetime` column in the GeoParquet file. Some files may not include this column, requiring a fallback to geopandas.

```{code-cell} ipython3
single_file = hf_hub_download(repo_id, embedding_files[0], repo_type="dataset")
ds = geoai.load_embedding_dataset("clay", root=single_file)

print(f"Dataset length: {len(ds)}")
print(f"Dataset type: {type(ds).__name__}")
```

```text
Dataset length: 1786
Dataset type: ClayEmbeddings
```

```{code-cell} ipython3
try:
    sample = ds[0]
    print(f"Sample keys: {list(sample.keys())}")
    print(f"Embedding shape: {sample['embedding'].shape}")

    fig = ds.plot(sample)
    plt.show()
except KeyError as e:
    print(
        f"Note: This parquet file is missing the '{e.args[0]}' column "
        f"expected by TorchGeo's ClayEmbeddings class."
    )
    print("For such files, use geopandas directly (as shown above).")
    print("The TorchGeo class works best with official Clay data products.")
```

## Working with TESSERA Temporal Embeddings

[TESSERA](https://github.com/ucam-eo/tessera) (Temporal Embeddings of Surface Spectra for Earth Representation and Analysis) generates 128-channel representation maps at 10-meter resolution globally from time-series Sentinel-1 and Sentinel-2 imagery. The `geoai` package includes functions for checking availability, downloading, visualizing, and sampling TESSERA embeddings.

```{code-cell} ipython3
# %pip install geoai-py geotessera
```

### Checking Data Availability

Check which years are available and how many tiles cover your region.

```{code-cell} ipython3
years = geoai.tessera_available_years()
print(f"Available years: {years}")
```

```text
Available years: [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
```

+++

Define a bounding box for Cambridge, UK and check tile count.

```{code-cell} ipython3
bbox = (0.05, 52.15, 0.25, 52.25)
count = geoai.tessera_tile_count(bbox=bbox, year=2024)
print(f"{count} tiles available for the specified region")
```

```text
6 tiles available for the specified region
```

Generate coverage maps.

```{code-cell} ipython3
geoai.tessera_coverage(year=2024, output_path="tessera_coverage_2024.png")
```

```{code-cell} ipython3
geoai.tessera_coverage(
    year=2024, region_bbox=(-10, 35, 40, 60), output_path="tessera_coverage_europe.png"
)
```

### Downloading Embeddings

TESSERA embeddings can be downloaded by bounding box, point coordinates, or region file in GeoTIFF or NumPy format.

```{code-cell} ipython3
bbox = (0.05, 52.15, 0.25, 52.25)

cambridge_files = geoai.tessera_download(
    bbox=bbox, year=2024, output_dir="./tessera_cambridge", output_format="tiff"
)
print(f"Downloaded {len(cambridge_files)} files")
for f in cambridge_files:
    print(f"  {f}")
```

Download a single tile by point coordinates.

```{code-cell} ipython3
files = geoai.tessera_download(
    lon=0.15, lat=52.05, year=2024, output_dir="./tessera_single_tile"
)
print(f"Downloaded {len(files)} file(s)")
```

Download only specific bands to reduce file size.

```{code-cell} ipython3
files = geoai.tessera_download(
    bbox=bbox, year=2024, bands=[0, 1, 2], output_dir="./tessera_rgb_only"
)
print(f"Downloaded {len(files)} files with 3 bands each")
```

### Visualizing TESSERA Embeddings

TESSERA embeddings can be rendered as false-color RGB composites by selecting three of the 128 channels. Areas with similar colors share similar temporal dynamics.

```{code-cell} ipython3
geoai.tessera_visualize_rgb(
    str(cambridge_files[0]), bands=(0, 1, 2), title="Cambridge - TESSERA Bands 0, 1, 2"
)
```

Try a different band combination to highlight other aspects of the temporal signal.

```{code-cell} ipython3
geoai.tessera_visualize_rgb(
    str(cambridge_files[0]),
    bands=(30, 60, 90),
    title="Cambridge - TESSERA Bands 30, 60, 90",
)
```

Different band combinations can reveal seasonal vegetation patterns, urban heat signatures, or water body dynamics.

### Fetching Embeddings to Memory

Use `tessera_fetch_embeddings()` to load embedding arrays directly into memory without saving files to disk.

```{code-cell} ipython3
tiles = geoai.tessera_fetch_embeddings(bbox=(0.05, 52.15, 0.25, 52.25), year=2024)

for tile in tiles:
    print(f"Tile ({tile['lon']:.2f}, {tile['lat']:.2f}):")
    print(f"  Shape: {tile['embedding'].shape}")
    print(f"  CRS: {tile['crs']}")
    print(f"  Mean: {tile['embedding'].mean():.4f}")
    print(f"  Std: {tile['embedding'].std():.4f}")
```

### Sampling Embeddings at Point Locations

Extract embedding values at specific points using `tessera_sample_points()`, which appends 128 columns (`tessera_0` through `tessera_127`) to the input GeoDataFrame.

```{code-cell} ipython3
from shapely.geometry import Point

points = gpd.GeoDataFrame(
    {"name": ["Point A", "Point B", "Point C"]},
    geometry=[Point(0.12, 52.20), Point(0.15, 52.18), Point(0.20, 52.22)],
    crs="EPSG:4326",
)

result = geoai.tessera_sample_points(points, year=2024)

print(f"Result shape: {result.shape}")
print(
    f"\nOriginal columns: {[c for c in result.columns if not c.startswith('tessera_')]}"
)
print(f"Embedding columns: tessera_0 through tessera_127")
print(f"\nFirst few embedding values for Point A:")
print(result.iloc[0][[f"tessera_{i}" for i in range(5)]])
```

```text
Result shape: (3, 130)

Original columns: ['name', 'geometry']
Embedding columns: tessera_0 through tessera_127

First few embedding values for Point A:
tessera_0    8.032325
tessera_1   -0.316233
tessera_2    0.505973
tessera_3    2.593113
tessera_4    0.442727
Name: 0, dtype: object
```

### Alternative Download Formats

Download as NumPy arrays with a JSON metadata file.

```{code-cell} ipython3
import json

files = geoai.tessera_download(
    bbox=bbox, year=2024, output_dir="./tessera_numpy", output_format="npy"
)

with open("./tessera_numpy/metadata.json") as f:
    meta = json.load(f)

print(f"Year: {meta['year']}")
print(f"Number of tiles: {len(meta['tiles'])}")
for tile_info in meta["tiles"]:
    arr = np.load(f"./tessera_numpy/{tile_info['file']}")
    print(f"  {tile_info['file']}: shape={arr.shape}, dtype={arr.dtype}")
```

Use a GeoJSON or Shapefile to define the download region.

```{code-cell} ipython3
from shapely.geometry import box

region = gpd.GeoDataFrame(geometry=[box(0.05, 52.15, 0.25, 52.25)], crs="EPSG:4326")
region.to_file("cambridge_region.geojson", driver="GeoJSON")

files = geoai.tessera_download(
    region_file="cambridge_region.geojson",
    year=2024,
    output_dir="./tessera_from_region",
)
print(f"Downloaded {len(files)} files")
```

## Exploring AlphaEarth Satellite Embeddings

[AlphaEarth](https://deepmind.google/discover/blog/alphaearth-foundations-helps-map-our-planet-in-unprecedented-detail) from Google DeepMind contains annual satellite embeddings from 2017 to 2024 at 10-meter resolution, available on Google Earth Engine. By comparing embeddings across years, you can identify landscape changes without downloading any data locally.

This section requires a Google Earth Engine account. Create one at <https://earthengine.google.com> if needed.

```{code-cell} ipython3
# %pip install -U geemap
```

```{code-cell} ipython3
import ee
import geoai
```

Authenticate and initialize with your project. Replace `"your-ee-project"` with your own project ID.

```{code-cell} ipython3
ee.Authenticate()
ee.Initialize(project="your-ee-project")
```

### Interactive Map with AlphaEarth GUI

The `geoai` package includes a built-in AlphaEarth GUI widget for interactive exploration. Click **Apply** to update the visualization.

```{code-cell} ipython3
m = geoai.Map(projection="globe", sidebar_visible=True)
m.add_basemap("USGS.Imagery")
m.add_alphaearth_gui()
m
```

### Visualizing Embeddings as RGB Composites

Load the AlphaEarth ImageCollection and filter by date to compare 2017 and 2024 embeddings.

```{code-cell} ipython3
m = geoai.Map(projection="globe", sidebar_visible=True)
m.add_basemap("USGS.Imagery")

lon = -121.8036
lat = 39.0372
m.set_center(lon, lat, zoom=12)
m
```

```{code-cell} ipython3
point = ee.Geometry.Point(lon, lat)
dataset = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")

image1 = dataset.filterDate("2017-01-01", "2018-01-01").filterBounds(point).first()
image2 = dataset.filterDate("2024-01-01", "2025-01-01").filterBounds(point).first()
```

Visualize both years as false-color RGB composites using bands A01, A16, and A09.

```{code-cell} ipython3
vis_params = {"min": -0.3, "max": 0.3, "bands": ["A01", "A16", "A09"]}
m.add_ee_layer(image1, vis_params, name="2017 embeddings")
m.add_ee_layer(image2, vis_params, name="2024 embeddings")
```

### Change Detection via Embedding Similarity

Compute the dot product between 2017 and 2024 embeddings for a pixel-level similarity map. High values (near 1) indicate stability; low values indicate significant change.

```{code-cell} ipython3
dot_prod = image1.multiply(image2).reduce(ee.Reducer.sum())
```

Add the similarity layer with a white-to-black palette where lighter pixels indicate more change.

```{code-cell} ipython3
vis_params = {"min": 0, "max": 1, "palette": ["#ffffff", "#000000"]}
m.add_ee_layer(dot_prod, vis_params, name="Similarity")
m.add_colorbar(cmap="gray", label="Similarity")
m
```

This cloud-based approach scales to any region on Earth covered by the AlphaEarth dataset without downloading raster data.

## Key Takeaways

1. Satellite embeddings are compact vectors from foundation models that enable similarity search, clustering, classification, and change detection without task-specific training.
2. Patch-based embeddings (GeoParquet) summarize tiles for scene-level analysis, while pixel-based embeddings (GeoTIFF) support fine-grained per-pixel work.
3. The `geoai` package provides a unified registry of embedding datasets from multiple foundation models, accessible through a consistent API.
4. PCA visualization reveals structure in embedding spaces, with clustered points representing locations sharing similar characteristics.
5. Cosine similarity search enables fast geographic retrieval of locations with matching characteristics.
6. Unsupervised clustering reveals natural groupings like land cover types without labeled data.
7. Lightweight classifiers trained on embeddings achieve strong accuracy with very few labeled samples.
8. Comparing embeddings across time or space provides efficient change detection without labeled change examples.
9. TESSERA temporal embeddings at 10-meter resolution capture seasonal dynamics globally, with multiple download formats and access patterns.
10. AlphaEarth embeddings in Google Earth Engine enable cloud-based multi-year comparison at global scale without local data downloads.
