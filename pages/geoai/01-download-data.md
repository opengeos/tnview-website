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
---

# Downloading Remote Sensing Data

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/opengeos/tnview-website/blob/main/pages/geoai/01-download-data.ipynb)

## Introduction

This tutorial is hands-on: you will search cloud archives, download satellite imagery, and retrieve vector datasets, all from Python.

Freely available Earth observation data has expanded dramatically. NASA, USGS, and ESA provide petabytes of imagery, and cloud platforms such as Microsoft Planetary Computer and Google Earth Engine host large portions of these archives with STAC-compliant APIs for programmatic access. The challenge is no longer data availability but knowing how to find, filter, and fetch exactly what you need.

Choosing the right imagery involves balancing spatial resolution, temporal coverage, cloud cover, and spectral bands. By the end of this tutorial, you will have a repeatable Python workflow for acquiring imagery and vector data for any area of interest.

## Learning Objectives

By the end of this tutorial, you will be able to:

- Identify the major free satellite imagery programs and their key characteristics
- Explain the STAC specification and its four core components (Catalog, Collection, Item, Asset)
- Search for satellite imagery using the STAC API and the Planetary Computer
- Download NAIP aerial imagery for a bounding box using the `geoai` package
- Download and merge Sentinel-2 and Landsat bands into multi-band GeoTIFFs
- Retrieve vector datasets such as building footprints from Overture Maps
- Organize downloaded data following best practices for GeoAI projects

## Satellite Imagery Sources

### NAIP (National Agriculture Imagery Program)

The [National Agriculture Imagery Program (NAIP)](https://naip-usdaonline.hub.arcgis.com/) acquires aerial imagery across the continental United States during the agricultural growing season.

- **Spatial resolution**: approximately 0.6 meters (60 cm)
- **Spectral bands**: 4 bands (Red, Green, Blue, and Near-Infrared)
- **Coverage**: Continental United States
- **Revisit cycle**: Every 2 to 3 years, depending on the state
- **Archive**: Imagery available from 2003 to the present. Pre-2009 imagery is RGB only.

NAIP's sub-meter resolution makes it ideal for building footprint extraction, tree canopy mapping, and urban land cover classification.

### Sentinel-2

The [Sentinel-2](https://dataspace.copernicus.eu/data-collections/copernicus-sentinel-missions/sentinel-2) mission, operated by ESA, currently includes Sentinel-2A, Sentinel-2B, and Sentinel-2C, providing global coverage with a roughly 5-day revisit time.

- **Spatial resolution**: 10 m (visible and NIR), 20 m (red edge and SWIR), 60 m (atmospheric)
- **Spectral bands**: 13 bands spanning visible through shortwave infrared
- **Coverage**: Global land surfaces between 84 N and 56 S
- **Archive**: Imagery available from 2015 to the present

Sentinel-2's combination of high revisit frequency, global coverage, and 10-meter resolution makes it the workhorse for many GeoAI applications.

### Landsat

The [Landsat program](https://www.usgs.gov/landsat-missions), a joint effort between NASA and USGS, is the longest-running civilian Earth observation program with continuous data since 1972.

- **Spatial resolution**: 30 m (multispectral), 15 m (panchromatic), 100 m (thermal)
- **Spectral bands**: 11 bands including visible, NIR, SWIR, and thermal infrared
- **Coverage**: Global
- **Revisit cycle**: 16 days per satellite (8 days combined)
- **Archive**: Over 50 years of continuous imagery

Landsat's unmatched temporal depth makes it essential for long-term change analysis such as urban expansion, deforestation, or glacier retreat.

### Commercial Imagery

Several commercial providers offer very high-resolution imagery (0.3-0.5 m) with rapid tasking capabilities. Major providers include Vantor (formerly Maxar), Planet (daily global coverage at 3-5 m), and Airbus (Pleiades and SPOT). These datasets generally require paid licenses, but some subsets are freely available for research.

### Vantor Open Data Program

[Vantor's Open Data Program](https://vantor.com/company/open-data-program) provides free access to high-resolution satellite imagery captured before and after major natural disasters under a CC-BY-NC-4.0 license.

The data is published as a STAC catalog on AWS S3, with each disaster event as a separate collection containing pre-event and post-event COGs. You can browse the catalog using the [STAC Browser](https://radiantearth.github.io/stac-browser/#/external/vantor-opendata.s3.amazonaws.com/events/catalog.json) or an [interactive web application](https://opengeos.org/maplibre-gl-vantor).

## Installation

Uncomment the following line to install the required package.

```{code-cell} ipython3
# %pip install geoai-py
```

## Searching with STAC

### What is STAC?

The [SpatioTemporal Asset Catalog (STAC)](https://stacspec.org) specification defines a common, JSON-based language for describing geospatial assets. Before STAC, every provider had its own search interface and metadata schema, making cross-provider searches impractical.

Today, major platforms have adopted STAC, including Microsoft Planetary Computer, NASA Earthdata, and Element 84 Earth Search. Learning STAC gives you a single set of skills that applies across all of these sources.

### The STAC Specification

The STAC specification organizes geospatial data into four hierarchical components:

- **Catalog**: The top-level entry point that links to collections. For example, the Planetary Computer catalog contains all of Microsoft's hosted Earth observation datasets.

- **Collection**: A group of related items sharing common metadata, such as [sentinel-2-l2a](https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a) or [naip](https://planetarycomputer.microsoft.com/dataset/naip). Collections describe the dataset's spatial/temporal extent, license, and filterable properties.

- **Item**: A GeoJSON Feature representing a single spatiotemporal observation. Each item contains a geometry, datetime, properties (cloud cover, platform, etc.), and links to its assets.

- **Asset**: An individual file associated with an item, such as spectral band files (COGs), thumbnails, or metadata documents. Each asset has a key, href, media type, and optional roles.

The workflow follows this hierarchy: connect to a **Catalog**, browse **Collections**, search for **Items** using spatial/temporal/property filters, then access specific **Assets**.

### STAC API Endpoints

A **static STAC** catalog is a set of linked JSON files on a web server. A **STAC API** extends this with RESTful search endpoints:

- `GET /` -- Landing page with links and API capabilities
- `GET /collections` -- Lists all available collections
- `GET /collections/{id}` -- Metadata for a specific collection
- `POST /search` -- Searches across collections using filters

The search endpoint accepts `bbox`, `datetime`, `collections`, `query`, and `max_items` parameters. In practice, the `pystac_client` library abstracts these into a Pythonic interface.

### Python Libraries for STAC

Two libraries form the foundation for working with STAC:

- [**pystac**](https://pystac.readthedocs.io/) is the core library for reading, creating, and manipulating STAC objects in Python.

- [**pystac_client**](https://pystac-client.readthedocs.io/) adds a `Client` class for connecting to STAC APIs and executing searches.

Connect to the Planetary Computer's STAC API and explore its collections:

```{code-cell} python
from pystac_client import Client

catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
print(f"Catalog title: {catalog.title}")
print(f"Catalog description: {catalog.description}")

collections = list(catalog.get_collections())
print(f"\nNumber of collections: {len(collections)}")
print("\nFirst 10 collections:")
for collection in collections[:10]:
    print(f"  {collection.id}: {collection.title}")
```

```text
Catalog title: Microsoft Planetary Computer STAC API
Catalog description: Searchable spatiotemporal metadata describing Earth science datasets hosted by the Microsoft Planetary Computer

Number of collections: 134

First 10 collections:
  daymet-annual-pr: Daymet Annual Puerto Rico
  daymet-daily-hi: Daymet Daily Hawaii
  3dep-seamless: USGS 3DEP Seamless DEMs
  3dep-lidar-dsm: USGS 3DEP Lidar Digital Surface Model
  fia: Forest Inventory and Analysis
  gridmet: gridMET
  daymet-annual-na: Daymet Annual North America
  daymet-monthly-na: Daymet Monthly North America
  daymet-annual-hi: Daymet Annual Hawaii
  daymet-monthly-hi: Daymet Monthly Hawaii
```

Because online catalogs are updated continuously, exact counts and IDs may differ from the examples shown here.

### Exploring a STAC Collection

Inspect a collection's metadata to understand its contents and filtering options.

```{code-cell} python
collection = catalog.get_collection("sentinel-2-l2a")
print(f"Title: {collection.title}")
print(f"Description: {collection.description[:200]}...")
print(f"License: {collection.license}")
print(f"Temporal extent: {collection.extent.temporal.intervals}")
print(f"Spatial extent: {collection.extent.spatial.bboxes}")
```

```text
Title: Sentinel-2 Level-2A
Description: The [Sentinel-2](https://sentinel.esa.int/web/sentinel/missions/sentinel-2) program provides global imagery in thirteen spectral bands at 10m-60m resolution and a revisit time of approximately five da...
License: proprietary
Temporal extent: [[datetime.datetime(2015, 6, 27, 10, 25, 31, tzinfo=tzutc()), None]]
Spatial extent: [[-180, -90, 180, 90]]
```

### Searching for Items

Search for Sentinel-2 scenes over Knoxville, Tennessee, during summer 2025 with low cloud cover:

```{code-cell} python
bbox = [-83.95, 35.94, -83.91, 35.98]  # Small area near Knoxville, TN

search = catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=bbox,
    datetime="2025-06-01/2025-08-31",
    query={"eo:cloud_cover": {"lt": 10}},
    max_items=3,
)

items = search.item_collection()
print(f"Found {len(items)} items\n")

for item in items:
    cloud_cover = item.properties.get("eo:cloud_cover")
    cloud_cover_text = (
        f"{cloud_cover:.1f}%" if cloud_cover is not None else "N/A"
    )
    print(f"ID: {item.id}")
    print(f"  Date: {item.datetime}")
    print(f"  Cloud cover: {cloud_cover_text}")
    print(f"  Bounding box: {item.bbox}")
    print()
```

```text
Found 3 items

ID: S2B_MSIL2A_20250816T161829_R040_T16SGE_20250816T201128
  Date: 2025-08-16 16:18:29.024000+00:00
  Cloud cover: 9.7%
  Bounding box: [-84.8052687, 35.1072182, -83.5595813, 36.1242843]

ID: S2A_MSIL2A_20250704T162711_R040_T16SGE_20250705T010417
  Date: 2025-07-04 16:27:11.024000+00:00
  Cloud cover: 8.0%
  Bounding box: [-84.8052687, 35.1072182, -83.5595813, 36.1242843]

ID: S2C_MSIL2A_20250622T161851_R040_T16SGE_20250622T215416
  Date: 2025-06-22 16:18:51.025000+00:00
  Cloud cover: 9.9%
  Bounding box: [-84.8052687, 35.1072182, -83.5595813, 36.1242843]
```

### Inspecting Items and Assets

Inspect an individual item to see its properties and available assets.

```{code-cell} python
if items:
    item = items[0]
    cloud_cover = item.properties.get("eo:cloud_cover")
    cloud_cover_text = (
        f"{cloud_cover}%" if cloud_cover is not None else "N/A"
    )
    print(f"Item ID: {item.id}")
    print(f"Date: {item.datetime}")
    print(f"Cloud cover: {cloud_cover_text}")
    print(f"Platform: {item.properties.get('platform', 'N/A')}")
    print(f"\nAvailable assets ({len(item.assets)}):")
    for key, asset in item.assets.items():
        roles = ", ".join(asset.roles) if asset.roles else "N/A"
        print(f"  {key}: {asset.title or 'No title'} [{roles}]")
```

```text
Item ID: S2B_MSIL2A_20250816T161829_R040_T16SGE_20250816T201128
Date: 2025-08-16 16:18:29.024000+00:00
Cloud cover: 9.708405%
Platform: Sentinel-2B

Available assets (23):
  AOT: Aerosol optical thickness (AOT) [data]
  B01: Band 1 - Coastal aerosol - 60m [data]
  B02: Band 2 - Blue - 10m [data]
  B03: Band 3 - Green - 10m [data]
  B04: Band 4 - Red - 10m [data]
  B05: Band 5 - Vegetation red edge 1 - 20m [data]
  B06: Band 6 - Vegetation red edge 2 - 20m [data]
  B07: Band 7 - Vegetation red edge 3 - 20m [data]
  B08: Band 8 - NIR - 10m [data]
  B09: Band 9 - Water vapor - 60m [data]
  B11: Band 11 - SWIR (1.6) - 20m [data]
  B12: Band 12 - SWIR (2.2) - 20m [data]
  B8A: Band 8A - Vegetation red edge 4 - 20m [data]
  SCL: Scene classification map (SCL) [data]
  WVP: Water vapour (WVP) [data]
  visual: True color image [data]
  safe-manifest: SAFE manifest [metadata]
  granule-metadata: Granule metadata [metadata]
  inspire-metadata: INSPIRE metadata [metadata]
  product-metadata: Product metadata [metadata]
  datastrip-metadata: Datastrip metadata [metadata]
  tilejson: TileJSON with default rendering [tiles]
  rendered_preview: Rendered preview [overview]
```

For most GeoAI applications, you will want the spectral band files (`data` role assets) rather than thumbnails or metadata.

## Downloading NAIP Imagery

The `geoai` package provides `download_naip()`, which wraps the STAC search and download workflow into a single call.

```{code-cell} python
import geoai

bbox = [-83.94, 35.96, -83.92, 35.98]  # Small area near Knoxville, TN
output_dir = "naip_data"

filepaths = geoai.download_naip(
    bbox=bbox,
    output_dir=output_dir,
    year=2021,
    max_items=1,
)
print(f"Downloaded {len(filepaths)} file(s):")
for fp in filepaths:
    print(f"  {fp}")
```

The function parameters are:

- **bbox**: A tuple of (min_lon, min_lat, max_lon, max_lat) in WGS84 coordinates
- **output_dir**: Directory where downloaded GeoTIFF files are saved
- **year**: Specific NAIP acquisition year (e.g., 2021). If `None`, returns imagery from all available years
- **max_items**: Maximum number of scene tiles to download

Inspect the downloaded imagery with `rasterio`:

```{code-cell} python
import rasterio

if filepaths:
    with rasterio.open(filepaths[0]) as src:
        print(f"Dimensions: {src.width} x {src.height}")
        print(f"Bands: {src.count}")
        print(f"CRS: {src.crs}")
        print(f"Resolution: {src.res[0]:.2f} m")
        print(f"Bounds: {src.bounds}")
        print(f"Data type: {src.dtypes[0]}")
```

```text
Dimensions: 10420 x 12520
Bands: 4
CRS: EPSG:26917
Resolution: 0.60 m
Bounds: BoundingBox(left=229164.0, bottom=3980802.0, right=235416.0, top=3988314.0)
Data type: uint8
```

NAIP imagery is delivered as 4-band GeoTIFFs (Red, Green, Blue, Near-Infrared) with `uint8` values ranging from 0 to 255.

You can also print raster metadata in one line:

```{code-cell} python
geoai.print_raster_info(filepaths[0])
```

The USDA NRCS also hosts NAIP imagery in a public Box folder at <https://nrcs.app.box.com/v/naip>, organized by year and state.

## Downloading Sentinel-2 Data

The workflow has two steps: search for items using `pystac_client`, then download specific bands using `geoai.download_pc_stac_item()`.

### Searching for Sentinel-2 Items

```{code-cell} python
from pystac_client import Client

catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
bbox = [-83.94, 35.96, -83.92, 35.98]

search = catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=bbox,
    datetime="2025-06-01/2025-08-31",
    query={"eo:cloud_cover": {"lt": 10}},
    max_items=1,
)

items = search.item_collection()
if items:
    item = items[0]
    cloud_cover = item.properties.get("eo:cloud_cover")
    cloud_cover_text = (
        f"{cloud_cover}%" if cloud_cover is not None else "N/A"
    )
    print(f"Selected item: {item.id}")
    print(f"Date: {item.datetime}")
    print(f"Cloud cover: {cloud_cover_text}")
    item_url = item.self_href
    print(f"Item URL: {item_url}")
```

```text
Selected item: S2B_MSIL2A_20250816T161829_R040_T16SGE_20250816T201128
Date: 2025-08-16 16:18:29.024000+00:00
Cloud cover: 9.708405%
Item URL: https://planetarycomputer.microsoft.com/api/stac/v1/collections/sentinel-2-l2a/items/S2B_MSIL2A_20250816T161829_R040_T16SGE_20250816T201128
```

### Downloading and Merging Bands

Use `geoai.download_pc_stac_item()` to download specific bands and optionally merge them into a single multi-band GeoTIFF:

```{code-cell} python
if items:
    result = geoai.download_pc_stac_item(
        item_url=item_url,
        bands=["B02", "B03", "B04", "B08"],  # Blue, Green, Red, NIR
        output_dir="sentinel2_data",
        merge_bands=True,
        merged_filename="knoxville_s2_rgbn.tif",
        overwrite=False,
    )

    for band_name, path in result.items():
        print(f"  {band_name}: {path}")
```

The `merge_bands=True` parameter resamples all requested bands to a common resolution and stacks them into a single GeoTIFF. You can control the output resolution with the `cell_size` parameter.

## Downloading Landsat Data

Landsat data follows the same pattern. The Planetary Computer hosts Landsat 8 and 9 in the `landsat-c2-l2` collection.

```{code-cell} python
from pystac_client import Client

catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
bbox = [-83.94, 35.96, -83.92, 35.98]

search = catalog.search(
    collections=["landsat-c2-l2"],
    bbox=bbox,
    datetime="2025-06-01/2025-08-31",
    query={"eo:cloud_cover": {"lt": 10}},
    max_items=1,
)

items = search.item_collection()
if items:
    item = items[0]
    cloud_cover = item.properties.get("eo:cloud_cover")
    cloud_cover_text = (
        f"{cloud_cover}%" if cloud_cover is not None else "N/A"
    )
    print(f"Selected item: {item.id}")
    print(f"Date: {item.datetime}")
    print(f"Cloud cover: {cloud_cover_text}")
    print(f"\nAvailable assets:")
    for key in list(item.assets.keys())[:10]:
        print(f"  {key}")
```

```text
Selected item: LC09_L2SP_019035_20250816_02_T1
Date: 2025-08-16 16:12:03.296497+00:00
Cloud cover: 8.2%

Available assets:
  qa
  ang
  red
  blue
  drad
  emis
  emsd
  trad
  urad
  atran
```

Landsat bands use different names than Sentinel-2 (e.g., "blue", "green", "red", "nir08" vs. "B02", "B03", "B04", "B08"):

```{code-cell} python
if items:
    landsat_url = items[0].self_href
    result = geoai.download_pc_stac_item(
        item_url=landsat_url,
        bands=["blue", "green", "red", "nir08"],
        output_dir="landsat_data",
        merge_bands=True,
        merged_filename="knoxville_landsat_rgbn.tif",
        overwrite=False,
    )

    for band_name, path in result.items():
        print(f"  {band_name}: {path}")
```

You can also use `geoai.pc_stac_download()` when you already have STAC item objects and want to download with multiple workers:

```{code-cell} python
if items:
    downloaded = geoai.pc_stac_download(
        items=items[0],
        output_dir="landsat_data_raw",
        assets=["blue", "green", "red", "nir08"],
        max_workers=4,
    )
    for item_id, assets in downloaded.items():
        print(f"Item: {item_id}")
        for asset_key, fpath in assets.items():
            print(f"  {asset_key}: {fpath}")
```

## Downloading Vantor Open Data

The Vantor Open Data catalog can be searched using the same `pystac_client` workflow.

```{code-cell} python
from pystac_client import Client

vantor_catalog_url = (
    "https://vantor-opendata.s3.amazonaws.com/events/catalog.json"
)
vantor_catalog = Client.open(vantor_catalog_url)

collections = list(vantor_catalog.get_collections())
print(f"Number of event collections: {len(collections)}")
for collection in collections:
    print(f"  {collection.id}: {collection.title}")
```

Inspect a specific event collection:

```{code-cell} python
if collections:
    event = collections[0]
    print(f"Event: {event.title}")
    print(f"Description: {event.description}")
    print(f"License: {event.license}")
    print(f"Temporal extent: {event.extent.temporal.intervals}")
    print(f"Spatial extent: {event.extent.spatial.bboxes}")
```

## Accessing Vector Data

Many GeoAI workflows require vector data alongside satellite imagery for training labels, features of interest, or classification targets.

### Overture Maps Buildings

The [Overture Maps Foundation](https://overturemaps.org) produces open map data, including a global building footprint dataset. The `geoai` package offers two ways to access this data.

The `download_overture_buildings()` function downloads building data for a bounding box:

```{code-cell} python
import geoai

bbox = (-83.94, 35.96, -83.92, 35.98)  # Knoxville area
output_path = "buildings.geojson"

geoai.download_overture_buildings(
    bbox=bbox,
    output=output_path,
    overture_type="building",
)
print(f"Buildings saved to {output_path}")
```

For more flexibility, `get_overture_data()` returns data directly as a GeoDataFrame and supports multiple data types (buildings, places, roads, land cover, water):

```{code-cell} python
gdf = geoai.get_overture_data(
    overture_type="building",
    bbox=(-83.94, 35.96, -83.92, 35.98),
    output="buildings.parquet",
)
print(f"Downloaded {len(gdf)} buildings")
gdf.head()
```

The Overture Maps functions require the `overturemaps` package (`pip install overturemaps`).

### OpenStreetMap Data

[OpenStreetMap (OSM)](https://www.openstreetmap.org) is a collaborative, open-source map maintained by volunteers. Libraries such as [osmnx](https://osmnx.readthedocs.io) and the [quackosm](https://kraina-ai.github.io/quackosm) library (which uses DuckDB for fast PBF extraction) provide Python interfaces for querying OSM data. The `leafmap` package wraps these tools for convenient access.

Download OSM building footprints by bounding box:

```{code-cell} python
import leafmap.osm as osm

bbox = (-83.94, 35.96, -83.92, 35.98)  # Knoxville area
buildings = osm.quackosm_gdf_from_bbox(bbox, tags={"building": True})
print(f"Downloaded {len(buildings)} buildings")
buildings.head()
```

Query by place name:

```{code-cell} python
roads = osm.quackosm_gdf_from_place("Knoxville, Tennessee", tags={"highway": True})
print(f"Downloaded {len(roads)} road segments")
roads.head()
```

Query by custom geometry:

```{code-cell} python
from shapely.geometry import Polygon

polygon = Polygon([
    (-83.94, 35.96),
    (-83.92, 35.96),
    (-83.92, 35.98),
    (-83.94, 35.98),
])
natural = osm.quackosm_gdf_from_geometry(polygon, tags={"natural": True})
print(f"Downloaded {len(natural)} natural features")
natural.head()
```

Common OSM tags for GeoAI include `building`, `highway`, `landuse`, `natural`, `waterway`, and `amenity`. The first download for a region may be slower while the PBF file is cached; subsequent queries reuse it.

## Organizing Your Data

A clean directory structure prevents confusion and keeps workflows reproducible. A recommended layout:

```text
project/
├── data/
│   ├── raw/              # Original downloaded files (never modified)
│   │   ├── naip/
│   │   ├── sentinel2/
│   │   ├── landsat/
│   │   └── vectors/
│   ├── processed/        # Clipped, reprojected, or merged files
│   └── training/         # Tiles and labels ready for model training
├── models/               # Saved model weights and configs
├── notebooks/            # Jupyter notebooks for exploration
├── scripts/              # Reusable Python scripts
└── results/              # Predictions, maps, and evaluation outputs
```

Follow these naming conventions:

- **Include the source**: `naip_2021_knoxville.tif`, `s2_20240615_B04.tif`
- **Include the date**: Use ISO format (YYYY-MM-DD or YYYYMMDD) so files sort chronologically
- **Include the area**: Add a short location identifier like a city name or tile ID
- **Preserve raw data**: Never modify original downloads. Create processed copies in a separate directory.
- **Use consistent CRS**: Reproject all datasets to a common CRS early in your workflow to avoid misalignment later

## Key Takeaways

1. NAIP provides sub-meter U.S. aerial imagery, Sentinel-2 offers global 10 m multispectral data, and Landsat provides over 50 years of 30 m imagery.
2. STAC is a community specification that standardizes geospatial data search across multiple providers.
3. The Planetary Computer hosts major satellite collections searchable via `pystac_client` and downloadable via `geoai`.
4. `geoai.download_naip()` simplifies NAIP acquisition to a single function call.
5. `geoai.download_pc_stac_item()` downloads and optionally merges specific bands from any Planetary Computer STAC item.
6. Overture Maps and OpenStreetMap provide global vector data accessible through `geoai` and `leafmap`.
7. Organized directory structures with clear naming conventions make projects reproducible.
8. Always inspect downloaded data (CRS, resolution, bands, extent, data type) before proceeding to analysis.
