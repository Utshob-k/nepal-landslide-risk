# Data

Raw downloads are gitignored. Versions, dates and licences are in the main
README.

| Layer | Source | Where it goes |
|---|---|---|
| Landslides | NASA Global Landslide Catalog legacy export, or current COOLR reports | `data/raw/Global_Landslide_Catalog_Export_rows.csv` or `data/raw/COOLR_Nepal_reports.csv` |
| Elevation | Copernicus GLO-30 DEM, 45 tiles | `data/raw/dem_glo30/` |
| Boundary | geoBoundaries NPL ADM0 | `data/raw/geoBoundaries-NPL-ADM0.geojson` |
| Roads, rivers | Geofabrik Nepal extract (OSM) | `nepal-latest.osm.pbf` in `data/raw/` or `$NEPAL_RAW_DIR` |
| Rainfall | CHIRPS v3.0 annual GeoTIFFs, 1991 to 2020 | `chirps_annual/` in `data/raw/` or `$NEPAL_RAW_DIR` |
| Land cover | ESA WorldCover 2021 v200, 8 tiles | `worldcover/` in `data/raw/` or `$NEPAL_RAW_DIR` |

`data/processed/` (gitignored) holds the road and river GeoPackages the table
builder extracts. `data/tables/` holds the feature tables and stays gitignored
because it is derived from the large raw downloads.

To use the current NASA COOLR reports catalogue, run:

```bash
python -m src.download_catalogue
python -m src.build_feature_table --catalogue data/raw/COOLR_Nepal_reports.csv
```

This changes the event inventory, so re-run every analysis script and label its
results as COOLR-based; do not compare them directly with the published legacy-
catalogue tables.
