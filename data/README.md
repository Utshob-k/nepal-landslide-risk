# Data

Raw downloads are gitignored. Versions, dates and licences are in the main
README.

| Layer | Source | Where it goes |
|---|---|---|
| Landslides | NASA Global Landslide Catalog, legacy CSV export | `data/raw/Global_Landslide_Catalog_Export_rows.csv` |
| Elevation | Copernicus GLO-30 DEM, 45 tiles | `data/raw/dem_glo30/` |
| Boundary | geoBoundaries NPL ADM0 | `data/raw/geoBoundaries-NPL-ADM0.geojson` |
| Roads, rivers | Geofabrik Nepal extract (OSM) | `nepal-latest.osm.pbf` in `data/raw/` or `$NEPAL_RAW_DIR` |
| Rainfall | CHIRPS v3.0 annual GeoTIFFs, 1991 to 2020 | `chirps_annual/` in `data/raw/` or `$NEPAL_RAW_DIR` |
| Land cover | ESA WorldCover 2021 v200, 8 tiles | `worldcover/` in `data/raw/` or `$NEPAL_RAW_DIR` |

`data/processed/` (gitignored) holds the road and river GeoPackages the table
builder extracts. `data/tables/` holds the feature tables; they aren't committed
until the catalogue and DEM licences are checked.
