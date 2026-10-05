# Data plan

Nothing here is downloaded for you, and I have not verified download links. Find
each source from its official page, read its licence, and record the version and
date you used. Large files are gitignored.

| Layer | Candidate source | Notes |
|---|---|---|
| Landslide events | NASA Global Landslide Catalog | Point events with dates and locations. Filter to Nepal. Reporting bias is real (see README). Check for a newer catalogue version. |
| Elevation | Copernicus GLO-30 DEM or SRTM | 30 m. Reproject to a metric CRS (UTM zone 44N/45N) before computing slope. |
| Rainfall | CHIRPS | Monthly or daily rainfall, around 5 km. Use long-term mean annual and the wettest-month mean as features. |
| Land cover | ESA WorldCover | 10 m, categorical. |
| Rivers and roads | OpenStreetMap (a Nepal extract) | Roads are used for the accessibility weight and the bias ablation. |
| Nepal boundary | geoBoundaries or similar | For the valid-area mask. Check the licence. |

## Layout once downloaded

```
data/raw/        untouched downloads (gitignored)
data/processed/  clipped, reprojected, aligned rasters (gitignored)
data/tables/     the final feature table as CSV/Parquet (small ones may be committed)
```

All rasters should end up on one shared grid and CRS before sampling.
Record the exact versions used in the README when you report results.
