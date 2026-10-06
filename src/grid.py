"""Shared grid, DEM warping and terrain features at points.

Everything is on one grid: EPSG:32645 (UTM 45N) at 30 m, for all of Nepal.
Far-west Nepal lies in zone 44N; using 45N throughout costs well under 1 %
scale error at the western edge, which is fine for slope.

The table itself is built by src/build_feature_table.py.
"""
from __future__ import annotations

import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject
from shapely.geometry import box

from .terrain import (
    aspect_degrees,
    aspect_to_sin_cos,
    plan_curvature_proxy,
    slope_degrees,
)

CRS = "EPSG:32645"
CELL = 30.0  # metres
COARSE = 300.0  # metres; grid used only to draw absence locations
HALF = 2  # patch half-width in cells used for the terrain derivatives
ACC_KM = {"exact": 0.0, "1km": 1.0, "5km": 5.0, "10km": 10.0, "25km": 25.0, "50km": 50.0}

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "tables"


def make_grid():
    """Nepal polygon in UTM 45N plus the shared 30 m grid (row 0 = north), padded 1 km."""
    nepal = gpd.read_file(RAW / "geoBoundaries-NPL-ADM0.geojson").to_crs(CRS)
    xmin, ymin, xmax, ymax = nepal.total_bounds
    pad = 1000.0
    xmin, ymin = math.floor((xmin - pad) / CELL) * CELL, math.floor((ymin - pad) / CELL) * CELL
    xmax, ymax = math.ceil((xmax + pad) / CELL) * CELL, math.ceil((ymax + pad) / CELL) * CELL
    shape = (int((ymax - ymin) / CELL), int((xmax - xmin) / CELL))
    return nepal.geometry.union_all(), xmin, ymax, shape, from_origin(xmin, ymax, CELL, CELL)


def load_presences(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df[df["country_name"].str.strip().str.lower() == "nepal"].copy()
    df["accuracy_km"] = df["location_accuracy"].map(ACC_KM)  # 'unknown' -> NaN
    df["event_date"] = pd.to_datetime(df["event_date"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
    return df[["event_id", "event_date", "longitude", "latitude", "accuracy_km"]].reset_index(drop=True)


def build_dem(nepal_utm, tiles: list[Path], transform, shape) -> np.ndarray:
    """Warp every tile that touches Nepal onto the shared 30 m UTM grid (NaN = no data)."""
    dem = np.full(shape, np.nan, dtype=np.float32)
    used = 0
    for tif in tiles:
        with rasterio.open(tif) as src:
            tb = gpd.GeoSeries([box(*src.bounds)], crs=src.crs).to_crs(CRS).iloc[0]
            if not tb.intersects(nepal_utm):
                continue
            reproject(
                source=rasterio.band(src, 1),
                destination=dem,
                dst_transform=transform,
                dst_crs=CRS,
                resampling=Resampling.bilinear,
                src_nodata=src.nodata,
                dst_nodata=np.nan,
                init_dest_nodata=False,
            )
            used += 1
    print(f"warped {used} of {len(tiles)} tiles")
    return dem


def terrain_at(dem: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> pd.DataFrame:
    """Terrain features at (row, col) from a small patch around each cell."""
    out = {k: np.full(len(rows), np.nan) for k in ("elevation", "slope", "aspect_sin", "aspect_cos", "curvature")}
    h, w = dem.shape
    for i, (r, c) in enumerate(zip(rows, cols, strict=True)):
        if r < HALF or c < HALF or r >= h - HALF or c >= w - HALF:
            continue
        patch = dem[r - HALF : r + HALF + 1, c - HALF : c + HALF + 1]
        if np.isnan(patch).any():
            continue
        asp = aspect_degrees(patch, CELL)[HALF, HALF]
        s, co = aspect_to_sin_cos(np.array([asp]))
        out["elevation"][i] = patch[HALF, HALF]
        out["slope"][i] = slope_degrees(patch, CELL)[HALF, HALF]
        out["aspect_sin"][i] = 0.0 if np.isnan(asp) else s[0]  # flat cell: no aspect
        out["aspect_cos"][i] = 0.0 if np.isnan(asp) else co[0]
        out["curvature"][i] = plan_curvature_proxy(patch, CELL)[HALF, HALF]
    return pd.DataFrame(out)
