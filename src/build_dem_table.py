"""Build the DEM-only feature table (presences + unweighted absences).

Run from the repo root with the Python 3.12 env:
    .venv312/Scripts/python -m src.build_dem_table

PARTIAL by design: terrain features only. There is no road layer yet, so the
absences are NOT accessibility-weighted and the road-distance ablation cannot
be run on this table. No rainfall, land cover or river features either.

Everything is on one grid: EPSG:32645 (UTM 45N) at 30 m, for all of Nepal.
Far-west Nepal lies in zone 44N; using 45N throughout costs well under 1 %
scale error at the western edge, which is fine for slope.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject
from shapely import contains_xy
from shapely.geometry import box

from .sampling import sample_background
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
    for i, (r, c) in enumerate(zip(rows, cols)):
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--absence-ratio", type=int, default=5, help="absences per presence with accuracy <= --max-acc-km")
    ap.add_argument("--max-acc-km", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tiles = sorted((RAW / "dem_glo30").glob("*.tif"))
    if len(tiles) != 45:
        raise SystemExit(f"Expected 45 DEM tiles in data/raw/dem_glo30, found {len(tiles)}.")

    nepal_geom, xmin, ymax, shape, transform = make_grid()
    print("grid", shape, f"{shape[0] * shape[1] * 4 / 1e9:.2f} GB float32")

    dem = build_dem(nepal_geom, tiles, transform, shape)

    # presences
    pres = load_presences(RAW / "Global_Landslide_Catalog_Export_rows.csv")
    to_utm = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
    pres["x"], pres["y"] = to_utm.transform(pres["longitude"].to_numpy(), pres["latitude"].to_numpy())
    pres["row"] = ((ymax - pres["y"]) / CELL).astype(int)
    pres["col"] = ((pres["x"] - xmin) / CELL).astype(int)
    n_all = len(pres)

    # absences: drawn on a coarse 300 m mask (a 30 m index array for ~160 M cells would not fit comfortably)
    ct = from_origin(xmin, ymax, COARSE, COARSE)
    cshape = (math.ceil(shape[0] * CELL / COARSE), math.ceil(shape[1] * CELL / COARSE))
    valid = rasterize([(nepal_geom, 1)], out_shape=cshape, transform=ct, dtype="uint8").astype(bool)
    near = pres[(pres["accuracy_km"] <= args.max_acc_km)]
    exclude = np.zeros(cshape, dtype=bool)
    for x, y in zip(pres["x"], pres["y"]):  # keep absences out of every presence's coarse cell
        r, c = int((ymax - y) / COARSE), int((x - xmin) / COARSE)
        if 0 <= r < cshape[0] and 0 <= c < cshape[1]:
            exclude[r, c] = True
    n_abs = args.absence_ratio * len(near)
    rng = np.random.default_rng(args.seed)
    ar, ac = sample_background(valid, n_abs, rng, weights=None, exclude=exclude)
    absn = pd.DataFrame({"x": xmin + (ac + 0.5) * COARSE, "y": ymax - (ar + 0.5) * COARSE})
    absn["row"] = ((ymax - absn["y"]) / CELL).astype(int)
    absn["col"] = ((absn["x"] - xmin) / CELL).astype(int)
    to_ll = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
    absn["longitude"], absn["latitude"] = to_ll.transform(absn["x"].to_numpy(), absn["y"].to_numpy())

    pres["label"], absn["label"] = 1, 0
    table = pd.concat([pres, absn], ignore_index=True)
    feats = terrain_at(dem, table["row"].to_numpy(), table["col"].to_numpy())
    table = pd.concat([table, feats], axis=1)

    n_before = len(table)
    table = table.dropna(subset=["elevation", "slope", "curvature"]).copy()
    dup = table.duplicated(subset=["label", "row", "col"])  # two presences in one 30 m cell
    table = table[~dup]
    # a catalogue point can fall just outside the boundary polygon (5-50 km location error): flag it, keep it
    table["inside_nepal"] = contains_xy(nepal_geom, table["x"].to_numpy(), table["y"].to_numpy())

    cols = ["label", "event_id", "event_date", "accuracy_km", "longitude", "latitude", "x", "y",
            "elevation", "slope", "aspect_sin", "aspect_cos", "curvature", "inside_nepal"]
    OUT.mkdir(parents=True, exist_ok=True)
    table[cols].to_csv(OUT / "features_dem_only.csv", index=False)

    meta = {
        "built": datetime.now().isoformat(timespec="seconds"),
        "crs": CRS,
        "cell_m": CELL,
        "partial": "terrain features only; absences NOT accessibility-weighted; no road ablation possible",
        "absence_ratio": args.absence_ratio,
        "max_acc_km_for_absence_count": args.max_acc_km,
        "seed": args.seed,
        "presences_in_catalogue": int(n_all),
        "rows_before_dropping_nan_or_dupes": int(n_before),
        "rows_written": int(len(table)),
        "presences_written": int((table["label"] == 1).sum()),
        "absences_written": int((table["label"] == 0).sum()),
        "presences_by_accuracy_km": {str(k): int(v) for k, v in table[table.label == 1]["accuracy_km"].value_counts(dropna=False).items()},
        "presences_outside_boundary": int(((table.label == 1) & ~table.inside_nepal).sum()),
        "sources": ["Global_Landslide_Catalog_Export_rows.csv (NASA GLC legacy export)",
                    "Copernicus GLO-30 DEM, 45 tiles N26-N30 E080-E088",
                    "geoBoundaries gbOpen NPL ADM0 (2019, CC BY 4.0)"],
    }
    (OUT / "features_dem_only.meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
