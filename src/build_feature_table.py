"""Feature table with terrain + road distance + river distance, and two absence sets.

    NEPAL_RAW_DIR=<dir holding the .pbf, if not data/raw> .venv312/Scripts/python -m src.build_feature_table

Rows: the catalogue presences, plus two absence sets drawn with the same count:
  uniform        every valid cell equally likely
  road_weighted  drawn with the presences' own road-distance profile (src.roads.presence_matched_weights)
`abs_set` says which. Run both through the models and compare: the gap is
how much of a score is road access (reporting bias) rather than terrain.

Still no rainfall, land cover or river features.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely import contains_xy

from .climate_cover import LC_COLS, landcover_shares, mean_annual_rainfall, sample_raster
from .grid import CELL, COARSE, CRS, OUT, RAW, build_dem, load_presences, make_grid, terrain_at
from .roads import BIN_EDGES_KM, distance_to_lines, extract_motor_roads, extract_rivers, presence_matched_weights
from .sampling import sample_background

ROADS_GPKG = RAW.parent / "processed" / "roads_motor.gpkg"
RIVERS_GPKG = RAW.parent / "processed" / "rivers.gpkg"
LC_RADIUS_KM = 2.5


def raw_dir() -> Path:
    """Where the big downloads live: NEPAL_RAW_DIR if set, else data/raw."""
    return Path(os.environ.get("NEPAL_RAW_DIR", RAW))


def bin_shares(dist_m: np.ndarray) -> list[float]:
    b = np.digitize(dist_m / 1000.0, BIN_EDGES_KM[1:-1])
    return (np.bincount(b, minlength=len(BIN_EDGES_KM) - 1) / len(b)).round(3).tolist()


def main(n_draws: int = 1) -> None:
    cutoff, ratio, seed = 5.0, 5, 0  # accuracy cutoff in km, absences per presence, rng seed

    if not ROADS_GPKG.exists():
        pbf = raw_dir() / "nepal-latest.osm.pbf"
        print("extracting roads from", pbf)
        extract_motor_roads(pbf, ROADS_GPKG, CRS)
    lines = gpd.read_file(ROADS_GPKG).geometry.values

    if not RIVERS_GPKG.exists():
        pbf = raw_dir() / "nepal-latest.osm.pbf"
        print("extracting rivers from", pbf)
        extract_rivers(pbf, RIVERS_GPKG, CRS)
    rivers = gpd.read_file(RIVERS_GPKG).geometry.values

    tiles = sorted((RAW / "dem_glo30").glob("*.tif"))
    if len(tiles) != 45:
        raise SystemExit(f"Expected 45 DEM tiles in data/raw/dem_glo30, found {len(tiles)}.")
    nepal_geom, xmin, ymax, shape, transform = make_grid()
    dem = build_dem(nepal_geom, tiles, transform, shape)

    pres = load_presences(RAW / "Global_Landslide_Catalog_Export_rows.csv")
    to_utm = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
    pres["x"], pres["y"] = to_utm.transform(pres["longitude"].to_numpy(), pres["latitude"].to_numpy())
    near = pres[pres["accuracy_km"] <= cutoff]

    # coarse 300 m mask over Nepal, the cells presences sit in are never drawn as absences
    cshape = (math.ceil(shape[0] * CELL / COARSE), math.ceil(shape[1] * CELL / COARSE))
    ct = from_origin(xmin, ymax, COARSE, COARSE)
    valid = rasterize([(nepal_geom, 1)], out_shape=cshape, transform=ct, dtype="uint8").astype(bool)
    exclude = np.zeros(cshape, dtype=bool)
    for x, y in zip(pres["x"], pres["y"], strict=True):
        r, c = int((ymax - y) / COARSE), int((x - xmin) / COARSE)
        if 0 <= r < cshape[0] and 0 <= c < cshape[1]:
            exclude[r, c] = True

    # absence weights from exact distances at every valid coarse cell
    vr, vc = np.nonzero(valid)
    bg_dist = distance_to_lines(xmin + (vc + 0.5) * COARSE, ymax - (vr + 0.5) * COARSE, lines)
    pres_dist = distance_to_lines(near["x"].to_numpy(), near["y"].to_numpy(), lines)
    weights = np.zeros(cshape)
    weights[vr, vc] = presence_matched_weights(bg_dist, pres_dist)

    n_abs = ratio * len(near)
    to_ll = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)

    def draw(abs_set: str, rng: np.random.Generator, w: np.ndarray | None) -> pd.DataFrame:
        ar, ac = sample_background(valid, n_abs, rng, weights=w, exclude=exclude)
        df = pd.DataFrame({"x": xmin + (ac + 0.5) * COARSE, "y": ymax - (ar + 0.5) * COARSE})
        df["longitude"], df["latitude"] = to_ll.transform(df["x"].to_numpy(), df["y"].to_numpy())
        df["abs_set"], df["label"] = abs_set, 0
        return df

    # draw 0 uses the original seeds, so the default table does not change
    absences = []
    for k in range(n_draws):
        uniform = draw("uniform", np.random.default_rng(seed + 2 * k), None)
        weighted = draw("road_weighted", np.random.default_rng(seed + 2 * k + 1), weights)
        absences += [uniform.assign(draw=k), weighted.assign(draw=k)]
    pres = pres.assign(abs_set="presence", label=1, draw=-1)

    table = pd.concat([pres, *absences], ignore_index=True)
    table["row"] = ((ymax - table["y"]) / CELL).astype(int)
    table["col"] = ((table["x"] - xmin) / CELL).astype(int)
    table = pd.concat([table, terrain_at(dem, table["row"].to_numpy(), table["col"].to_numpy())], axis=1)
    table["road_dist_km"] = distance_to_lines(table["x"].to_numpy(), table["y"].to_numpy(), lines) / 1000.0
    table["river_dist_km"] = distance_to_lines(table["x"].to_numpy(), table["y"].to_numpy(), rivers) / 1000.0

    # rainfall: mean of the annual totals, read at the cell the point falls in
    chirps = sorted((raw_dir() / "chirps_annual").glob("chirps-v3.0.*.tif"))
    years = [int(f.stem.rsplit(".", 1)[1]) for f in chirps]
    if years != list(range(1991, 2021)):
        raise SystemExit(f"Expected CHIRPS annual files 1991 to 2020 in {raw_dir() / 'chirps_annual'}, found {years}.")
    rain, rain_t = mean_annual_rainfall(chirps, (79.9, 26.2, 88.4, 30.6))
    table["rain_mm"] = sample_raster(rain, rain_t, table["longitude"].to_numpy(), table["latitude"].to_numpy())

    # land cover: share of each class within LC_RADIUS_KM of the point
    wc_tiles = sorted((raw_dir() / "worldcover").glob("ESA_WorldCover_10m_2021_v200_*_Map.tif"))
    if len(wc_tiles) != 8:
        raise SystemExit(f"Expected 8 WorldCover tiles in {raw_dir() / 'worldcover'}, found {len(wc_tiles)}.")
    shares = landcover_shares(table["longitude"].to_numpy(), table["latitude"].to_numpy(), wc_tiles, LC_RADIUS_KM)
    for name, values in shares.items():
        table[name] = values

    n_before = len(table)
    table = table.dropna(subset=["elevation", "slope", "curvature", "rain_mm", *LC_COLS])
    table = table[~table.duplicated(subset=["abs_set", "draw", "row", "col"])].copy()
    table["inside_nepal"] = contains_xy(nepal_geom, table["x"].to_numpy(), table["y"].to_numpy())

    cols = ["label", "abs_set", "event_id", "event_date", "accuracy_km", "longitude", "latitude", "x", "y",
            "elevation", "slope", "aspect_sin", "aspect_cos", "curvature", "road_dist_km", "river_dist_km",
            "rain_mm", *LC_COLS, "inside_nepal"]
    OUT.mkdir(parents=True, exist_ok=True)
    if n_draws > 1:  # extra absence draws for src/absence_variation.py; the main table and its meta stay as they are
        table[[*cols, "draw"]].to_csv(OUT / "features_roads_draws.csv", index=False)
        print(f"wrote {len(table)} rows, {n_draws} draws per scheme -> features_roads_draws.csv")
        return
    table[cols].to_csv(OUT / "features_roads.csv", index=False)

    near_t = table[(table.abs_set == "presence") & (table.accuracy_km <= cutoff)]
    meta = {
        "built": datetime.now().isoformat(timespec="seconds"),
        "crs": CRS, "cell_m": CELL, "absence_ratio": ratio, "seed": seed, "presence_cutoff_km": cutoff,
        "road_definition": "OSM highway in motorway/trunk/primary/secondary/tertiary/unclassified/residential (+_link); "
                           "no track/path/footway/steps/service",
        "motor_ways": int(len(lines)),
        "river_definition": "OSM waterway=river only (no stream, canal, ditch or drain)",
        "river_ways": int(len(rivers)),
        "rainfall": "CHIRPS v3.0 annual totals 1991 to 2020, mean, 0.05 degree cell read at the point (mm/yr)",
        "land_cover": f"ESA WorldCover 2021 v200 class shares within {LC_RADIUS_KM} km of the point; "
                      "groups tree, shrub, grass, crop, built, bare, snow, other (water, wetland, mangrove, moss); "
                      "no-data pixels left out of the shares",
        "median_rain_mm": {k: round(float(table[table.abs_set == k]["rain_mm"].median()))
                           for k in ("presence", "uniform", "road_weighted")},
        "mean_land_cover_share": {c: {k: round(float(table[table.abs_set == k][c].mean()), 3)
                                      for k in ("presence", "uniform", "road_weighted")} for c in LC_COLS},
        "median_river_dist_km": {k: round(float(table[table.abs_set == k]["river_dist_km"].median()), 2)
                                 for k in ("presence", "uniform", "road_weighted")},
        "road_distance_bins_km": BIN_EDGES_KM.tolist()[:-1] + ["inf"],
        "share_by_distance_bin": {
            "background_valid_cells": bin_shares(bg_dist),
            "presences_le_cutoff": bin_shares(near_t["road_dist_km"].to_numpy() * 1000.0),
            "uniform_absences": bin_shares(table[table.abs_set == "uniform"]["road_dist_km"].to_numpy() * 1000.0),
            "road_weighted_absences": bin_shares(table[table.abs_set == "road_weighted"]["road_dist_km"].to_numpy() * 1000.0),
        },
        "median_road_dist_km": {k: round(float(table[table.abs_set == k]["road_dist_km"].median()), 2)
                                for k in ("presence", "uniform", "road_weighted")},
        "rows_written": int(len(table)), "rows_dropped_nan_or_dupes": int(n_before - len(table)),
        "counts_by_set": table["abs_set"].value_counts().to_dict(),
        "caveats": "weights match only the road-distance marginal; presence locations are 5 km or worse so their "
                   "road distances are noisy; weights use the presences' own distances "
                   "(mild label information, set once before CV)",
        "osm": "(c) OpenStreetMap contributors, ODbL; Geofabrik Nepal extract, md5 d098ee3d64113fbc596a99fb6cf55232",
    }
    (OUT / "features_roads.meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--draws", type=int, default=1,
                        help="absence draws per scheme (more than 1 writes features_roads_draws.csv)")
    main(parser.parse_args().draws)
