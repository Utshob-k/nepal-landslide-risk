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

from .grid import CELL, COARSE, CRS, OUT, RAW, build_dem, load_presences, make_grid, terrain_at
from .roads import BIN_EDGES_KM, distance_to_lines, extract_motor_roads, extract_rivers, presence_matched_weights
from .sampling import sample_background

ROADS_GPKG = RAW.parent / "processed" / "roads_motor.gpkg"
RIVERS_GPKG = RAW.parent / "processed" / "rivers.gpkg"


def bin_shares(dist_m: np.ndarray) -> list[float]:
    b = np.digitize(dist_m / 1000.0, BIN_EDGES_KM[1:-1])
    return (np.bincount(b, minlength=len(BIN_EDGES_KM) - 1) / len(b)).round(3).tolist()


def main() -> None:
    cutoff, ratio, seed = 5.0, 5, 0  # accuracy cutoff in km, absences per presence, rng seed

    if not ROADS_GPKG.exists():
        pbf = Path(os.environ.get("NEPAL_RAW_DIR", RAW)) / "nepal-latest.osm.pbf"
        print("extracting roads from", pbf)
        extract_motor_roads(pbf, ROADS_GPKG, CRS)
    lines = gpd.read_file(ROADS_GPKG).geometry.values

    if not RIVERS_GPKG.exists():
        pbf = Path(os.environ.get("NEPAL_RAW_DIR", RAW)) / "nepal-latest.osm.pbf"
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

    uniform = draw("uniform", np.random.default_rng(seed), None)
    weighted = draw("road_weighted", np.random.default_rng(seed + 1), weights)
    pres = pres.assign(abs_set="presence", label=1)

    table = pd.concat([pres, uniform, weighted], ignore_index=True)
    table["row"] = ((ymax - table["y"]) / CELL).astype(int)
    table["col"] = ((table["x"] - xmin) / CELL).astype(int)
    table = pd.concat([table, terrain_at(dem, table["row"].to_numpy(), table["col"].to_numpy())], axis=1)
    table["road_dist_km"] = distance_to_lines(table["x"].to_numpy(), table["y"].to_numpy(), lines) / 1000.0
    table["river_dist_km"] = distance_to_lines(table["x"].to_numpy(), table["y"].to_numpy(), rivers) / 1000.0

    n_before = len(table)
    table = table.dropna(subset=["elevation", "slope", "curvature"])
    table = table[~table.duplicated(subset=["abs_set", "row", "col"])].copy()
    table["inside_nepal"] = contains_xy(nepal_geom, table["x"].to_numpy(), table["y"].to_numpy())

    cols = ["label", "abs_set", "event_id", "event_date", "accuracy_km", "longitude", "latitude", "x", "y",
            "elevation", "slope", "aspect_sin", "aspect_cos", "curvature", "road_dist_km", "river_dist_km", "inside_nepal"]
    OUT.mkdir(parents=True, exist_ok=True)
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
    main()
