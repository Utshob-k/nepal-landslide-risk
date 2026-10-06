"""Can the CV code see leakage on this data? Give the model the coordinates and find out.

    python -m src.leakage_check

The baselines show a random-vs-spatial gap near zero. That is only believable if
the same code does show a gap when the model can use location. So: score a forest
on terrain, on terrain + x, y, and on x, y alone, under random and spatial CV, at
the headline setting. Coordinates should score well under random CV (neighbours
sit in both train and test) and fall under spatial CV. Also prints how clustered
the presences are. Then repeats terrain and x, y only at larger blocks (100, 200 km):
if x, y keeps scoring there, the signal is regional, not local. Writes
results/leakage_check.csv.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .modeling import compare_cv
from .run_baselines import HEADLINE, SEEDS, TERRAIN, _forest

ROOT = Path(__file__).resolve().parents[1]
BLOCKS_KM = (50, 100, 200)
SETS = {
    "terrain": TERRAIN,
    "terrain + x, y": [*TERRAIN, "x", "y"],
    "x, y only": ["x", "y"],
}


def nn_km(xy: np.ndarray) -> np.ndarray:
    """Distance in km from each point to its nearest other point."""
    d, _ = cKDTree(xy).query(xy, k=2)
    return d[:, 1] / 1000.0


def main() -> None:
    table = pd.read_csv(ROOT / "data" / "tables" / "features_roads.csv")
    pres = table[(table.label == 1) & (table.accuracy_km <= HEADLINE["cutoff"])]

    rows = []
    for abs_set in ("uniform", "road_weighted"):
        d = pd.concat([pres, table[table.abs_set == abs_set]])
        y, coords = d.label.to_numpy(), d[["x", "y"]].to_numpy()
        for block_km in BLOCKS_KM:
            for name, cols in SETS.items():
                if block_km != HEADLINE["block_km"] and name == "terrain + x, y":
                    continue  # the larger blocks only need terrain and x, y alone
                res = [compare_cv(_forest(), d[cols].to_numpy(), y, coords, block_km * 1000.0, 5,
                                  HEADLINE["buffer_km"] * 1000.0, seed) for seed in SEEDS]
                rows.append({
                    "abs_set": abs_set, "block_km": block_km, "features": name,
                    "random_auroc": np.mean([r["random"]["auroc_mean"] for r in res]),
                    "spatial_auroc": np.mean([r["spatial"]["auroc_mean"] for r in res]),
                    "gap": np.mean([r["auroc_gap"] for r in res]),
                    "spatial_folds_scored": np.mean([r["spatial"]["folds"] for r in res]),
                })
        print("done", abs_set, flush=True)

    df = pd.DataFrame(rows)
    (ROOT / "results").mkdir(exist_ok=True)
    df.to_csv(ROOT / "results" / "leakage_check.csv", index=False)
    pd.set_option("display.width", 200)
    print(df.round(3).to_string(index=False))

    print("\nnearest-neighbour distance, km (same point set, itself excluded)")
    for label, pts in [("presences", pres), ("uniform absences", table[table.abs_set == "uniform"]),
                       ("road_weighted absences", table[table.abs_set == "road_weighted"])]:
        q = np.percentile(nn_km(pts[["x", "y"]].to_numpy()), [10, 50, 90])
        print(f"  {label:24s} n={len(pts):4d}  p10 {q[0]:5.1f}  median {q[1]:5.1f}  p90 {q[2]:5.1f}")
    print(f"  (blocks are {HEADLINE['block_km']} km, buffer {HEADLINE['buffer_km']} km)")


if __name__ == "__main__":
    main()
