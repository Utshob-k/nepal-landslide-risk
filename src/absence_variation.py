"""Measure how background sampling changes the headline score.

    python -m src.build_feature_table --draws 20
    python -m src.absence_variation

The cross-validation seed stays fixed, so differences come only from the
background draws. The script writes ``results/absence_variation.csv``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .climate_cover import LC_COLS
from .modeling import compare_cv
from .run_baselines import HEADLINE, MODELS, TERRAIN, _forest

ROOT = Path(__file__).resolve().parents[1]
USE = [
    "Slope only (logistic)",
    "Logistic regression",
    "Random forest",
    "Random forest + road distance (ablation)",
    "Random forest + rainfall",
    "Random forest + land cover",
    "Gradient boosting",
    "Gradient boosting + land cover",
]
# This variant removes classes that may primarily represent access and farmland.
NO_ACCESS = [c for c in LC_COLS if c not in ("lc_built", "lc_crop")]
EXTRA = {
    "Random forest + land cover (no built, crop)": (TERRAIN + NO_ACCESS, _forest())
}


def main() -> None:
    """Run the selected models once for every saved background draw."""
    table = pd.read_csv(ROOT / "data" / "tables" / "features_roads_draws.csv")
    table["log_road_dist"] = np.log(table["road_dist_km"] + 0.01)
    table["log_river_dist"] = np.log(table["river_dist_km"] + 0.01)
    pres = table[(table.label == 1) & (table.accuracy_km <= HEADLINE["cutoff"])]

    models = {**{n: MODELS[n] for n in USE}, **EXTRA}
    rows = []
    for (abs_set, k), absences in table[table.label == 0].groupby(["abs_set", "draw"]):
        d = pd.concat([pres, absences])
        y, coords = d.label.to_numpy(), d[["x", "y"]].to_numpy()
        for name, (cols, make) in models.items():
            res = compare_cv(
                make,
                d[cols].to_numpy(),
                y,
                coords,
                HEADLINE["block_km"] * 1000.0,
                5,
                HEADLINE["buffer_km"] * 1000.0,
                0,
            )
            rows.append(
                {
                    "abs_set": abs_set,
                    "draw": k,
                    "model": name,
                    "n_abs": int((1 - y).sum()),
                    "spatial_auroc": res["spatial"]["auroc_mean"],
                    "gap": res["auroc_gap"],
                }
            )
        print("done", abs_set, k, flush=True)

    df = pd.DataFrame(rows)
    (ROOT / "results").mkdir(exist_ok=True)
    df.to_csv(ROOT / "results" / "absence_variation.csv", index=False)
    summary = (
        df.groupby(["abs_set", "model"])
        .spatial_auroc.agg(["mean", "std", "min", "max", "count"])
        .round(3)
    )
    pd.set_option("display.width", 200)
    print(summary.to_string())


if __name__ == "__main__":
    main()
