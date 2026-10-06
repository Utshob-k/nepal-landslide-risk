"""Baselines under random vs spatial-block CV, on both absence sets, with a road-distance ablation.

    .venv312/Scripts/python -m src.run_baselines

Headline setting is fixed in advance: 50 km blocks, 10 km buffer, presences
with location accuracy <= 5 km. Every other setting is reported as sensitivity.
Table: data/tables/features_roads.csv (src/build_feature_table.py). Still PARTIAL: terrain and
road distance only. Two absence sets: `uniform` and `road_weighted`; comparing them shows how much
of a score is road access. CPU only.
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .modeling import compare_cv

ROOT = Path(__file__).resolve().parents[1]
TERRAIN = ["elevation", "slope", "aspect_sin", "aspect_cos", "curvature"]
SEEDS = range(5)
HEADLINE = {"cutoff": 5.0, "block_km": 50, "buffer_km": 10}

def _logit(max_iter: int = 100):
    return lambda: make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=max_iter))


def _forest():
    return lambda: RandomForestClassifier(200, min_samples_leaf=5, class_weight="balanced", n_jobs=4, random_state=0)


MODELS = {
    "Slope only (logistic)": (["slope"], _logit()),
    "Elevation only (logistic)": (["elevation"], _logit()),
    "Logistic regression": (TERRAIN, _logit(1000)),
    "Random forest": (TERRAIN, _forest()),
    "Random forest + road distance (ablation)": (TERRAIN + ["log_road_dist"], _forest()),
    "Road distance only (logistic)": (["log_road_dist"], _logit()),
}


def run(table: pd.DataFrame, abs_set: str, cutoff: float, block_km: int, buffer_km: int) -> list[dict]:
    d = table[((table.label == 1) & (table.accuracy_km <= cutoff)) | (table.abs_set == abs_set)]
    y = d.label.to_numpy()
    coords = d[["x", "y"]].to_numpy()  # UTM 45N metres
    rows = []
    for name, (cols, make) in MODELS.items():
        X = d[cols].to_numpy()
        res = [compare_cv(make, X, y, coords, block_km * 1000.0, 5, buffer_km * 1000.0, seed) for seed in SEEDS]
        spatial = [r["spatial"]["auroc_mean"] for r in res]
        rows.append({
            "abs_set": abs_set,
            "model": name,
            "cutoff_km": cutoff,
            "block_km": block_km,
            "buffer_km": buffer_km,
            "n_pres": int(y.sum()),
            "n_abs": int((1 - y).sum()),
            "random_auroc": np.mean([r["random"]["auroc_mean"] for r in res]),
            "spatial_auroc": np.mean(spatial),
            "spatial_auroc_sd_over_seeds": np.std(spatial),
            "gap": np.mean([r["auroc_gap"] for r in res]),
            "random_prauc": np.mean([r["random"]["pr_auc_mean"] for r in res]),
            "spatial_prauc": np.mean([r["spatial"]["pr_auc_mean"] for r in res]),
            "spatial_folds_scored": np.mean([r["spatial"]["folds"] for r in res]),
        })
    return rows


def main() -> None:
    table = pd.read_csv(ROOT / "data" / "tables" / "features_roads.csv")
    table["log_road_dist"] = np.log(table["road_dist_km"] + 0.01)
    out = []
    grid = itertools.product(["uniform", "road_weighted"], [5.0, 10.0], [25, 50, 100], [0, 10])
    for abs_set, cutoff, block_km, buffer_km in grid:
        print("running", abs_set, cutoff, block_km, buffer_km, flush=True)
        out += run(table, abs_set, cutoff, block_km, buffer_km)
    df = pd.DataFrame(out)
    (ROOT / "results").mkdir(exist_ok=True)
    df.to_csv(ROOT / "results" / "baselines_roads.csv", index=False)

    pd.set_option("display.width", 200)
    h = df[(df.cutoff_km == HEADLINE["cutoff"]) & (df.block_km == HEADLINE["block_km"]) & (df.buffer_km == HEADLINE["buffer_km"])]
    print("\nHEADLINE (50 km blocks, 10 km buffer, <=5 km accuracy), mean over 5 seeds")
    show = ["abs_set", "model", "spatial_auroc", "spatial_auroc_sd_over_seeds", "random_auroc", "gap",
            "spatial_prauc", "n_pres", "n_abs"]
    print(h[show].round(3).to_string(index=False))
    print("\nSENSITIVITY: spatial AUROC / gap")
    for value in ("spatial_auroc", "gap"):
        piv = df.pivot_table(index=["abs_set", "model"], columns=["cutoff_km", "block_km", "buffer_km"], values=value)
        print(piv.round(2).to_string())


if __name__ == "__main__":
    main()
