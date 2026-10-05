"""Compare random vs spatial cross-validation for the same model.

The gap between the two numbers is the project's headline finding: how
much of a naive score is just spatial autocorrelation.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import average_precision_score, roc_auc_score

from .cv import random_kfold, spatial_block_kfold


def _score(model: BaseEstimator, X, y, splits) -> dict:
    aucs, aps = [], []
    for train, test in splits:
        if len(set(y[test])) < 2 or len(set(y[train])) < 2:
            continue  # a fold with one class has no AUROC
        model.fit(X[train], y[train])
        p = model.predict_proba(X[test])[:, 1]
        aucs.append(roc_auc_score(y[test], p))
        aps.append(average_precision_score(y[test], p))
    return {
        "auroc_mean": float(np.mean(aucs)),
        "auroc_std": float(np.std(aucs)),
        "pr_auc_mean": float(np.mean(aps)),
        "folds": len(aucs),
    }


def compare_cv(
    make_model: Callable[[], BaseEstimator],
    X: np.ndarray,
    y: np.ndarray,
    coords: np.ndarray,
    block_size: float,
    k: int = 5,
    buffer: float = 0.0,
    seed: int = 0,
) -> dict:
    """Score a freshly built model under random and spatial-block CV."""
    random_res = _score(make_model(), X, y, random_kfold(y, k, seed))
    spatial_res = _score(make_model(), X, y, spatial_block_kfold(coords, block_size, k, buffer, seed))
    return {
        "random": random_res,
        "spatial": spatial_res,
        "auroc_gap": random_res["auroc_mean"] - spatial_res["auroc_mean"],
    }
