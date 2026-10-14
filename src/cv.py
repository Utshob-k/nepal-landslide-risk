"""Random and spatial-block k-fold. Coordinates must be in metres (projected CRS).

Random folds put near-twins of the test points in training, so scores look
too good. Holding out whole blocks doesn't.
"""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from scipy.spatial import cKDTree
from sklearn.model_selection import StratifiedKFold


def block_ids(coords: np.ndarray, block_size: float) -> np.ndarray:
    """Square block id for each point. coords is (n, 2) in metres."""
    ix = np.floor(coords[:, 0] / block_size).astype(np.int64)
    iy = np.floor(coords[:, 1] / block_size).astype(np.int64)
    _, ids = np.unique(np.stack([ix, iy], axis=1), axis=0, return_inverse=True)
    return ids.ravel()


def spatial_block_kfold(
    coords: np.ndarray,
    block_size: float,
    k: int = 5,
    buffer: float = 0.0,
    seed: int = 0,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (train, test) with whole blocks held out.

    buffer > 0 also drops training points within that many metres of a test point.
    """
    ids = block_ids(coords, block_size)
    n_blocks = ids.max() + 1
    if n_blocks < k:
        raise ValueError(f"Only {n_blocks} blocks for {k} folds; use a smaller block_size.")
    rng = np.random.default_rng(seed)
    fold_of_block = rng.permutation(n_blocks) % k
    fold_of_point = fold_of_block[ids]
    for f in range(k):
        test = np.where(fold_of_point == f)[0]
        train = np.where(fold_of_point != f)[0]
        if buffer > 0 and len(test) and len(train):
            dist, _ = cKDTree(coords[test]).query(coords[train])
            train = train[dist > buffer]
        yield train, test


def random_kfold(y: np.ndarray, k: int = 5, seed: int = 0) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Plain stratified k-fold, for comparison."""
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    yield from skf.split(np.zeros(len(y)), y)
