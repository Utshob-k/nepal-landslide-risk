"""Choosing 'no landslide' points.

The catalogue only lists landslides someone REPORTED, and reports cluster
near roads and towns. Sampling absences uniformly across the country
therefore teaches a model "near a road = landslide", which is reporting
behaviour, not geology. Pass `weights` (for example an accessibility
surface) to draw absences with the same bias as the presences.
"""
from __future__ import annotations

import numpy as np


def sample_background(
    valid_mask: np.ndarray,
    n: int,
    rng: np.random.Generator,
    weights: np.ndarray | None = None,
    exclude: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw n distinct (row, col) cells from the valid area.

    valid_mask: bool array, True where sampling is allowed (inside Nepal, not water).
    weights:    optional non-negative array, same shape; higher = more likely.
    exclude:    optional bool array of cells to never draw (e.g. known landslide cells).
    """
    allowed = valid_mask.copy()
    if exclude is not None:
        allowed &= ~exclude
    rows, cols = np.nonzero(allowed)
    if n > len(rows):
        raise ValueError(f"Asked for {n} samples but only {len(rows)} cells are allowed.")
    if weights is None:
        p = None
    else:
        w = np.clip(weights[rows, cols].astype(float), 0, None)
        if w.sum() == 0:
            raise ValueError("All sampling weights are zero inside the allowed area.")
        p = w / w.sum()
        if np.count_nonzero(p) < n:
            raise ValueError("Fewer non-zero-weight cells than requested samples.")
    pick = rng.choice(len(rows), size=n, replace=False, p=p)
    return rows[pick], cols[pick]
