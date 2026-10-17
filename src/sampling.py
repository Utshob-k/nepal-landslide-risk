"""Draw background points for a presence-only landslide catalogue.

Reported landslides tend to be closer to roads than the country as a whole. A
uniform background can therefore teach a model that road access is a landslide
signal. Optional weights let the background follow the observed road-distance
pattern instead.
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
    """Choose distinct valid cells, excluding blocked cells and using weights when given."""
    allowed = valid_mask.copy()
    if exclude is not None:
        allowed &= ~exclude
    rows, cols = np.nonzero(allowed)
    if n > len(rows):
        raise ValueError(
            f"Asked for {n} samples but only {len(rows)} cells are allowed."
        )
    if weights is None:
        probabilities = None
    else:
        w = np.clip(weights[rows, cols].astype(float), 0, None)
        if w.sum() == 0:
            raise ValueError("All sampling weights are zero inside the allowed area.")
        probabilities = w / w.sum()
        if np.count_nonzero(probabilities) < n:
            raise ValueError("Fewer non-zero-weight cells than requested samples.")
    pick = rng.choice(len(rows), size=n, replace=False, p=probabilities)
    return rows[pick], cols[pick]
