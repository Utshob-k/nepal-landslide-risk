"""Derive local terrain features from an elevation raster.

The first row represents the northern edge of the raster. Elevation values must
be on a projected grid with cell sizes measured in metres.
"""

from __future__ import annotations

import numpy as np


def _gradients(dem: np.ndarray, cell_size: float) -> tuple[np.ndarray, np.ndarray]:
    """Return elevation change to the east and north, in metres per metre."""
    dz_rows, dz_cols = np.gradient(dem.astype(float), cell_size)
    return dz_cols, -dz_rows  # Raster rows increase towards the south.


def slope_degrees(dem: np.ndarray, cell_size: float) -> np.ndarray:
    """Return the downhill steepness at every raster cell, in degrees."""
    dz_e, dz_n = _gradients(dem, cell_size)
    return np.degrees(np.arctan(np.hypot(dz_e, dz_n)))


def aspect_degrees(dem: np.ndarray, cell_size: float) -> np.ndarray:
    """Return the downhill compass bearing, with north as 0 degrees.

    Flat cells have no meaningful direction, so they are returned as ``NaN``.
    """
    dz_e, dz_n = _gradients(dem, cell_size)
    bearing = np.degrees(np.arctan2(-dz_e, -dz_n)) % 360.0
    flat = np.hypot(dz_e, dz_n) < 1e-9
    return np.where(flat, np.nan, bearing)


def plan_curvature_proxy(dem: np.ndarray, cell_size: float) -> np.ndarray:
    """Approximate curvature from neighbouring elevation values.

    Positive values tend to occur in hollows and negative values on ridges. This
    is a simple proxy, not a full geomorphometric curvature calculation.
    """
    d = dem.astype(float)
    padded = np.pad(d, 1, mode="edge")
    lap = (
        padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
        - 4 * d
    ) / (cell_size**2)
    return lap


def aspect_to_sin_cos(aspect: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Represent circular aspect values as sine and cosine features."""
    rad = np.radians(aspect)
    return np.sin(rad), np.cos(rad)
