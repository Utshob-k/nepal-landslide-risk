"""Terrain derivatives from a DEM held as a 2-D numpy array.

Row 0 is the NORTH edge (the usual raster layout), so row index increases
southward. Cell size is in metres, so reproject a geographic DEM to a
metric CRS first, or slopes will be wrong.
"""
from __future__ import annotations

import numpy as np


def _gradients(dem: np.ndarray, cell_size: float) -> tuple[np.ndarray, np.ndarray]:
    """(dz/dx towards east, dz/dy towards north), both in metres per metre."""
    dz_rows, dz_cols = np.gradient(dem.astype(float), cell_size)
    return dz_cols, -dz_rows  # moving down a row goes south, so north = -rows


def slope_degrees(dem: np.ndarray, cell_size: float) -> np.ndarray:
    dz_e, dz_n = _gradients(dem, cell_size)
    return np.degrees(np.arctan(np.hypot(dz_e, dz_n)))


def aspect_degrees(dem: np.ndarray, cell_size: float) -> np.ndarray:
    """Compass bearing (0 = north, clockwise) the slope FACES, i.e. downhill.

    Flat cells have no aspect and come back as NaN.
    """
    dz_e, dz_n = _gradients(dem, cell_size)
    bearing = np.degrees(np.arctan2(-dz_e, -dz_n)) % 360.0
    flat = np.hypot(dz_e, dz_n) < 1e-9
    return np.where(flat, np.nan, bearing)


def plan_curvature_proxy(dem: np.ndarray, cell_size: float) -> np.ndarray:
    """Laplacian of the DEM: positive in hollows (convergent), negative on ridges.

    A cheap stand-in for curvature, good enough as one input feature.
    """
    d = dem.astype(float)
    padded = np.pad(d, 1, mode="edge")
    lap = (
        padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:] - 4 * d
    ) / (cell_size**2)
    return lap


def aspect_to_sin_cos(aspect: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Aspect is circular (359 is next to 1); feed models sin/cos, not raw degrees."""
    rad = np.radians(aspect)
    return np.sin(rad), np.cos(rad)
