"""Slope, aspect and a curvature proxy from a DEM array.

Row 0 is the north edge. Cell size is in metres, so reproject first.
"""
from __future__ import annotations

import numpy as np


def _gradients(dem: np.ndarray, cell_size: float) -> tuple[np.ndarray, np.ndarray]:
    """dz/dx (east) and dz/dy (north), metres per metre."""
    dz_rows, dz_cols = np.gradient(dem.astype(float), cell_size)
    return dz_cols, -dz_rows  # a row down is south


def slope_degrees(dem: np.ndarray, cell_size: float) -> np.ndarray:
    dz_e, dz_n = _gradients(dem, cell_size)
    return np.degrees(np.arctan(np.hypot(dz_e, dz_n)))


def aspect_degrees(dem: np.ndarray, cell_size: float) -> np.ndarray:
    """Compass bearing the slope faces (downhill), 0 = north. NaN on flat cells."""
    dz_e, dz_n = _gradients(dem, cell_size)
    bearing = np.degrees(np.arctan2(-dz_e, -dz_n)) % 360.0
    flat = np.hypot(dz_e, dz_n) < 1e-9
    return np.where(flat, np.nan, bearing)


def plan_curvature_proxy(dem: np.ndarray, cell_size: float) -> np.ndarray:
    """Laplacian of the DEM: positive in hollows, negative on ridges. Cheap stand-in for curvature."""
    d = dem.astype(float)
    padded = np.pad(d, 1, mode="edge")
    lap = (
        padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:] - 4 * d
    ) / (cell_size**2)
    return lap


def aspect_to_sin_cos(aspect: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Aspect is circular, so give models sin and cos."""
    rad = np.radians(aspect)
    return np.sin(rad), np.cos(rad)
