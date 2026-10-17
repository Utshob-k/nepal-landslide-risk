"""Read rainfall and land-cover features at longitude/latitude locations.

Rainfall is the mean of annual CHIRPS totals. Its cells are roughly 5.5 km wide,
so it describes broad climate rather than local storms.

Single WorldCover pixels are unreliable for events whose locations may be up to
five kilometres off. The project therefore uses the share of each class inside a
2.5 km circle. Only the needed windows are read. WorldCover tiles must share the
standard pixel grid anchored at ``(-180, 90)``. Zeros represent no data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import from_bounds

M_PER_DEG = 111_320.0

# WorldCover classes grouped into the features used by the models.
GROUPS = {
    "lc_tree": (10,),
    "lc_shrub": (20,),
    "lc_grass": (30,),
    "lc_crop": (40,),
    "lc_built": (50,),
    "lc_bare": (60,),
    "lc_snow": (70,),
    "lc_other": (80, 90, 95, 100),
}
LC_COLS = list(GROUPS)


def mean_annual_rainfall(files: list[Path], bounds: tuple[float, float, float, float]):
    """Average annual rainfall over matching rasters within geographic bounds.

    ``bounds`` is ``(left, bottom, right, top)`` in degrees. Missing values are
    excluded from each cell's average. The function returns the cropped raster
    and its transform.
    """
    if not files:
        raise ValueError("At least one annual rainfall raster is required.")
    with rasterio.open(files[0]) as first:
        window = from_bounds(*bounds, first.transform).round_offsets().round_lengths()
        transform = first.window_transform(window)
        reference = (first.crs, first.transform, first.width, first.height)
    total = np.zeros((int(window.height), int(window.width)), dtype="float64")
    counts = np.zeros_like(total, dtype=np.uint16)
    for f in files:
        with rasterio.open(f) as src:
            if (src.crs, src.transform, src.width, src.height) != reference:
                raise ValueError(
                    f"Rainfall raster {f.name} is not on the same grid as {files[0].name}."
                )
            a = src.read(1, window=window, masked=True).astype("float64")
        valid = ~np.ma.getmaskarray(a) & np.isfinite(a.data)
        total[valid] += a.data[valid]
        counts[valid] += 1
    mean = np.full_like(total, np.nan)
    np.divide(total, counts, out=mean, where=counts > 0)
    return mean, transform


def sample_raster(
    arr: np.ndarray, transform, lon: np.ndarray, lat: np.ndarray
) -> np.ndarray:
    """Return each point's raster-cell value, or ``NaN`` outside the raster."""
    rows = np.floor((transform.f - np.asarray(lat)) / -transform.e).astype(int)
    cols = np.floor((np.asarray(lon) - transform.c) / transform.a).astype(int)
    ok = (rows >= 0) & (rows < arr.shape[0]) & (cols >= 0) & (cols < arr.shape[1])
    out = np.full(len(rows), np.nan)
    out[ok] = arr[rows[ok], cols[ok]]
    return out


def landcover_shares(
    lon: np.ndarray, lat: np.ndarray, tiles: list[Path], radius_km: float = 2.5
) -> dict[str, np.ndarray]:
    """Calculate land-cover shares within a radius of every point.

    A point with no valid WorldCover pixels receives ``NaN`` for every class.
    """
    if not tiles:
        raise ValueError("At least one WorldCover tile is required.")
    if radius_km <= 0:
        raise ValueError("The land-cover radius must be greater than zero.")
    srcs = [rasterio.open(t) for t in tiles]
    try:
        res = srcs[0].res[0]
        origins = []
        for s in srcs:
            r0 = (90.0 - s.bounds.top) / res
            c0 = (s.bounds.left + 180.0) / res
            if not (
                np.isclose(r0, round(r0), rtol=0, atol=1e-3)
                and np.isclose(c0, round(c0), rtol=0, atol=1e-3)
            ):
                raise ValueError(
                    f"{s.name} is not on the pixel grid anchored at (-180, 90)"
                )
            origins.append((round(r0), round(c0), s.height, s.width))

        lon, lat = np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)
        out = {k: np.full(len(lon), np.nan) for k in GROUPS}
        radius_m = radius_km * 1000.0
        for i, (x, y) in enumerate(zip(lon, lat, strict=True)):
            half_rows = radius_m / (res * M_PER_DEG)
            half_cols = half_rows / np.cos(np.radians(y))
            prow, pcol = (90.0 - y) / res, (x + 180.0) / res
            r0, r1 = int(np.floor(prow - half_rows)), int(np.ceil(prow + half_rows))
            c0, c1 = int(np.floor(pcol - half_cols)), int(np.ceil(pcol + half_cols))
            canvas = np.zeros((r1 - r0, c1 - c0), dtype=np.uint8)
            for s, (tr0, tc0, th, tw) in zip(srcs, origins, strict=True):
                a, b = max(r0, tr0), min(r1, tr0 + th)
                c, d = max(c0, tc0), min(c1, tc0 + tw)
                if a >= b or c >= d:
                    continue
                win = ((a - tr0, b - tr0), (c - tc0, d - tc0))
                canvas[a - r0 : b - r0, c - c0 : d - c0] = s.read(1, window=win)
            rr = (np.arange(r0, r1) + 0.5 - prow) * res * M_PER_DEG
            cc = (
                (np.arange(c0, c1) + 0.5 - pcol)
                * res
                * M_PER_DEG
                * np.cos(np.radians(y))
            )
            inside = (rr[:, None] ** 2 + cc[None, :] ** 2) <= radius_m**2
            vals = canvas[inside & (canvas > 0)]
            if vals.size == 0:
                continue
            counts = np.bincount(vals, minlength=101)
            for k, codes in GROUPS.items():
                out[k][i] = counts[list(codes)].sum() / vals.size
        return out
    finally:
        for s in srcs:
            s.close()
