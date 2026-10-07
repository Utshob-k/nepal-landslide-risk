"""Tests for rainfall averaging and land-cover shares, on small synthetic rasters."""
import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin  # noqa: E402

from src.climate_cover import LC_COLS, landcover_shares, mean_annual_rainfall, sample_raster  # noqa: E402


def _write(path, arr, left, top, res, dtype):
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1, dtype=dtype,
                       crs="EPSG:4326", transform=from_origin(left, top, res, res)) as dst:
        dst.write(arr.astype(dtype), 1)


# ---- rainfall --------------------------------------------------------------

def test_mean_rainfall_averages_the_years_and_crops(tmp_path):
    files = []
    for k, value in enumerate([1000.0, 2000.0, 3000.0]):
        grid = np.full((40, 60), value, dtype="float32")
        grid[:, 20:] += 600.0  # lon 81.0 and east is wetter in every year
        f = tmp_path / f"y{k}.tif"
        _write(f, grid, 80.0, 30.0, 0.05, "float32")
        files.append(f)
    arr, transform = mean_annual_rainfall(files, (80.5, 29.0, 81.5, 29.5))
    assert arr.shape == (10, 20)
    assert arr[:, :10].mean() == pytest.approx(2000.0)
    assert arr[:, 10:].mean() == pytest.approx(2600.0)  # lon 81.0 and east
    got = sample_raster(arr, transform, np.array([80.6, 81.4]), np.array([29.3, 29.3]))
    assert got == pytest.approx([2000.0, 2600.0])


def test_sample_raster_gives_nan_outside_the_array(tmp_path):
    arr = np.arange(12, dtype="float64").reshape(3, 4)
    t = from_origin(80.0, 30.0, 0.05, 0.05)
    got = sample_raster(arr, t, np.array([80.01, 79.9, 80.01, 80.21]), np.array([29.99, 29.99, 30.5, 29.99]))
    assert got[0] == 0
    assert np.isnan(got[1:]).all()  # west of, north of and east of the array


# ---- land cover ------------------------------------------------------------

def test_shares_sum_to_one_and_follow_the_split(tmp_path):
    tile = np.full((400, 400), 10, dtype="uint8")  # tree
    tile[:, 200:] = 40  # crop in the east half
    _write(tmp_path / "t.tif", tile, 84.0, 30.0, 0.002, "uint8")
    lon = np.array([84.4, 84.2, 84.7])  # on the split (col 200), well inside the west, well inside the east
    lat = np.full(3, 29.6)
    s = landcover_shares(lon, lat, [tmp_path / "t.tif"], radius_km=2.5)
    assert sum(s[k] for k in LC_COLS) == pytest.approx(np.ones(3))
    assert s["lc_tree"][0] == pytest.approx(0.5, abs=0.03)
    assert s["lc_tree"][1] == pytest.approx(1.0) and s["lc_crop"][2] == pytest.approx(1.0)


def test_circle_spans_two_tiles_and_ignores_missing_and_nodata(tmp_path):
    west = np.full((400, 400), 30, dtype="uint8")  # grass
    east = np.full((400, 400), 60, dtype="uint8")  # bare
    east[:, :5] = 0  # a thin no-data strip along the seam
    _write(tmp_path / "w.tif", west, 84.0, 30.0, 0.002, "uint8")
    _write(tmp_path / "e.tif", east, 84.8, 30.0, 0.002, "uint8")
    tiles = [tmp_path / "w.tif", tmp_path / "e.tif"]
    s = landcover_shares(np.array([84.8, 85.3]), np.array([29.6, 29.6]), tiles, radius_km=2.5)
    assert s["lc_grass"][0] > 0.3 and s["lc_bare"][0] > 0.3  # the circle crosses the seam
    assert s["lc_grass"][0] + s["lc_bare"][0] == pytest.approx(1.0)  # the 0 pixels are not counted
    assert s["lc_bare"][1] == pytest.approx(1.0)

    far = landcover_shares(np.array([90.0]), np.array([29.6]), tiles, radius_km=2.5)
    assert np.isnan(far["lc_tree"][0])  # no tile there at all


def test_a_tile_off_the_pixel_grid_is_rejected(tmp_path):
    _write(tmp_path / "t.tif", np.ones((50, 50), dtype="uint8"), 84.0005, 30.0, 0.002, "uint8")
    with pytest.raises(ValueError, match="pixel grid"):
        landcover_shares(np.array([84.05]), np.array([29.95]), [tmp_path / "t.tif"])
