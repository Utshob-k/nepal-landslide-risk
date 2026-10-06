"""Tests for the road helpers, the table builders and the baselines runner.

All synthetic: nothing here needs the files in data/. Tests that need the geo
stack (shapely, rasterio, geopandas) skip themselves when it is not installed.
"""
import numpy as np
import pytest

from src.roads import BIN_EDGES_KM, MOTOR, presence_matched_weights


def _bin_shares(dist_m):
    b = np.digitize(dist_m / 1000.0, BIN_EDGES_KM[1:-1])
    return np.bincount(b, minlength=len(BIN_EDGES_KM) - 1) / len(b)


# ---- road weights ----------------------------------------------------------

def test_motor_roads_exclude_footpaths_and_tracks():
    for kind in ("footway", "path", "steps", "track", "cycleway", "service"):
        assert kind not in MOTOR
    assert {"primary", "residential", "unclassified"} <= MOTOR


def test_weights_make_drawn_absences_follow_the_presence_profile():
    rng = np.random.default_rng(0)
    background = rng.uniform(0, 60_000, 100_000)  # metres, spread over every distance bin
    presences = np.concatenate([rng.uniform(0, 150, 150), rng.uniform(150, 3000, 50)])
    w = presence_matched_weights(background, presences)
    assert (w >= 0).all() and np.isfinite(w).all()

    drawn = rng.choice(len(background), size=20_000, p=w / w.sum())
    got, want = _bin_shares(background[drawn]), _bin_shares(presences)
    assert np.abs(got - want).max() < 0.02


def test_cells_in_bins_without_presences_get_zero_weight():
    background = np.array([50.0, 400.0, 30_000.0, 45_000.0])
    presences = np.array([20.0, 80.0, 300.0])  # nobody reports from 20 km out
    w = presence_matched_weights(background, presences)
    assert w[2] == 0 and w[3] == 0
    assert w[0] > 0 and w[1] > 0


def test_weights_do_not_blow_up_when_background_misses_a_presence_bin():
    background = np.array([50.0, 60.0, 70.0])  # nothing farther than 100 m
    presences = np.array([20.0, 5_000.0])  # one presence sits in a bin with no background
    w = presence_matched_weights(background, presences)
    assert np.isfinite(w).all()


# ---- distance to roads -----------------------------------------------------

def test_distance_to_roads_is_exact_and_chunk_independent():
    shapely = pytest.importorskip("shapely")
    from src.roads import distance_to_roads

    lines = [shapely.LineString([(0, 0), (1000, 0)]), shapely.LineString([(0, 5000), (1000, 5000)])]
    x = np.array([500.0, -400.0, 500.0, 1300.0, 500.0])
    y = np.array([300.0, 0.0, 4000.0, 400.0, 2400.0])
    want = np.array([300.0, 400.0, 1000.0, 500.0, 2400.0])
    assert distance_to_roads(x, y, lines) == pytest.approx(want)
    assert distance_to_roads(x, y, lines, chunk=2) == pytest.approx(want)


def test_a_point_on_a_road_is_zero_metres_away():
    shapely = pytest.importorskip("shapely")
    from src.roads import distance_to_roads

    line = [shapely.LineString([(0, 0), (1000, 1000)])]
    assert distance_to_roads(np.array([500.0]), np.array([500.0]), line)[0] == pytest.approx(0.0)


# ---- table builders --------------------------------------------------------

def test_terrain_at_reads_slope_aspect_and_curvature_from_a_patch():
    pytest.importorskip("rasterio")
    pytest.importorskip("geopandas")
    from src.grid import CELL, terrain_at

    dem = np.tile(np.arange(30) * CELL, (30, 1)).astype("float32")  # rises 45 degrees eastward
    t = terrain_at(dem, np.array([15]), np.array([10]))
    assert t.slope[0] == pytest.approx(45.0, abs=1e-4)
    assert t.elevation[0] == pytest.approx(10 * CELL)
    assert t.aspect_sin[0] == pytest.approx(-1.0)  # faces west
    assert t.aspect_cos[0] == pytest.approx(0.0, abs=1e-6)
    assert t.curvature[0] == pytest.approx(0.0, abs=1e-9)


def test_terrain_at_returns_nan_at_the_edge_and_next_to_missing_data():
    pytest.importorskip("rasterio")
    pytest.importorskip("geopandas")
    from src.grid import terrain_at

    dem = np.random.default_rng(0).uniform(100, 200, (30, 30)).astype("float32")
    dem[15, 15] = np.nan
    t = terrain_at(dem, np.array([0, 15, 14, 29]), np.array([15, 15, 15, 15]))
    assert t.elevation.isna().tolist() == [True, True, True, True]  # edge, hole, neighbour of hole, edge


def test_flat_cell_gets_zero_aspect_not_nan():
    pytest.importorskip("rasterio")
    pytest.importorskip("geopandas")
    from src.grid import terrain_at

    t = terrain_at(np.full((20, 20), 500.0, dtype="float32"), np.array([10]), np.array([10]))
    assert t.slope[0] == 0 and t.aspect_sin[0] == 0 and t.aspect_cos[0] == 0


def test_load_presences_keeps_nepal_and_maps_accuracy(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("rasterio")
    pytest.importorskip("geopandas")
    from src.grid import load_presences

    csv = tmp_path / "glc.csv"
    pd.DataFrame({
        "country_name": ["Nepal", " nepal ", "India", "Nepal", "Nepal"],
        "location_accuracy": ["exact", "5km", "1km", "unknown", "50km"],
        "event_id": [1, 2, 3, 4, 5],
        "event_date": ["08/01/2008 12:00:00 AM", "07/15/2012 12:00:00 AM", "01/01/2010 12:00:00 AM",
                       "not a date", "03/02/2015 12:00:00 AM"],
        "longitude": [85.0, 84.0, 77.0, 83.0, 86.0],
        "latitude": [27.0, 28.0, 20.0, 29.0, 27.5],
    }).to_csv(csv, index=False)
    out = load_presences(csv)
    assert out.event_id.tolist() == [1, 2, 4, 5]  # India dropped
    assert out.accuracy_km.iloc[:2].tolist() == [0.0, 5.0]
    assert np.isnan(out.accuracy_km.iloc[2]) and out.accuracy_km.iloc[3] == 50.0
    assert pd.isna(out.event_date.iloc[2]) and out.event_date.iloc[0].year == 2008


def test_bin_shares_sum_to_one_and_count_correctly():
    pytest.importorskip("rasterio")
    pytest.importorskip("geopandas")
    from src.build_feature_table import bin_shares

    shares = bin_shares(np.array([10.0, 20.0, 700.0, 3000.0]))
    assert sum(shares) == pytest.approx(1.0, abs=0.01)
    assert shares[0] == 0.5  # two of four within 100 m


# ---- baselines runner ------------------------------------------------------

def test_run_uses_the_right_presences_and_absence_set(monkeypatch):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("sklearn")
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from src import run_baselines

    rng = np.random.default_rng(0)
    n = 120
    rows = []
    for abs_set, label, count in [("presence", 1, n), ("uniform", 0, 200), ("road_weighted", 0, 200)]:
        rows.append(pd.DataFrame({
            "label": label, "abs_set": abs_set,
            "accuracy_km": rng.choice([1.0, 5.0, 25.0], count) if label else np.nan,
            "x": rng.uniform(0, 400_000, count), "y": rng.uniform(0, 200_000, count),
            "slope": rng.normal(20 + 5 * label, 8, count),
        }))
    table = pd.concat(rows, ignore_index=True)

    one_model = {"Slope only": (["slope"], lambda: make_pipeline(StandardScaler(), LogisticRegression()))}
    monkeypatch.setattr(run_baselines, "MODELS", one_model)
    monkeypatch.setattr(run_baselines, "SEEDS", range(2))

    res = run_baselines.run(table, "road_weighted", cutoff=5.0, block_km=50, buffer_km=0)[0]
    n_close = int(((table.label == 1) & (table.accuracy_km <= 5.0)).sum())
    assert res["n_pres"] == n_close
    assert res["n_abs"] == 200  # only the road_weighted absences, not the uniform ones
    assert res["spatial_folds_scored"] > 0
    assert 0.5 < res["random_auroc"] <= 1.0  # slope carries real signal in this fake table
