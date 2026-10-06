import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.cv import block_ids, spatial_block_kfold
from src.modeling import compare_cv
from src.sampling import sample_background
from src.terrain import aspect_degrees, slope_degrees


# ---- terrain ---------------------------------------------------------------

def test_slope_of_45_degree_plane():
    cell = 30.0
    dem = np.tile(np.arange(20) * cell, (20, 1))  # rises 30 m per 30 m cell, eastward
    assert slope_degrees(dem, cell)[10, 10] == pytest.approx(45.0)


def test_aspect_faces_downhill_in_compass_terms():
    cell = 30.0
    rising_east = np.tile(np.arange(20) * cell, (20, 1))
    assert aspect_degrees(rising_east, cell)[10, 10] == pytest.approx(270.0)  # faces west

    # Row 0 is north, so values that DEcrease with row index rise toward the north.
    rising_north = np.tile((19 - np.arange(20))[:, None] * cell, (1, 20))
    assert aspect_degrees(rising_north, cell)[10, 10] == pytest.approx(180.0)  # faces south


def test_flat_ground_has_no_aspect():
    assert np.isnan(aspect_degrees(np.zeros((5, 5)), 30.0)).all()


# ---- sampling --------------------------------------------------------------

def test_background_respects_mask_exclusions_and_uniqueness():
    rng = np.random.default_rng(0)
    mask = np.zeros((50, 50), bool)
    mask[10:40, 10:40] = True
    exclude = np.zeros_like(mask)
    exclude[20:30, 20:30] = True
    rows, cols = sample_background(mask, 200, rng, exclude=exclude)
    assert len(set(zip(rows, cols, strict=True))) == 200
    assert mask[rows, cols].all()
    assert not exclude[rows, cols].any()


def test_background_weights_bias_the_draw():
    rng = np.random.default_rng(1)
    mask = np.ones((40, 40), bool)
    weights = np.zeros((40, 40))
    weights[:, :10] = 1.0  # only the west quarter may be drawn
    _, cols = sample_background(mask, 100, rng, weights=weights)
    assert (cols < 10).all()


# ---- cross-validation ------------------------------------------------------

def test_blocks_are_never_split_across_train_and_test():
    rng = np.random.default_rng(0)
    coords = rng.uniform(0, 100_000, size=(500, 2))
    ids = block_ids(coords, 10_000)
    for train, test in spatial_block_kfold(coords, 10_000, k=5):
        assert not set(ids[train]) & set(ids[test])


def test_buffer_removes_nearby_training_points():
    rng = np.random.default_rng(0)
    coords = rng.uniform(0, 100_000, size=(800, 2))
    from scipy.spatial import cKDTree

    for train, test in spatial_block_kfold(coords, 20_000, k=4, buffer=3_000):
        d, _ = cKDTree(coords[test]).query(coords[train])
        assert (d > 3_000).all()


def test_too_few_blocks_is_an_error():
    coords = np.random.default_rng(0).uniform(0, 1_000, size=(50, 2))
    with pytest.raises(ValueError):
        list(spatial_block_kfold(coords, 10_000, k=5))


def _smooth_field(coords, rng):
    """A spatially smooth surface with ~30 km wavelength: the thing that fools random CV."""
    out = np.zeros(len(coords))
    for _ in range(6):
        kx, ky = rng.normal(0, 2 * np.pi / 30_000, size=2)
        phase = rng.uniform(0, 2 * np.pi)
        out += np.sin(kx * coords[:, 0] + ky * coords[:, 1] + phase)
    return out


def test_random_cv_overstates_skill_on_spatially_clustered_data():
    rng = np.random.default_rng(42)
    coords = rng.uniform(0, 150_000, size=(1500, 2))
    field = _smooth_field(coords, rng)
    y = (field + rng.normal(0, 0.3, len(coords)) > np.median(field)).astype(int)
    # Location is the only feature: a model can only score by memorising neighbours.
    res = compare_cv(
        lambda: RandomForestClassifier(n_estimators=80, random_state=0, n_jobs=1),
        coords, y, coords, block_size=30_000, k=5, buffer=5_000, seed=0,
    )
    # Measured over several seeds: random ~0.84-0.92, spatial ~0.53-0.60, gap 0.28-0.38.
    assert res["random"]["auroc_mean"] > 0.80
    assert res["auroc_gap"] > 0.15, res
