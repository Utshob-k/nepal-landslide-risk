# Nepal landslide susceptibility: how much of the score is geography?

A landslide susceptibility map for Nepal built from open terrain, rainfall and
land-cover data, with one question running through it: **how much of a typical
"model accuracy" is just spatial autocorrelation, and how much is real skill?**

> **Student research project. Not a hazard forecast and not for planning,
> evacuation or construction decisions.**

## Why this question

Landslides cluster in space, so a random train/test split puts a near-twin of
every test point into the training set. Scores come out flattering and the model
looks better than it is. Splitting by geographic block gives the honest number.
On a synthetic benchmark in `tests/` the gap between the two is about 0.3 AUROC
(random ~0.84-0.92 against spatial ~0.53-0.60) even though the model has no
terrain knowledge at all. The study measures the same gap on real data.

## Plan

1. **Presence points.** Landslide events inside Nepal from a public catalogue.
2. **Absence points.** Background cells sampled from the rest of Nepal. The
   catalogue lists only *reported* slides, and reports cluster near roads and
   towns, so absences are also drawn with an accessibility weight
   (`src/sampling.py`), and a road-distance ablation is reported.
3. **Features.** Elevation, slope, aspect (as sin/cos), a curvature proxy
   (`src/terrain.py`), mean annual rainfall, land cover, distance to rivers.
4. **Models.** A slope-only baseline, logistic regression, random forest, and
   gradient boosting. Simple first; complexity must earn its place.
5. **Evaluation.** AUROC and PR-AUC under **random** CV and under **spatial
   block** CV with a buffer (`src/cv.py`, `src/modeling.py`). The headline
   number is the spatial one and the gap to the random one.
6. **Map.** A susceptibility surface over Nepal, exported as an interactive web
   map, with the training points and an uncertainty note.

## Results

_To fill in. Keep the failures and the gap in._

| Model | Random-CV AUROC | Spatial-CV AUROC | Gap |
|---|---|---|---|
| Slope only | | | |
| Logistic regression | | | |
| Random forest | | | |
| Random forest + road distance (ablation) | | | |

## Limitations (write these honestly)

- The catalogue is a sample of reported events, not a census of landslides.
  It is biased toward places people can see and report.
- Susceptibility is not hazard: it says where slopes are prone to fail, not
  when, and not how big or how damaging.
- No geology or soil layer unless a trustworthy open one is found.
- Block size, buffer and the absence-sampling choice all move the numbers; the
  sensitivity to each is reported rather than hidden.

## Data

See [`data/README.md`](data/README.md). Nothing large is committed; each
source's licence applies.

## Run

```bash
# Python 3.12 recommended for the geospatial stack (rasterio etc.)
pip install -r requirements.txt            # core: numpy, scipy, scikit-learn
pip install -r requirements-geo.txt        # raster/vector libraries, when you reach the data step
pytest
```

## Repository layout

```
src/terrain.py    slope, aspect, curvature proxy from a DEM array
src/sampling.py   background (absence) points, optionally bias-weighted
src/cv.py         spatial block k-fold with an optional buffer, plus random k-fold
src/modeling.py   random-vs-spatial CV comparison for any sklearn model
tests/            synthetic-data tests, including the leakage demonstration
```
