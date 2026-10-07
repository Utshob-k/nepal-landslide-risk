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

1. Presence points. Landslide events inside Nepal from a public catalogue.
2. Absence points. Background cells sampled from the rest of Nepal. The
   catalogue lists only *reported* slides, and reports cluster near roads and
   towns, so absences are also drawn with an accessibility weight
   (`src/sampling.py`), and a road-distance ablation is reported.
3. Features. Elevation, slope, aspect (as sin/cos), a curvature proxy
   (`src/terrain.py`), mean annual rainfall, land cover, distance to rivers.
4. Models. A slope-only baseline, logistic regression, random forest, and
   gradient boosting. Simple first; complexity must earn its place.
5. Evaluation. AUROC and PR-AUC under random CV and under spatial
   block CV with a buffer (`src/cv.py`, `src/modeling.py`). The headline
   number is the spatial one and the gap to the random one.
6. Map. A susceptibility surface over Nepal, exported as an interactive web
   map, with the training points and an uncertainty note.

## Results

Work in progress. So far the features are terrain (elevation, slope, aspect,
a curvature proxy), distance to the nearest motor road and to the nearest
river, mean annual rainfall, and land-cover shares around the point. Gradient
boosting hasn't been run. These numbers describe a partial model and are not a
hazard forecast.

Setup, fixed before running: 5-fold spatial block CV, 50 km blocks, 10 km
buffer, 194 presences (catalogue location accuracy 5 km or better), 975
absences, averaged over 5 seeds for the fold assignment. Gap is random-CV AUROC
minus spatial-CV AUROC. The spatial number is the one to read; the random column
is there to show the gap. Every setting is in `results/baselines_roads.csv`.
The tables below use one absence draw per scheme; the next section redraws the
absences 20 times.

Absences drawn to match the presences' distance to roads (my preferred setting,
see Limitations):

| Model | Spatial-CV AUROC | Random-CV AUROC | Gap |
|---|---|---|---|
| Slope only | 0.67 | 0.67 | -0.01 |
| Logistic regression (terrain) | 0.68 | 0.68 | 0.00 |
| Random forest (terrain) | 0.72 | 0.73 | 0.01 |
| Random forest + road distance (ablation) | 0.72 | 0.74 | 0.02 |
| Road distance only (diagnostic) | 0.44 | 0.47 | 0.03 |
| Random forest + river distance | 0.73 | 0.74 | 0.02 |
| Random forest + road + river distance | 0.73 | 0.75 | 0.02 |
| River distance only (diagnostic) | 0.49 | 0.51 | 0.02 |
| Random forest + rainfall | 0.71 | 0.74 | 0.03 |
| Random forest + land cover | 0.76 | 0.80 | 0.04 |
| Random forest + rainfall + land cover | 0.75 | 0.79 | 0.05 |
| Rainfall only (diagnostic) | 0.55 | 0.57 | 0.03 |
| Land cover only (diagnostic) | 0.73 | 0.78 | 0.05 |

Absences drawn uniformly over Nepal:

| Model | Spatial-CV AUROC | Random-CV AUROC | Gap |
|---|---|---|---|
| Slope only | 0.53 | 0.53 | 0.00 |
| Logistic regression (terrain) | 0.66 | 0.68 | 0.02 |
| Random forest (terrain) | 0.77 | 0.77 | 0.00 |
| Random forest + road distance (ablation) | 0.81 | 0.82 | 0.01 |
| Road distance only (diagnostic) | 0.74 | 0.74 | 0.00 |
| Random forest + river distance | 0.77 | 0.78 | 0.01 |
| Random forest + road + river distance | 0.81 | 0.83 | 0.01 |
| River distance only (diagnostic) | 0.58 | 0.59 | 0.01 |
| Random forest + rainfall | 0.77 | 0.80 | 0.02 |
| Random forest + land cover | 0.80 | 0.83 | 0.03 |
| Random forest + rainfall + land cover | 0.81 | 0.84 | 0.03 |
| Rainfall only (diagnostic) | 0.64 | 0.66 | 0.02 |
| Land cover only (diagnostic) | 0.79 | 0.83 | 0.04 |

Reading the tables:

- Road access is a big shortcut. With uniform absences, road distance alone
  scores 0.74, and adding it takes the random forest from 0.77 to 0.81. With
  matched absences it adds nothing (0.72 either way). The terrain-only random
  forest at about 0.70, give or take 0.02, is the number I'd quote. The 0.72 in
  the table is one absence draw and a little high (see the redraw results
  below). It stays between 0.69 and 0.73 across the 12 block, buffer and cutoff
  settings.
- River distance adds almost nothing. With matched absences the terrain-only
  random forest goes from 0.721 to 0.728, which is smaller than the seed spread
  (0.017 to 0.021), and river distance alone scores 0.49, which is chance. With
  uniform absences it alone scores 0.58, so it looks like a weak access proxy
  and not terrain skill.
- Land cover is the first feature that adds clear skill. With matched absences
  the terrain forest goes from 0.721 to 0.758, and land cover alone scores 0.731.
  Redrawing the absences (below) puts the gain at +0.045, and it shows up in
  all 20 draws. Most of it comes from two classes, though: without the built-up
  and crop shares it falls to +0.010, which is inside the noise. Built-up share
  is probably a settlement and access proxy, and crop share separates the flat
  farmland of the Terai from the hills. So I wouldn't call the full gain skill
  from land.
- Rainfall adds nothing: 0.710 with matched absences against 0.721 for terrain
  alone, and 0.546 on its own. CHIRPS cells are about 5.5 km, so it is smooth
  across most of a block.
- The random-vs-spatial gap is close to zero for the terrain features
  (-0.02 to +0.05 over everything here; the land-cover models have the biggest
  gaps, 0.03 to 0.05), where
  the synthetic benchmark in `tests/` shows about 0.3. I checked that the CV
  code can see a gap at all by giving the model the coordinates (see the leakage
  check below); it can, so the small terrain gap is a result and not a bug.
- Some of the uniform-absence skill was road access and not terrain: the random
  forest drops from 0.77 to about 0.70 once absences are no longer easy to find
  in remote terrain.

Redrawing the absences, 20 draws per scheme at the headline setting, with the
CV fold seed fixed at 0 (`src/absence_variation.py`, `results/absence_variation.csv`).
Spatial-CV AUROC, mean ± sd across draws:

| Model | Matched absences | Uniform absences |
|---|---|---|
| Slope only | 0.652 ± 0.011 | 0.534 ± 0.019 |
| Logistic regression (terrain) | 0.665 ± 0.015 | 0.659 ± 0.020 |
| Random forest (terrain) | 0.704 ± 0.017 | 0.774 ± 0.014 |
| Random forest + road distance (ablation) | 0.704 ± 0.016 | 0.814 ± 0.014 |
| Random forest + rainfall | 0.711 ± 0.018 | 0.776 ± 0.011 |
| Random forest + land cover | 0.749 ± 0.013 | 0.819 ± 0.013 |
| Random forest + land cover, no built-up or crop | 0.713 ± 0.014 | 0.788 ± 0.016 |

- The spread from redrawing absences (0.011 to 0.020) is as large as the spread
  from fold assignment (0.017 to 0.021), so the real uncertainty on a single
  score is closer to ±0.02 or ±0.03 than to ±0.01.
- The one draw in the tables above scored 0.711 with this fold seed against a
  mean of 0.704, so it was a bit lucky. The conclusions don't change: road
  distance still adds nothing with matched absences (0.704 either way) and
  still adds about 0.04 with uniform ones.
- Paired against the terrain forest in the same draw (change in spatial AUROC, and how
  many of the 20 draws came out higher):

  | Added | Matched absences | Uniform absences |
  |---|---|---|
  | land cover | +0.045 (20/20) | +0.045 (20/20) |
  | land cover without built-up and crop | +0.010 (15/20) | +0.014 (18/20) |
  | rainfall | +0.007 (14/20) | +0.003 (12/20) |
  | road distance | +0.001 (13/20) | +0.040 (20/20) |
- The gap stays small in every draw (-0.03 to +0.06).

Leakage check (`src/leakage_check.py`, `results/leakage_check.csv`): the same
random forest, headline setting, 5 fold seeds, with and without coordinates.
Spatial-CV AUROC, and in brackets the gap to random CV:

| Features | Matched absences | Uniform absences |
|---|---|---|
| Terrain | 0.721 (0.010) | 0.767 (0.003) |
| Terrain + x, y | 0.737 (0.031) | 0.778 (0.048) |
| x, y only | 0.699 (0.056) | 0.686 (0.126) |

- With coordinates the gap opens up (0.03 to 0.13), so the CV code does detect
  leakage on this data. Terrain alone gives almost nothing to leak, probably
  because 30 m terrain values at points located to 5 km or worse don't
  identify a place. That is my reading; I haven't proved it.
- Coordinates alone score 0.70 under spatial CV, about the same as the terrain
  model, so regional pattern carries as much signal as terrain at 50 km blocks.
  Larger blocks separate the two:

| Block size | x, y only (matched) | Terrain (matched) | x, y only (uniform) | Terrain (uniform) |
|---|---|---|---|---|
| 50 km | 0.699 | 0.721 | 0.686 | 0.767 |
| 100 km | 0.680 | 0.721 | 0.663 | 0.770 |
| 200 km | 0.502 | 0.717 | 0.506 | 0.755 |

  At 200 km blocks coordinates drop to chance while terrain barely moves, so
  the terrain skill isn't just regional location. At 200 km only 4.4 to 4.6 of
  5 folds had both classes to score, so treat that row as rough.

## Limitations

- The catalogue is reported events, not every landslide, so it leans toward
  places people can see and report.
- Susceptibility isn't hazard. It says where slopes are prone to fail, not when
  or how badly.
- No geology or soil layer yet.
- Block size, buffer and the absence scheme all move the numbers, so the
  sensitivity is reported.
- Catalogue locations are coarse. Of 481 Nepal events (2007 to 2016), about 29
  are within 1 km and most are 5 to 50 km off. I use the 194 at 5 km or better.
  Such points are often snapped to towns, which probably makes presences look
  closer to roads than they are (44% within 100 m of a road, against 13% of the
  country). This is the NASA legacy export; a newer COOLR version may have more
  events and I haven't checked.
- I chose the road-matched absences as the preferred setting after seeing the
  results. Block size, buffer and cutoff were fixed beforehand.
- The matched absences over-correct a little: their median distance to a road
  is 0.14 km against 0.19 km for presences, which is why road distance alone
  scores 0.44. Only the distance-to-road profile is matched, using the
  presences' own distances once before CV.
- Part of the matched-absence skill is probably hills versus plains. Road-side
  absences include flat lowland such as the Terai, which is why slope-only goes
  from 0.53 to 0.67. The two schemes ask slightly different questions, so both
  are shown.
- "Road" is a choice: OSM motorway through tertiary, unclassified and
  residential (plus links), about 133,000 km, mostly village lanes. Tracks,
  paths, steps and footways are left out.
- "River" is also a choice: OSM `waterway=river` only, about 26,800 km. Streams
  (about 66,600 km) are left out because how well they are mapped probably
  follows mapping effort, and canals, ditches and drains are man-made. Results
  could differ with streams included.
- The tables use one absence draw per scheme. The redraw section above shows
  how much that matters (sd 0.01 to 0.02), but it runs one fold seed and a
  handful of models, not the whole grid of settings.
- Land cover is read as class shares within 2.5 km of the point, because
  catalogue locations are only good to 5 km. I didn't tune that radius.
  WorldCover is the 2021 map and the events are 2007 to 2016, so land cover may
  have changed in between.
- Rainfall is the mean of CHIRPS annual totals for 1991 to 2020 at 0.05 degree
  cells (about 5.5 km), so it can't resolve local differences.
- The absence weights match road distance only. Land cover also differs between
  presences and matched absences (more tree, less crop), and I haven't tried
  matching on it.
- No water mask, so some absences may land on rivers or lakes. Absence locations
  come from a 300 m grid. One UTM zone (45N) covers all of Nepal; the scale error
  at the western edge is under 1%.

## Data

See [`data/README.md`](data/README.md). Nothing large is committed; each
source's licence applies.

Versions used so far (downloaded 2026-10-05, and 2026-10-06 for rainfall and land cover):

- Landslides: NASA Global Landslide Catalog, legacy CSV export
  (`Global_Landslide_Catalog_Export_rows.csv`, Nepal 2007 to 2016). Licence not
  yet verified.
- Elevation: Copernicus GLO-30 DEM, 45 tiles N26 to N30, E080 to E088 (AWS
  open bucket `copernicus-dem-30m`). Licence not yet verified; attribution to
  Copernicus is expected.
- Boundary: geoBoundaries gbOpen NPL ADM0, year 2019, CC BY 4.0.
- Roads and rivers: Geofabrik Nepal extract (`nepal-latest.osm.pbf`, last modified
  2026-10-03, md5 `d098ee3d64113fbc596a99fb6cf55232`). Contains data from
  OpenStreetMap contributors, ODbL.
- Rainfall: CHIRPS v3.0 annual totals, global GeoTIFFs for 1991 to 2020
  (`chirps-v3.0.YYYY.tif` from `data.chc.ucsb.edu/products/CHIRPS/v3.0/annual/global/tifs/`),
  put in a `chirps_annual/` folder. Climate Hazards Center, CC BY 4.0,
  doi 10.15780/G2JQ0P.
- Land cover: ESA WorldCover 2021 v200, 10 m, the 8 tiles N24E084, N24E087,
  N27E078, N27E081, N27E084, N27E087, N30E078 and N30E081 (AWS open bucket
  `esa-worldcover`, `v200/2021/map/`), put in a `worldcover/` folder. CC BY 4.0.

## Run

```bash
# Python 3.12 recommended for the geospatial stack (rasterio etc.)
pip install -r requirements.txt            # core: numpy, scipy, scikit-learn
pip install -r requirements-geo.txt        # raster/vector libraries, when you reach the data step
pytest

# Build the tables and run the baselines (raw data in data/raw, see data/README.md).
# If the OSM .pbf, chirps_annual/ and worldcover/ live elsewhere, point NEPAL_RAW_DIR at their folder.
python -m src.build_feature_table    # terrain, road and river distance, rainfall, land cover; two absence sets
python -m src.run_baselines          # writes results/baselines_roads.csv
python -m src.build_feature_table --draws 20   # 20 absence draws per scheme
python -m src.absence_variation      # writes results/absence_variation.csv
python -m src.leakage_check          # writes results/leakage_check.csv
```

## Repository layout

```
src/terrain.py    slope, aspect, curvature proxy from a DEM array
src/sampling.py   background (absence) points, optionally bias-weighted
src/cv.py         spatial block k-fold with an optional buffer, plus random k-fold
src/modeling.py   random-vs-spatial CV comparison for any sklearn model
src/roads.py      OSM roads and rivers, distance to nearest line, presence-matched absence weights
src/climate_cover.py        mean annual rainfall (CHIRPS) and land-cover shares (WorldCover) at points
src/grid.py       shared 30 m UTM grid, DEM warping, terrain features at points
src/build_feature_table.py  terrain + road and river distance, uniform and road-weighted absences
src/run_baselines.py        baselines under random vs spatial CV, with the road ablation
src/absence_variation.py    score repeated absence draws at the headline setting
src/leakage_check.py        does the CV show a gap when the model gets coordinates?
tests/            synthetic-data tests, including the leakage demonstration
```
