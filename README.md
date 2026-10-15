# Nepal landslide susceptibility: how much of the score is geography?

Landslide susceptibility for Nepal from open terrain, rainfall and land-cover
data. The question running through it: how much of a typical "model accuracy"
is just spatial autocorrelation, and how much is real skill?

> **Student project. Not a hazard forecast, and not for planning, evacuation or
> construction decisions.**

## Why this question

Landslides cluster in space, so a random train/test split puts a near-twin of
every test point in training and the score looks better than it is. Splitting by
geographic block gives the honest number. On a synthetic benchmark in `tests/`
the gap is about 0.3 AUROC (random 0.84-0.92, spatial 0.53-0.60) even though the
model knows nothing about terrain. This project measures the same gap on real
data.

## Plan

1. Presences: landslide events in Nepal from a public catalogue.
2. Absences: background cells from the rest of Nepal. The catalogue only has
   reported slides and reports cluster near roads and towns, so absences can also
   be drawn to follow that bias (`src/sampling.py`), and there's a road-distance
   ablation.
3. Features: elevation, slope, aspect (sin/cos), a curvature proxy, distance to
   roads and rivers, mean annual rainfall, land cover.
4. Models: slope-only baseline, logistic regression, random forest, gradient
   boosting. Simple first.
5. Evaluation: AUROC and PR-AUC under random CV and spatial block CV with a
   buffer. The headline is the spatial number and its gap to the random one.
6. Map: an interactive map of terrain-model scores at the sampled locations,
   with the catalogue points and a clear uncertainty note.

## Results

Work in progress. Features so far: terrain, distance to the nearest motor road
and river, mean annual rainfall, and land-cover shares around the point. This is
a partial model, not a hazard forecast.

Setup, fixed before running: 5-fold spatial block CV, 50 km blocks, 10 km
buffer, 194 presences (location accuracy 5 km or better), 975 absences, averaged
over 5 fold-assignment seeds. Gap is random-CV AUROC minus spatial-CV AUROC. Read
the spatial column; the random one is there to show the gap. Every setting is in
`results/baselines_roads.csv`. These tables use one absence draw per scheme, the
redraw further down uses 20.

Absences matched to the presences' distance to roads (preferred setting, see
Limitations):

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
| Gradient boosting (terrain) | 0.72 | 0.73 | 0.01 |
| Gradient boosting + land cover | 0.77 | 0.80 | 0.03 |
| Gradient boosting + rainfall + land cover | 0.76 | 0.80 | 0.04 |

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
| Gradient boosting (terrain) | 0.77 | 0.78 | 0.01 |
| Gradient boosting + land cover | 0.80 | 0.82 | 0.02 |
| Gradient boosting + rainfall + land cover | 0.80 | 0.83 | 0.03 |

Reading the tables:

- Road access is a big shortcut. With uniform absences, road distance alone
  scores 0.74 and adding it lifts the random forest from 0.77 to 0.81. With
  matched absences it adds nothing (0.72 either way). The number to quote is the
  terrain-only forest at about 0.70, give or take 0.02. The 0.72 in the table is
  one absence draw and a bit high (see the redraw below). Across the 12 block,
  buffer and cutoff settings it stays between 0.69 and 0.73.
- River distance adds almost nothing: 0.721 to 0.728 with matched absences,
  inside the seed spread (0.017 to 0.021). Alone it scores 0.49 (chance) with
  matched absences and 0.58 with uniform ones, so it looks more like a weak
  access proxy than terrain skill.
- Land cover is the first thing that adds clear skill. With matched absences the
  forest goes from 0.721 to 0.758, and land cover alone scores 0.731. Redrawing
  the absences gives +0.045 in all 20 draws. Most of that comes from two classes,
  though: without built-up and crop it drops to +0.010, inside the noise.
  Built-up is probably a settlement and access proxy, and crop separates the flat
  Terai farmland from the hills, so the full gain isn't clean skill from land.
- Gradient boosting does about the same as the random forest (0.724 against 0.721
  for terrain only with matched absences, 0.771 against 0.758 with land cover).
  The redraw below says the difference is noise, so the more complex model
  doesn't earn its place here.
- Rainfall adds nothing: 0.710 with matched absences against 0.721 for terrain
  alone, and 0.546 on its own. CHIRPS cells are about 5.5 km, so it barely
  changes across a block.
- The random-vs-spatial gap is small for the terrain features (-0.02 to +0.05
  across everything here, with the land-cover models the biggest at 0.03 to
  0.05), against about 0.3 on the synthetic benchmark. Giving the model the
  coordinates does open a gap (leakage check below), so the small terrain gap is
  a result and not a bug.
- Some of the uniform-absence skill was road access, not terrain: the forest
  drops from 0.77 to about 0.70 once absences stop being easy to find in remote
  terrain.

Redrawing the absences: 20 draws per scheme at the headline setting, fold seed
fixed at 0 (`src/absence_variation.py`, `results/absence_variation.csv`).
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
| Gradient boosting (terrain) | 0.703 ± 0.016 | 0.780 ± 0.014 |
| Gradient boosting + land cover | 0.750 ± 0.013 | 0.819 ± 0.013 |

- Redrawing moves the score (sd 0.011 to 0.020) about as much as changing the
  fold assignment (0.017 to 0.021), so the uncertainty on one score is closer to
  ±0.02 to ±0.03 than ±0.01.
- The draw used in the tables above scored 0.711 with this fold seed against a
  mean of 0.704, so it was a bit lucky. The conclusions hold: road distance adds
  nothing with matched absences (0.704 either way) and about 0.04 with uniform
  ones.
- Paired against the terrain forest within each draw, the change in spatial AUROC
  and how many of the 20 draws came out higher:

  | Added | Matched absences | Uniform absences |
  |---|---|---|
  | land cover | +0.045 (20/20) | +0.045 (20/20) |
  | land cover without built-up and crop | +0.010 (15/20) | +0.014 (18/20) |
  | rainfall | +0.007 (14/20) | +0.003 (12/20) |
  | road distance | +0.001 (13/20) | +0.040 (20/20) |

- Gradient boosting against the random forest in the same draw: -0.001 (10 of
  20 higher) with matched absences and +0.007 (14 of 20) with uniform; with land
  cover +0.001 (9 of 20) and +0.000 (10 of 20). So the single-draw edge for
  gradient boosting with land cover in the tables above (0.771 against 0.758) is
  noise.
- The gap stays small in every draw (-0.03 to +0.06).

Leakage check (`src/leakage_check.py`, `results/leakage_check.csv`): the same
random forest, headline setting, 5 fold seeds, with and without coordinates.
Spatial-CV AUROC, with the gap to random CV in brackets:

| Features | Matched absences | Uniform absences |
|---|---|---|
| Terrain | 0.721 (0.010) | 0.767 (0.003) |
| Terrain + x, y | 0.737 (0.031) | 0.778 (0.048) |
| x, y only | 0.699 (0.056) | 0.686 (0.126) |

- With coordinates the gap opens up (0.03 to 0.13), so the CV code does see
  leakage on this data. Terrain alone gives it little to leak, probably because
  30 m terrain values at points located to 5 km or worse don't identify a place.
  That's a guess, not proven.
- Coordinates alone score 0.70 under spatial CV, about the same as the terrain
  model, so regional pattern carries as much signal as terrain at 50 km blocks.
  Bigger blocks separate them:

| Block size | x, y only (matched) | Terrain (matched) | x, y only (uniform) | Terrain (uniform) |
|---|---|---|---|---|
| 50 km | 0.699 | 0.721 | 0.686 | 0.767 |
| 100 km | 0.680 | 0.721 | 0.663 | 0.770 |
| 200 km | 0.502 | 0.717 | 0.506 | 0.755 |

  At 200 km blocks coordinates fall to chance and terrain barely moves, so the
  terrain skill isn't just regional location. Only 4.4 to 4.6 of 5 folds had both
  classes to score at 200 km, so treat that row as rough.

## Limitations

- The catalogue is reported events, not every landslide, so it leans toward
  places people can see and report.
- Susceptibility isn't hazard. It says where slopes are prone to fail, not when
  or how badly.
- No geology or soil layer yet.
- Block size, buffer and absence scheme all move the numbers, so the sensitivity
  is reported.
- Catalogue locations are coarse. Of 481 Nepal events (2007 to 2016), about 29
  are within 1 km and most are 5 to 50 km off. The 194 at 5 km or better are
  used. Those points are often snapped to towns, which probably makes presences
  look closer to roads than they are (44% within 100 m of a road, against 13% of
  the country). `src.download_catalogue` can fetch the current NASA COOLR
  reports catalogue, but it is a different inventory and needs a complete rerun.
- The matched absences were picked as the preferred setting after seeing the
  results. Block size, buffer and cutoff were fixed beforehand.
- The matched absences over-correct a little: their median distance to a road is
  0.14 km against 0.19 km for presences, which is why road distance alone scores
  0.44. Only the distance profile is matched, using the presences' own distances
  once before CV.
- Part of the matched-absence skill is probably hills versus plains. Road-side
  absences include flat lowland like the Terai, which is why slope-only goes from
  0.53 to 0.67. The two schemes ask slightly different questions, so both are
  shown.
- "Road" is a choice: OSM motorway through tertiary, unclassified and residential
  (plus links), about 133,000 km, mostly village lanes. Tracks, paths, steps and
  footways are out.
- "River" is a choice too: OSM `waterway=river` only, about 26,800 km. Streams
  (about 66,600 km) are out because how well they're mapped probably follows
  mapping effort, and canals, ditches and drains are man-made. Results could
  differ with streams.
- The tables use one absence draw per scheme. The redraw shows that matters (sd
  0.01 to 0.02), but it runs one fold seed and a handful of models, not the whole
  settings grid.
- Land cover is class shares within 2.5 km of the point, because locations are
  only good to 5 km. The radius wasn't tuned. WorldCover is the 2021 map and the
  events are 2007 to 2016, so land cover may have changed.
- Rainfall is the mean of CHIRPS annual totals for 1991 to 2020 at 0.05 degree
  cells (about 5.5 km), so no local detail.
- The absence weights match road distance only. Land cover also differs between
  presences and matched absences (more tree, less crop), and matching on it
  hasn't been tried.
- Gradient boosting uses fixed settings (depth 3, learning rate 0.05, 200
  iterations) that weren't tuned, so a tuned model might do a bit better.
- No water mask, so some absences may land on rivers or lakes. Absence locations
  come from a 300 m grid. One UTM zone (45N) covers Nepal; the scale error at the
  western edge is under 1%.

## Data

See [`data/README.md`](data/README.md). Nothing large is committed. Each
source's licence applies.

Versions used so far (downloaded 2026-10-05, rainfall and land cover on
2026-10-06):

- Landslides: NASA Global Landslide Catalog, legacy CSV export
  (`Global_Landslide_Catalog_Export_rows.csv`, Nepal 2007 to 2016). NASA Earth
  Science data are generally CC0 unless marked otherwise; cite NASA as the
  source. The current COOLR reports service can be downloaded with
  `python -m src.download_catalogue`; it combines GLC and later reports, so it
  is not interchangeable with these results.
- Elevation: Copernicus GLO-30 DEM, 45 tiles N26 to N30, E080 to E088 (AWS open
  bucket `copernicus-dem-30m`). Free to use under the COP-DEM-GLO-30-F licence.
  Required notice: “produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014
  and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the
  European Union and ESA; all rights reserved”.
- Boundary: geoBoundaries gbOpen NPL ADM0, year 2019, CC BY 4.0.
- Roads and rivers: Geofabrik Nepal extract (`nepal-latest.osm.pbf`, last
  modified 2026-10-03, md5 `d098ee3d64113fbc596a99fb6cf55232`). Contains data
  from OpenStreetMap contributors, ODbL.
- Rainfall: CHIRPS v3.0 annual totals, global GeoTIFFs for 1991 to 2020
  (`chirps-v3.0.YYYY.tif` from
  `data.chc.ucsb.edu/products/CHIRPS/v3.0/annual/global/tifs/`), in a
  `chirps_annual/` folder. Climate Hazards Center, CC BY 4.0, doi
  10.15780/G2JQ0P.
- Land cover: ESA WorldCover 2021 v200, 10 m, the 8 tiles N24E084, N24E087,
  N27E078, N27E081, N27E084, N27E087, N30E078 and N30E081 (AWS open bucket
  `esa-worldcover`, `v200/2021/map/`), in a `worldcover/` folder. CC BY 4.0.

## Run

```bash
# Python 3.12 recommended for the geospatial stack (rasterio etc.)
pip install -r requirements.txt            # core: numpy, scipy, scikit-learn
pip install -r requirements-geo.txt        # raster and vector libraries
pytest

# Raw data goes in data/raw (see data/README.md). If the OSM .pbf, chirps_annual/
# and worldcover/ live somewhere else, point NEPAL_RAW_DIR at that folder.
python -m src.build_feature_table              # terrain, road and river distance, rainfall, land cover
python -m src.run_baselines                    # writes results/baselines_roads.csv
python -m src.build_feature_table --draws 20   # 20 absence draws per scheme
python -m src.absence_variation                # writes results/absence_variation.csv
python -m src.leakage_check                    # writes results/leakage_check.csv
python -m src.make_map                          # writes results/susceptibility_map.html
python -m src.make_figures                      # writes report-ready PNG figures under results/figures/
```

For a current NASA catalogue rather than the legacy export, run
`python -m src.download_catalogue`, then pass
`--catalogue data/raw/COOLR_Nepal_reports.csv` to the table builder and rerun
the analysis scripts. Do not mix its output with the published legacy-catalogue
results.

## Layout

```
src/terrain.py              slope, aspect, curvature proxy from a DEM array
src/sampling.py             background (absence) points, optionally weighted
src/cv.py                   spatial block k-fold with an optional buffer, plus random k-fold
src/modeling.py             random vs spatial CV for any sklearn model
src/roads.py                OSM roads and rivers, distance to nearest line, absence weights
src/climate_cover.py        mean annual rainfall (CHIRPS), land-cover shares (WorldCover)
src/grid.py                 shared 30 m UTM grid, DEM warping, terrain at points
src/build_feature_table.py  builds the feature table, uniform and road-matched absences
src/run_baselines.py        baselines under random vs spatial CV, with the road ablation
src/absence_variation.py    repeated absence draws at the headline setting
src/leakage_check.py        does the CV show a gap when the model gets coordinates?
src/download_catalogue.py   downloads current Nepal reports from NASA COOLR
src/make_map.py             writes an interactive map of sampled model scores
src/make_figures.py         writes validation and absence-draw figures for a report
tests/                      synthetic-data tests, including the leakage demonstration
```
