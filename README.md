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
a curvature proxy) and distance to the nearest motor road. Rainfall, land cover
and rivers aren't in yet and gradient boosting hasn't been run. These numbers
describe a partial model and are not a hazard forecast.

Setup, fixed before running: 5-fold spatial block CV, 50 km blocks, 10 km
buffer, 194 presences (catalogue location accuracy 5 km or better), 975
absences, averaged over 5 seeds for the fold assignment. Gap is random-CV AUROC
minus spatial-CV AUROC. The spatial number is the one to read; the random column
is there to show the gap. Every setting is in `results/baselines_roads.csv`.

Absences drawn to match the presences' distance to roads (my preferred setting,
see Limitations):

| Model | Spatial-CV AUROC | Random-CV AUROC | Gap |
|---|---|---|---|
| Slope only | 0.67 | 0.67 | -0.01 |
| Logistic regression (terrain) | 0.68 | 0.68 | 0.00 |
| Random forest (terrain) | 0.72 | 0.73 | 0.01 |
| Random forest + road distance (ablation) | 0.72 | 0.74 | 0.02 |
| Road distance only (diagnostic) | 0.44 | 0.47 | 0.03 |

Absences drawn uniformly over Nepal:

| Model | Spatial-CV AUROC | Random-CV AUROC | Gap |
|---|---|---|---|
| Slope only | 0.53 | 0.53 | 0.00 |
| Logistic regression (terrain) | 0.66 | 0.68 | 0.02 |
| Random forest (terrain) | 0.77 | 0.77 | 0.00 |
| Random forest + road distance (ablation) | 0.81 | 0.82 | 0.01 |
| Road distance only (diagnostic) | 0.74 | 0.74 | 0.00 |

Reading the tables:

- Road access is a big shortcut. With uniform absences, road distance alone
  scores 0.74, and adding it takes the random forest from 0.77 to 0.81. With
  matched absences it adds nothing (0.72 either way). The terrain-only random
  forest at about 0.72 is the number I'd quote; it stays between 0.69 and 0.73
  across the 12 block, buffer and cutoff settings.
- The random-vs-spatial gap is close to zero everywhere (-0.02 to +0.03), where
  the synthetic benchmark in `tests/` shows about 0.3. My guess is that the
  features are smooth terrain values with no coordinates, and the catalogue
  locations are only good to 5 km or worse, so presences don't cluster at the
  30 m scale the features see. I haven't tested that.
- Some of the uniform-absence skill was road access and not terrain: the random
  forest drops from 0.77 to 0.72 once absences are no longer easy to find in
  remote terrain.

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
- Each scheme is one absence draw. The seed spread (0.01 to 0.02) covers fold
  assignment only, not drawing different absences.
- No water mask, so some absences may land on rivers or lakes. Absence locations
  come from a 300 m grid. One UTM zone (45N) covers all of Nepal; the scale error
  at the western edge is under 1%.

## Data

See [`data/README.md`](data/README.md). Nothing large is committed; each
source's licence applies.

Versions used so far (downloaded 2026-10-05):

- Landslides: NASA Global Landslide Catalog, legacy CSV export
  (`Global_Landslide_Catalog_Export_rows.csv`, Nepal 2007 to 2016). Licence not
  yet verified.
- Elevation: Copernicus GLO-30 DEM, 45 tiles N26 to N30, E080 to E088 (AWS
  open bucket `copernicus-dem-30m`). Licence not yet verified; attribution to
  Copernicus is expected.
- Boundary: geoBoundaries gbOpen NPL ADM0, year 2019, CC BY 4.0.
- Roads: Geofabrik Nepal extract (`nepal-latest.osm.pbf`, last modified
  2026-10-03, md5 `d098ee3d64113fbc596a99fb6cf55232`). Contains data from
  OpenStreetMap contributors, ODbL.

## Run

```bash
# Python 3.12 recommended for the geospatial stack (rasterio etc.)
pip install -r requirements.txt            # core: numpy, scipy, scikit-learn
pip install -r requirements-geo.txt        # raster/vector libraries, when you reach the data step
pytest

# Build the tables and run the baselines (raw data in data/raw, see data/README.md).
# If the OSM .pbf lives elsewhere, point NEPAL_RAW_DIR at its folder.
python -m src.build_dem_table        # DEM-only table (terrain, uniform absences)
python -m src.build_feature_table    # terrain + road distance, two absence sets
python -m src.run_baselines          # writes results/baselines_roads.csv
```

## Repository layout

```
src/terrain.py    slope, aspect, curvature proxy from a DEM array
src/sampling.py   background (absence) points, optionally bias-weighted
src/cv.py         spatial block k-fold with an optional buffer, plus random k-fold
src/modeling.py   random-vs-spatial CV comparison for any sklearn model
src/roads.py      motor roads from OSM, distance to road, presence-matched absence weights
src/build_dem_table.py      DEM-only feature table (shared grid and terrain helpers)
src/build_feature_table.py  terrain + road distance, uniform and road-weighted absences
src/run_baselines.py        baselines under random vs spatial CV, with the road ablation
tests/            synthetic-data tests, including the leakage demonstration
```
