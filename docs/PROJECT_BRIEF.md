# Project brief

## Question

How much apparent landslide-model skill in Nepal remains when evaluation keeps
nearby places out of the training set?

## What was built

- A reproducible feature pipeline from elevation, roads, rivers, rainfall and
  land cover.
- Logistic-regression, random-forest and gradient-boosting baselines.
- Random and spatial-block cross-validation with a 10 km buffer.
- A road-matched background scheme to test reporting and access bias.
- A small interactive map and scripts for report-ready figures.

## Headline result

The terrain-only random forest scores about **0.70 spatial-CV AUROC** with
road-matched background samples. It is the useful headline because a random
split is easier and can overstate what transfers to new places.

Land cover raises the score by about 0.045 across all 20 background redraws,
but much of that may represent settlement access and lowland agriculture. Road
distance boosts results with uniform backgrounds but adds almost nothing after
road matching. Gradient boosting does not reliably beat the random forest.

## Why it matters

The project shows a common geospatial-ML failure mode: a high score can reflect
where observations were reported instead of a feature that genuinely generalises
to new places. The analysis makes that uncertainty visible rather than hiding it.

## Evidence to show

Generate the two figures with:

```bash
python -m src.make_figures
```

`cv_comparison.png` compares random and spatial CV. `absence_variation.png`
shows the score spread across 20 independently redrawn background samples.

Generate the interactive demonstration with:

```bash
python -m src.make_map
```

The map shows model scores at sampled locations and the catalogue events. It is
not a calibrated probability surface or a hazard forecast.

## Next research step

Rerun the entire workflow with the current NASA COOLR catalogue, then compare
the results as a separate experiment. The code supports this through
`python -m src.download_catalogue`; it must not be mixed with the legacy
catalogue results currently reported here.
