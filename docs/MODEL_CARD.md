# Model card

## Model identity

This project compares a logistic regression, random forest and gradient-
boosting classifier for **landslide susceptibility research in Nepal**. The
published headline model is a terrain-only random forest. It is intentionally
simple so the validation and data-bias analysis are easier to inspect.

## Intended use

Use the project to learn from, reproduce, or extend a spatial machine-learning
workflow. It can support an academic discussion of susceptibility modelling and
of how spatial validation and background sampling change model scores.

## Out of scope

Do not use it to forecast a landslide, choose a safe route or building site,
plan evacuation, set insurance or make emergency decisions. The output scores
rank sampled records from this study; they are not calibrated probabilities of
landslide occurrence.

## Inputs

The terrain model uses elevation, slope, sine and cosine of aspect, and a plan-
curvature proxy. Other experiments add distance to roads and rivers, mean annual
rainfall, and land-cover shares. Event locations come from the NASA Global
Landslide Catalog legacy export. The analysis keeps only events located to 5 km
accuracy or better.

## Evaluation

The fixed headline setting is five-fold spatial-block cross-validation, 50 km
blocks and a 10 km training buffer. The result averages five fold assignments.
The preferred background scheme matches the event locations' road-distance
distribution to reduce reporting/access bias.

| Model | Spatial-CV AUROC | Random-CV AUROC |
|---|---:|---:|
| Terrain random forest | 0.72 | 0.73 |
| Terrain random forest + land cover | 0.76 | 0.80 |
| Terrain gradient boosting | 0.72 | 0.73 |
| Terrain gradient boosting + land cover | 0.77 | 0.80 |

Across 20 independent background redraws, the terrain random forest has mean
spatial-CV AUROC 0.704 ± 0.017. This is a better uncertainty summary than any
single score in the table.

## Known limitations

- The catalogue records reported landslides, not a complete inventory.
- Event coordinates are often coarse and may be snapped to towns or roads.
- Road matching addresses only one observable reporting bias.
- Land cover can encode settlement access and flat agricultural land, so its
  apparent gain should not be read as purely physical susceptibility.
- Rainfall is a long-term mean at roughly 5.5 km resolution; it cannot model a
  storm trigger.
- The current work has no geology, soil, water mask or time-varying trigger.

## Responsible communication

State the background scheme, spatial-CV setting and uncertainty whenever using
these results. Refer to the map as an exploratory visualisation of sampled model
scores, never as a hazard map.
