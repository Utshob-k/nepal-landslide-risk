# Data card

## Purpose

This card records the datasets combined into the Nepal landslide-susceptibility
feature table, their role, and the main constraints on reuse.

| Layer | Role | Spatial or temporal detail | Key constraint |
|---|---|---|---|
| NASA Global Landslide Catalog | Reported event presences | Nepal, 2007–2016 | Media and reporting bias; locations can be coarse |
| Copernicus GLO-30 DEM | Terrain derivatives | 30 m | Surface model; derived slope is local while events may be imprecise |
| OpenStreetMap via Geofabrik | Road and river distance | Varies with mapping coverage | Roads may proxy access/reporting |
| CHIRPS v3 annual totals | Mean annual rainfall | 0.05°; 1991–2020 mean | Too coarse for local storm triggers |
| ESA WorldCover 2021 | Land-cover shares | 10 m | Later than the events and may proxy settlement/agriculture |
| geoBoundaries NPL ADM0 | Nepal mask | 2019 boundary | Administrative boundary only |

## Construction

The table contains catalogue events and sampled background cells. For each
event, five backgrounds are drawn under two schemes: uniform across Nepal and
matched to the events' road-distance distribution. Sampling occurs on a 300 m
grid, and samples sharing an occupied event cell are excluded.

Land cover is summarised as class shares inside 2.5 km of each point. This radius
is a deliberate compromise for the 5 km location-accuracy cutoff. It was not
tuned against model performance.

## Access and licences

Raw downloads are excluded from Git. The root README has version dates, source
links and source-specific notices. NASA Earth Science data are generally CC0
unless marked otherwise; Copernicus DEM carries its own source-notice terms;
OpenStreetMap data are ODbL. Review each source's current terms before a new
distribution.

## Privacy and sensitive information

The inputs are public geospatial datasets. Still, use care when displaying exact
reported-event locations: an event coordinate may represent a community or
roadside report, not a precisely surveyed failure point.

## Reuse guidance

The existing results use the legacy catalogue and must remain labelled as such.
`python -m src.download_catalogue` fetches current NASA COOLR reports, but that
creates a different dataset and requires a full new analysis run. Do not append
new records to the existing results or compare the two inventories as though
they were one experiment.
