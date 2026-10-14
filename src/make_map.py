"""Create an interactive, exploratory map of scores at the sampled locations.

    python -m src.make_map

The map uses the terrain-only random forest and the road-matched background
scheme. Its scores are model outputs for the sampled rows, not calibrated
landslide probabilities or a hazard forecast.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
TERRAIN = ["elevation", "slope", "aspect_sin", "aspect_cos", "curvature"]


def score_samples(table: pd.DataFrame) -> pd.DataFrame:
    """Fit the headline terrain model and return scores for road-matched samples."""
    d = table[((table.label == 1) & (table.accuracy_km <= 5)) | (table.abs_set == "road_weighted")].copy()
    model = RandomForestClassifier(200, min_samples_leaf=5, class_weight="balanced", n_jobs=4, random_state=0)
    model.fit(d[TERRAIN], d["label"])
    d["score"] = model.predict_proba(d[TERRAIN])[:, 1]
    return d


def _colour(score: float) -> str:
    """Return a readable three-class colour for a model score."""
    if score < 1 / 3:
        return "#2c7bb6"
    if score < 2 / 3:
        return "#fdae61"
    return "#d7191c"


def build_map(samples: pd.DataFrame, output: Path) -> None:
    """Write a Leaflet map with a model-score layer and catalogue presences."""
    import folium

    centre = [float(samples.latitude.mean()), float(samples.longitude.mean())]
    m = folium.Map(location=centre, zoom_start=7, tiles="OpenStreetMap", control_scale=True)
    score_layer = folium.FeatureGroup(name="Terrain-model score at sampled locations", show=True)
    pres_layer = folium.FeatureGroup(name="Catalogue presences", show=True)
    for row in samples.itertuples():
        location = [row.latitude, row.longitude]
        if row.label:
            folium.CircleMarker(location, radius=4, color="#1b1b1b", weight=1, fill=True,
                                fill_color="#7b3294", fill_opacity=0.9,
                                popup=f"Reported landslide<br>Model score: {row.score:.2f}").add_to(pres_layer)
        else:
            folium.CircleMarker(location, radius=3, color=_colour(row.score), weight=0, fill=True,
                                fill_color=_colour(row.score), fill_opacity=0.55,
                                popup=f"Background sample<br>Model score: {row.score:.2f}").add_to(score_layer)
    score_layer.add_to(m)
    pres_layer.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    title = """
    <div style="position: fixed; top: 10px; left: 50px; z-index: 1000; width: 360px;
                background: white; border: 1px solid #777; border-radius: 4px; padding: 10px; font-size: 13px;">
      <b>Nepal landslide susceptibility: exploratory model map</b><br>
      Terrain-only random-forest scores at sampled locations. Purple marks reported catalogue events;
      blue, orange and red show low, medium and high model scores for road-matched background samples.<br>
      <b>Not a hazard forecast or a tool for planning, evacuation or construction.</b>
    </div>
    """
    m.get_root().html.add_child(folium.Element(title))
    output.parent.mkdir(exist_ok=True)
    m.save(output)


def main() -> None:
    table = pd.read_csv(ROOT / "data" / "tables" / "features_roads.csv")
    samples = score_samples(table)
    output = ROOT / "results" / "susceptibility_map.html"
    build_map(samples, output)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
