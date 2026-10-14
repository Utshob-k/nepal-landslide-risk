"""Download current NASA COOLR landslide reports for Nepal.

    python -m src.download_catalogue

Writes data/raw/COOLR_Nepal_reports.csv in the column layout read by
src.build_feature_table. The downloaded file is intentionally gitignored.
"""
from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

from .grid import RAW

ENDPOINT = "https://gis.earthdata.nasa.gov/portal/rest/services/Landslides/COOLR_Reports_Points/MapServer/0/query"
NEPAL_BOUNDS = (80.0, 26.0, 89.0, 31.0)


def _legacy_rows(payload: dict) -> pd.DataFrame:
    """Convert a COOLR GeoJSON response to the project catalogue schema."""
    rows = []
    for feature in payload.get("features", []):
        props = feature.get("properties", {})
        coords = feature.get("geometry", {}).get("coordinates")
        if not coords or len(coords) < 2:
            continue
        rows.append({
            "event_id": props.get("event_id"),
            "event_date": props.get("event_date"),
            "longitude": coords[0],
            "latitude": coords[1],
            "location_accuracy": props.get("location_accuracy"),
            "country_name": "Nepal",
            "event_import_source": props.get("event_import_source"),
            "source_name": props.get("source_name"),
            "source_link": props.get("source_link"),
        })
    return pd.DataFrame(rows)


def download() -> pd.DataFrame:
    """Fetch all reports inside Nepal's bounding box from NASA's public service."""
    params = {
        "where": "1=1",
        "geometry": ",".join(map(str, NEPAL_BOUNDS)),
        "geometryType": "esriGeometryEnvelope",
        "inSR": 4326,
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "*",
        "returnGeometry": "true",
        "f": "geojson",
    }
    with urlopen(f"{ENDPOINT}?{urlencode(params)}", timeout=60) as response:
        payload = json.load(response)
    return _legacy_rows(payload)


def main() -> None:
    table = download()
    if table.empty:
        raise SystemExit("COOLR returned no Nepal reports; check the service and try again.")
    RAW.mkdir(parents=True, exist_ok=True)
    target = RAW / "COOLR_Nepal_reports.csv"
    table.to_csv(target, index=False)
    print(f"wrote {len(table)} reports to {target}")


if __name__ == "__main__":
    main()
