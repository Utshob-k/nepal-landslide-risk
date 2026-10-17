"""Download current NASA COOLR reports for Nepal.

    python -m src.download_catalogue

The result is written in the column layout used by ``src.build_feature_table``.
The downloaded file stays out of version control because it is source data.
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

ENDPOINT = "https://gis.earthdata.nasa.gov/portal/rest/services/Landslides/COOLR_Reports_Points/MapServer/0/query"
NEPAL_BOUNDS = (80.0, 26.0, 89.0, 31.0)
FETCH_SIZE = 2_000


def _legacy_rows(payload: dict) -> pd.DataFrame:
    """Convert a COOLR GeoJSON response to the feature builder's CSV layout."""
    rows = []
    for feature in payload.get("features", []):
        props = feature.get("properties", {})
        coords = feature.get("geometry", {}).get("coordinates")
        if not coords or len(coords) < 2:
            continue
        rows.append(
            {
                "event_id": props.get("event_id"),
                "event_date": props.get("event_date"),
                "longitude": coords[0],
                "latitude": coords[1],
                "location_accuracy": props.get("location_accuracy"),
                "country_name": "Nepal",
                "event_import_source": props.get("event_import_source"),
                "source_name": props.get("source_name"),
                "source_link": props.get("source_link"),
            }
        )
    return pd.DataFrame(rows)


def _request(params: dict) -> dict:
    """Call the NASA service and turn an ArcGIS error into a useful exception."""
    with urlopen(f"{ENDPOINT}?{urlencode(params)}", timeout=60) as response:
        payload = json.load(response)
    if "error" in payload:
        raise RuntimeError(f"NASA COOLR request failed: {payload['error']}")
    return payload


def download() -> pd.DataFrame:
    """Download every report inside Nepal's bounding box from NASA's public service."""
    params = {
        "where": "1=1",
        "geometry": ",".join(map(str, NEPAL_BOUNDS)),
        "geometryType": "esriGeometryEnvelope",
        "inSR": 4326,
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "*",
        "returnGeometry": "true",
    }
    count = _request({**params, "returnCountOnly": "true", "f": "json"})["count"]
    features = []
    for offset in range(0, count, FETCH_SIZE):
        page = _request(
            {
                **params,
                "resultOffset": offset,
                "resultRecordCount": FETCH_SIZE,
                "f": "geojson",
            }
        )
        features.extend(page.get("features", []))
    return _legacy_rows({"features": features})


def main() -> None:
    table = download()
    if table.empty:
        raise SystemExit(
            "COOLR returned no Nepal reports; check the service and try again."
        )
    RAW.mkdir(parents=True, exist_ok=True)
    target = RAW / "COOLR_Nepal_reports.csv"
    table.to_csv(target, index=False)
    print(f"wrote {len(table)} reports to {target}")


if __name__ == "__main__":
    main()
