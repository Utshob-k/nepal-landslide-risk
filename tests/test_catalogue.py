from src.download_catalogue import _legacy_rows


def test_coolr_geojson_is_converted_to_the_project_catalogue_schema():
    payload = {
        "features": [{
            "geometry": {"coordinates": [85.3, 27.7]},
            "properties": {
                "event_id": 12, "event_date": 1704067200000,
                "location_accuracy": "5km", "event_import_source": "GLC",
                "source_name": "Example", "source_link": "https://example.test/event",
            },
        }],
    }
    rows = _legacy_rows(payload)
    assert rows.loc[0, "longitude"] == 85.3
    assert rows.loc[0, "latitude"] == 27.7
    assert rows.loc[0, "country_name"] == "Nepal"
    assert rows.loc[0, "location_accuracy"] == "5km"
