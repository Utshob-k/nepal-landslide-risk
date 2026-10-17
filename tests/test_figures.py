import pandas as pd

from src.make_figures import headline_rows


def test_headline_rows_selects_only_the_fixed_setting_and_report_models():
    rows = pd.DataFrame(
        {
            "cutoff_km": [5.0, 10.0, 5.0],
            "block_km": [50, 50, 50],
            "buffer_km": [10, 10, 0],
            "model": ["Random forest", "Random forest", "Gradient boosting"],
        }
    )
    out = headline_rows(rows)
    assert out.model.tolist() == ["Random forest"]
