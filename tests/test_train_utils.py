import pandas as pd

from automl_app.tabs.train import _dedupe_columns, _profile_dataset


def test_dedupe_columns_handles_duplicates_and_empty_names() -> None:
    cols = ["name", "name", "", "name", " "]
    deduped = _dedupe_columns(cols)

    assert deduped[0] == "name"
    assert deduped[1] == "name_1"
    assert deduped[2] == "unnamed"
    assert deduped[3] == "name_2"
    assert deduped[4] == "unnamed_1"


def test_profile_dataset_reports_drops_and_task() -> None:
    df = pd.DataFrame(
        {
            "feature_ok": [1, 2, 3, 4],
            "feature_constant": [9, 9, 9, 9],
            "feature_null": [None, None, None, None],
            "target": ["yes", "no", "yes", None],
        }
    )

    profile = _profile_dataset(df, "target")

    assert profile["task"] == "classification"
    assert profile["rows"] == 4
    assert profile["removed_target_na"] == 1
    assert "feature_constant" in profile["dropped_cols"]
    assert "feature_null" in profile["dropped_cols"]
    assert 0 <= profile["health_score"] <= 100
