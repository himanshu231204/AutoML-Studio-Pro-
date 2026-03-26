import pandas as pd

from automl_app.tabs.train import _clip_numeric_outliers, _dedupe_columns, _drop_high_corr_features, _profile_dataset


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


def test_drop_high_corr_features_removes_duplicate_signal() -> None:
    X_train = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": [2, 4, 6, 8, 10],
            "c": [5, 3, 4, 2, 1],
        }
    )
    X_test = pd.DataFrame(
        {
            "a": [6, 7],
            "b": [12, 14],
            "c": [0, -1],
        }
    )

    new_train, new_test, dropped = _drop_high_corr_features(X_train, X_test, threshold=0.95)

    assert len(dropped) == 1
    assert dropped[0] in ["a", "b"]
    assert new_train.shape[1] == 2
    assert new_test.shape[1] == 2


def test_clip_numeric_outliers_caps_extreme_values() -> None:
    X_train = pd.DataFrame({"x": [1, 2, 3, 4, 500], "cat": ["a", "b", "a", "b", "a"]})
    X_test = pd.DataFrame({"x": [0, 1000], "cat": ["a", "b"]})

    clipped_train, clipped_test = _clip_numeric_outliers(X_train, X_test, q_low=0.05, q_high=0.95)

    assert clipped_train["x"].max() < 500
    assert clipped_test["x"].max() < 1000
