import json

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from automl_app.core import helpers


def test_quick_dtype_buckets_excludes_target() -> None:
    df = pd.DataFrame(
        {
            "age": [22, 35, 41],
            "city": ["A", "B", "C"],
            "target": [1, 0, 1],
        }
    )

    num_cols, cat_cols = helpers.quick_dtype_buckets(df, "target")

    assert "target" not in num_cols
    assert "target" not in cat_cols
    assert "age" in num_cols
    assert "city" in cat_cols


def test_build_preprocessor_detects_numeric_and_categorical() -> None:
    df = pd.DataFrame(
        {
            "num_feature": [1.0, 2.0, 3.0],
            "cat_feature": ["x", "y", "x"],
            "target": [10.2, 20.1, 30.0],
        }
    )

    preprocessor, num_cols, cat_cols = helpers.build_preprocessor(df, "target")

    assert num_cols == ["num_feature"]
    assert cat_cols == ["cat_feature"]
    transformed = preprocessor.fit_transform(df.drop(columns=["target"]))
    assert transformed.shape[0] == len(df)


def test_save_schema_handles_nan_numeric_and_empty_categories(tmp_path) -> None:
    old_dir = helpers.ARTIFACTS_DIR
    helpers.ARTIFACTS_DIR = str(tmp_path)
    try:
        X = pd.DataFrame(
            {
                "num_feature": [None, None],
                "cat_feature": [None, None],
            }
        )

        helpers.save_schema(
            X=X,
            num_cols=["num_feature"],
            cat_cols=["cat_feature"],
            task="classification",
            y=pd.Series([0, 1]),
        )

        schema = json.loads((tmp_path / "app_schema.json").read_text(encoding="utf-8"))
        features = {item["name"]: item for item in schema["features"]}

        assert features["num_feature"]["mean"] == 0.0
        assert features["cat_feature"]["options"] == ["Unknown"]
        assert schema["target_mapping"] == {"0": "0", "1": "1"}
    finally:
        helpers.ARTIFACTS_DIR = old_dir


def test_select_best_model_returns_sorted_leaderboard() -> None:
    df = pd.DataFrame(
        {
            "num_feature": [1, 2, 3, 4, 5, 6, 7, 8],
            "cat_feature": ["a", "a", "b", "b", "c", "c", "a", "b"],
            "target": [2.0, 3.5, 5.2, 8.1, 10.0, 11.8, 14.2, 16.0],
        }
    )

    preprocessor, _, _ = helpers.build_preprocessor(df, "target")
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train = X.iloc[:6]
    y_train = y.iloc[:6]
    X_test = X.iloc[6:]
    y_test = y.iloc[6:]

    models = {
        "rf": RandomForestRegressor(n_estimators=20, random_state=42),
        "knn": helpers.KNeighborsRegressor(n_neighbors=2),
    }

    best_name, best_pipeline, best_score, leaderboard = helpers.select_best_model(
        task="regression",
        preprocessor=preprocessor,
        models=models,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    assert best_name in models
    assert best_pipeline is not None
    assert isinstance(best_score, float)
    assert len(leaderboard) >= 1
    assert leaderboard == sorted(leaderboard, key=lambda item: item["score"], reverse=True)
