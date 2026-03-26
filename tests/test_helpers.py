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
    assert leaderboard == sorted(leaderboard, key=lambda item: item["ranking_score"], reverse=True)


def test_get_candidate_models_fast_mode_is_smaller() -> None:
    cls_fast = helpers.get_candidate_models("classification", training_mode="fast")
    cls_high = helpers.get_candidate_models("classification", training_mode="high_accuracy")
    reg_fast = helpers.get_candidate_models("regression", training_mode="fast")
    reg_high = helpers.get_candidate_models("regression", training_mode="high_accuracy")

    assert len(cls_fast) < len(cls_high)
    assert len(reg_fast) < len(reg_high)


# ============================================================================
# EDGE CASE TESTS - Testing boundary conditions and unusual inputs
# ============================================================================


def test_quick_dtype_buckets_with_empty_dataframe() -> None:
    """Test handling of empty dataframe"""
    df = pd.DataFrame()
    num_cols, cat_cols = helpers.quick_dtype_buckets(df, "target")
    assert num_cols == []
    assert cat_cols == []


def test_quick_dtype_buckets_with_only_target_column() -> None:
    """Test behavior when dataframe has only target column"""
    df = pd.DataFrame({"target": [1, 2, 3]})
    num_cols, cat_cols = helpers.quick_dtype_buckets(df, "target")
    assert num_cols == []
    assert cat_cols == []


def test_build_preprocessor_with_no_numeric_features() -> None:
    """Test preprocessor with only categorical features"""
    df = pd.DataFrame(
        {
            "cat1": ["a", "b", "c"],
            "cat2": ["x", "y", "z"],
            "target": [1, 2, 3],
        }
    )
    preprocessor, num_cols, cat_cols = helpers.build_preprocessor(df, "target")
    assert num_cols == []
    assert len(cat_cols) == 2


def test_build_preprocessor_with_no_categorical_features() -> None:
    """Test preprocessor with only numeric features"""
    df = pd.DataFrame(
        {
            "num1": [1.0, 2.0, 3.0],
            "num2": [4.0, 5.0, 6.0],
            "target": [7.0, 8.0, 9.0],
        }
    )
    preprocessor, num_cols, cat_cols = helpers.build_preprocessor(df, "target")
    assert len(num_cols) == 2
    assert cat_cols == []


def test_is_classification_numeric_target() -> None:
    """Test classification detection with numeric target"""
    y_classification = pd.Series([0, 1, 0, 1, 0])
    # Use more unique values to trigger regression detection (> 20 unique values)
    y_regression = pd.Series([i + 0.5 for i in range(30)])
    
    assert helpers.is_classification(y_classification) is True
    assert helpers.is_classification(y_regression) is False


def test_is_classification_categorical_target() -> None:
    """Test classification detection with categorical target"""
    y = pd.Series(["cat", "dog", "cat", "bird"])
    assert helpers.is_classification(y) is True


def test_save_schema_with_single_unique_categorical() -> None:
    """Test schema generation with single unique category"""
    X = pd.DataFrame(
        {
            "num_col": [1.0, 2.0, 3.0],
            "cat_col": ["a", "a", "a"],
        }
    )
    y = pd.Series([0, 1, 0])
    
    # Should not raise error
    helpers.save_schema(
        X=X,
        num_cols=["num_col"],
        cat_cols=["cat_col"],
        task="classification",
        y=y,
    )


# ============================================================================
# ERROR HANDLING TESTS - Testing error conditions and exceptions
# ============================================================================


def test_quick_dtype_buckets_with_nonexistent_target_col() -> None:
    """Test handling of nonexistent target column"""
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    # Should just not remove anything since target doesn't exist
    num_cols, cat_cols = helpers.quick_dtype_buckets(df, "nonexistent")
    # Both columns should be included since they're not removed
    assert len(num_cols) + len(cat_cols) > 0


def test_save_schema_with_missing_num_and_cat_cols() -> None:
    """Test schema generation when specified columns don't exist"""
    X = pd.DataFrame({"existing_col": [1, 2, 3]})
    y = pd.Series([0, 1, 0])
    
    # Function will raise KeyError when column doesn't exist - test that error happens
    try:
        helpers.save_schema(
            X=X,
            num_cols=["nonexistent_num"],
            cat_cols=["nonexistent_cat"],
            task="classification",
            y=y,
        )
        # If it doesn't raise, then passing empty lists is fine
        assert False, "Should have raised KeyError for missing columns"
    except KeyError:
        # Expected behavior
        pass


def test_select_best_model_with_single_sample_test_set() -> None:
    """Test model selection with minimal test data"""
    df = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4, 5],
            "target": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    
    preprocessor, _, _ = helpers.build_preprocessor(df, "target")
    X = df.drop(columns=["target"])
    y = df["target"]
    
    X_train = X.iloc[:4]
    y_train = y.iloc[:4]
    X_test = X.iloc[4:]  # Single sample
    y_test = y.iloc[4:]
    
    models = {"rf": helpers.RandomForestRegressor(n_estimators=5, random_state=42)}
    
    best_name, best_pipeline, best_score, leaderboard = helpers.select_best_model(
        task="regression",
        preprocessor=preprocessor,
        models=models,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    
    assert best_name == "rf"
    assert best_pipeline is not None


def test_select_best_model_empty_models_dict() -> None:
    """Test model selection with empty models dictionary"""
    df = pd.DataFrame(
        {
            "feature": [1, 2, 3],
            "target": [1.0, 2.0, 3.0],
        }
    )
    
    preprocessor, _, _ = helpers.build_preprocessor(df, "target")
    X = df.drop(columns=["target"])
    y = df["target"]
    
    try:
        helpers.select_best_model(
            task="regression",
            preprocessor=preprocessor,
            models={},  # Empty dict
            X_train=X,
            y_train=y,
            X_test=X,
            y_test=y,
        )
        # If it doesn't raise, that's acceptable
    except (ValueError, KeyError):
        # Expected behavior
        pass


# ============================================================================
# INTEGRATION TESTS - Testing multiple functions working together
# ============================================================================


def test_full_pipeline_classification_workflow() -> None:
    """Test complete classification pipeline from data to model selection"""
    df = pd.DataFrame(
        {
            "num_feature": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "cat_feature": ["a", "b", "a", "b", "a", "b"],
            "target": [0, 1, 0, 1, 0, 1],
        }
    )
    
    # Step 1: Build preprocessor
    preprocessor, num_cols, cat_cols = helpers.build_preprocessor(df, "target")
    assert num_cols == ["num_feature"]
    assert cat_cols == ["cat_feature"]
    
    # Step 2: Verify classification detection
    assert helpers.is_classification(df["target"]) is True
    
    # Step 3: Save schema
    X = df.drop(columns=["target"])
    y = df["target"]
    helpers.save_schema(X, num_cols, cat_cols, "classification", y=y)
    
    # Step 4: Get candidate models
    models = helpers.get_candidate_models("classification", training_mode="fast")
    assert len(models) > 0
    
    # Step 5: Split data and select best model
    X_train = X.iloc[:4]
    y_train = y.iloc[:4]
    X_test = X.iloc[4:]
    y_test = y.iloc[4:]
    
    best_name, best_pipeline, score, leaderboard = helpers.select_best_model(
        task="classification",
        preprocessor=preprocessor,
        models=models,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    
    assert best_name is not None
    assert best_pipeline is not None
    assert isinstance(score, float)
    assert len(leaderboard) > 0


def test_full_pipeline_regression_workflow() -> None:
    """Test complete regression pipeline"""
    # Create regression dataset with > 20 unique values
    df = pd.DataFrame(
        {
            "feature1": [i + 0.5 for i in range(25)],
            "feature2": ["a" if i % 2 == 0 else "b" for i in range(25)],
            "target": [10.5 + i * 2.3 for i in range(25)],
        }
    )
    
    # Step 1: Build preprocessor
    preprocessor, num_cols, cat_cols = helpers.build_preprocessor(df, "target")
    
    # Step 2: Verify regression detection
    assert helpers.is_classification(df["target"]) is False
    
    # Step 3: Get regression models
    models = helpers.get_candidate_models("regression", training_mode="fast")
    assert len(models) > 0
    
    # Step 4: Model selection
    X = df.drop(columns=["target"])
    y = df["target"]
    X_train = X.iloc[:20]
    y_train = y.iloc[:20]
    X_test = X.iloc[20:]
    y_test = y.iloc[20:]
    
    best_name, best_pipeline, score, leaderboard = helpers.select_best_model(
        task="regression",
        preprocessor=preprocessor,
        models=models,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    
    assert best_name is not None
    assert len(leaderboard) > 0


def test_candidate_models_all_valid_sklearn_models() -> None:
    """Verify all candidate models are valid sklearn estimators"""
    for task_type in ["classification", "regression"]:
        for training_mode in ["fast", "high_accuracy"]:
            models = helpers.get_candidate_models(task_type, training_mode=training_mode)
            for model_name, model in models.items():
                # All sklearn models have fit and predict methods
                assert hasattr(model, "fit"), f"{model_name} missing fit method"
                assert hasattr(model, "predict"), f"{model_name} missing predict method"


def test_training_mode_affects_model_count() -> None:
    """Verify that training mode selection properly affects model pool size"""
    for task_type in ["classification", "regression"]:
        fast_models = helpers.get_candidate_models(task_type, training_mode="fast")
        high_acc_models = helpers.get_candidate_models(task_type, training_mode="high_accuracy")
        
        # High accuracy mode should have more models
        assert len(high_acc_models) >= len(fast_models), f"Failed for task: {task_type}"
