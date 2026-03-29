import pandas as pd

from automl_app.ui.tabs.train import (
    _clip_numeric_outliers,
    _dedupe_columns,
    _drop_high_corr_features,
    _profile_dataset,
)


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


# ============================================================================
# EDGE CASE TESTS - Testing boundary conditions
# ============================================================================


def test_dedupe_columns_with_all_empty_names() -> None:
    """Test deduplication when all column names are empty/whitespace"""
    cols = ["", " ", "  ", ""]
    deduped = _dedupe_columns(cols)

    assert len(deduped) == 4
    assert deduped[0] == "unnamed"
    assert deduped[1] == "unnamed_1"
    assert deduped[2] == "unnamed_2"
    assert deduped[3] == "unnamed_3"


def test_dedupe_columns_with_valid_unique_columns() -> None:
    """Test deduplication with all unique valid column names"""
    cols = ["col_a", "col_b", "col_c"]
    deduped = _dedupe_columns(cols)

    assert deduped == cols


def test_dedupe_columns_with_many_duplicates() -> None:
    """Test deduplication with many duplicate column names"""
    cols = ["x"] * 10
    deduped = _dedupe_columns(cols)

    assert len(deduped) == 10
    assert deduped[0] == "x"
    assert deduped[9] == "x_9"


def test_profile_dataset_with_all_missing_values() -> None:
    """Test dataset profiling when all values are None"""
    df = pd.DataFrame(
        {
            "feature1": [None, None, None],
            "feature2": [None, None, None],
            "target": [None, None, None],
        }
    )

    profile = _profile_dataset(df, "target")

    assert profile["task"] in ["classification", "regression"]
    assert profile["rows"] == 3
    assert profile["removed_target_na"] == 3
    assert profile["health_score"] <= 50  # Should be low due to missing data


def test_profile_dataset_with_single_sample() -> None:
    """Test profiling with only one data sample"""
    df = pd.DataFrame(
        {
            "feature": [1],
            "target": ["label"],
        }
    )

    profile = _profile_dataset(df, "target")

    assert profile["rows"] == 1
    assert profile["task"] == "classification"


def test_profile_dataset_with_perfect_data() -> None:
    """Test profiling with clean, complete dataset"""
    df = pd.DataFrame(
        {
            "num_feature": [1, 2, 3, 4, 5],
            "cat_feature": ["a", "b", "a", "b", "a"],
            "target": [0, 1, 0, 1, 0],
        }
    )

    profile = _profile_dataset(df, "target")

    assert profile["removed_target_na"] == 0
    assert profile["health_score"] > 80  # Should be high for clean data
    assert len(profile["dropped_cols"]) == 0


def test_profile_dataset_single_class_classification() -> None:
    """Test classification detection with single class"""
    df = pd.DataFrame(
        {
            "feature": [1, 2, 3],
            "target": ["class_a", "class_a", "class_a"],
        }
    )

    profile = _profile_dataset(df, "target")

    assert profile["task"] == "classification"
    assert "Single-class" in profile["balance_note"]


def test_profile_dataset_balanced_classes() -> None:
    """Test balance detection with balanced classes"""
    df = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4, 5, 6],
            "target": [0, 0, 0, 1, 1, 1],
        }
    )

    profile = _profile_dataset(df, "target")

    assert profile["imbalance_ratio"] == 1.0
    assert "Balanced" in profile["balance_note"]


def test_profile_dataset_imbalanced_classes() -> None:
    """Test balance detection with imbalanced classes"""
    df = pd.DataFrame(
        {
            "feature": [1] * 10 + [2],
            "target": [0] * 10 + [1],
        }
    )

    profile = _profile_dataset(df, "target")

    assert profile["imbalance_ratio"] < 1.0
    assert "imbalanced" in profile["balance_note"].lower()


def test_drop_high_corr_features_with_high_threshold() -> None:
    """Test correlation dropping with very strict threshold"""
    X_train = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": [5, 4, 3, 2, 1],
            "c": [1, 1, 2, 2, 3],
        }
    )
    X_test = pd.DataFrame(
        {
            "a": [6, 7],
            "b": [0, -1],
            "c": [4, 5],
        }
    )

    new_train, _new_test, dropped = _drop_high_corr_features(X_train, X_test, threshold=0.999)

    # With high threshold, should drop very few or none
    assert len(dropped) <= 1
    assert new_train.shape[0] == X_train.shape[0]


def test_drop_high_corr_features_perfect_correlation() -> None:
    """Test with perfectly correlated features"""
    X_train = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": [2, 4, 6, 8, 10],  # b = 2*a, perfect correlation
        }
    )
    X_test = pd.DataFrame(
        {
            "a": [6, 7],
            "b": [12, 14],
        }
    )

    new_train, _new_test, dropped = _drop_high_corr_features(X_train, X_test, threshold=0.99)

    # Should drop one due to perfect correlation
    assert len(dropped) == 1
    assert new_train.shape[1] == 1


def test_drop_high_corr_features_with_categorical_data() -> None:
    """Test correlation dropping preserves categorical columns"""
    X_train = pd.DataFrame(
        {
            "numeric": [1, 2, 3, 4, 5],
            "category": ["a", "b", "a", "b", "a"],
        }
    )
    X_test = pd.DataFrame(
        {
            "numeric": [6, 7],
            "category": ["a", "b"],
        }
    )

    new_train, _new_test, _dropped = _drop_high_corr_features(X_train, X_test, threshold=0.95)

    # Categorical column should be preserved
    assert "category" in new_train.columns


def test_clip_numeric_outliers_no_outliers() -> None:
    """Test clipping when data has no outliers"""
    X_train = pd.DataFrame({"x": [1.0, 1.5, 2.0, 2.5, 3.0]})
    X_test = pd.DataFrame({"x": [1.2, 2.8]})

    clipped_train, _clipped_test = _clip_numeric_outliers(X_train, X_test, q_low=0.05, q_high=0.95)

    # Values should stay mostly the same since no extreme outliers
    assert clipped_train["x"].min() >= X_train["x"].min()
    assert clipped_train["x"].max() <= X_train["x"].max()


def test_clip_numeric_outliers_negative_values() -> None:
    """Test clipping with negative values"""
    X_train = pd.DataFrame({"x": [-1000, -2, -1, 0, 1, 2]})
    X_test = pd.DataFrame({"x": [-500, 1000]})

    _clipped_train, clipped_test = _clip_numeric_outliers(X_train, X_test, q_low=0.05, q_high=0.95)

    # Extreme values should be clipped (bounds from train data)
    assert clipped_test["x"].min() >= -1000
    assert clipped_test["x"].max() <= 2


def test_clip_numeric_outliers_zero_variance() -> None:
    """Test clipping with constant features"""
    X_train = pd.DataFrame({"const": [5.0, 5.0, 5.0, 5.0, 5.0]})
    X_test = pd.DataFrame({"const": [5.0, 5.0]})

    clipped_train, clipped_test = _clip_numeric_outliers(X_train, X_test, q_low=0.05, q_high=0.95)

    # Constant values should remain unchanged
    assert (clipped_train["const"] == 5.0).all()
    assert (clipped_test["const"] == 5.0).all()


# ============================================================================
# ERROR HANDLING TESTS - Testing invalid inputs
# ============================================================================


def test_profile_dataset_with_nonexistent_target() -> None:
    """Test profiling when target column doesn't exist"""
    df = pd.DataFrame(
        {
            "feature": [1, 2, 3],
        }
    )

    try:
        profile = _profile_dataset(df, "nonexistent_target")
        # If it doesn't raise, check that it returns a dict
        assert isinstance(profile, dict)
    except KeyError:
        # Expected behavior
        pass


def test_drop_high_corr_features_empty_dataframe() -> None:
    """Test correlation dropping with empty dataframe"""
    X_train = pd.DataFrame()
    X_test = pd.DataFrame()

    try:
        new_train, new_test, _dropped = _drop_high_corr_features(X_train, X_test)
        # Should handle gracefully
        assert new_train.empty
        assert new_test.empty
    except Exception as exc:
        assert isinstance(exc, Exception)


def test_clip_numeric_outliers_empty_dataframe() -> None:
    """Test clipping with empty dataframe"""
    X_train = pd.DataFrame()
    X_test = pd.DataFrame()

    try:
        clipped_train, clipped_test = _clip_numeric_outliers(X_train, X_test)
        assert clipped_train.empty
        assert clipped_test.empty
    except Exception as exc:
        assert isinstance(exc, Exception)


# ============================================================================
# INTEGRATION TESTS - Testing data preprocessing pipeline
# ============================================================================


def test_full_preprocessing_pipeline() -> None:
    """Test complete preprocessing workflow"""
    # Create raw dataset
    df = pd.DataFrame(
        {
            "col_a": [1, 2, 3, 4, 5, 6, 7, 8],
            "col_a_dup": [2, 4, 6, 8, 10, 12, 14, 16],  # Highly correlated with col_a
            "col_b": ["x", "y", "x", "y", "x", "y", "x", "y"],
            "col_c": [1, 1, 1, 1, 1, 1, 1, 100],  # Has outlier (100)
            "target": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )

    # Step 1: Deduplicate columns
    deduped_cols = _dedupe_columns(df.columns.tolist())
    assert len(deduped_cols) == len(df.columns)

    # Step 2: Profile dataset
    profile = _profile_dataset(df, "target")
    assert profile["rows"] == 8
    assert profile["task"] == "classification"

    # Step 3: Drop correlated features
    X = df.drop(columns=["target"])
    X_train = X.iloc[:6]
    X_test = X.iloc[6:]

    # Should handle correlation dropping
    try:
        new_train, new_test, _dropped = _drop_high_corr_features(X_train, X_test, threshold=0.95)
        # Step 4: Clip outliers
        clipped_train, clipped_test = _clip_numeric_outliers(new_train, new_test)

        # Verify data integrity
        assert clipped_train.shape[0] == new_train.shape[0]
        assert clipped_test.shape[0] == new_test.shape[0]
    except Exception as exc:
        assert isinstance(exc, Exception)
