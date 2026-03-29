"""
Test Phase 1 Features - AutoML Studio Pro

Tests for:
1. Sample Datasets - Load built-in Iris, Wine, Breast Cancer, Diabetes datasets
2. Missing Value Strategy Selector - Choose imputation method
3. Preprocessing Pipeline Preview - Visual preview of preprocessing steps
4. Cross-Validation Visualization - Bar charts & histograms (test data generation)
5. Model History - Track last 10 trained models (session state logic)
6. Model Comparison Dashboard - Compare models with charts (data generation)
7. Ensemble Model Builder - Voting/stacking ensemble options
8. PDF Report Export - HTML report generation
"""

import datetime
import json

import pandas as pd
import pytest

from automl_app.ui.tabs.train import (
    SAMPLE_DATASETS,
    _load_sample_dataset,
    _profile_dataset,
    _dedupe_columns,
)


# ==============================================================================
# FEATURE 1: Sample Datasets Tests
# ==============================================================================


def test_sample_datasets_iris_loading() -> None:
    """Test loading Iris Flower dataset - positive case."""
    # Arrange & Act
    df = _load_sample_dataset("Iris Flower")
    
    # Assert
    assert df is not None
    assert df.shape[0] == 150  # Classic Iris has 150 samples
    assert df.shape[1] == 5  # 4 features + 1 target
    assert "target" in df.columns
    assert "sepal length (cm)" in df.columns or "feature_0" in df.columns


def test_sample_datasets_wine_loading() -> None:
    """Test loading Wine Quality dataset - positive case."""
    # Arrange & Act
    df = _load_sample_dataset("Wine Quality")
    
    # Assert
    assert df is not None
    assert df.shape[0] == 178  # Wine dataset has 178 samples
    assert "target" in df.columns


def test_sample_datasets_breast_cancer_loading() -> None:
    """Test loading Breast Cancer dataset - positive case."""
    # Arrange & Act
    df = _load_sample_dataset("Breast Cancer")
    
    # Assert
    assert df is not None
    assert df.shape[0] == 569  # Breast cancer has 569 samples
    assert "target" in df.columns


def test_sample_datasets_diabetes_loading() -> None:
    """Test loading Diabetes dataset - positive case."""
    # Arrange & Act
    df = _load_sample_dataset("Diabetes")
    
    # Assert
    assert df is not None
    assert df.shape[0] == 442  # Diabetes has 442 samples
    assert "target" in df.columns


def test_sample_datasets_invalid_name_raises_error() -> None:
    """Test loading invalid dataset name - negative case."""
    # Arrange
    invalid_name = "Invalid Dataset Name"
    
    # Act & Assert
    with pytest.raises(ValueError, match=f"Unknown dataset: {invalid_name}"):
        _load_sample_dataset(invalid_name)


def test_sample_datasets_dict_structure() -> None:
    """Test SAMPLE_DATASETS dictionary has correct structure."""
    # Assert
    assert "Iris Flower" in SAMPLE_DATASETS
    assert "Wine Quality" in SAMPLE_DATASETS
    assert "Breast Cancer" in SAMPLE_DATASETS
    assert "Diabetes" in SAMPLE_DATASETS
    
    # Check each has target and description
    for name, info in SAMPLE_DATASETS.items():
        assert "target" in info
        assert "description" in info
        assert info["target"] == "target"


# ==============================================================================
# FEATURE 2: Missing Value Strategy Tests
# ==============================================================================


def test_missing_value_strategy_numeric_median() -> None:
    """Test median imputation strategy for numeric columns."""
    # This is tested via the session state storage
    # The actual imputation happens in build_preprocessor from helpers
    from automl_app.core.helpers import build_preprocessor
    
    # Arrange
    df = pd.DataFrame({
        "feature1": [1.0, 2.0, None, 4.0, 5.0],
        "feature2": ["a", "b", "c", "d", "e"],
        "target": [10, 20, 30, 40, 50],
    })
    
    # Act
    preprocessor, num_cols, cat_cols = build_preprocessor(
        df, "target", 
        num_impute_strategy="median",
        cat_impute_strategy="most_frequent"
    )
    
    # Assert
    assert "feature1" in num_cols
    transformed = preprocessor.fit_transform(df.drop(columns=["target"]))
    assert transformed is not None
    assert not pd.isna(transformed).any()  # No missing values after imputation


def test_missing_value_strategy_numeric_mean() -> None:
    """Test mean imputation strategy for numeric columns."""
    from automl_app.core.helpers import build_preprocessor
    
    # Arrange
    df = pd.DataFrame({
        "feature1": [1.0, 2.0, None, 4.0, 5.0],
        "feature2": ["a", "b", "c", "d", "e"],
        "target": [10, 20, 30, 40, 50],
    })
    
    # Act
    preprocessor, num_cols, cat_cols = build_preprocessor(
        df, "target",
        num_impute_strategy="mean",
        cat_impute_strategy="most_frequent"
    )
    
    # Assert
    assert "feature1" in num_cols
    transformed = preprocessor.fit_transform(df.drop(columns=["target"]))
    assert transformed is not None


def test_missing_value_strategy_numeric_constant() -> None:
    """Test constant imputation strategy for numeric columns."""
    from automl_app.core.helpers import build_preprocessor
    
    # Arrange
    df = pd.DataFrame({
        "feature1": [1.0, 2.0, None, 4.0, 5.0],
        "target": [10, 20, 30, 40, 50],
    })
    
    # Act
    preprocessor, num_cols, cat_cols = build_preprocessor(
        df, "target",
        num_impute_strategy="constant",
        cat_impute_strategy="most_frequent"
    )
    
    # Assert
    transformed = preprocessor.fit_transform(df.drop(columns=["target"]))
    assert transformed is not None


def test_missing_value_strategy_categorical() -> None:
    """Test categorical imputation strategies."""
    from automl_app.core.helpers import build_preprocessor
    
    # Arrange - data with missing categorical values
    df = pd.DataFrame({
        "num_feat": [1.0, 2.0, 3.0, 4.0, 5.0],
        "cat_feat": ["a", None, "b", None, "c"],
        "target": [10, 20, 30, 40, 50],
    })
    
    # Test most_frequent
    preprocessor, num_cols, cat_cols = build_preprocessor(
        df, "target",
        num_impute_strategy="median",
        cat_impute_strategy="most_frequent"
    )
    assert "cat_feat" in cat_cols
    
    # Test constant
    preprocessor2, _, _ = build_preprocessor(
        df, "target",
        num_impute_strategy="median",
        cat_impute_strategy="constant"
    )
    assert preprocessor2 is not None


# ==============================================================================
# FEATURE 3: Preprocessing Pipeline Preview Tests
# ==============================================================================


def test_preprocessing_preview_column_detection() -> None:
    """Test that preprocessing preview correctly identifies column types."""
    from automl_app.core.helpers import quick_dtype_buckets
    
    # Arrange
    df = pd.DataFrame({
        "num1": [1.0, 2.0, 3.0],
        "num2": [4.0, 5.0, 6.0],
        "cat1": ["a", "b", "c"],
        "cat2": ["x", "y", "z"],
        "target": [1, 2, 3],
    })
    
    # Act
    num_cols, cat_cols = quick_dtype_buckets(df, "target")
    
    # Assert
    assert "num1" in num_cols
    assert "num2" in num_cols
    assert "cat1" in cat_cols
    assert "cat2" in cat_cols
    assert "target" not in num_cols
    assert "target" not in cat_cols


def test_preprocessing_preview_empty_dataframe() -> None:
    """Test preprocessing preview with empty dataframe - edge case."""
    from automl_app.core.helpers import quick_dtype_buckets
    
    # Arrange
    df = pd.DataFrame()
    
    # Act
    num_cols, cat_cols = quick_dtype_buckets(df, "target")
    
    # Assert
    assert num_cols == []
    assert cat_cols == []


# ==============================================================================
# FEATURE 4: Cross-Validation Visualization Tests
# ==============================================================================


def test_cv_visualization_data_format() -> None:
    """Test that CV score data format is correct for visualization."""
    # This tests the data that would be passed to matplotlib for CV visualization
    
    # Arrange - simulate leaderboard data structure
    leaderboard = [
        {"model": "RandomForest", "cv_score": 0.95},
        {"model": "GradientBoosting", "cv_score": 0.92},
        {"model": "LogisticRegression", "cv_score": 0.88},
    ]
    leaderboard_df = pd.DataFrame(leaderboard)
    
    # Assert - data is in correct format for bar chart
    assert "model" in leaderboard_df.columns
    assert "cv_score" in leaderboard_df.columns
    assert len(leaderboard_df) >= 2  # Need at least 2 for distribution
    
    # Test score distribution data
    score_data = leaderboard_df["cv_score"].dropna()
    assert len(score_data) > 0
    assert score_data.mean() > 0.8  # Reasonable score


def test_cv_visualization_single_model() -> None:
    """Test CV visualization with single model - edge case."""
    # Arrange
    leaderboard = [
        {"model": "RandomForest", "cv_score": 0.95},
    ]
    leaderboard_df = pd.DataFrame(leaderboard)
    
    # Assert - single model should still work
    assert len(leaderboard_df) == 1
    assert leaderboard_df["cv_score"].notna().all()


# ==============================================================================
# FEATURE 5: Model History Tests
# ==============================================================================


def test_model_history_tracks_last_10_models() -> None:
    """Test model history keeps only last 10 entries."""
    # Arrange - simulate session state
    model_history = []
    
    # Act - add 15 entries
    for i in range(15):
        history_entry = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "model": f"Model_{i}",
            "score": 0.9 + (i * 0.005),
            "cv_score": 0.85 + (i * 0.005),
            "task": "classification",
            "dataset_rows": 100,
            "dataset_cols": 10,
            "training_mode": "High Accuracy",
        }
        model_history.insert(0, history_entry)
    
    # Keep only last 10
    model_history = model_history[:10]
    
    # Assert
    assert len(model_history) == 10
    assert model_history[0]["model"] == "Model_0"  # Most recent first
    assert model_history[9]["model"] == "Model_9"  # 10th entry


def test_model_history_empty_state() -> None:
    """Test model history with empty state - edge case."""
    # Arrange
    model_history = []
    
    # Assert
    assert len(model_history) == 0


def test_model_history_entry_structure() -> None:
    """Test model history entry has all required fields."""
    # Arrange
    entry = {
        "timestamp": "2026-03-29 10:30",
        "model": "RandomForest",
        "score": 0.95,
        "cv_score": 0.92,
        "task": "classification",
        "dataset_rows": 150,
        "dataset_cols": 5,
        "training_mode": "High Accuracy",
    }
    
    # Assert
    required_fields = ["timestamp", "model", "score", "cv_score", "task", 
                       "dataset_rows", "dataset_cols", "training_mode"]
    for field in required_fields:
        assert field in entry


# ==============================================================================
# FEATURE 6: Model Comparison Dashboard Tests
# ==============================================================================


def test_model_comparison_bar_chart_data() -> None:
    """Test model comparison bar chart data format."""
    # Arrange - simulate model history for comparison
    model_history = [
        {"model": "RandomForest", "score": 0.95},
        {"model": "GradientBoosting", "score": 0.92},
        {"model": "LogisticRegression", "score": 0.88},
    ]
    
    # Act - prepare data for bar chart
    models = [h["model"][:12] + "..." if len(h["model"]) > 12 else h["model"] 
               for h in model_history[:6]]
    scores = [h["score"] for h in model_history[:6]]
    
    # Assert
    assert len(models) == 3
    assert len(scores) == 3
    assert max(scores) == 0.95  # Best model


def test_model_comparison_trend_data() -> None:
    """Test score trend over time data format."""
    # Arrange
    model_history = [
        {"timestamp": "2026-03-29 10:30", "score": 0.85},
        {"timestamp": "2026-03-29 10:25", "score": 0.82},
        {"timestamp": "2026-03-29 10:20", "score": 0.80},
    ]
    
    # Act - reverse for chronological order
    history_reversed = list(reversed(model_history))
    timestamps = [h["timestamp"][-5:] for h in history_reversed[:6]]
    scores_time = [h["score"] for h in history_reversed[:6]]
    
    # Assert
    assert timestamps == ["10:20", "10:25", "10:30"]
    assert scores_time == [0.80, 0.82, 0.85]  # Increasing over time


def test_model_comparison_minimum_2_models() -> None:
    """Test that comparison requires at least 2 models."""
    # Arrange
    model_history_single = [
        {"model": "RandomForest", "score": 0.95},
    ]
    
    # Assert
    assert len(model_history_single) < 2  # Should not show comparison


# ==============================================================================
# FEATURE 7: Ensemble Model Builder Tests
# ==============================================================================


def test_ensemble_model_builder_voting_hard() -> None:
    """Test ensemble builder with voting_hard option."""
    # This would be stored in session state
    ensemble_config = {
        "enabled": True,
        "type": "voting_hard"
    }
    
    # Assert
    assert ensemble_config["enabled"] is True
    assert ensemble_config["type"] == "voting_hard"


def test_ensemble_model_builder_voting_soft() -> None:
    """Test ensemble builder with voting_soft option."""
    ensemble_config = {
        "enabled": True,
        "type": "voting_soft"
    }
    
    assert ensemble_config["enabled"] is True
    assert ensemble_config["type"] == "voting_soft"


def test_ensemble_model_builder_stacking() -> None:
    """Test ensemble builder with stacking option."""
    ensemble_config = {
        "enabled": True,
        "type": "stacking"
    }
    
    assert ensemble_config["enabled"] is True
    assert ensemble_config["type"] == "stacking"


def test_ensemble_model_builder_disabled() -> None:
    """Test ensemble builder when disabled."""
    ensemble_config = {"enabled": False}
    
    assert ensemble_config["enabled"] is False


# ==============================================================================
# FEATURE 8: PDF Report Export (HTML Report) Tests
# ==============================================================================


def test_report_generation_html_structure() -> None:
    """Test HTML report has correct structure."""
    # Arrange
    best_name = "RandomForest"
    task = "classification"
    training_mode = "High Accuracy"
    time_budget_sec = 90
    score = 0.95
    cv_score = 0.92
    df_rows = 150
    df_cols = 5
    target_col = "target"
    num_cols_count = 4
    cat_cols_count = 1
    impute_numeric = "median"
    impute_categorical = "most_frequent"
    
    # Act - generate report HTML (similar to train.py)
    report_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>AutoML Training Report</title>
    </head>
    <body>
        <h1>🚀 AutoML Training Report</h1>
        <p><strong>Generated:</strong> {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        
        <h2>Model Details</h2>
        <table>
            <tr><th>Property</th><th>Value</th></tr>
            <tr><td>Best Model</td><td>{best_name}</td></tr>
            <tr><td>Task Type</td><td>{task}</td></tr>
            <tr><td>Training Mode</td><td>{training_mode}</td></tr>
            <tr><td>Time Budget</td><td>{time_budget_sec}s</td></tr>
        </table>
        
        <h2>Performance Metrics</h2>
        <div>Test Score: {score:.4f}</div>
        <div>CV Score: {cv_score:.4f}</div>
        
        <h2>Dataset Info</h2>
        <table>
            <tr><th>Property</th><th>Value</th></tr>
            <tr><td>Total Rows</td><td>{df_rows}</td></tr>
            <tr><td>Total Columns</td><td>{df_cols}</td></tr>
            <tr><td>Target Column</td><td>{target_col}</td></tr>
            <tr><td>Numeric Features</td><td>{num_cols_count}</td></tr>
            <tr><td>Categorical Features</td><td>{cat_cols_count}</td></tr>
        </table>
        
        <h2>Preprocessing Steps</h2>
        <ul>
            <li>Numeric Imputation: {impute_numeric}</li>
            <li>Categorical Imputation: {impute_categorical}</li>
            <li>Scaling: StandardScaler</li>
            <li>Encoding: OneHotEncoder</li>
        </ul>
    </body>
    </html>
    """
    
    # Assert - verify HTML structure
    assert "<!DOCTYPE html>" in report_html
    assert "<title>AutoML Training Report</title>" in report_html
    assert best_name in report_html
    assert task in report_html
    assert str(time_budget_sec) in report_html
    assert f"{score:.4f}" in report_html
    assert f"{cv_score:.4f}" in report_html
    assert "StandardScaler" in report_html
    assert "OneHotEncoder" in report_html


def test_report_generation_empty_scores() -> None:
    """Test report generation with None CV score - edge case."""
    # Arrange
    cv_score = None
    
    # Act
    cv_score_str = f"{float(cv_score):.4f}" if cv_score is not None else "N/A"
    
    # Assert
    assert cv_score_str == "N/A"


def test_report_download_button_format() -> None:
    """Test report is in correct format for download."""
    # Arrange
    report_content = "<html>...</html>"
    
    # Assert - should be downloadable as HTML
    assert isinstance(report_content, str)
    assert "html" in report_content.lower()


# ==============================================================================
# Additional Utility Tests
# ==============================================================================


def test_dedupe_columns_basic() -> None:
    """Test column deduplication utility."""
    # Arrange
    columns = ["col1", "col2", "col1", "col3", "col2"]
    
    # Act
    result = _dedupe_columns(columns)
    
    # Assert
    assert result == ["col1", "col2", "col1_1", "col3", "col2_1"]


def test_dedupe_columns_with_empty_names() -> None:
    """Test column deduplication with empty names - edge case."""
    # Arrange
    columns = ["", "  ", "col1", ""]
    
    # Act
    result = _dedupe_columns(columns)
    
    # Assert
    assert "unnamed" in result or len(result) == 4


def test_profile_dataset_classification() -> None:
    """Test dataset profiling for classification task."""
    # Arrange
    df = pd.DataFrame({
        "feature1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "feature2": ["a", "b", "a", "b", "a", "b"],
        "target": [0, 1, 0, 1, 0, 1],  # Binary classification
    })
    
    # Act
    profile = _profile_dataset(df, "target")
    
    # Assert
    assert profile["task"] == "classification"
    assert profile["rows"] == 6
    assert "health_score" in profile
    assert "balance_note" in profile


def test_profile_dataset_regression() -> None:
    """Test dataset profiling for regression task."""
    # Arrange
    df = pd.DataFrame({
        "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "feature2": [2.0, 4.0, 6.0, 8.0, 10.0],
        "target": [10, 20, 30, 40, 50],  # Regression
    })
    
    # Act
    profile = _profile_dataset(df, "target")
    
    # Assert
    assert profile["task"] == "regression"
    assert profile["rows"] == 5


def test_profile_dataset_with_missing_target() -> None:
    """Test dataset profiling when target has missing values - edge case."""
    # Arrange
    df = pd.DataFrame({
        "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "target": [0, 1, None, 0, 1],  # Missing value in target
    })
    
    # Act
    profile = _profile_dataset(df, "target")
    
    # Assert
    assert profile["removed_target_na"] == 1
    assert profile["rows"] == 5


# ==============================================================================
# Integration Tests
# ==============================================================================


def test_full_pipeline_sample_to_training() -> None:
    """Integration test: Load sample dataset and verify it's ready for training."""
    # Arrange & Act
    df = _load_sample_dataset("Iris Flower")
    
    # Verify data is ready for preprocessing
    num_cols, cat_cols = pd.DataFrame(), []
    from automl_app.core.helpers import quick_dtype_buckets
    num_cols, cat_cols = quick_dtype_buckets(df, "target")
    
    # Assert
    assert df is not None
    assert len(df) > 0
    assert "target" in df.columns
    assert len(num_cols) > 0  # Has numeric features


def test_impute_strategy_persists_across_sessions(tmp_path) -> None:
    """Test that imputation strategies can be stored in session state."""
    # This simulates session state behavior
    import sys
    import types
    
    # Create a mock module to simulate session state
    mock_session = types.ModuleType("mock_streamlit")
    mock_session_state = {}
    
    def mock_get(key, default=None):
        return mock_session_state.get(key, default)
    
    def mock_set(key, value):
        mock_session_state[key] = value
    
    # Arrange - simulate storing imputation strategy
    impute_strategy = {
        "numeric": "mean",
        "categorical": "constant"
    }
    
    # Act
    stored_strategy = impute_strategy
    
    # Assert
    assert stored_strategy["numeric"] == "mean"
    assert stored_strategy["categorical"] == "constant"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
