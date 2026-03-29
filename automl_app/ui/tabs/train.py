import datetime
import io
import os
import zipfile

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris, load_wine
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures

from automl_app.core.config import ARTIFACTS_DIR
from automl_app.core.helpers import (
    build_preprocessor,
    generate_python_code,
    get_candidate_models,
    is_classification,
    quick_dtype_buckets,
    save_schema,
    select_best_model,
    tune_top_models,
)

# Sample datasets for quick demo
SAMPLE_DATASETS = {
    "Iris Flower": {
        "target": "target",
        "description": "Classic iris flower dataset - 150 samples, 4 features, 3 classes",
    },
    "Wine Quality": {
        "target": "target",
        "description": "Wine quality dataset - 178 samples, 13 features",
    },
    "Breast Cancer": {
        "target": "target",
        "description": "Breast cancer Wisconsin - 569 samples, 30 features",
    },
    "Diabetes": {
        "target": "target",
        "description": "Diabetes progression - 442 samples, 10 features",
    },
}


def _load_sample_dataset(name: str) -> pd.DataFrame:
    """Load a built-in sample dataset."""
    loaders = {
        "Iris Flower": load_iris,
        "Wine Quality": load_wine,
        "Breast Cancer": load_breast_cancer,
        "Diabetes": load_diabetes,
    }

    if name not in loaders:
        raise ValueError(f"Unknown dataset: {name}")

    bunch = loaders[name](as_frame=True)

    # Handle different return types (sklearn version compatibility)
    try:
        return bunch.frame
    except Exception:
        # Fallback: manually construct DataFrame
        feature_names = list(bunch.feature_names) if hasattr(bunch, "feature_names") else [f"feature_{i}" for i in range(bunch.data.shape[1])]
        df = pd.DataFrame(bunch.data, columns=feature_names)
        target_name = "target"
        df[target_name] = bunch.target
        return df


def _load_csv_robust(uploaded_file) -> pd.DataFrame:
    uploaded_file.seek(0)
    try:
        return pd.read_csv(uploaded_file)
    except Exception:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, encoding="latin-1")


def _dedupe_columns(columns: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    final_cols: list[str] = []
    for col in columns:
        base = str(col).strip() or "unnamed"
        if base not in seen:
            seen[base] = 0
            final_cols.append(base)
            continue
        seen[base] += 1
        final_cols.append(f"{base}_{seen[base]}")
    return final_cols


def _profile_dataset(df: pd.DataFrame, target_col: str) -> dict[str, object]:
    profile: dict[str, object] = {}
    total_rows = int(df.shape[0])
    total_features = int(max(df.shape[1] - 1, 0))
    cleaned_df = df.dropna(subset=[target_col])
    removed_target_na = int(total_rows - cleaned_df.shape[0])

    X = cleaned_df.drop(columns=[target_col], errors="ignore")
    all_null_cols = X.columns[X.isna().all()].tolist()
    nunique = X.nunique(dropna=False)
    constant_cols = nunique[nunique <= 1].index.tolist()
    dropped_cols = sorted(set(all_null_cols).union(constant_cols))
    usable_features = max(X.shape[1] - len(dropped_cols), 0)

    task = "classification" if is_classification(cleaned_df[target_col]) else "regression"
    balance_note = "N/A"
    imbalance_ratio = None
    if task == "classification" and not cleaned_df.empty:
        counts = cleaned_df[target_col].value_counts(dropna=False)
        if counts.shape[0] >= 2 and counts.max() > 0:
            imbalance_ratio = float(counts.min() / counts.max())
            if imbalance_ratio >= 0.6:
                balance_note = "Balanced"
            elif imbalance_ratio >= 0.3:
                balance_note = "Moderately imbalanced"
            else:
                balance_note = "Highly imbalanced"
        else:
            balance_note = "Single-class target"

    completeness = 0.0
    if total_rows > 0:
        completeness = max(0.0, 1.0 - (removed_target_na / total_rows))
    feature_quality = 0.0
    if total_features > 0:
        feature_quality = max(0.0, min(1.0, usable_features / total_features))
    score = round((0.55 * completeness + 0.45 * feature_quality) * 100)

    profile["task"] = task
    profile["rows"] = total_rows
    profile["removed_target_na"] = removed_target_na
    profile["usable_features"] = usable_features
    profile["dropped_cols"] = dropped_cols
    profile["health_score"] = score
    profile["balance_note"] = balance_note
    profile["imbalance_ratio"] = imbalance_ratio
    return profile


def _drop_high_corr_features(X_train: pd.DataFrame, X_test: pd.DataFrame, threshold: float = 0.98):
    num_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
    if len(num_cols) < 2:
        return X_train, X_test, []

    corr = X_train[num_cols].corr().abs()
    mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
    upper = corr.where(mask)
    to_drop = [col for col in upper.columns if (upper[col] > threshold).any()]

    if not to_drop:
        return X_train, X_test, []
    return X_train.drop(columns=to_drop, errors="ignore"), X_test.drop(columns=to_drop, errors="ignore"), to_drop


def _clip_numeric_outliers(X_train: pd.DataFrame, X_test: pd.DataFrame, q_low: float = 0.01, q_high: float = 0.99):
    num_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
    if not num_cols:
        return X_train, X_test

    train_clipped = X_train.copy()
    test_clipped = X_test.copy()
    low = train_clipped[num_cols].quantile(q_low)
    high = train_clipped[num_cols].quantile(q_high)

    for col in num_cols:
        train_clipped[col] = train_clipped[col].clip(lower=low[col], upper=high[col])
        test_clipped[col] = test_clipped[col].clip(lower=low[col], upper=high[col])
    return train_clipped, test_clipped


def render_train_tab() -> None:
    st.markdown("#### 1. Build Your Model")
    st.info("Upload data, train the AI, and inspect how it works.")

    # Sample datasets section
    with st.expander("📂 Load Sample Dataset", expanded=False):
        sample_options = ["-- Select a sample dataset --", *list(SAMPLE_DATASETS.keys())]
        selected_sample = st.selectbox("Choose a dataset", sample_options, key="sample_dataset")

        if selected_sample != "-- Select a sample dataset --":
            dataset_info = SAMPLE_DATASETS[selected_sample]
            st.caption(dataset_info["description"])
            if st.button("Load Dataset", key="load_sample"):
                try:
                    df = _load_sample_dataset(selected_sample)
                    st.session_state["df_train"] = df
                    st.session_state["sample_loaded"] = selected_sample
                    st.rerun()
                except Exception as e:
                    st.error(f"Error loading dataset: {e}")

    # Check if sample dataset was previously loaded
    if "sample_loaded" in st.session_state and st.session_state.get("df_train") is not None:
        df = st.session_state["df_train"]
        st.success(f"✅ Loaded: {st.session_state['sample_loaded']}")

        c1, c2 = st.columns([3, 1])
        with c1:
            st.write(f"**Data Preview:** {df.shape[0]} rows, {df.shape[1]} columns")
            st.dataframe(df.head(3), use_container_width=True)
        with c2:
            all_cols = df.columns.tolist()
            default_idx = all_cols.index(dataset_info["target"]) if dataset_info["target"] in all_cols else len(all_cols) - 1
            target_col = st.selectbox("🎯 Target Column", all_cols, index=default_idx, key="target_col_sample")

        # Clear sample dataset button
        if st.button("Clear Dataset", key="clear_sample"):
            del st.session_state["df_train"]
            if "sample_loaded" in st.session_state:
                del st.session_state["sample_loaded"]
            st.rerun()

    else:
        # Normal file upload
        uploaded_file = st.file_uploader("Drop your CSV file here", type=["csv"], key="train_up")

        if not uploaded_file:
            return

        df = _load_csv_robust(uploaded_file)
        df.columns = _dedupe_columns(df.columns.tolist())
        st.session_state["df_train"] = df

    c1, c2 = st.columns([3, 1])
    with c1:
        st.write(f"**Data Preview:** {df.shape[0]} rows, {df.shape[1]} columns")
        st.dataframe(df.head(3), use_container_width=True)
    with c2:
        all_cols = df.columns.tolist()
        target_col = st.selectbox("🎯 Target Column", all_cols, index=len(all_cols) - 1, key="target_col_csv")

    profile = _profile_dataset(df, target_col)
    st.markdown("##### 📈 Auto Data Profiler")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Health Score", f"{profile['health_score']} / 100")
    p2.metric("Detected Task", str(profile["task"]).title())
    p3.metric("Rows Removed", profile["removed_target_na"])
    p4.metric("Usable Features", profile["usable_features"])

    dropped_cols = profile["dropped_cols"]
    if dropped_cols:
        preview = ", ".join(dropped_cols[:6])
        suffix = "..." if len(dropped_cols) > 6 else ""
        st.caption(f"Dropped columns (estimate): {preview}{suffix}")

    if profile["task"] == "classification":
        ratio = profile["imbalance_ratio"]
        if ratio is None:
            st.caption(f"Class Balance: {profile['balance_note']}")
        else:
            st.caption(f"Class Balance: {profile['balance_note']} (minor/major ratio: {ratio:.2f})")

    training_mode = st.selectbox("⚡ Training Mode", ["Fast", "High Accuracy"], index=1, key="training_mode")
    training_mode_value = "fast" if training_mode == "Fast" else "high_accuracy"

    time_budget_sec = st.slider("⏱️ Training Time Budget (seconds)", min_value=15, max_value=300, value=90, step=15)

    # Missing Value Strategy Selector
    with st.expander("🔧 Missing Value Strategy", expanded=False):
        st.caption("Choose how to handle missing values in your data")
        num_impute_strategy = st.selectbox(
            "Numeric Columns Imputation",
            ["median", "mean", "most_frequent", "constant"],
            index=0,
            help="Strategy for imputing missing numeric values",
            key="num_impute_strategy"
        )
        cat_impute_strategy = st.selectbox(
            "Categorical Columns Imputation",
            ["most_frequent", "constant"],
            index=0,
            help="Strategy for imputing missing categorical values",
            key="cat_impute_strategy"
        )

        # Store in session for use in preprocessing
        st.session_state["impute_strategy"] = {
            "numeric": num_impute_strategy,
            "categorical": cat_impute_strategy
        }

        # Preprocessing Pipeline Preview
        st.markdown("---")
        st.markdown("##### 🔍 Preprocessing Pipeline Preview")

        # Get column info
        num_cols_preview, cat_cols_preview = quick_dtype_buckets(df, target_col)

        # Display preprocessing steps
        preview_col1, preview_col2 = st.columns(2)
        with preview_col1:
            st.markdown("**Numeric Columns**")
            if num_cols_preview:
                for col in num_cols_preview[:5]:
                    st.caption(f"• {col}")
                if len(num_cols_preview) > 5:
                    st.caption(f"... and {len(num_cols_preview) - 5} more")
                st.success(f"✓ Imputation: {num_impute_strategy}")
                st.success("✓ Scaling: StandardScaler")
            else:
                st.caption("No numeric columns")

        with preview_col2:
            st.markdown("**Categorical Columns**")
            if cat_cols_preview:
                for col in cat_cols_preview[:5]:
                    st.caption(f"• {col}")
                if len(cat_cols_preview) > 5:
                    st.caption(f"... and {len(cat_cols_preview) - 5} more")
                st.success(f"✓ Imputation: {cat_impute_strategy}")
                st.success("✓ Encoding: OneHotEncoder")
            else:
                st.caption("No categorical columns")

    classification_metric = "accuracy"
    if profile["task"] == "classification":
        metric_map = {
            "Accuracy": "accuracy",
            "F1 (Weighted)": "f1",
            "ROC AUC": "roc_auc",
        }
        selected_metric = st.selectbox("🎯 Optimization Metric", list(metric_map.keys()), index=0, key="optimization_metric")
        classification_metric = metric_map[selected_metric]

    enable_tuning = st.checkbox("⚙️ Enable lightweight hyperparameter tuning (top models)", value=True)

    # Advanced AutoML with Optuna
    with st.expander("🧠 Advanced AutoML (Optuna)", expanded=False):
        st.caption("Use Optuna for advanced hyperparameter optimization")

        enable_optuna = st.checkbox("Enable Optuna Optimization", value=False, key="enable_optuna")

        if enable_optuna:
            optuna_trials = st.slider("Number of Trials", min_value=10, max_value=100, value=30, key="optuna_trials")
            optuna_timeout = st.slider("Timeout (seconds)", min_value=30, max_value=300, value=60, key="optuna_timeout")
            st.session_state["optuna_config"] = {
                "enabled": True,
                "trials": optuna_trials,
                "timeout": optuna_timeout
            }
            st.success(f"✓ Optuna enabled: {optuna_trials} trials, {optuna_timeout}s timeout")
        else:
            st.session_state["optuna_config"] = {"enabled": False}

    # Feature Engineering Module
    with st.expander("🔧 Feature Engineering", expanded=False):
        st.caption("Automatically create new features to improve model performance")

        fe_col1, fe_col2 = st.columns(2)

        with fe_col1:
            enable_poly = st.checkbox("Polynomial Features", value=False, key="enable_poly")
            if enable_poly:
                poly_degree = st.slider("Polynomial Degree", min_value=2, max_value=3, value=2, key="poly_degree")
                st.caption(f"Creates {poly_degree}-degree polynomial features")

        with fe_col2:
            enable_interactions = st.checkbox("Interaction Features", value=False, key="enable_interactions")
            if enable_interactions:
                max_interactions = st.slider("Max Interactions", min_value=2, max_value=5, value=3, key="max_interactions")
                st.caption(f"Creates up to {max_interactions} interaction features")

        enable_aggregations = st.checkbox("Statistical Aggregations", value=False, key="enable_aggregations")
        if enable_aggregations:
            st.caption("Creates mean, std, min, max aggregations for numeric features")

        # Store feature engineering config
        st.session_state["feature_engineering"] = {
            "polynomial": {"enabled": enable_poly, "degree": poly_degree if enable_poly else 2},
            "interactions": {"enabled": enable_interactions, "max_features": max_interactions if enable_interactions else 3},
            "aggregations": {"enabled": enable_aggregations}
        }

        if enable_poly or enable_interactions or enable_aggregations:
            st.success("✓ Feature Engineering enabled")

    # Ensemble Model Builder
    with st.expander("🎛️ Ensemble Model Builder", expanded=False):
        st.caption("Combine multiple models into an ensemble for potentially better performance")
        enable_ensemble = st.checkbox("Enable Ensemble (Voting Classifier)", value=False, help="Combines multiple models using voting")

        if enable_ensemble:
            ensemble_type = st.selectbox("Ensemble Type", ["voting_hard", "voting_soft", "stacking"], index=0, key="ensemble_type")
            st.session_state["ensemble_config"] = {
                "enabled": True,
                "type": ensemble_type
            }
            st.success(f"✓ Ensemble enabled: {ensemble_type}")
        else:
            st.session_state["ensemble_config"] = {"enabled": False}

    # NLP/Text Classification Support
    with st.expander("📝 NLP/Text Classification", expanded=False):
        st.caption("Enable text preprocessing for text-based classification")

        enable_nlp = st.checkbox("Enable Text Processing", value=False, key="enable_nlp")

        if enable_nlp:
            # Detect potential text columns
            text_cols = df.select_dtypes(include=["object"]).columns.tolist()
            if target_col in text_cols:
                text_cols.remove(target_col)

            if text_cols:
                selected_text_cols = st.multiselect(
                    "Select Text Columns",
                    options=text_cols,
                    default=text_cols[:2] if len(text_cols) >= 2 else text_cols,
                    key="text_cols"
                )

                nlp_col1, nlp_col2 = st.columns(2)
                with nlp_col1:
                    max_features = st.slider("Max TF-IDF Features", min_value=100, max_value=5000, value=1000, key="max_tfidf")
                with nlp_col2:
                    ngram_range = st.selectbox("N-gram Range", ["(1,1)", "(1,2)", "(1,3)"], index=1, key="ngram_range")

                import ast
                st.session_state["nlp_config"] = {
                    "enabled": True,
                    "text_columns": selected_text_cols,
                    "max_features": max_features,
                    "ngram_range": ast.literal_eval(ngram_range)
                }
                st.success(f"✓ NLP enabled for {len(selected_text_cols)} text column(s)")
            else:
                st.warning("No text columns detected in dataset")
                st.session_state["nlp_config"] = {"enabled": False}
        else:
            st.session_state["nlp_config"] = {"enabled": False}

    # Time Series Support
    with st.expander("📈 Time Series Forecasting", expanded=False):
        st.caption("Enable time series forecasting for temporal data")

        enable_timeseries = st.checkbox("Enable Time Series Mode", value=False, key="enable_timeseries")

        if enable_timeseries:
            # Detect potential date columns
            date_cols = []
            for col in df.columns:
                if df[col].dtype == 'object':
                    is_date_col = False
                    try:
                        pd.to_datetime(df[col].head(5))
                        is_date_col = True
                    except (ValueError, TypeError):
                        is_date_col = False
                    if is_date_col:
                        date_cols.append(col)
                elif 'datetime' in str(df[col].dtype):
                    date_cols.append(col)

            if date_cols:
                date_col = st.selectbox("Select Date Column", options=date_cols, key="date_col")

                ts_col1, ts_col2 = st.columns(2)
                with ts_col1:
                    forecast_periods = st.slider("Forecast Periods", min_value=1, max_value=30, value=7, key="forecast_periods")
                with ts_col2:
                    ts_model = st.selectbox("Model", ["ARIMA", "Exponential Smoothing"], index=0, key="ts_model")

                st.session_state["timeseries_config"] = {
                    "enabled": True,
                    "date_column": date_col,
                    "forecast_periods": forecast_periods,
                    "model": ts_model
                }
                st.success(f"✓ Time Series enabled: {ts_model} forecasting {forecast_periods} periods")
            else:
                st.warning("No date columns detected in dataset")
                st.session_state["timeseries_config"] = {"enabled": False}
        else:
            st.session_state["timeseries_config"] = {"enabled": False}

    # Data Versioning
    with st.expander("📁 Data Versioning", expanded=False):
        st.caption("Track and compare datasets across versions")

        # Initialize data history in session state
        if "data_history" not in st.session_state:
            st.session_state["data_history"] = []

        # Save current dataset
        if st.button("💾 Save Current Dataset Version", key="save_data_version"):
            import hashlib
            data_hash = hashlib.sha256(df.to_csv().encode()).hexdigest()[:8]
            version_entry = {
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                "name": f"Version {len(st.session_state['data_history']) + 1}",
                "rows": df.shape[0],
                "cols": df.shape[1],
                "hash": data_hash,
                "columns": list(df.columns)
            }
            st.session_state["data_history"].insert(0, version_entry)
            st.session_state["data_history"] = st.session_state["data_history"][:10]  # Keep last 10
            st.success(f"✓ Dataset saved (hash: {data_hash})")

        # Display data history
        if st.session_state["data_history"]:
            st.markdown("**Dataset Versions:**")
            for entry in st.session_state["data_history"][:5]:
                st.caption(f"• {entry['name']} - {entry['rows']} rows, {entry['cols']} cols ({entry['timestamp']})")

            if st.button("Clear History", key="clear_data_history"):
                st.session_state["data_history"] = []
                st.rerun()
        else:
            st.caption("No saved versions yet")

    train_btn = st.button("🚀 Start Training")

    if not train_btn:
        return

    status = st.status("Running AutoML Pipeline...", expanded=True)
    try:
        status.write("🧹 Cleaning & Preprocessing...")
        df = df.dropna(subset=[target_col])
        if df.empty:
            raise ValueError("Target column me valid data nahi mila. Missing values remove karne ke baad rows 0 reh gaye.")

        X = df.drop(columns=[target_col]).copy()
        y_raw = df[target_col].copy()

        # Remove fully-empty and constant columns to keep training stable across noisy datasets.
        X = X.dropna(axis=1, how="all")
        nunique = X.nunique(dropna=False)
        X = X.loc[:, nunique > 1]
        if X.shape[1] == 0:
            raise ValueError("Training ke liye koi useful feature column nahi mila.")

        task = "classification" if is_classification(df[target_col]) else "regression"
        status.write(f"🧭 Detected Task: **{task.upper()}**")

        y = y_raw

        label_encoder = None
        y_orig = y
        if task == "classification" and y.dtype.kind in ("O", "b"):
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y.astype(str))
            y_orig = None

        stratify_y = None
        use_smote = False
        smote_k_neighbors = 1
        if task == "classification":
            y_series = pd.Series(y)
            class_counts = y_series.value_counts(dropna=False)
            if class_counts.shape[0] < 2:
                raise ValueError("Classification ke liye kam se kam 2 classes chahiye.")

            min_class_count = int(class_counts.min())
            if min_class_count >= 2:
                stratify_y = y
            use_smote = min_class_count >= 3
            smote_k_neighbors = max(1, min(5, min_class_count - 1))

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=stratify_y,
        )

        if X_train.shape[0] < 2 or X_test.shape[0] < 1:
            raise ValueError("Dataset bohot chhota hai. Kam se kam 3-5 rows ka usable data chahiye.")

        X_train, X_test, dropped_corr_cols = _drop_high_corr_features(X_train, X_test, threshold=0.98)
        X_train, X_test = _clip_numeric_outliers(X_train, X_test)
        if dropped_corr_cols:
            status.write(f"🧪 Auto Feature Selection: {len(dropped_corr_cols)} highly-correlated feature(s) removed")

        # Feature Engineering
        fe_config = st.session_state.get("feature_engineering", {})

        # Polynomial Features
        if fe_config.get("polynomial", {}).get("enabled", False):
            status.write("🔧 Creating Polynomial Features...")
            num_cols_fe = X_train.select_dtypes(include=["number"]).columns.tolist()
            if num_cols_fe:
                poly_degree = fe_config["polynomial"].get("degree", 2)
                poly = PolynomialFeatures(degree=poly_degree, include_bias=False, interaction_only=False)
                poly_train = poly.fit_transform(X_train[num_cols_fe].fillna(0))
                poly_test = poly.transform(X_test[num_cols_fe].fillna(0))
                poly_feature_names = poly.get_feature_names_out(num_cols_fe)

                # Add polynomial features (limit to avoid explosion)
                max_poly_features = 20
                if len(poly_feature_names) > max_poly_features:
                    poly_feature_names = poly_feature_names[:max_poly_features]
                    poly_train = poly_train[:, :max_poly_features]
                    poly_test = poly_test[:, :max_poly_features]

                for i, name in enumerate(poly_feature_names):
                    if name not in X_train.columns:
                        X_train[name] = poly_train[:, i]
                        X_test[name] = poly_test[:, i]

                status.write(f"✓ Created {len(poly_feature_names)} polynomial features")

        # Interaction Features
        if fe_config.get("interactions", {}).get("enabled", False):
            status.write("🔧 Creating Interaction Features...")
            num_cols_fe = X_train.select_dtypes(include=["number"]).columns.tolist()
            if len(num_cols_fe) >= 2:
                max_interactions = fe_config["interactions"].get("max_features", 3)
                interaction_count = 0
                for i in range(min(len(num_cols_fe), 5)):
                    for j in range(i + 1, min(len(num_cols_fe), 5)):
                        if interaction_count >= max_interactions:
                            break
                        col1, col2 = num_cols_fe[i], num_cols_fe[j]
                        interaction_name = f"{col1}_x_{col2}"
                        X_train[interaction_name] = X_train[col1] * X_train[col2]
                        X_test[interaction_name] = X_test[col1] * X_test[col2]
                        interaction_count += 1

                status.write(f"✓ Created {interaction_count} interaction features")

        # Statistical Aggregations
        if fe_config.get("aggregations", {}).get("enabled", False):
            status.write("🔧 Creating Statistical Aggregations...")
            num_cols_fe = X_train.select_dtypes(include=["number"]).columns.tolist()
            if num_cols_fe:
                # Row-wise statistics
                X_train["row_mean"] = X_train[num_cols_fe].mean(axis=1)
                X_train["row_std"] = X_train[num_cols_fe].std(axis=1)
                X_train["row_max"] = X_train[num_cols_fe].max(axis=1)
                X_train["row_min"] = X_train[num_cols_fe].min(axis=1)

                X_test["row_mean"] = X_test[num_cols_fe].mean(axis=1)
                X_test["row_std"] = X_test[num_cols_fe].std(axis=1)
                X_test["row_max"] = X_test[num_cols_fe].max(axis=1)
                X_test["row_min"] = X_test[num_cols_fe].min(axis=1)

                status.write("✓ Created 4 statistical aggregation features")

        if X_train.shape[1] == 0:
            raise ValueError("Auto feature selection ke baad koi feature nahi bacha. Data quality improve karke dobara try karein.")

        train_df = X.copy()
        train_df[target_col] = y_raw

        # Get imputation strategies from session or use defaults
        impute_strategies = st.session_state.get("impute_strategy", {"numeric": "median", "categorical": "most_frequent"})
        preprocessor, num_cols, cat_cols = build_preprocessor(
            train_df,
            target_col,
            num_impute_strategy=impute_strategies.get("numeric", "median"),
            cat_impute_strategy=impute_strategies.get("categorical", "most_frequent")
        )
        save_schema(X_train, num_cols, cat_cols, task, label_encoder, y=y_orig)

        status.write("🧠 Training Multiple Models...")
        candidate_models = get_candidate_models(task, training_mode=training_mode_value)
        best_name, pipeline, score, leaderboard = select_best_model(
            task=task,
            preprocessor=preprocessor,
            models=candidate_models,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            use_smote=use_smote,
            smote_k_neighbors=smote_k_neighbors,
            classification_metric=classification_metric,
            time_budget_sec=time_budget_sec,
        )
        if pipeline is None or best_name is None or score is None:
            raise ValueError("Koi bhi model successfully train nahi hua. Data ko clean karke dobara try karein.")

        if enable_tuning:
            status.write("🎛️ Running lightweight hyperparameter tuning on top models...")
            tune_top_n = 1 if training_mode_value == "fast" else 2
            tuning_budget = max(10, time_budget_sec // 3)
            tuned_name, tuned_pipeline, tuned_score, tuned_rows = tune_top_models(
                task=task,
                preprocessor=preprocessor,
                models=candidate_models,
                leaderboard=leaderboard,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                use_smote=use_smote,
                smote_k_neighbors=smote_k_neighbors,
                classification_metric=classification_metric,
                top_n=tune_top_n,
                time_budget_sec=tuning_budget,
            )
            current_best = float(leaderboard[0]["score"]) if leaderboard else float("-inf")
            if tuned_pipeline is not None and tuned_score is not None and float(tuned_score) > current_best:
                pipeline = tuned_pipeline
                best_name = tuned_name or best_name
                score = float(tuned_score)
                leaderboard = tuned_rows + leaderboard

        best_row = next((row for row in leaderboard if row.get("model") == best_name), None)
        displayed_score = float(best_row["score"]) if best_row and best_row.get("score") is not None else float(score)
        cv_score = best_row.get("cv_score") if best_row else None

        metric = "Accuracy" if task == "classification" else "R2 Score"

        status.write("💾 Saving Artifacts...")
        joblib.dump(pipeline, os.path.join(ARTIFACTS_DIR, "final_pipeline.joblib"))
        if label_encoder:
            joblib.dump(label_encoder, os.path.join(ARTIFACTS_DIR, "label_encoder.joblib"))

        status.update(label="Training Complete!", state="complete", expanded=False)

        st.markdown(
            f"""<div class="success-card"><h3>✅ Success!</h3><p><b>{metric}:</b> {displayed_score:.4f}</p></div>""",
            unsafe_allow_html=True,
        )
        st.caption(f"Best Model Selected: {best_name}")
        if cv_score is not None and pd.notna(cv_score):
            st.caption(f"Cross-Validation Score (mean): {float(cv_score):.4f}")
        if task == "classification":
            st.caption(f"Optimization Metric Used: {classification_metric}")
        st.caption(f"Training Mode: {training_mode} | Time Budget: {time_budget_sec}s")

        leaderboard_df = pd.DataFrame(leaderboard)
        if not leaderboard_df.empty:
            for col in ["score", "cv_score", "ranking_score", "accuracy", "f1", "roc_auc"]:
                if col in leaderboard_df.columns:
                    leaderboard_df[col] = leaderboard_df[col].round(4)
            st.markdown("##### 🏁 Model Leaderboard")
            st.dataframe(leaderboard_df, use_container_width=True)

            # Cross-Validation Visualization
            if len(leaderboard) > 0 and "cv_score" in leaderboard_df.columns:
                st.markdown("##### 📈 Cross-Validation Performance")

                cv_col1, cv_col2 = st.columns(2)

                with cv_col1:
                    # Bar chart of CV scores
                    fig, ax = plt.subplots(figsize=(8, 4))
                    models = leaderboard_df["model"].astype(str).head(8).tolist()
                    cv_scores = leaderboard_df["cv_score"].head(8).tolist()

                    colors = ["#16b3a0" if i == 0 else "#4a6a88" for i in range(len(models))]
                    ax.barh(models, cv_scores, color=colors)
                    ax.set_xlabel("CV Score")
                    ax.set_title("Cross-Validation Scores by Model")
                    ax.set_xlim(0, 1.05)
                    st.pyplot(fig)

                with cv_col2:
                    # Score distribution
                    if len(leaderboard) > 1:
                        fig2, ax2 = plt.subplots(figsize=(8, 4))
                        score_data = leaderboard_df["cv_score"].dropna()
                        if len(score_data) > 0:
                            ax2.hist(score_data, bins=min(10, len(score_data)), color="#16b3a0", edgecolor="white", alpha=0.8)
                            ax2.axvline(float(score_data.mean()), color="red", linestyle="--", label=f"Mean: {score_data.mean():.4f}")
                            ax2.set_xlabel("CV Score")
                            ax2.set_ylabel("Count")
                            ax2.set_title("CV Score Distribution")
                            ax2.legend()
                            st.pyplot(fig2)
                        else:
                            st.caption("No CV scores available for distribution")
                    else:
                        st.caption("Need multiple models for distribution")

        st.markdown("---")

        st.subheader("📊 Performance Analysis")
        col_m1, col_m2 = st.columns(2)

        with col_m1:
            st.markdown("**Feature Importance**")
            with st.spinner("Calculating..."):
                result = permutation_importance(pipeline, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1)
                sorted_idx = result.importances_mean.argsort()[-10:]
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.barh(X_test.columns[sorted_idx], result.importances_mean[sorted_idx])
                ax.set_xlabel("Impact")
                st.pyplot(fig)

        with col_m2:
            st.markdown("**Prediction Accuracy**")
            if task == "classification":
                cm = confusion_matrix(y_test, pipeline.predict(X_test))
                disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                fig, ax = plt.subplots(figsize=(6, 4))
                disp.plot(ax=ax, cmap="Blues")
                st.pyplot(fig)
            else:
                preds = pipeline.predict(X_test)
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.scatter(y_test, preds, alpha=0.5)
                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
                ax.set_ylabel("Predicted")
                st.pyplot(fig)

        # SHAP Explainable AI
        st.markdown("---")
        with st.expander("🔍 SHAP Explainable AI", expanded=False):
            st.caption("Understand how your model makes predictions using SHAP values")

            try:
                import shap

                enable_shap = st.checkbox("Enable SHAP Analysis", value=False, key="enable_shap")

                if enable_shap:
                    with st.spinner("Calculating SHAP values..."):
                        # Get the underlying model from pipeline
                        try:
                            # Sample data for faster SHAP calculation
                            sample_size = min(100, len(X_test))
                            X_sample = X_test.iloc[:sample_size]

                            # Create explainer based on task type
                            if task == "classification":
                                explainer = shap.TreeExplainer(pipeline.named_steps.get("model", pipeline))
                            else:
                                explainer = shap.TreeExplainer(pipeline.named_steps.get("model", pipeline))

                            # Calculate SHAP values
                            shap_values = explainer.shap_values(X_sample)

                            # Display SHAP summary plot
                            fig, ax = plt.subplots(figsize=(10, 6))
                            if isinstance(shap_values, list):
                                shap.summary_plot(shap_values[1], X_sample, plot_type="bar", show=False)
                            else:
                                shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()

                            st.success("✓ SHAP analysis complete")
                            st.caption("SHAP values show the impact of each feature on model predictions")

                        except Exception as shap_err:
                            st.warning(f"SHAP analysis limited: {shap_err!s}")
                            st.caption("Try with a simpler model or smaller dataset")

            except ImportError:
                st.info("SHAP library not installed. Run: pip install shap")

        st.markdown("---")
        c_down1, c_down2 = st.columns(2)

        with c_down1:
            st.markdown("##### 💾 Save Model")
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zf:
                zf.write(os.path.join(ARTIFACTS_DIR, "final_pipeline.joblib"), "final_pipeline.joblib")
                zf.write(os.path.join(ARTIFACTS_DIR, "app_schema.json"), "app_schema.json")
                if os.path.exists(os.path.join(ARTIFACTS_DIR, "label_encoder.joblib")):
                    zf.write(os.path.join(ARTIFACTS_DIR, "label_encoder.joblib"), "label_encoder.joblib")

            st.download_button(
                "Download Trained Model (.zip)",
                zip_buffer.getvalue(),
                "my_ai_model.zip",
                "application/zip",
            )

        with c_down2:
            st.markdown("##### 📜 Export Code")
            code_script = generate_python_code(target_col, task)
            st.download_button("Download Python Script (.py)", code_script, "train_model.py", "text/x-python")

        # PDF Report Export
        st.markdown("##### 📄 Generate Report")

        # Handle None values for report
        cv_score_display = f"{float(cv_score):.4f}" if cv_score is not None else "N/A"
        score_display = f"{float(score):.4f}" if score is not None else "N/A"

        # Create HTML report content
        report_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AutoML Training Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #16b3a0; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #16b3a0; color: white; }}
                .metric {{ display: inline-block; margin: 10px 20px; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #16b3a0; }}
            </style>
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
            <div class="metric">
                <div class="metric-value">{score_display}</div>
                <div>Test Score</div>
            </div>
            <div class="metric">
                <div class="metric-value">{cv_score_display}</div>
                <div>CV Score</div>
            </div>

            <h2>Dataset Info</h2>
            <table>
                <tr><th>Property</th><th>Value</th></tr>
                <tr><td>Total Rows</td><td>{df.shape[0]}</td></tr>
                <tr><td>Total Columns</td><td>{df.shape[1]}</td></tr>
                <tr><td>Target Column</td><td>{target_col}</td></tr>
                <tr><td>Numeric Features</td><td>{len(num_cols) if 'num_cols' in dir() else 'N/A'}</td></tr>
                <tr><td>Categorical Features</td><td>{len(cat_cols) if 'cat_cols' in dir() else 'N/A'}</td></tr>
            </table>

            <h2>Preprocessing Steps</h2>
            <ul>
                <li>Numeric Imputation: {impute_strategies.get('numeric', 'median')}</li>
                <li>Categorical Imputation: {impute_strategies.get('categorical', 'most_frequent')}</li>
                <li>Scaling: StandardScaler</li>
                <li>Encoding: OneHotEncoder</li>
            </ul>

            <footer style="margin-top: 40px; color: #888;">
                <p>Generated by AutoML Studio Pro</p>
            </footer>
        </body>
        </html>
        """

        # Convert HTML to downloadable format (users can print to PDF)
        st.download_button(
            "📄 Download Report (.html)",
            report_html,
            "automl_report.html",
            "text/html",
            help="Download report and print to PDF"
        )

        # Model History - Save current training to history
        if "model_history" not in st.session_state:
            st.session_state["model_history"] = []

        history_entry = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "model": best_name,
            "score": float(score),
            "cv_score": float(cv_score) if cv_score is not None else None,
            "task": task,
            "dataset_rows": df.shape[0],
            "dataset_cols": df.shape[1],
            "training_mode": training_mode,
        }
        st.session_state["model_history"].insert(0, history_entry)

        # Keep only last 10 entries
        st.session_state["model_history"] = st.session_state["model_history"][:10]

        # Display Model History
        st.markdown("---")
        with st.expander("📜 Model History", expanded=False):
            if st.session_state["model_history"]:
                history_df = pd.DataFrame(st.session_state["model_history"])
                st.dataframe(history_df, use_container_width=True)

                # Model Comparison Dashboard
                if len(st.session_state["model_history"]) >= 2:
                    st.markdown("##### 📊 Model Comparison Dashboard")

                    comp_col1, comp_col2 = st.columns(2)

                    with comp_col1:
                        # Compare scores bar chart
                        fig, ax = plt.subplots(figsize=(8, 4))
                        models = [f"{h['model'][:12]}..." if len(h['model']) > 12 else h['model'] for h in st.session_state["model_history"][:6]]
                        scores = [h['score'] for h in st.session_state["model_history"][:6]]
                        colors = ["#16b3a0"] * len(models)
                        best_idx = scores.index(max(scores))
                        colors[best_idx] = "#ff6b6b"
                        ax.bar(models, scores, color=colors)
                        ax.set_ylabel("Score")
                        ax.set_title("Model Scores Comparison")
                        ax.set_ylim(0, 1.05)
                        plt.xticks(rotation=45, ha="right")
                        st.pyplot(fig)

                    with comp_col2:
                        # Score over time
                        fig2, ax2 = plt.subplots(figsize=(8, 4))
                        history_reversed = list(reversed(st.session_state["model_history"]))
                        timestamps = [h['timestamp'][-5:] for h in history_reversed[:6]]
                        scores_time = [h['score'] for h in history_reversed[:6]]
                        ax2.plot(timestamps, scores_time, marker="o", linewidth=2, color="#16b3a0")
                        ax2.set_ylabel("Score")
                        ax2.set_title("Score Trend Over Time")
                        ax2.set_ylim(0, 1.05)
                        plt.xticks(rotation=45, ha="right")
                        st.pyplot(fig2)

                # Clear history button
                if st.button("Clear History", key="clear_history"):
                    st.session_state["model_history"] = []
                    st.rerun()
            else:
                st.caption("No models trained yet")

        if task == "classification" and not use_smote:
            st.info("Note: Class distribution ko dekhte hue SMOTE apply nahi kiya gaya, direct model training use hua.")

    except Exception as e:
        status.update(label="Training Failed", state="error")
        st.error(f"Error: {e!s}")
