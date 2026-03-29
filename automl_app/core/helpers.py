import json
import logging
import os
import time

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from automl_app.core.config import ARTIFACTS_DIR

logger = logging.getLogger(__name__)


def quick_dtype_buckets(df: pd.DataFrame, target_col: str) -> tuple[list[str], list[str]]:
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    if target_col in num_cols:
        num_cols.remove(target_col)
    if target_col in cat_cols:
        cat_cols.remove(target_col)
    return num_cols, cat_cols


def is_classification(y: pd.Series) -> bool:
    return (pd.Series(y).dtype.kind in ("O", "b")) or (pd.Series(y).nunique() <= 20)


def build_preprocessor(
    df: pd.DataFrame,
    target_col: str,
    num_impute_strategy: str = "median",
    cat_impute_strategy: str = "most_frequent"
) -> tuple[ColumnTransformer, list[str], list[str]]:
    num_cols, cat_cols = quick_dtype_buckets(df, target_col)

    # Use user-specified imputation strategies with fallback to defaults
    num_strategy = num_impute_strategy if num_impute_strategy in ["mean", "median", "most_frequent", "constant"] else "median"
    cat_strategy = cat_impute_strategy if cat_impute_strategy in ["most_frequent", "constant"] else "most_frequent"

    steps_num = [("imputer", SimpleImputer(strategy=num_strategy)), ("scaler", StandardScaler())]

    # Reduce overfitting from very rare categories when supported.
    try:
        encoder = OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=0.01,
            sparse_output=False,
        )
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    steps_cat = [
        ("imputer", SimpleImputer(strategy=cat_strategy)),
        ("encoder", encoder),
    ]

    preprocessor = ColumnTransformer(
        transformers=[("num", Pipeline(steps_num), num_cols), ("cat", Pipeline(steps_cat), cat_cols)],
        remainder="drop",
    )
    return preprocessor, num_cols, cat_cols


def save_schema(
    X: pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
    task: str,
    label_encoder: LabelEncoder | None = None,
    y: pd.Series | None = None,
) -> None:
    schema = {"task": task, "features": []}
    for col in num_cols:
        mean_val = X[col].mean()
        if pd.isna(mean_val):
            mean_val = 0.0
        schema["features"].append({"name": col, "type": "numeric", "mean": float(mean_val)})
    for col in cat_cols:
        unique_vals = X[col].dropna().unique().tolist()
        if not unique_vals:
            unique_vals = ["Unknown"]
        if len(unique_vals) > 50:
            unique_vals = unique_vals[:50]
        schema["features"].append({"name": col, "type": "categorical", "options": unique_vals})

    if label_encoder:
        schema["target_mapping"] = {i: str(label) for i, label in enumerate(label_encoder.classes_)}
    elif task == "classification" and y is not None:
        unique_targets = sorted(y.unique())
        schema["target_mapping"] = {str(val): str(val) for val in unique_targets}

    with open(os.path.join(ARTIFACTS_DIR, "app_schema.json"), "w", encoding="utf-8") as f:
        json.dump(schema, f)


def generate_python_code(target_col: str, task: str) -> str:
    return f"""import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge

# 1. Load Data
df = pd.read_csv('your_dataset.csv')
target = '{target_col}'

# 2. Preprocessing
df = df.dropna(subset=[target])
X = df.drop(columns=[target])
y = df[target]

num_cols = X.select_dtypes(include=['number']).columns.tolist()
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
cat_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                     ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
preprocessor = ColumnTransformer([('num', num_pipe, num_cols), ('cat', cat_pipe, cat_cols)])

# 3. Multiple Models (pick one)
if '{task}' == 'classification':
    model = HistGradientBoostingClassifier(learning_rate=0.1, max_depth=5)  # or RandomForestClassifier(), LogisticRegression(max_iter=1200)
    pipeline = ImbPipeline([('pre', preprocessor), ('smote', SMOTE()), ('model', model)])
else:
    model = HistGradientBoostingRegressor(learning_rate=0.1, max_depth=5)  # or RandomForestRegressor(), Ridge()
    pipeline = Pipeline([('pre', preprocessor), ('model', model)])

# 4. Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
if '{task}' == 'classification' and y.dtype == 'object':
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

pipeline.fit(X_train, y_train)
print(f\"Model Score: {{pipeline.score(X_test, y_test)}}\")
"""


def get_candidate_models(task: str, training_mode: str = "high_accuracy"):
    mode = (training_mode or "high_accuracy").lower()
    if task == "classification":
        if mode == "fast":
            return {
                "HistGradientBoosting": HistGradientBoostingClassifier(
                    learning_rate=0.06,
                    max_leaf_nodes=31,
                    min_samples_leaf=20,
                    random_state=42,
                ),
                "RandomForest": RandomForestClassifier(
                    n_estimators=250,
                    max_features="sqrt",
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=-1,
                ),
                "LogisticRegression": LogisticRegression(max_iter=1500, class_weight="balanced"),
                "KNN": KNeighborsClassifier(n_neighbors=5, weights="distance"),
            }
        return {
            "HistGradientBoosting": HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_depth=None,
                max_leaf_nodes=31,
                min_samples_leaf=20,
                l2_regularization=0.05,
                random_state=42,
            ),
            "GradientBoosting": GradientBoostingClassifier(random_state=42),
            "RandomForest": RandomForestClassifier(
                n_estimators=500,
                max_features="sqrt",
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            ),
            "ExtraTrees": ExtraTreesClassifier(
                n_estimators=600,
                max_features="sqrt",
                min_samples_leaf=1,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            ),
            "AdaBoost": AdaBoostClassifier(random_state=42),
            "DecisionTree": DecisionTreeClassifier(class_weight="balanced", random_state=42),
            "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced"),
            "KNN": KNeighborsClassifier(n_neighbors=5, weights="distance"),
        }

    if mode == "fast":
        return {
            "HistGradientBoosting": HistGradientBoostingRegressor(
                learning_rate=0.06,
                max_leaf_nodes=31,
                min_samples_leaf=20,
                random_state=42,
            ),
            "RandomForest": RandomForestRegressor(
                n_estimators=250,
                max_features="sqrt",
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            ),
            "Ridge": Ridge(alpha=1.0),
            "KNN": KNeighborsRegressor(n_neighbors=5, weights="distance"),
        }

    return {
        "HistGradientBoosting": HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_depth=None,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=0.05,
            random_state=42,
        ),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
        "RandomForest": RandomForestRegressor(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=600,
            max_features="sqrt",
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1,
        ),
        "AdaBoost": AdaBoostRegressor(random_state=42),
        "DecisionTree": DecisionTreeRegressor(random_state=42),
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.0005),
        "ElasticNet": ElasticNet(alpha=0.005, l1_ratio=0.2, random_state=42),
        "KNN": KNeighborsRegressor(n_neighbors=5, weights="distance"),
    }


def build_model_pipeline(
    task: str,
    preprocessor: ColumnTransformer,
    model,
    use_smote: bool = True,
    smote_k_neighbors: int = 3,
):
    if task == "classification":
        steps = [("pre", preprocessor)]
        if use_smote:
            steps.append(("smote", SMOTE(k_neighbors=smote_k_neighbors)))
        steps.append(("model", model))
        return ImbPipeline(steps)

    steps = [("pre", preprocessor), ("model", model)]
    return Pipeline(steps)


def _get_classification_scoring_name(classification_metric: str, y_train) -> str:
    metric = (classification_metric or "accuracy").lower()
    if metric == "f1":
        return "f1_weighted"
    if metric == "roc_auc":
        classes = pd.Series(y_train).nunique(dropna=False)
        return "roc_auc" if int(classes) == 2 else "roc_auc_ovr_weighted"
    return "accuracy"


def _evaluate_pipeline_score(
    task: str,
    pipeline,
    X_test: pd.DataFrame,
    y_test,
    classification_metric: str = "accuracy",
) -> float:
    if task != "classification":
        return float(pipeline.score(X_test, y_test))

    metric = (classification_metric or "accuracy").lower()
    y_pred = pipeline.predict(X_test)

    if metric == "f1":
        return float(f1_score(y_test, y_pred, average="weighted"))
    if metric == "roc_auc":
        try:
            y_prob = pipeline.predict_proba(X_test)
            classes = pd.Series(y_test).nunique(dropna=False)
            if int(classes) == 2:
                return float(roc_auc_score(y_test, y_prob[:, 1]))
            return float(roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted"))
        except Exception:
            return float("nan")
    return float(accuracy_score(y_test, y_pred))


def _classification_metric_breakdown(pipeline, X_test: pd.DataFrame, y_test) -> dict[str, float]:
    y_pred = pipeline.predict(X_test)
    out = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred, average="weighted")),
        "roc_auc": float("nan"),
    }
    try:
        y_prob = pipeline.predict_proba(X_test)
        classes = pd.Series(y_test).nunique(dropna=False)
        if int(classes) == 2:
            out["roc_auc"] = float(roc_auc_score(y_test, y_prob[:, 1]))
        else:
            out["roc_auc"] = float(roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted"))
    except Exception as e:
        logger.debug(f"ROC AUC calculation failed: {e}")
    return out


def _build_tuned_variants(task: str, model_name: str):
    if task == "classification":
        if model_name == "RandomForest":
            return [
                RandomForestClassifier(n_estimators=300, max_features="sqrt", min_samples_leaf=1, random_state=42, n_jobs=-1),
                RandomForestClassifier(n_estimators=700, max_features="sqrt", min_samples_leaf=2, random_state=42, n_jobs=-1),
            ]
        if model_name == "ExtraTrees":
            return [
                ExtraTreesClassifier(n_estimators=400, max_features="sqrt", min_samples_leaf=1, random_state=42, n_jobs=-1),
                ExtraTreesClassifier(n_estimators=800, max_features="sqrt", min_samples_leaf=2, random_state=42, n_jobs=-1),
            ]
        if model_name == "HistGradientBoosting":
            return [
                HistGradientBoostingClassifier(learning_rate=0.03, max_leaf_nodes=31, min_samples_leaf=20, random_state=42),
                HistGradientBoostingClassifier(learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=10, random_state=42),
            ]
        if model_name == "LogisticRegression":
            return [
                LogisticRegression(C=0.7, max_iter=2500, class_weight="balanced"),
                LogisticRegression(C=1.5, max_iter=2500, class_weight="balanced"),
            ]
        if model_name == "KNN":
            return [
                KNeighborsClassifier(n_neighbors=3, weights="distance"),
                KNeighborsClassifier(n_neighbors=9, weights="distance"),
            ]
        return []

    if model_name == "RandomForest":
        return [
            RandomForestRegressor(n_estimators=300, max_features="sqrt", min_samples_leaf=1, random_state=42, n_jobs=-1),
            RandomForestRegressor(n_estimators=700, max_features="sqrt", min_samples_leaf=2, random_state=42, n_jobs=-1),
        ]
    if model_name == "ExtraTrees":
        return [
            ExtraTreesRegressor(n_estimators=400, max_features="sqrt", min_samples_leaf=1, random_state=42, n_jobs=-1),
            ExtraTreesRegressor(n_estimators=800, max_features="sqrt", min_samples_leaf=2, random_state=42, n_jobs=-1),
        ]
    if model_name == "HistGradientBoosting":
        return [
            HistGradientBoostingRegressor(learning_rate=0.03, max_leaf_nodes=31, min_samples_leaf=20, random_state=42),
            HistGradientBoostingRegressor(learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=10, random_state=42),
        ]
    if model_name == "KNN":
        return [
            KNeighborsRegressor(n_neighbors=3, weights="distance"),
            KNeighborsRegressor(n_neighbors=9, weights="distance"),
        ]
    return []


def tune_top_models(
    task: str,
    preprocessor: ColumnTransformer,
    models: dict,
    leaderboard: list[dict[str, object]],
    X_train: pd.DataFrame,
    y_train,
    X_test: pd.DataFrame,
    y_test,
    use_smote: bool = False,
    smote_k_neighbors: int = 3,
    classification_metric: str = "accuracy",
    top_n: int = 2,
    time_budget_sec: int | None = None,
):
    start_time = time.perf_counter()

    tuned_rows: list[dict[str, object]] = []
    top_names = [str(row.get("model")) for row in leaderboard[:top_n] if row.get("model") in models]

    best_name = None
    best_pipeline = None
    best_score = None

    for name in top_names:
        if time_budget_sec is not None and (time.perf_counter() - start_time) >= float(time_budget_sec):
            break
        base_model = models[name]
        variants = [clone(base_model), *_build_tuned_variants(task, name)]
        for idx, variant in enumerate(variants):
            if time_budget_sec is not None and (time.perf_counter() - start_time) >= float(time_budget_sec):
                break
            display_name = f"{name}-tuned-{idx}"
            pipeline = build_model_pipeline(
                task,
                preprocessor,
                variant,
                use_smote=use_smote,
                smote_k_neighbors=smote_k_neighbors,
            )
            try:
                pipeline.fit(X_train, y_train)
                score = _evaluate_pipeline_score(
                    task,
                    pipeline,
                    X_test,
                    y_test,
                    classification_metric=classification_metric,
                )
            except Exception as e:
                logger.debug(f"Model {display_name} training failed: {e}")
                continue

            row = {"model": display_name, "score": float(score)}
            if task == "classification":
                row.update(_classification_metric_breakdown(pipeline, X_test, y_test))
            tuned_rows.append(row)
            if best_score is None or score > best_score:
                best_name = display_name
                best_score = float(score)
                best_pipeline = pipeline

    tuned_rows.sort(key=lambda item: item["score"], reverse=True)
    return best_name, best_pipeline, best_score, tuned_rows


def select_best_model(
    task: str,
    preprocessor: ColumnTransformer,
    models: dict,
    X_train: pd.DataFrame,
    y_train,
    X_test: pd.DataFrame,
    y_test,
    use_smote: bool = False,
    smote_k_neighbors: int = 3,
    classification_metric: str = "accuracy",
    time_budget_sec: int | None = None,
):
    start_time = time.perf_counter()

    leaderboard: list[dict[str, object]] = []
    best_name = None
    best_pipeline = None
    best_score = None

    min_samples = len(X_train)
    use_cv = min_samples >= 30
    cv_splitter = None

    if use_cv:
        if task == "classification":
            class_counts = pd.Series(y_train).value_counts(dropna=False)
            min_class = int(class_counts.min()) if not class_counts.empty else 0
            if min_class >= 2:
                cv_n_splits = max(2, min(5, min_class))
                cv_splitter = StratifiedKFold(n_splits=cv_n_splits, shuffle=True, random_state=42)
        else:
            cv_n_splits = max(2, min(5, min_samples // 5))
            cv_splitter = KFold(n_splits=cv_n_splits, shuffle=True, random_state=42)

    for model_name, model in models.items():
        if time_budget_sec is not None and (time.perf_counter() - start_time) >= float(time_budget_sec):
            break
        pipeline = build_model_pipeline(
            task,
            preprocessor,
            model,
            use_smote=use_smote,
            smote_k_neighbors=smote_k_neighbors,
        )
        try:
            pipeline.fit(X_train, y_train)
            score = _evaluate_pipeline_score(
                task,
                pipeline,
                X_test,
                y_test,
                classification_metric=classification_metric,
            )
        except Exception:
            if task != "classification" or not use_smote:
                continue
            try:
                pipeline = build_model_pipeline(task, preprocessor, model, use_smote=False)
                pipeline.fit(X_train, y_train)
                score = _evaluate_pipeline_score(
                    task,
                    pipeline,
                    X_test,
                    y_test,
                    classification_metric=classification_metric,
                )
            except Exception as e:
                logger.debug(f"Model evaluation failed: {e}")
                continue

        cv_score = np.nan
        if cv_splitter is not None:
            try:
                scoring_name = _get_classification_scoring_name(classification_metric, y_train) if task == "classification" else "r2"
                cv_pipeline = build_model_pipeline(
                    task,
                    preprocessor,
                    clone(model),
                    use_smote=use_smote,
                    smote_k_neighbors=smote_k_neighbors,
                )
                cv_values = cross_val_score(cv_pipeline, X_train, y_train, cv=cv_splitter, scoring=scoring_name, n_jobs=-1)
                cv_score = float(np.mean(cv_values))
            except Exception:
                cv_score = np.nan

        ranking_score = score
        if not np.isnan(cv_score):
            ranking_score = float((0.7 * cv_score) + (0.3 * score))

        row = {
            "model": model_name,
            "score": score,
            "cv_score": cv_score,
            "ranking_score": ranking_score,
        }
        if task == "classification":
            row.update(_classification_metric_breakdown(pipeline, X_test, y_test))

        leaderboard.append(row)
        if best_score is None or ranking_score > best_score:
            best_name = model_name
            best_score = ranking_score
            best_pipeline = pipeline

    leaderboard.sort(key=lambda item: item["ranking_score"], reverse=True)
    return best_name, best_pipeline, best_score, leaderboard
