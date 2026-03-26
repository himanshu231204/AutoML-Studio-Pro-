import json
import os

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
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from automl_app.core.config import ARTIFACTS_DIR


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


def build_preprocessor(df: pd.DataFrame, target_col: str) -> tuple[ColumnTransformer, list[str], list[str]]:
    num_cols, cat_cols = quick_dtype_buckets(df, target_col)
    steps_num = [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]

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
        ("imputer", SimpleImputer(strategy="most_frequent")),
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


def get_candidate_models(task: str):
    if task == "classification":
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
):
    leaderboard: list[dict[str, object]] = []
    best_name = None
    best_pipeline = None
    best_score = None

    min_samples = int(len(X_train))
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
        pipeline = build_model_pipeline(
            task,
            preprocessor,
            model,
            use_smote=use_smote,
            smote_k_neighbors=smote_k_neighbors,
        )
        try:
            pipeline.fit(X_train, y_train)
            score = float(pipeline.score(X_test, y_test))
        except Exception:
            if task != "classification" or not use_smote:
                continue
            try:
                pipeline = build_model_pipeline(task, preprocessor, model, use_smote=False)
                pipeline.fit(X_train, y_train)
                score = float(pipeline.score(X_test, y_test))
            except Exception:
                continue

        cv_score = np.nan
        if cv_splitter is not None:
            try:
                cv_pipeline = build_model_pipeline(
                    task,
                    preprocessor,
                    clone(model),
                    use_smote=use_smote,
                    smote_k_neighbors=smote_k_neighbors,
                )
                cv_values = cross_val_score(cv_pipeline, X_train, y_train, cv=cv_splitter, n_jobs=-1)
                cv_score = float(np.mean(cv_values))
            except Exception:
                cv_score = np.nan

        ranking_score = score
        if not np.isnan(cv_score):
            ranking_score = float((0.7 * cv_score) + (0.3 * score))

        leaderboard.append({
            "model": model_name,
            "score": score,
            "cv_score": cv_score,
            "ranking_score": ranking_score,
        })
        if best_score is None or ranking_score > best_score:
            best_name = model_name
            best_score = ranking_score
            best_pipeline = pipeline

    leaderboard.sort(key=lambda item: item["ranking_score"], reverse=True)
    return best_name, best_pipeline, best_score, leaderboard
