import io
import os
import zipfile

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from automl_app.core.config import ARTIFACTS_DIR
from automl_app.core.helpers import (
    build_preprocessor,
    generate_python_code,
    get_candidate_models,
    is_classification,
    save_schema,
    select_best_model,
)


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
    score = int(round((0.55 * completeness + 0.45 * feature_quality) * 100))

    profile["task"] = task
    profile["rows"] = total_rows
    profile["removed_target_na"] = removed_target_na
    profile["usable_features"] = usable_features
    profile["dropped_cols"] = dropped_cols
    profile["health_score"] = score
    profile["balance_note"] = balance_note
    profile["imbalance_ratio"] = imbalance_ratio
    return profile


def render_train_tab() -> None:
    st.markdown("#### 1. Build Your Model")
    st.info("Upload data, train the AI, and inspect how it works.")

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
        target_col = st.selectbox("🎯 Target Column", all_cols, index=len(all_cols) - 1)

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

        train_df = X.copy()
        train_df[target_col] = y_raw
        preprocessor, num_cols, cat_cols = build_preprocessor(train_df, target_col)
        save_schema(X_train, num_cols, cat_cols, task, label_encoder, y=y_orig)

        status.write("🧠 Training Multiple Models...")
        candidate_models = get_candidate_models(task)
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
        )
        if pipeline is None or best_name is None or score is None:
            raise ValueError("Koi bhi model successfully train nahi hua. Data ko clean karke dobara try karein.")

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

        leaderboard_df = pd.DataFrame(leaderboard)
        if not leaderboard_df.empty:
            for col in ["score", "cv_score", "ranking_score"]:
                if col in leaderboard_df.columns:
                    leaderboard_df[col] = leaderboard_df[col].round(4)
            st.markdown("##### 🏁 Model Leaderboard")
            st.dataframe(leaderboard_df, use_container_width=True)

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

        if task == "classification" and not use_smote:
            st.info("Note: Class distribution ko dekhte hue SMOTE apply nahi kiya gaya, direct model training use hua.")

    except Exception as e:
        status.update(label="Training Failed", state="error")
        st.error(f"Error: {str(e)}")
