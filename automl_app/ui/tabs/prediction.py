import json
import os
import zipfile

import joblib
import pandas as pd
import streamlit as st

from automl_app.core.config import ARTIFACTS_DIR


def render_prediction_tab() -> None:
    st.markdown("#### 🔮 AI Production Engine")

    source = st.radio("Model Source:", ["Use Current Training", "Upload Pre-Trained (.zip)"], horizontal=True)

    if source == "Upload Pre-Trained (.zip)":
        uploaded_zip = st.file_uploader("Upload .zip model", type="zip")
        if uploaded_zip:
            with zipfile.ZipFile(uploaded_zip, "r") as z:
                z.extractall(ARTIFACTS_DIR)
            st.success("✅ Model Loaded!")

    if not os.path.exists(os.path.join(ARTIFACTS_DIR, "app_schema.json")):
        st.warning("⚠️ No model found. Train one or Upload one.")
        st.stop()

    pipeline = joblib.load(os.path.join(ARTIFACTS_DIR, "final_pipeline.joblib"))
    with open(os.path.join(ARTIFACTS_DIR, "app_schema.json"), encoding="utf-8") as f:
        schema = json.load(f)

    pred_mode = st.radio("Input Mode:", ["Single Entry", "Batch Upload (CSV)"], horizontal=True)

    if pred_mode == "Single Entry":
        render_single_prediction(schema, pipeline)
        return

    render_batch_prediction(schema, pipeline)


def render_single_prediction(schema: dict, pipeline) -> None:
    with st.form("single_pred"):
        cols = st.columns(3)
        input_data = {}
        for i, feat in enumerate(schema["features"]):
            with cols[i % 3]:
                if feat["type"] == "numeric":
                    input_data[feat["name"]] = st.number_input(feat["name"], value=float(feat["mean"]))
                else:
                    input_data[feat["name"]] = st.selectbox(feat["name"], options=feat["options"])
        submitted = st.form_submit_button("⚡ Predict")

    if not submitted:
        return

    df_input = pd.DataFrame([input_data])
    try:
        pred = pipeline.predict(df_input)[0]
        if "target_mapping" in schema:
            mapping = schema["target_mapping"]
            if str(pred) in mapping:
                pred = mapping[str(pred)]

        disp = f"{pred:,.2f}" if isinstance(pred, (int, float)) else pred

        st.markdown("---")
        st.markdown(
            f"""<div class="metric-card"><small>Predicted Outcome</small><h1>{disp}</h1></div>""",
            unsafe_allow_html=True,
        )

        csv = df_input.copy()
        csv["Prediction"] = pred
        st.download_button("⬇️ Download Result", csv.to_csv(index=False), "result.csv", "text/csv")
    except Exception as e:
        st.error(f"Error: {e}")


def render_batch_prediction(schema: dict, pipeline) -> None:
    st.info("Upload a CSV file containing the input columns (without the target).")
    batch_file = st.file_uploader("Upload CSV", type=["csv"], key="batch_up")
    if not batch_file:
        return

    batch_df = pd.read_csv(batch_file)
    if not st.button("🚀 Process Batch"):
        return

    try:
        preds = pipeline.predict(batch_df)
        final_preds = preds
        if "target_mapping" in schema:
            mapping = schema["target_mapping"]
            final_preds = [mapping.get(str(p), p) for p in preds]

        batch_df["Predicted_Result"] = final_preds
        st.success("Processing Complete!")
        st.dataframe(batch_df.head(), use_container_width=True)
        st.download_button(
            "⬇️ Download Full Report",
            batch_df.to_csv(index=False).encode("utf-8"),
            "batch_predictions.csv",
            "text/csv",
        )
    except Exception as e:
        st.error(f"Batch Error: {e}")
