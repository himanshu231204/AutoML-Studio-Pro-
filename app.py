import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import joblib
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import zipfile
import io

# SKLEARN & ML IMPORTS
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

warnings.filterwarnings("ignore")

# ================= CONFIGURATION =================
st.set_page_config(
    page_title="AutoML Studio Pro",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

ARTIFACTS_DIR = "artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# ================= CUSTOM CSS =================
st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .stButton>button { 
        width: 100%; border-radius: 8px; font-weight: 600; height: 3em; 
        background-color: #ff4b4b; color: white; border: none;
    }
    .stButton>button:hover { background-color: #ff3333; color: white; }
    
    .metric-card {
        background-color: #ffffff !important; 
        color: #000000 !important; 
        padding: 20px;
        border-radius: 10px;
        border-left: 8px solid #ff4b4b;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .metric-card h1 { color: #000000 !important; margin: 0; font-size: 2.2rem; }
    .metric-card small { color: #555555 !important; font-weight: bold; font-size: 1rem; }
    
    .success-card {
        background-color: #d1e7dd; color: #0f5132; padding: 15px; border-radius: 8px; border: 1px solid #badbcc;
    }
</style>
""", unsafe_allow_html=True)

# ================= HELPER FUNCTIONS =================

def quick_dtype_buckets(df, target_col):
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    if target_col in num_cols: num_cols.remove(target_col)
    if target_col in cat_cols: cat_cols.remove(target_col)
    return num_cols, cat_cols

def is_classification(y):
    return (pd.Series(y).dtype.kind in ("O","b")) or (pd.Series(y).nunique() <= 20)

def build_preprocessor(df, target_col):
    num_cols, cat_cols = quick_dtype_buckets(df, target_col)
    steps_num = [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    steps_cat = [("imputer", SimpleImputer(strategy="most_frequent")), 
                 ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]

    preprocessor = ColumnTransformer(
        transformers=[("num", Pipeline(steps_num), num_cols),
                      ("cat", Pipeline(steps_cat), cat_cols)],
        remainder="drop"
    )
    return preprocessor, num_cols, cat_cols

def save_schema(X, num_cols, cat_cols, task, label_encoder=None, y=None):
    schema = {"task": task, "features": []}
    for col in num_cols:
        schema["features"].append({"name": col, "type": "numeric", "mean": float(X[col].mean())})
    for col in cat_cols:
        unique_vals = X[col].dropna().unique().tolist()
        if len(unique_vals) > 50: unique_vals = unique_vals[:50]
        schema["features"].append({"name": col, "type": "categorical", "options": unique_vals})
    
    if label_encoder:
        schema["target_mapping"] = {i: str(label) for i, label in enumerate(label_encoder.classes_)}
    elif task == "classification" and y is not None:
        unique_targets = sorted(y.unique())
        schema["target_mapping"] = {int(val): str(val) for val in unique_targets}
    
    with open(os.path.join(ARTIFACTS_DIR, "app_schema.json"), "w") as f:
        json.dump(schema, f)

def generate_python_code(target_col, task):
    code = f"""import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

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

# 3. Model
if '{task}' == 'classification':
    model = HistGradientBoostingClassifier(learning_rate=0.1, max_depth=5)
    pipeline = ImbPipeline([('pre', preprocessor), ('smote', SMOTE()), ('model', model)])
else:
    model = HistGradientBoostingRegressor(learning_rate=0.1, max_depth=5)
    pipeline = Pipeline([('pre', preprocessor), ('model', model)])

# 4. Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
if '{task}' == 'classification' and y.dtype == 'object':
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

pipeline.fit(X_train, y_train)
print(f"Model Score: {{pipeline.score(X_test, y_test)}}")
"""
    return code

# ================= MAIN APP =================
def main():
    st.title("🎓 AutoML Studio Pro")
    st.markdown("### The Productive Learning & ML Platform")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🚀 Train & Learn", "📊 Data Analysis", "🔮 Production Engine", "📘 User Manual"])

    # --- TAB 1: TRAIN & LEARN ---
    with tab1:
        st.markdown("#### 1. Build Your Model")
        st.info("Upload data, train the AI, and inspect how it works.")

        uploaded_file = st.file_uploader("Drop your CSV file here", type=["csv"], key="train_up")
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.session_state['df_train'] = df 
            
            c1, c2 = st.columns([3, 1])
            with c1:
                st.write(f"**Data Preview:** {df.shape[0]} rows, {df.shape[1]} columns")
                st.dataframe(df.head(3), use_container_width=True)
            with c2:
                all_cols = df.columns.tolist()
                target_col = st.selectbox("🎯 Target Column", all_cols, index=len(all_cols)-1)
                train_btn = st.button("🚀 Start Training")
            
            if train_btn:
                status = st.status("Running AutoML Pipeline...", expanded=True)
                try:
                    status.write("🧹 Cleaning & Preprocessing...")
                    df = df.dropna(subset=[target_col])
                    task = "classification" if is_classification(df[target_col]) else "regression"
                    status.write(f"🧭 Detected Task: **{task.upper()}**")
                    
                    y = df[target_col]
                    X = df.drop(columns=[target_col])
                    
                    label_encoder = None
                    y_orig = y 
                    if task == "classification" and y.dtype.kind in ("O", "b"):
                        label_encoder = LabelEncoder()
                        y = label_encoder.fit_transform(y.astype(str))
                        y_orig = None 

                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    
                    preprocessor, num_cols, cat_cols = build_preprocessor(df, target_col)
                    save_schema(X_train, num_cols, cat_cols, task, label_encoder, y=y_orig)

                    status.write("🧠 Training Gradient Boosting Model...")
                    if task == "classification":
                        model = HistGradientBoostingClassifier(learning_rate=0.1, max_depth=5, random_state=42)
                        steps = [("pre", preprocessor), ("smote", SMOTE(k_neighbors=3)), ("model", model)]
                        pipeline = ImbPipeline(steps)
                        metric = "Accuracy"
                    else:
                        model = HistGradientBoostingRegressor(learning_rate=0.1, max_depth=5, random_state=42)
                        steps = [("pre", preprocessor), ("model", model)]
                        pipeline = Pipeline(steps)
                        metric = "R2 Score"

                    pipeline.fit(X_train, y_train)
                    
                    status.write("💾 Saving Artifacts...")
                    joblib.dump(pipeline, os.path.join(ARTIFACTS_DIR, "final_pipeline.joblib"))
                    if label_encoder:
                        joblib.dump(label_encoder, os.path.join(ARTIFACTS_DIR, "label_encoder.joblib"))

                    score = pipeline.score(X_test, y_test)
                    status.update(label="Training Complete!", state="complete", expanded=False)
                    
                    st.markdown(f"""<div class="success-card"><h3>✅ Success!</h3><p><b>{metric}:</b> {score:.4f}</p></div>""", unsafe_allow_html=True)
                    st.markdown("---")
                    
                    # RESULTS & EXPORTS
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
                            disp.plot(ax=ax, cmap='Blues')
                            st.pyplot(fig)
                        else:
                            preds = pipeline.predict(X_test)
                            fig, ax = plt.subplots(figsize=(6, 4))
                            ax.scatter(y_test, preds, alpha=0.5)
                            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                            ax.set_ylabel("Predicted")
                            st.pyplot(fig)

                    st.markdown("---")
                    c_down1, c_down2 = st.columns(2)
                    
                    with c_down1:
                        st.markdown("##### 💾 Save Model")
                        # Zip Model
                        zip_buffer = io.BytesIO()
                        with zipfile.ZipFile(zip_buffer, "w") as zf:
                            zf.write(os.path.join(ARTIFACTS_DIR, "final_pipeline.joblib"), "final_pipeline.joblib")
                            zf.write(os.path.join(ARTIFACTS_DIR, "app_schema.json"), "app_schema.json")
                            if os.path.exists(os.path.join(ARTIFACTS_DIR, "label_encoder.joblib")):
                                zf.write(os.path.join(ARTIFACTS_DIR, "label_encoder.joblib"), "label_encoder.joblib")
                        
                        st.download_button("Download Trained Model (.zip)", zip_buffer.getvalue(), "my_ai_model.zip", "application/zip")
                        
                    with c_down2:
                        st.markdown("##### 📜 Export Code")
                        code_script = generate_python_code(target_col, task)
                        st.download_button("Download Python Script (.py)", code_script, "train_model.py", "text/x-python")

                except Exception as e:
                    status.update(label="Training Failed", state="error")
                    st.error(f"Error: {str(e)}")

    # --- TAB 2: DATA ANALYSIS ---
    with tab2:
        st.markdown("#### 📊 Exploratory Data Analysis")
        if 'df_train' not in st.session_state:
            st.warning("Please upload a CSV in the 'Train & Learn' tab first.")
        else:
            df = st.session_state['df_train']
            st.subheader("1. Data Summary")
            st.dataframe(df.describe(), use_container_width=True)
            
            st.subheader("2. Correlation Matrix")
            numeric_df = df.select_dtypes(include=[np.number])
            if numeric_df.shape[1] > 1:
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
                st.pyplot(fig)
            else:
                st.info("Not enough numeric columns for correlation.")

    # --- TAB 3: PRODUCTION ENGINE ---
    with tab3:
        st.markdown("#### 🔮 AI Production Engine")
        
        # --- MODEL LOADING LOGIC ---
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
        with open(os.path.join(ARTIFACTS_DIR, "app_schema.json"), "r") as f:
            schema = json.load(f)

        # --- PREDICTION LOGIC ---
        pred_mode = st.radio("Input Mode:", ["Single Entry", "Batch Upload (CSV)"], horizontal=True)

        if pred_mode == "Single Entry":
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

            if submitted:
                df_input = pd.DataFrame([input_data])
                try:
                    pred = pipeline.predict(df_input)[0]
                    if "target_mapping" in schema:
                        mapping = schema["target_mapping"]
                        if str(pred) in mapping: pred = mapping[str(pred)]
                    
                    disp = f"{pred:,.2f}" if isinstance(pred, (int, float)) else pred
                    
                    st.markdown("---")
                    st.markdown(f"""<div class="metric-card"><small>Predicted Outcome</small><h1>{disp}</h1></div>""", unsafe_allow_html=True)
                    
                    csv = df_input.copy()
                    csv["Prediction"] = pred
                    st.download_button("⬇️ Download Result", csv.to_csv(index=False), "result.csv", "text/csv")
                except Exception as e:
                    st.error(f"Error: {e}")

        else:
            st.info("Upload a CSV file containing the input columns (without the target).")
            batch_file = st.file_uploader("Upload CSV", type=["csv"], key="batch_up")
            if batch_file:
                batch_df = pd.read_csv(batch_file)
                if st.button("🚀 Process Batch"):
                    try:
                        preds = pipeline.predict(batch_df)
                        final_preds = preds
                        if "target_mapping" in schema:
                            mapping = schema["target_mapping"]
                            final_preds = [mapping.get(str(p), p) for p in preds]
                        
                        batch_df["Predicted_Result"] = final_preds
                        st.success("Processing Complete!")
                        st.dataframe(batch_df.head(), use_container_width=True)
                        st.download_button("⬇️ Download Full Report", batch_df.to_csv(index=False).encode('utf-8'), "batch_predictions.csv", "text/csv")
                    except Exception as e:
                        st.error(f"Batch Error: {e}")

    # --- TAB 4: MANUAL ---
    with tab4:
        st.markdown("""
        ### 📘 User Manual
        
        **1. Train & Learn**
        * Upload your clean CSV data.
        * Train the model to see Accuracy and Feature Importance charts.
        * **Save Model:** Download the .zip file to use later.
        * **Export Code:** Download the Python script to learn how it works.
        
        **2. Data Analysis**
        * Check correlations and data distribution before training.
        
        **3. Production Engine**
        * **Use Current Training:** Use the model you just trained.
        * **Upload Pre-Trained:** Upload a .zip file you downloaded previously.
        * **Batch Mode:** Upload a CSV to predict hundreds of items at once.
        """)

if __name__ == "__main__":
    main()
    # ---------------- FOOTER ----------------
st.divider()

st.markdown(
    """
    <div style="text-align:center; font-size:14px;">
        👨‍💻 Developed by <b>Himanshu Kumar</b><br><br>
        🔗 
        <a href="https://www.linkedin.com/in/himanshu231204" target="_blank">LinkedIn</a> |
        <a href="https://github.com/himanshu231204" target="_blank">GitHub</a>
    </div>
    """,
    unsafe_allow_html=True
)
