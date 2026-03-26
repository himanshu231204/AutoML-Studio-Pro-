import streamlit as st


def render_manual_tab() -> None:
    st.markdown(
        """
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

        **4. Model Selection (AutoML Leaderboard)**
        * App multiple algorithms ko train karke best score wala model automatically choose karti hai.
        * **Classification** me score = Accuracy, **Regression** me score = R2 Score.
        * Leaderboard me top model first row me hota hai; wahi final model save hota hai.
        * Agar dataset imbalanced ho aur SMOTE fail ho, app safe fallback use karke training continue karti hai.

        **5. Quick Model Hints**
        * **HistGradientBoosting:** Mixed tabular data ke liye strong default.
        * **RandomForest / ExtraTrees:** Non-linear patterns aur noisy data me kaafi robust.
        * **LogisticRegression / Ridge:** Fast baseline aur simpler patterns ke liye useful.
        * **KNN:** Small-medium data me useful, but large datasets par slow ho sakta hai.
        """
    )
