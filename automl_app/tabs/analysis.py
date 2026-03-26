import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import streamlit as st


def render_analysis_tab() -> None:
    st.markdown("#### 📊 Exploratory Data Analysis")
    if "df_train" not in st.session_state:
        st.warning("Please upload a CSV in the 'Train & Learn' tab first.")
        return

    df = st.session_state["df_train"]
    st.subheader("1. Data Summary")
    st.dataframe(df.describe(), use_container_width=True)

    st.subheader("2. Correlation Matrix")
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] > 1:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)
        return

    st.info("Not enough numeric columns for correlation.")
