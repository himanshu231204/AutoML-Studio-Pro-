import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st


def _safe_sample(df: pd.DataFrame, max_rows: int = 50000) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df
    return df.sample(max_rows, random_state=42)


def _column_profile(df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for col in df.columns:
        series = df[col]
        missing = int(series.isna().sum())
        missing_pct = float((missing / len(df)) * 100) if len(df) else 0.0
        nunique = int(series.nunique(dropna=True))
        sample_values = series.dropna().astype(str).head(3).tolist()
        records.append(
            {
                "column": col,
                "dtype": str(series.dtype),
                "missing": missing,
                "missing_%": round(missing_pct, 2),
                "unique": nunique,
                "sample_values": ", ".join(sample_values) if sample_values else "-",
            }
        )
    return pd.DataFrame(records)


def _outlier_summary(numeric_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for col in numeric_df.columns:
        series = numeric_df[col].dropna()
        if series.empty:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            outliers = 0
        else:
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = int(((series < lower) | (series > upper)).sum())
        outlier_pct = (outliers / len(series)) * 100 if len(series) else 0
        rows.append({"column": col, "outliers": outliers, "outlier_%": round(outlier_pct, 2)})
    return pd.DataFrame(rows).sort_values("outlier_%", ascending=False)


def render_analysis_tab() -> None:
    st.markdown("#### 📊 Exploratory Data Analysis")
    if "df_train" not in st.session_state:
        st.warning("Please upload a CSV in the 'Train & Learn' tab first.")
        return

    df = st.session_state["df_train"].copy()
    if df.empty:
        st.warning("Uploaded dataset is empty.")
        return

    sampled_df = _safe_sample(df)
    numeric_df = sampled_df.select_dtypes(include=[np.number])
    categorical_df = sampled_df.select_dtypes(include=["object", "category", "bool"])

    st.subheader("1. Dataset Health")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", f"{df.shape[1]:,}")
    c3.metric("Missing Cells", f"{int(df.isna().sum().sum()):,}")
    c4.metric("Duplicate Rows", f"{int(df.duplicated().sum()):,}")
    c5.metric("Memory", f"{(df.memory_usage(deep=True).sum() / (1024**2)):.2f} MB")
    if len(df) != len(sampled_df):
        st.caption(f"Large dataset detected. Visual analysis is computed on a random sample of {len(sampled_df):,} rows.")

    st.subheader("2. Column Profile")
    profile_df = _column_profile(df)
    st.dataframe(profile_df, use_container_width=True)

    st.subheader("3. Missing Values Diagnostics")
    missing_df = profile_df[profile_df["missing"] > 0].sort_values("missing_%", ascending=False)
    if missing_df.empty:
        st.success("No missing values found in this dataset.")
    else:
        top_missing = missing_df.head(20)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(top_missing["column"], top_missing["missing_%"], color="#1da58f")
        ax.set_xlabel("Missing %")
        ax.invert_yaxis()
        st.pyplot(fig)

    st.subheader("4. Numeric Analysis")
    if numeric_df.empty:
        st.info("No numeric columns detected.")
    else:
        st.dataframe(numeric_df.describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).T, use_container_width=True)

        outlier_df = _outlier_summary(numeric_df)
        if not outlier_df.empty:
            st.markdown("**Outlier Summary (IQR method)**")
            st.dataframe(outlier_df.head(15), use_container_width=True)

        selected_num_col = st.selectbox("Numeric column plot", numeric_df.columns.tolist(), key="eda_num_col")
        plot_type = st.radio("Plot type", ["Histogram", "Box Plot"], horizontal=True, key="eda_num_plot")

        fig, ax = plt.subplots(figsize=(8, 4))
        if plot_type == "Histogram":
            sns.histplot(numeric_df[selected_num_col].dropna(), kde=True, ax=ax, color="#1da58f")
        else:
            sns.boxplot(x=numeric_df[selected_num_col].dropna(), ax=ax, color="#5ab8aa")
        ax.set_title(f"{plot_type} - {selected_num_col}")
        st.pyplot(fig)

    st.subheader("5. Categorical Analysis")
    if categorical_df.empty:
        st.info("No categorical columns detected.")
    else:
        cat_summary = pd.DataFrame(
            {
                "column": categorical_df.columns,
                "unique_values": [categorical_df[col].nunique(dropna=True) for col in categorical_df.columns],
            }
        ).sort_values("unique_values", ascending=False)
        st.dataframe(cat_summary, use_container_width=True)

        selected_cat_col = st.selectbox("Categorical column", categorical_df.columns.tolist(), key="eda_cat_col")
        top_n = st.slider("Top categories", min_value=5, max_value=30, value=10, key="eda_cat_topn")
        top_values = categorical_df[selected_cat_col].astype(str).value_counts().head(top_n)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(top_values.index, top_values.values, color="#4e9ecf")
        ax.set_title(f"Top {top_n} categories - {selected_cat_col}")
        ax.tick_params(axis="x", rotation=45)
        st.pyplot(fig)

    st.subheader("6. Correlation Matrix")
    if numeric_df.shape[1] < 2:
        st.info("At least 2 numeric columns are required for correlation analysis.")
        return

    corr_method = st.selectbox("Correlation method", ["pearson", "spearman"], key="eda_corr_method")
    max_cols = st.slider("Max numeric columns in heatmap", min_value=5, max_value=30, value=15, key="eda_corr_max")
    selected_numeric = numeric_df.columns.tolist()
    if len(selected_numeric) > max_cols:
        variance_ranked = numeric_df.var(numeric_only=True).sort_values(ascending=False).index.tolist()
        selected_numeric = variance_ranked[:max_cols]
        st.caption("Showing top-variance numeric columns for readability.")

    corr = numeric_df[selected_numeric].corr(method=corr_method)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(corr, annot=False, cmap="coolwarm", center=0, ax=ax)
    ax.set_title(f"Correlation Heatmap ({corr_method.title()})")
    st.pyplot(fig)
