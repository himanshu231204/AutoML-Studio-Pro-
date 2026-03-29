import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy import stats


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


def _advanced_numeric_stats(numeric_df: pd.DataFrame) -> pd.DataFrame:
    """Compute advanced statistical metrics for numeric columns."""
    rows: list[dict[str, object]] = []
    for col in numeric_df.columns:
        series = numeric_df[col].dropna()
        if series.empty:
            continue
        rows.append({
            "column": col,
            "mean": round(series.mean(), 4),
            "median": round(series.median(), 4),
            "std": round(series.std(), 4),
            "skewness": round(stats.skew(series), 4),
            "kurtosis": round(stats.kurtosis(series), 4),
            "min": round(series.min(), 4),
            "max": round(series.max(), 4),
            "q25": round(series.quantile(0.25), 4),
            "q75": round(series.quantile(0.75), 4),
        })
    return pd.DataFrame(rows)


def _data_quality_score(df: pd.DataFrame) -> dict[str, float]:
    """Calculate comprehensive data quality metrics."""
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    completeness = ((total_cells - missing_cells) / total_cells * 100) if total_cells > 0 else 0
    uniqueness = ((df.shape[0] - duplicate_rows) / df.shape[0] * 100) if df.shape[0] > 0 else 0

    numeric_df = df.select_dtypes(include=[np.number])
    outlier_count = 0
    for col in numeric_df.columns:
        series = numeric_df[col].dropna()
        if len(series) > 0:
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                outlier_count += ((series < (q1 - 1.5 * iqr)) | (series > (q3 + 1.5 * iqr))).sum()

    total_numeric_cells = len(numeric_df) * len(numeric_df.columns) if len(numeric_df.columns) > 0 else 1
    outlier_ratio = (outlier_count / total_numeric_cells * 100) if total_numeric_cells > 0 else 0
    consistency = max(0, 100 - outlier_ratio)

    quality_score = (completeness * 0.4 + uniqueness * 0.3 + consistency * 0.3)

    return {
        "completeness": round(completeness, 2),
        "uniqueness": round(uniqueness, 2),
        "consistency": round(consistency, 2),
        "quality_score": round(quality_score, 2),
    }


def _top_correlations(numeric_df: pd.DataFrame, method: str = "pearson", top_n: int = 10) -> pd.DataFrame:
    """Extract top correlated features (excluding self-correlation)."""
    if numeric_df.shape[1] < 2:
        return pd.DataFrame()

    corr_matrix = numeric_df.corr(method=method)

    pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
            correlation = corr_matrix.iloc[i, j]
            pairs.append({"feature_1": col1, "feature_2": col2, "correlation": round(correlation, 4)})

    if not pairs:
        return pd.DataFrame()

    pairs_df = pd.DataFrame(pairs)
    pairs_df["abs_correlation"] = pairs_df["correlation"].abs()
    return pairs_df.nlargest(top_n, "abs_correlation")[["feature_1", "feature_2", "correlation"]]


def _class_distribution(categorical_df: pd.DataFrame) -> dict[str, dict]:
    """Analyze class balance for categorical columns."""
    distribution_data = {}
    for col in categorical_df.columns:
        value_counts = categorical_df[col].value_counts()
        total = len(categorical_df[col].dropna())
        distribution_data[col] = {
            "classes": len(value_counts),
            "max_class_pct": round((value_counts.iloc[0] / total * 100) if total > 0 else 0, 2),
            "min_class_pct": round((value_counts.iloc[-1] / total * 100) if total > 0 else 0, 2),
            "balance_ratio": round(value_counts.iloc[0] / value_counts.iloc[-1] if len(value_counts) > 1 else 1, 2),
        }
    return distribution_data


def _variance_analysis(numeric_df: pd.DataFrame) -> pd.DataFrame:
    """Analyze variance and relative importance of numeric features."""
    rows: list[dict[str, object]] = []
    total_variance = numeric_df.var(numeric_only=True).sum()

    for col in numeric_df.columns:
        variance = numeric_df[col].var()
        variance_pct = (variance / total_variance * 100) if total_variance > 0 else 0
        rows.append({
            "column": col,
            "variance": round(variance, 4),
            "variance_%": round(variance_pct, 2),
        })

    return pd.DataFrame(rows).sort_values("variance_%", ascending=False)


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
    else:
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

    # Advanced EDA Section
    st.markdown("---")
    st.markdown("### 🔬 Advanced EDA Analytics")

    adv_tab1, adv_tab2, adv_tab3, adv_tab4, adv_tab5, adv_tab6 = st.tabs([
        "📈 Statistics",
        "🎯 Target Analysis",
        "🔗 Correlations",
        "📊 Distributions",
        "🗂️ Data Quality",
        "🔍 Variance Analysis"
    ])

    with adv_tab1:
        st.markdown("#### Advanced Statistical Summary")
        if not numeric_df.empty:
            stats_df = _advanced_numeric_stats(numeric_df)
            st.dataframe(stats_df, use_container_width=True)

            st.markdown("**Interpretation Guide:**")
            col1, col2 = st.columns(2)
            with col1:
                st.write("""
                - **Skewness**:
                  - 0-0.5: Fairly symmetric
                  - 0.5-1: Moderately skewed
                  - >1: Highly skewed
                """)
            with col2:
                st.write("""
                - **Kurtosis**:
                  - 0: Normal distribution
                  - >0: Heavy tails
                  - <0: Light tails
                """)
        else:
            st.info("No numeric columns found.")

    with adv_tab2:
        st.markdown("#### Target Variable Analysis")
        if not categorical_df.empty:
            target_col = st.selectbox("Select potential target column", categorical_df.columns.tolist())

            class_dist = _class_distribution(categorical_df[[target_col]])
            target_info = class_dist[target_col]

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Classes", target_info["classes"])
            col2.metric("Dominant Class %", f"{target_info['max_class_pct']}%")
            col3.metric("Minority Class %", f"{target_info['min_class_pct']}%")
            col4.metric("Class Balance Ratio", f"{target_info['balance_ratio']:.1f}:1")

            value_counts = categorical_df[target_col].value_counts()
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.barh(value_counts.index, value_counts.values, color="#1da58f")
            ax.set_xlabel("Count")
            ax.set_title(f"Class Distribution - {target_col}")

            for i, (_idx, val) in enumerate(zip(value_counts.index, value_counts.values)):
                pct = val / value_counts.sum() * 100
                ax.text(val, i, f" {pct:.1f}%", va="center", fontsize=9)

            st.pyplot(fig)

            if target_info["balance_ratio"] > 3:
                st.warning(f"⚠️ **Class Imbalance Detected**: Ratio is {target_info['balance_ratio']:.1f}:1. Consider using SMOTE or class weights.")
        else:
            st.info("No categorical columns found for target analysis.")

    with adv_tab3:
        st.markdown("#### Correlation Deep Dive")
        if not numeric_df.empty and numeric_df.shape[1] >= 2:
            corr_method = st.selectbox("Method", ["pearson", "spearman"], key="adv_corr_method")
            top_n = st.slider("Top N correlations", 5, 30, 10, key="adv_top_corr")

            top_corr_df = _top_correlations(numeric_df, method=corr_method, top_n=top_n)
            if not top_corr_df.empty:
                st.dataframe(top_corr_df, use_container_width=True)

                if len(numeric_df.columns) > 1:
                    corr = numeric_df.corr(method=corr_method)
                    fig, ax = plt.subplots(figsize=(11, 5))
                    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, cbar_kws={"label": "Correlation"})
                    ax.set_title(f"Full Correlation Matrix ({corr_method.title()})")
                    st.pyplot(fig)
            else:
                st.info("No correlations to display.")
        else:
            st.info("At least 2 numeric columns required.")

    with adv_tab4:
        st.markdown("#### Distribution Analysis")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Numeric Distributions**")
            if not numeric_df.empty:
                dist_col = st.selectbox("Select numeric column", numeric_df.columns.tolist(), key="adv_dist_num")
                series = numeric_df[dist_col].dropna()

                fig, axes = plt.subplots(1, 2, figsize=(12, 4))

                axes[0].hist(series, bins=30, alpha=0.7, edgecolor='black', color="#1da58f")
                axes[0].set_xlabel(dist_col)
                axes[0].set_ylabel("Frequency")
                axes[0].set_title(f"Histogram - {dist_col}")

                stats.probplot(series, dist="norm", plot=axes[1])
                axes[1].set_title(f"Q-Q Plot - {dist_col} (vs Normal)")

                st.pyplot(fig)

        with col2:
            st.markdown("**Categorical Distributions**")
            if not categorical_df.empty:
                cat_dist_col = st.selectbox("Select categorical column", categorical_df.columns.tolist(), key="adv_dist_cat")
                value_counts = categorical_df[cat_dist_col].value_counts().head(15)

                fig, ax = plt.subplots(figsize=(10, 5))
                ax.bar(range(len(value_counts)), value_counts.values, color="#4e9ecf")
                ax.set_xticks(range(len(value_counts)))
                ax.set_xticklabels(value_counts.index, rotation=45, ha='right')
                ax.set_ylabel("Count")
                ax.set_title(f"Top Categories - {cat_dist_col}")
                st.pyplot(fig)

    with adv_tab5:
        st.markdown("#### Data Quality Assessment")
        quality_metrics = _data_quality_score(df)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Completeness", f"{quality_metrics['completeness']}%")
        col2.metric("Uniqueness", f"{quality_metrics['uniqueness']}%")
        col3.metric("Consistency", f"{quality_metrics['consistency']}%")
        col4.metric("Overall Quality", f"{quality_metrics['quality_score']}%", delta=f"{quality_metrics['quality_score']-50:.1f}")

        fig, ax = plt.subplots(figsize=(10, 2))
        quality_score = quality_metrics['quality_score']
        colors = ["#d62728" if quality_score < 50 else "#ff7f0e" if quality_score < 75 else "#2ca02c"]
        ax.barh([0], [quality_score], color=colors[0], height=0.5)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xticklabels(["0", "25", "50", "75", "100"])
        ax.set_yticks([])
        ax.set_xlabel("Quality Score (%)")
        ax.set_title("Data Quality Gauge")

        for v in [25, 50, 75]:
            ax.axvline(v, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)

        st.pyplot(fig)

        st.markdown("**Data Quality Insights:**")
        insights = []
        if quality_metrics['completeness'] < 95:
            insights.append(f"❌ High missing data ({100 - quality_metrics['completeness']:.1f}%). Consider imputation or removal.")
        else:
            insights.append("✅ Data completeness is good.")

        if quality_metrics['uniqueness'] < 95:
            insights.append(f"❌ Significant duplicates detected ({100 - quality_metrics['uniqueness']:.1f}%). Consider deduplication.")
        else:
            insights.append("✅ Low duplicate records.")

        if quality_metrics['consistency'] < 85:
            insights.append("⚠️ Outliers detected. Review or clip extreme values.")
        else:
            insights.append("✅ Data consistency is good.")

        for insight in insights:
            st.write(insight)

    with adv_tab6:
        st.markdown("#### Feature Variance Analysis")
        if not numeric_df.empty:
            var_df = _variance_analysis(numeric_df)
            st.dataframe(var_df, use_container_width=True)

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.pie(var_df["variance"], labels=var_df["column"], autopct="%.1f%%", startangle=90)
            ax.set_title("Feature Variance Contribution")
            st.pyplot(fig)

            st.markdown("**Variance Insights:**")
            st.write("""
            - Features with high variance contribute more to model decisions
            - Low-variance features may be redundant
            - Consider feature scaling or selection based on variance
            """)
        else:
            st.info("No numeric features to analyze.")
