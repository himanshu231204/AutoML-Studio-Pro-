import warnings

import streamlit as st

from automl_app.core.config import setup_page
from automl_app.tabs.analysis import render_analysis_tab
from automl_app.tabs.developer import render_developer_tab
from automl_app.tabs.manual import render_manual_tab
from automl_app.tabs.prediction import render_prediction_tab
from automl_app.tabs.train import render_train_tab
from automl_app.ui.footer import render_footer

warnings.filterwarnings("ignore")


def main() -> None:
    setup_page(theme_mode="dark")

    st.markdown(
        """
        <section class="app-hero">
            <h1>🚀 AutoML Studio Pro</h1>
            <p>Enterprise-grade machine learning platform. Train robust models, inspect data health, compare algorithms, and deploy predictions — all from one production-ready workspace.</p>
        </section>
        <section class="quick-strip">
            <div class="quick-card">
                <b>⚡ Adaptive Training</b>
                <span>Intelligent task detection with multi-model benchmarking, automatic preprocessing, and best-model selection optimized for your metric.</span>
            </div>
            <div class="quick-card">
                <b>📊 Data Health First</b>
                <span>Built-in profiling, advanced EDA analytics, schema generation, and resilient preprocessing designed for real-world, noisy datasets.</span>
            </div>
            <div class="quick-card">
                <b>🔮 Production Ready</b>
                <span>Single and batch predictions with exportable artifacts, reusable Python code, and integrated model versioning.</span>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🚀 Train & Learn",
        "📊 Data Analysis",
        "🔮 Predictions",
        "📘 Guide",
        "👨‍💻 Developer",
    ])

    with tab1:
        render_train_tab()

    with tab2:
        render_analysis_tab()

    with tab3:
        render_prediction_tab()

    with tab4:
        render_manual_tab()

    with tab5:
        render_developer_tab()

    render_footer()


if __name__ == "__main__":
    main()
