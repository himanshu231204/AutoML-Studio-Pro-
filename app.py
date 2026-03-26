import warnings

import streamlit as st

from automl_app.core.config import setup_page
from automl_app.tabs.analysis import render_analysis_tab
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
            <h1>AutoML Studio Pro</h1>
            <p>Train robust ML models, inspect data health, compare algorithms, and deploy predictions from one production-ready workspace.</p>
        </section>
        <section class="quick-strip">
            <div class="quick-card">
                <b>Adaptive Training</b>
                <span>Auto task detection with multi-model benchmarking and best-model selection.</span>
            </div>
            <div class="quick-card">
                <b>Data Health First</b>
                <span>Built-in profiling, schema generation, and resilient preprocessing for noisy datasets.</span>
            </div>
            <div class="quick-card">
                <b>Production Ready</b>
                <span>Single and batch predictions with exportable artifacts and reusable Python code.</span>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "🚀 Train & Learn",
        "📊 Data Analysis",
        "🔮 Production Engine",
        "📘 User Manual",
    ])

    with tab1:
        render_train_tab()

    with tab2:
        render_analysis_tab()

    with tab3:
        render_prediction_tab()

    with tab4:
        render_manual_tab()

    render_footer()


if __name__ == "__main__":
    main()
