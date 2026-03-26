import os

import streamlit as st

ARTIFACTS_DIR = "artifacts"
BASE_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@500;700;800&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

    :root {
        __THEME_VARIABLES__
    }

    .stApp {
        background:
            radial-gradient(1200px 500px at 10% -10%, var(--orb-1) 0%, transparent 55%),
            radial-gradient(900px 500px at 90% -15%, var(--orb-2) 0%, transparent 58%),
            var(--bg);
        color: var(--ink);
        font-family: 'IBM Plex Sans', sans-serif;
    }

    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2.2rem;
        max-width: 1220px;
    }

    h1, h2, h3 {
        font-family: 'Manrope', sans-serif;
        letter-spacing: -0.02em;
        color: var(--ink);
    }

    p, li, label, .stCaption, .stMarkdown {
        color: var(--muted);
    }

    .app-hero {
        background: linear-gradient(130deg, var(--hero-1) 0%, var(--hero-2) 100%);
        border-radius: 20px;
        padding: 1.35rem 1.45rem;
        margin-bottom: 1.1rem;
        box-shadow: 0 16px 34px rgba(11, 28, 48, 0.23);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    .app-hero h1 {
        color: var(--hero-text) !important;
        margin: 0;
        font-size: clamp(1.45rem, 2.2vw, 2.25rem);
        font-weight: 800;
    }

    .app-hero p {
        margin: 0.45rem 0 0;
        color: var(--hero-subtext);
        font-size: 0.98rem;
    }

    .quick-strip {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.75rem;
        margin: 0.15rem 0 1.05rem;
    }

    .quick-card {
        background: var(--glass-surface);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 0.7rem 0.85rem;
        box-shadow: 0 8px 20px rgba(9, 28, 46, 0.08);
        backdrop-filter: blur(2px);
    }

    .quick-card b {
        display: block;
        color: var(--ink);
        font-size: 0.94rem;
        font-family: 'Manrope', sans-serif;
        margin-bottom: 0.16rem;
    }

    .quick-card span {
        color: var(--muted);
        font-size: 0.84rem;
    }

    @keyframes fadeSlideIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.45rem;
        background: var(--surface-soft);
        border: 1px solid var(--line);
        border-radius: 999px;
        padding: 0.35rem;
        margin-bottom: 1rem;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 999px;
        font-weight: 600;
        color: var(--muted);
    }

    .stTabs [aria-selected="true"] {
        background: var(--surface);
        color: var(--ink);
        box-shadow: 0 4px 12px rgba(8, 30, 52, 0.1);
    }

    .stTabs [data-baseweb="tab-panel"] {
        background: var(--panel-bg);
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 1rem 1rem 1.15rem;
        box-shadow: 0 10px 22px rgba(8, 26, 45, 0.08);
        animation: fadeSlideIn 0.34s ease;
    }

    .stButton > button {
        width: 100%;
        border-radius: 12px;
        font-weight: 700;
        min-height: 2.95rem;
        background: linear-gradient(135deg, var(--accent) 0%, var(--accent-dark) 100%);
        color: #f8ffff;
        border: none;
        transition: transform 0.18s ease, box-shadow 0.18s ease;
        box-shadow: 0 10px 20px rgba(15, 157, 138, 0.22);
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 12px 25px rgba(15, 157, 138, 0.3);
    }

    .stTextInput input,
    .stNumberInput input,
    .stTextArea textarea,
    .stSelectbox [data-baseweb="select"] > div,
    .stMultiSelect [data-baseweb="select"] > div {
        border-radius: 10px !important;
        border: 1px solid var(--line) !important;
        background: var(--surface) !important;
    }

    .stFileUploader {
        border: 1px dashed var(--upload-line);
        border-radius: 12px;
        background: var(--upload-bg);
        padding: 0.4rem;
    }

    .metric-card {
        background: var(--surface) !important;
        color: var(--ink) !important;
        padding: 1rem 1.15rem;
        border-radius: 12px;
        border: 1px solid var(--line);
        box-shadow: 0 10px 24px rgba(10, 28, 45, 0.09);
        margin-bottom: 20px;
    }

    .metric-card h1 {
        color: var(--ink) !important;
        margin: 0;
        font-size: clamp(1.3rem, 2.1vw, 2rem);
    }

    .metric-card small {
        color: var(--muted) !important;
        font-weight: 600;
        font-size: 0.9rem;
    }

    .success-card {
        background: var(--success-bg);
        color: var(--success-ink);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid var(--success-line);
    }

    .stDataFrame, .stTable {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid var(--line);
    }

    [data-testid="stAlert"] {
        border-radius: 12px;
        border: 1px solid var(--line);
    }

    [data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 0.5rem 0.65rem;
    }

    @media (max-width: 768px) {
        .block-container {
            padding-top: 0.8rem;
            padding-bottom: 1.5rem;
        }
        .app-hero {
            border-radius: 14px;
            padding: 1rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            border-radius: 14px;
        }
        .quick-strip {
            grid-template-columns: 1fr;
            gap: 0.55rem;
        }
    }
</style>
"""


def _theme_variables(theme_mode: str) -> str:
    if theme_mode == "dark":
        return """
        --bg: #0d1722;
        --surface: #132231;
        --surface-soft: #172b3f;
        --ink: #e8f2ff;
        --muted: #a9bfd6;
        --line: #2b4257;
        --accent: #16b3a0;
        --accent-dark: #0f8e7f;
        --hero-1: #0f2233;
        --hero-2: #214765;
        --hero-text: #f2f8ff;
        --hero-subtext: #d2e5f7;
        --orb-1: #1f3d59;
        --orb-2: #18443f;
        --glass-surface: rgba(19, 34, 49, 0.82);
        --panel-bg: rgba(19, 34, 49, 0.86);
        --upload-bg: rgba(19, 34, 49, 0.72);
        --upload-line: #4a6a88;
        --success-bg: #14382f;
        --success-ink: #9ee8d7;
        --success-line: #246555;
        """

    return """
    --bg: #f4f7fb;
    --surface: #ffffff;
    --surface-soft: #f8fbff;
    --ink: #122236;
    --muted: #4b5d73;
    --line: #d6e2ef;
    --accent: #0f9d8a;
    --accent-dark: #0b7b6c;
    --hero-1: #122236;
    --hero-2: #1e3a5b;
    --hero-text: #f6fbff;
    --hero-subtext: #d7e6f5;
    --orb-1: #dceeff;
    --orb-2: #d8f7f1;
    --glass-surface: rgba(255, 255, 255, 0.86);
    --panel-bg: rgba(255, 255, 255, 0.88);
    --upload-bg: rgba(255, 255, 255, 0.72);
    --upload-line: #8bb3d8;
    --success-bg: #e8f8f5;
    --success-ink: #0b5d52;
    --success-line: #bde9e1;
    """


def setup_page(theme_mode: str = "light") -> None:
    st.set_page_config(
        page_title="AutoML Studio Pro",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    themed_css = BASE_CSS.replace("__THEME_VARIABLES__", _theme_variables(theme_mode))
    st.markdown(themed_css, unsafe_allow_html=True)
