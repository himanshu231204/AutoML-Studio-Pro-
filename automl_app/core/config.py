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
        min-height: 100vh;
    }

    .main {
        max-width: 100%;
        width: 100%;
    }

    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2.2rem;
        max-width: 100%;
        width: 100%;
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
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        margin-left: 1.5rem;
        margin-right: 1.5rem;
        box-shadow: 0 20px 50px rgba(11, 28, 48, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.15);
        position: relative;
        overflow: hidden;
    }

    .app-hero::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(22, 179, 160, 0.1) 0%, transparent 70%);
        border-radius: 50%;
    }

    .app-hero h1 {
        color: var(--hero-text) !important;
        margin: 0;
        font-size: clamp(1.8rem, 3vw, 2.8rem);
        font-weight: 800;
        letter-spacing: -0.03em;
        position: relative;
        z-index: 1;
    }

    .app-hero p {
        margin: 0.6rem 0 0;
        color: var(--hero-subtext);
        font-size: 1.05rem;
        position: relative;
        z-index: 1;
        max-width: 90%;
        line-height: 1.6;
    }

    .quick-strip {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1.2rem;
        margin: 1.5rem 1.5rem 1.5rem 1.5rem;
        padding: 0;
    }

    .quick-card {
        background: var(--glass-surface);
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 1.2rem 1.4rem;
        box-shadow: 0 12px 32px rgba(9, 28, 46, 0.1);
        backdrop-filter: blur(3px);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }

    .quick-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        transition: left 0.5s;
    }

    .quick-card:hover {
        transform: translateY(-6px);
        box-shadow: 0 20px 48px rgba(9, 28, 46, 0.15);
        border-color: var(--accent);
    }

    .quick-card:hover::before {
        left: 100%;
    }

    .quick-card b {
        display: block;
        color: var(--ink);
        font-size: 1.1rem;
        font-family: 'Manrope', sans-serif;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }

    .quick-card span {
        color: var(--muted);
        font-size: 0.92rem;
        line-height: 1.5;
        display: block;
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

    @keyframes shimmer {
        0% { background-position: -1000px 0; }
        100% { background-position: 1000px 0; }
    }

    @keyframes glow {
        0%, 100% {
            box-shadow: 0 0 20px rgba(22, 179, 160, 0.3);
        }
        50% {
            box-shadow: 0 0 30px rgba(22, 179, 160, 0.5);
        }
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.8; }
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: var(--surface-soft);
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 0.5rem;
        margin: 1.5rem 1.5rem 1.2rem 1.5rem;
        box-shadow: 0 8px 20px rgba(9, 28, 46, 0.08);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 12px;
        font-weight: 600;
        color: var(--muted);
        padding: 0.6rem 1.2rem;
        font-size: 0.95rem;
        transition: all 0.25s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background: var(--surface);
        color: var(--accent);
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--surface) 0%, rgba(22, 179, 160, 0.05) 100%);
        color: var(--accent);
        box-shadow: 0 6px 16px rgba(22, 179, 160, 0.15);
        font-weight: 700;
    }

    .stTabs [data-baseweb="tab-panel"] {
        background: var(--panel-bg);
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 1.5rem 1.8rem;
        margin: 0 1.5rem 1.5rem 1.5rem;
        box-shadow: 0 12px 32px rgba(8, 26, 45, 0.09);
        animation: fadeSlideIn 0.35s cubic-bezier(0.34, 1.56, 0.64, 1);
    }

    .stButton > button {
        width: 100%;
        border-radius: 12px;
        font-weight: 700;
        min-height: 2.95rem;
        background: linear-gradient(135deg, var(--accent) 0%, var(--accent-dark) 100%);
        color: #f8ffff;
        border: none;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 12px 28px rgba(22, 179, 160, 0.25);
        position: relative;
        overflow: hidden;
        font-size: 0.95rem;
        letter-spacing: 0.3px;
    }

    .stButton > button::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255, 255, 255, 0.3), transparent);
        animation: shimmer 3s infinite;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 16px 36px rgba(22, 179, 160, 0.35);
    }

    .stButton > button:active {
        transform: translateY(0px);
        box-shadow: 0 8px 20px rgba(22, 179, 160, 0.25);
    }

    .stTextInput input,
    .stNumberInput input,
    .stTextArea textarea,
    .stSelectbox [data-baseweb="select"] > div,
    .stMultiSelect [data-baseweb="select"] > div {
        border-radius: 10px !important;
        border: 1.5px solid var(--line) !important;
        background: var(--surface) !important;
        transition: all 0.25s ease !important;
        font-size: 0.95rem;
    }

    .stTextInput input:focus,
    .stNumberInput input:focus,
    .stTextArea textarea:focus,
    .stSelectbox [data-baseweb="select"] > div:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px rgba(22, 179, 160, 0.1) !important;
    }

    .stSelectbox:focus-within [data-baseweb="select"] > div {
        border-color: var(--accent) !important;
    }

    .stSlider [data-testid="stTickBar"] {
        background-image: linear-gradient(to right, var(--accent), var(--accent-dark));
    }

    .stFileUploader {
        border: 2px dashed var(--upload-line);
        border-radius: 14px;
        background: var(--upload-bg);
        padding: 1.5rem;
        transition: all 0.3s ease;
    }

    .stFileUploader:hover {
        border-color: var(--accent);
        background: rgba(22, 179, 160, 0.05);
    }

    .metric-card {
        background: var(--surface) !important;
        color: var(--ink) !important;
        padding: 1.2rem 1.4rem;
        border-radius: 12px;
        border: 1px solid var(--line);
        box-shadow: 0 10px 28px rgba(10, 28, 45, 0.08);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 14px 36px rgba(10, 28, 45, 0.12);
    }

    .metric-card h1 {
        color: var(--accent) !important;
        margin: 0;
        font-size: clamp(1.4rem, 2vw, 2rem);
        font-weight: 800;
    }

    .metric-card small {
        color: var(--muted) !important;
        font-weight: 600;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
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
        box-shadow: 0 8px 24px rgba(10, 28, 45, 0.08);
    }

    [data-testid="stAlert"] {
        border-radius: 12px;
        border: 1.5px solid;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.08);
        padding: 1rem 1.2rem;
    }

    [data-testid="stAlert"] > div:first-child {
        font-weight: 600;
        font-size: 0.95rem;
    }

    [data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 1rem 1.1rem;
        box-shadow: 0 8px 20px rgba(10, 28, 45, 0.08);
        transition: all 0.3s ease;
    }

    [data-testid="stMetric"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 10px 28px rgba(10, 28, 45, 0.12);
    }

    .stDivider {
        border-top: 1px solid var(--line) !important;
        margin: 1.5rem 0 !important;
    }

    .stSubheader {
        color: var(--ink) !important;
        font-size: 1.3rem !important;
        font-weight: 700 !important;
        margin-top: 1.2rem !important;
    }

    .stMarkdown h4 {
        color: var(--accent) !important;
        font-weight: 700;
        margin-top: 1rem;
    }

    @media (max-width: 768px) {
        .block-container {
            padding-top: 0.8rem;
            padding-bottom: 1.5rem;
            padding-left: 0;
            padding-right: 0;
        }
        .app-hero {
            border-radius: 14px;
            padding: 1.4rem;
            margin: 1rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            border-radius: 12px;
            margin: 1rem;
        }
        .stTabs [data-baseweb="tab-panel"] {
            margin: 0 1rem 1rem 1rem;
            padding: 1rem;
        }
        .quick-strip {
            grid-template-columns: 1fr;
            gap: 0.8rem;
            margin: 1rem;
        }
        .stButton > button {
            min-height: 2.5rem;
            font-size: 0.9rem;
        }
        [data-testid="stMetric"] {
            padding: 0.8rem 0.9rem;
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
