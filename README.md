<div align="center">

# AutoML Studio Pro

**A no-code, end-to-end automated machine learning platform for training, evaluating, and deploying ML models—right from your browser.**

[![Version](https://img.shields.io/badge/Version-1.3.0-0078D4?style=for-the-badge)](CHANGELOG.md)
&nbsp;
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Visit%20App-0078D4?style=for-the-badge&logo=streamlit&logoColor=white)](https://automl-studio-pro-on5vfj7azahvyvdj9fyh7b.streamlit.app/)
&nbsp;
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
&nbsp;
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
&nbsp;
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)

</div>

---

> **Short Description:** AutoML Studio Pro is an interactive, browser-based platform that automates the complete machine learning workflow—from data upload and exploratory analysis to model training, evaluation, and prediction—without writing a single line of code.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Architecture](#architecture)
- [Project Governance](#project-governance)
- [Suggested Repository Topics](#suggested-repository-topics)
- [Contributing](#contributing)
- [Support This Project](#support-this-project)
- [License](#license)
- [Contact](#contact)

---

## Overview

**AutoML Studio Pro** eliminates the complexity of building machine learning models. Upload a CSV dataset, select a target column, and let the platform handle the rest—preprocessing, model selection, training, evaluation, and export.

Built with [Streamlit](https://streamlit.io/) and [Scikit-Learn](https://scikit-learn.org/), the application is designed for:

- **Beginners** who want to explore ML without writing code.
- **Students** looking for an educational tool with exportable Python scripts.
- **Practitioners** who need quick baseline models and batch-prediction capabilities.

---

## Key Features

### 🚀 AutoML Engine

| Feature | Description |
|---|---|
| Automatic Task Detection | Determines whether the problem is classification or regression based on the target column. |
| Automated Preprocessing | Handles missing values, encoding, and feature scaling via Scikit-Learn pipelines. |
| Intelligent Model Selection | Trains multiple models (RandomForest, GradientBoosting, XGBoost, etc.) and selects the best. |
| Imbalanced Data Handling | Applies SMOTE oversampling for skewed classification datasets. |
| Hyperparameter Tuning | Lightweight tuning with Optuna integration for advanced optimization. |
| Ensemble Models | Combine multiple models using voting or stacking ensembles. |

### 📊 Data Insights

| Feature | Description |
|---|---|
| Exploratory Data Analysis | Generates descriptive statistics, outlier detection, and correlation heatmaps. |
| Advanced EDA Analytics | Six comprehensive tabs: Statistics, Target Analysis, Correlations, Distributions, Data Quality, Variance. |
| Feature Importance (XAI) | Uses permutation importance and SHAP values to explain model predictions. |
| Performance Metrics | Displays confusion matrices, accuracy scores, R² scores, and prediction plots. |
| Cross-Validation Visualization | Bar charts and histograms showing CV scores across models. |

### 🔧 Feature Engineering

| Feature | Description |
|---|---|
| Polynomial Features | Automatically creates polynomial features to capture non-linear relationships. |
| Interaction Features | Generates feature interactions to discover combined effects. |
| Statistical Aggregations | Creates row-wise statistics (mean, std, min, max) for numeric features. |
| Missing Value Strategies | Choose from median, mean, most_frequent, or constant imputation. |
| Preprocessing Preview | Visual preview of all preprocessing steps before training. |

### 🧠 Advanced ML

| Feature | Description |
|---|---|
| SHAP Explainable AI | Model interpretability with SHAP values and feature explanations. |
| Optuna Optimization | Advanced hyperparameter optimization with configurable trials. |
| NLP/Text Classification | TF-IDF text preprocessing with configurable n-gram ranges. |
| Time Series Forecasting | ARIMA and Exponential Smoothing models for temporal data. |
| Data Versioning | Track and compare datasets across versions with MD5 hashing. |

### 📁 Sample Datasets

| Feature | Description |
|---|---|
| Built-in Datasets | Iris, Wine, Breast Cancer, and Diabetes datasets for quick demos. |
| One-Click Loading | Load sample datasets instantly without uploading files. |

### 💾 Export & Deploy

| Feature | Description |
|---|---|
| Model Export | Download trained models as `.zip` archives for reuse. |
| Python Code Export | Export a ready-to-run `train_model.py` script for learning and customization. |
| PDF Report Export | Generate HTML reports with model details and metrics. |
| Batch Predictions | Upload CSV files to generate predictions at scale. |
| Dynamic Prediction Form | Auto-generated input form based on dataset schema for single predictions. |
| Model History | Track and compare previously trained models with visualizations. |

### 🎨 User Experience

| Feature | Description |
|---|---|
| Dark/Light Theme | Toggle between dark and light themes via sidebar. |
| Responsive Design | Optimized for desktop, tablet, and mobile devices. |
| Real-time Feedback | Live status updates during training with progress indicators. |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend / UI | [Streamlit](https://streamlit.io/) |
| Machine Learning | [Scikit-Learn](https://scikit-learn.org/) — RandomForest, GradientBoosting, XGBoost, Pipelines |
| Hyperparameter Optimization | [Optuna](https://optuna.org/) |
| Explainable AI | [SHAP](https://shap.readthedocs.io/) |
| Time Series | [Statsmodels](https://www.statsmodels.org/) — ARIMA, Exponential Smoothing |
| Data Processing | [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/) |
| Visualization | [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/) |
| Model Serialization | [Joblib](https://joblib.readthedocs.io/) |
| Imbalance Handling | [Imbalanced-Learn](https://imbalanced-learn.org/) (SMOTE) |

---

## Getting Started

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/himanshu231204/AutoML-Studio-Pro-.git
cd AutoML-Studio-Pro-

# 2. (Recommended) Create and activate a virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the application
streamlit run app.py
```

The application will open automatically in your default browser at `http://localhost:8501`.

### Run with Docker

If you prefer running the app in a container, use one of the following options.

Prerequisites:
- Docker Desktop installed and running

Option 1: Docker Compose (recommended)

```bash
# Build and run in detached mode
docker compose up -d --build

# Open the app
# http://localhost:8501

# Stop containers
docker compose down
```

Option 2: Docker CLI

```bash
# Build image
docker build -t automl-studio-pro .

# Run container
docker run -d -p 8501:8501 --name automl_studio_pro automl-studio-pro

# Stop and remove container
docker stop automl_studio_pro
docker rm automl_studio_pro
```

Notes:
- The app is exposed on port `8501`.
- `requirements.txt` is UTF-8 encoded for Linux container compatibility.

Troubleshooting:
- Docker command not found:
	Install Docker Desktop and restart terminal, then run `docker --version`.
- Docker daemon is not running:
	Start Docker Desktop, wait until status is "Engine running", then retry.
- Port 8501 already in use:
	Run with a different host port, for example `docker run -d -p 8502:8501 --name automl_studio_pro automl-studio-pro`.
- Container exits immediately:
	Check logs with `docker logs automl_studio_pro`.
- Dependency changes are not reflected:
	Rebuild image with `docker compose up -d --build` or `docker build --no-cache -t automl-studio-pro .`.

---

## Project Structure

```
├── .github/                    # CI/CD and community files
│   ├── ISSUE_TEMPLATE/         # Bug report and feature request templates
│   └── workflows/              # GitHub Actions pipelines (ci.yml, cd.yml, tests.yml)
├── .streamlit/                 # Streamlit Cloud configuration
│   └── config.toml             # Server and theme settings
├── artifacts/                  # Auto-generated models & schema files
├── assets/
│   └── images/
│       ├── badges/             # Local badge assets (optional)
│       └── screenshots/        # UI screenshots for README/docs
├── automl_app/
│   ├── core/                   # Shared config and helper utilities
│   │   ├── config.py           # Page setup, theming, CSS
│   │   └── helpers.py          # ML utilities (preprocessing, model selection, etc.)
│   └── ui/                     # Reusable UI components and tab modules
│       ├── tabs/               # Streamlit tab modules
│       │   ├── train.py        # Training tab with all Phase 1 & 2 features
│       │   ├── analysis.py     # Advanced EDA with 6 sub-tabs
│       │   ├── prediction.py   # Single and batch predictions
│       │   ├── manual.py       # User guide
│       │   └── developer.py    # Developer info
│       └── footer.py           # Shared footer component
├── docs/
│   ├── api/                    # API/exported interface docs
│   └── guides/                 # User and developer guides
├── tests/                      # Unit tests
│   ├── test_phase1_features.py # Phase 1 feature tests
│   ├── test_helpers.py         # Helper function tests
│   └── test_train_utils.py     # Training utility tests
├── app.py                      # Main Streamlit application
├── Dockerfile                  # Multi-stage Docker build
├── docker-compose.yml          # Docker Compose configuration
├── requirements.txt            # Production dependencies
├── requirements-dev.txt        # Development dependencies
├── FEATURES_ROADMAP.md         # Feature roadmap and status
├── CHANGELOG.md                # Version history
└── README.md                   # Project documentation
```

### App Flow
- `app.py` initializes the page and routes each Streamlit tab to its dedicated module.
- `automl_app/core` holds reusable configuration and ML utility functions.
- `automl_app/ui/tabs` keeps each product area isolated for easier maintenance.
- `automl_app/ui` contains shared UI components used across the app.

---

## Usage

### 1. Train & Learn

Upload a CSV file or load a sample dataset, then configure your training:

- **Select Target Column** — Choose the column to predict
- **Training Mode** — Fast or High Accuracy mode
- **Missing Value Strategy** — Choose imputation method (median/mean/most_frequent/constant)
- **Feature Engineering** — Enable polynomial features, interactions, or aggregations
- **Advanced AutoML** — Enable Optuna for hyperparameter optimization
- **Ensemble Models** — Combine multiple models with voting/stacking
- **NLP/Text** — Enable TF-IDF for text classification
- **Time Series** — Enable ARIMA/ETS for temporal forecasting

Click **Start Training** to run the AutoML pipeline. View results including:
- Model leaderboard with cross-validation scores
- Feature importance and SHAP explanations
- Confusion matrices and prediction plots
- Model history and comparison dashboard

Download the trained model or export the equivalent Python code.

### 2. Data Analysis

Explore the uploaded dataset through six comprehensive analysis tabs:
- **Statistics** — Skewness, kurtosis, and detailed statistical summaries
- **Target Analysis** — Class balance detection and distribution visualization
- **Correlations** — Feature correlations with heatmap visualization
- **Distributions** — Histograms, KDE, and Q-Q plots
- **Data Quality** — Completeness, uniqueness, and quality scores
- **Variance Analysis** — Feature variance contribution analysis

### 3. Production Engine

Load a previously trained model or use the current session model:
- **Single Predictions** — Dynamic form based on dataset schema
- **Batch Predictions** — Upload CSV for bulk inference
- **Model Export** — Download as `.zip` archive
- **Report Export** — Generate HTML report with metrics

---

## Architecture

For technical design details, see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Project Governance

- Contribution workflow: [CONTRIBUTING.md](CONTRIBUTING.md)
- Community standards: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)

---

## Suggested Repository Topics

`automl` · `machine-learning` · `no-code` · `streamlit` · `scikit-learn` · `data-science` · `python` · `automated-machine-learning` · `eda` · `classification` · `regression` · `model-training` · `feature-importance` · `smote` · `gradient-boosting` · `shap` · `optuna` · `time-series` · `nlp` · `feature-engineering` · `explainable-ai` · `hyperparameter-optimization`

---

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a PR.

Quick start:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a Pull Request.

When you open a PR, GitHub will auto-load the pull request template to keep reviews consistent.

Please use the GitHub issue templates for bug reports and feature requests:
- **Bug report** template for reproducible defects.
- **Feature request** template for enhancements and roadmap ideas.

You can open a new issue here: [Issues](https://github.com/himanshu231204/AutoML-Studio-Pro-/issues).

---

## 💖 Support This Project

If this project helped you, consider supporting my work!

[![Sponsor](https://img.shields.io/badge/Sponsor-💖-ff69b4?style=for-the-badge)](https://github.com/sponsors/himanshu231204)

Every contribution helps me:
- ⏰ Spend more time on open-source
- 🆓 Keep all tools free for everyone
- 📚 Create more tutorials and guides
- 🚀 Build new developer tools

**[⭐ Star this repo](../../stargazers)** if you find it useful — it means a lot!

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## Contact

| Platform | Link |
|---|---|
| GitHub | [himanshu231204](https://github.com/himanshu231204) |
| LinkedIn | [himanshu231204](https://www.linkedin.com/in/himanshu231204/) |
| X (Twitter) | [himanshu231204](https://x.com/himanshu231204) |