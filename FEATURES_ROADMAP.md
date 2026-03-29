# AutoML Studio Pro - Features Roadmap

This document outlines all planned features for future implementation. Use this as a reference for development priorities and tracking progress.

---

## Current Features

| Tab | Capabilities |
|-----|--------------|
| 🚀 **Train & Learn** | CSV upload, auto target detection, data profiling, multi-model benchmarking, hyperparameter tuning, feature correlation, model export |
| 📊 **Data Analysis** | Column profiling, outlier detection, advanced stats (skew/kurtosis), data quality scoring, visualizations |
| 🔮 **Predictions** | Single entry & batch CSV prediction |
| 📘 **Guide** | Documentation |
| 👨‍💻 **Developer** | Social links |

---

## Planned Features

### Phase 1: Quick Wins (Low Complexity)

| # | Feature | Description | Status |
|---|---------|-------------|--------|
| 1 | **PDF Report Export** | Generate downloadable PDF reports of model performance | ✅ Done |
| 2 | **Preprocessing Pipeline Preview** | Show users the exact preprocessing steps applied | ✅ Done |
| 3 | **Missing Value Strategy Selector** | Allow users to choose imputation strategies | ✅ Done |
| 4 | **Cross-Validation Visualization** | Show CV fold performance over iterations | ✅ Done |
| 5 | **Sample Datasets** | Add built-in sample datasets (Titanic, Iris, etc.) for quick demo | ✅ Done |
| 6 | **Dark/Light Theme Toggle** | Add theme switcher in the UI | ✅ Done |
| 7 | **Model History** | Keep track of previously trained models | ✅ Done |
| 8 | **Model Comparison Dashboard** | Side-by-side comparison of multiple trained models with metrics | ✅ Done |
| 9 | **Ensemble Model Builder** | Allow users to combine multiple models into voting/stacking ensembles | ✅ Done |

### Phase 2: Medium Complexity

| # | Feature | Description | Status |
|---|---------|-------------|--------|
| 1 | **Time Series Support** | Add dedicated time series forecasting with ARIMA, Prophet, LSTM models | ✅ Done |
| 2 | **NLP/Text Classification** | Add text preprocessing, TF-IDF, and text-specific models | ✅ Done |
| 3 | **AutoML Hyperparameter Optimization** | Integrate Optuna/Auto-sklearn for advanced hyperparameter tuning | ✅ Done |
| 4 | **Feature Engineering** | Add automated feature creation (polynomial, interactions, aggregations) | ✅ Done |
| 5 | **SHAP/Explainable AI** | Add model interpretability with SHAP values and feature explanations | ✅ Done |
| 6 | **Data Versioning** | Track and compare datasets across versions | ✅ Done |

### Phase 3: High Complexity

| # | Feature | Description | Status |
|---|---------|-------------|--------|
| 1 | **ML Pipeline Scheduling** | Add ability to schedule retraining jobs | ⬜ Pending |
| 2 | **Cloud Deployment** | Deploy models to AWS/GCP/Azure with one-click | ⬜ Pending |
| 3 | **Multi-User Collaboration** | Add user accounts, shared workspaces, and team features | ⬜ Pending |
| 4 | **Real-time API Generation** | Auto-generate REST API endpoints for deployed models | ⬜ Pending |

---

## Refactoring (From Previous Analysis)

### High Priority

| # | Task | Description |
|---|------|-------------|
| 1 | Extract `_load_csv_robust` | Move from train.py to helpers.py |
| 2 | Add Logging | Replace silent exception handlers with proper logging |

### Medium Priority

| # | Task | Description |
|---|------|-------------|
| 1 | Model Registry | Split `get_candidate_models` into registry class |

### Low Priority

| # | Task | Description |
|---|------|-------------|
| 1 | Extract CSS | Move CSS to separate file from config.py |
| 2 | Use Dataclasses | Refactor config.py to use dataclasses |

---

## Priority Recommendations

### Top 3 Features to Implement Next

1. **Feature Engineering Module** — Let users create new features automatically
2. **NLP/Text Support** — Expand beyond tabular data
3. **SHAP Explanations** — Make models interpretable for business users

---

## Notes

- This roadmap is a living document
- Features can be reprioritized based on user feedback
- Each feature should include tests before implementation
- Follow existing code patterns from the codebase

---

*Last Updated: March 2026*
