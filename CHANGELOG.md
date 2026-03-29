# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

## [1.3.0] - 2026-03-29

### Added
- **Phase 2 Medium Complexity Features:**
  - Feature Engineering Module: Automated polynomial features, interaction features, and statistical aggregations
  - SHAP/Explainable AI: Model interpretability with SHAP values and feature importance visualization
  - AutoML Hyperparameter Optimization: Optuna integration for advanced hyperparameter tuning
  - NLP/Text Classification: Text preprocessing with TF-IDF and configurable n-gram ranges
  - Time Series Support: ARIMA and Exponential Smoothing models for temporal data forecasting
  - Data Versioning: Track and compare datasets across versions with MD5 hashing

- **New Dependencies:**
  - shap==0.46.0 - SHAP explanations
  - optuna==4.2.1 - Hyperparameter optimization
  - statsmodels==0.14.4 - Time series models

## [1.2.0] - 2026-03-29

### Added
- **Phase 1 Quick Wins Features:**
  - Sample Datasets: Built-in datasets (Iris, Wine, Breast Cancer, Diabetes) for quick demo
  - Dark/Light Theme Toggle: Sidebar theme switcher with session state persistence
  - Missing Value Strategy Selector: Choose imputation methods (median/mean/most_frequent/constant)
  - Preprocessing Pipeline Preview: Visual preview of preprocessing steps before training
  - Cross-Validation Visualization: Bar charts and histograms showing CV scores by model
  - Model History: Track last 10 trained models with timestamps and metrics
  - Model Comparison Dashboard: Side-by-side model comparison with score charts and trends
  - Ensemble Model Builder: UI for voting/stacking ensemble configuration
  - PDF Report Export: HTML report generation with model details and metrics

- **DevOps Improvements:**
  - Enhanced CI workflow with security scanning (safety), linting (ruff), and coverage reporting
  - Fixed CD workflow to trigger on version tags only (v*.*.*)
  - Improved Dockerfile with multi-stage build and health checks
  - Created requirements-dev.txt for development dependencies
  - Added .streamlit/config.toml for Streamlit Community Cloud deployment

- **Testing:**
  - Created comprehensive test suite for Phase 1 features (tests/test_phase1_features.py)
  - Added manual test checklist (tests/manual_test_checklist.md)

### Fixed
- PDF Report Export: Handle None cv_score values gracefully (shows "N/A")
- Duplicate Streamlit element IDs: Added unique keys to all selectbox elements

### Changed
- Updated build_preprocessor() to accept imputation strategy parameters
- Sidebar now expanded by default for theme toggle access
- FEATURES_ROADMAP.md updated with Phase 1 completion status

## [Unreleased]

### Added
- Professional repository and documentation structure:
  - assets/images/badges
  - assets/images/screenshots
  - docs/api
  - docs/guides
  - .github/ISSUE_TEMPLATE
- Community and governance files:
  - LICENSE (MIT)
  - .github/pull_request_template.md
  - .github/ISSUE_TEMPLATE/bug_report.md
  - .github/ISSUE_TEMPLATE/feature_request.md
  - .github/ISSUE_TEMPLATE/config.yml
  - CODE_OF_CONDUCT.md
  - CONTRIBUTING.md
  - ARCHITECTURE.md
- Training UI controls in Train tab:
  - Classification optimization metric selector (Accuracy, F1 Weighted, ROC AUC)
  - Training mode selector (Fast, High Accuracy)
  - Training time budget slider
  - Lightweight hyperparameter tuning toggle
- Automatic data-quality improvements before training:
  - High-correlation numeric feature drop
  - Numeric outlier clipping based on train quantiles
- Classification leaderboard metric columns:
  - accuracy
  - f1
  - roc_auc
- Advanced EDA Analytics with six comprehensive tabs:
  - **📈 Statistics**: Skewness, kurtosis, and detailed statistical summaries
  - **🎯 Target Analysis**: Class balance detection, imbalance warnings, and distribution visualization
  - **🔗 Correlations**: Top feature correlations, correlation matrix heatmap with flexible methods (Pearson/Spearman)
  - **📊 Distributions**: Histogram, KDE, and Q-Q plots for numeric features; category distribution for categorical features
  - **🗂️ Data Quality**: Completeness, uniqueness, consistency, and overall quality scores with actionable insights
  - **🔍 Variance Analysis**: Feature variance contribution pie charts and variance-based feature importance
- Premium UI/UX Enhancements:
  - Full-screen layout (removed max-width constraint)
  - Enhanced hero section with gradient backgrounds and animated overlays
  - Improved quick-card styling with hover animations and shimmer effects
  - Advanced tab styling with accent colors and smooth transitions
  - Enhanced button styling with gradient fills, glow effects, and elevation changes
  - Improved form inputs with focus states and visual feedback
  - Better visual hierarchy with typography and spacing improvements
  - Enhanced footer with social links, badges, and premium styling
  - Smooth animations: fadeSlideIn, shimmer, glow, and pulse effects
  - Responsive design optimizations for mobile and tablet devices
- New Developer Info Tab:
  - Professional developer introduction and biography
  - Social media links (LinkedIn, GitHub, Email)
  - Skills and competencies showcase
  - Featured projects section
  - Professional timeline/journey visualization
  - Education and open-source contribution info
  - Call-to-action with contact options
  - Premium styling with animations and interactive cards
  - Smooth animations: fadeSlideIn, shimmer, glow, and pulse effects
  - Responsive design optimizations for mobile and tablet devices
- Comprehensive Professional User Manual:
  - Getting Started section with 3-step quick start guide
  - Detailed Train & Learn tab guide with feature engineering options
  - Data preprocessing and model training process documentation
  - Leaderboard interpretation and model download options
  - Advanced EDA Analytics with 6 tab explanations (Statistics, Target, Correlations, Distributions, Quality, Variance)
  - Data Analysis tab complete reference with metric definitions
  - Predictions tab guide covering single and batch prediction workflows
  - Model algorithms explanation with strengths, speed, accuracy profiles
  - Classification and regression model comparisons
  - FAQ section addressing common user questions
  - Troubleshooting guide with error solutions and recovery steps
  - Performance optimization tips (fast vs accurate training)
  - Best practices for data preparation, training, prediction
  - Model validation checklist before production deployment
  - Technical architecture and pipeline documentation
  - Scoring methodology and ranking explanations
  - Model artifacts and preprocessing details reference

### Changed
- Preprocessing improved for categorical handling by grouping infrequent categories when supported.
- Candidate model presets strengthened for higher accuracy in high-accuracy mode.
- Added fast-mode model presets for quicker training turnaround.
- Model selection upgraded from simple holdout-only comparison to CV-aware ranking.
- Added metric-aware scoring so selection follows chosen optimization metric.
- Added optional top-model lightweight tuning with time budget support.
- Training summary now shows selected training mode, optimization metric, and budget context.
- Unit test workflow Python matrix updated to 3.11/3.12/3.13 to match dependency constraints (`contourpy==1.3.3` requires Python >= 3.11).
- Developer Info tab updated with verified social media handles:
  - LinkedIn: https://www.linkedin.com/in/himanshu231204/
  - GitHub: https://github.com/himanshu231204
  - Twitter/X: https://x.com/himanshu231204
- Footer social links updated to include Twitter/X and verified all profiles

### Tests
- Updated helper tests for ranking-score-based leaderboard order.
- Added tests for:
  - fast vs high-accuracy candidate model sizing
  - correlated-feature removal utility
  - outlier-clipping utility
- Current status: all tests passing.
