# Changelog

All notable changes to this project are documented in this file.

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

### Changed
- Preprocessing improved for categorical handling by grouping infrequent categories when supported.
- Candidate model presets strengthened for higher accuracy in high-accuracy mode.
- Added fast-mode model presets for quicker training turnaround.
- Model selection upgraded from simple holdout-only comparison to CV-aware ranking.
- Added metric-aware scoring so selection follows chosen optimization metric.
- Added optional top-model lightweight tuning with time budget support.
- Training summary now shows selected training mode, optimization metric, and budget context.
- Unit test workflow Python matrix updated to 3.11/3.12/3.13 to match dependency constraints (`contourpy==1.3.3` requires Python >= 3.11).

### Tests
- Updated helper tests for ranking-score-based leaderboard order.
- Added tests for:
  - fast vs high-accuracy candidate model sizing
  - correlated-feature removal utility
  - outlier-clipping utility
- Current status: all tests passing.
