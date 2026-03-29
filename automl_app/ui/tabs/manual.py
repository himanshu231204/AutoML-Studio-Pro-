import streamlit as st


def render_manual_tab() -> None:
    # Overview
    st.markdown(
        """
        # 📘 AutoML Studio Pro - Complete User Manual

        AutoML Studio Pro is an enterprise-grade machine learning platform that automates model training,
        evaluation, and deployment. This guide covers all features and workflows to help you build
        production-ready ML models efficiently.

        **Version**: 1.3.0 | **Last Updated**: March 2026
        """,
        unsafe_allow_html=True
    )

    # Create expandable sections
    with st.expander("🚀 **Getting Started** - New User Checklist", expanded=True):
        st.markdown(
            """
            ### Prerequisites
            - **Clean CSV Dataset**: Your data should have features (X) and target column (y)
            - **Data Format**: Numeric and categorical columns supported
            - **Minimum Records**: At least 30 records recommended for training

            ### Quick Start (3 Steps)

            **Step 1: Prepare Your Data**
            - Ensure your CSV is clean (handle missing values if needed)
            - First row should contain column headers
            - Target column must be clearly labeled
            - **OR** use built-in sample datasets (Iris, Wine, Breast Cancer, Diabetes)

            **Step 2: Train Model**
            - Go to **Train & Learn** tab
            - Upload your CSV file OR load a sample dataset
            - Configure training options (see Training Options below)
            - Click **Start Training** button
            - Wait for training to complete (1-5 minutes depending on data size)

            **Step 3: Make Predictions**
            - Go to **Predictions** tab
            - Upload new data for prediction
            - Download results as CSV

            ✅ **That's it!** You now have a trained production-ready model.
            """
        )

    with st.expander("📂 **Sample Datasets** - Quick Demo", expanded=False):
        st.markdown(
            """
            ### Built-in Datasets

            Don't have data ready? Use our built-in sample datasets to explore the platform:

            | Dataset | Description | Task | Samples | Features |
            |---------|-------------|------|---------|----------|
            | **Iris Flower** | Classic flower classification | Classification | 150 | 4 |
            | **Wine Quality** | Wine characteristics analysis | Classification | 178 | 13 |
            | **Breast Cancer** | Cancer diagnosis prediction | Classification | 569 | 30 |
            | **Diabetes** | Disease progression prediction | Regression | 442 | 10 |

            ### How to Load Sample Datasets

            1. Go to **Train & Learn** tab
            2. Click on **📂 Load Sample Dataset** expander
            3. Select a dataset from the dropdown
            4. Click **Load Dataset** button
            5. The dataset loads instantly with target column pre-selected

            ### When to Use Sample Datasets

            - ✅ Learning how the platform works
            - ✅ Testing new features before using your data
            - ✅ Comparing model performance on known benchmarks
            - ✅ Demonstrating the platform to others
            """
        )

    with st.expander("🎨 **Theme Toggle** - Dark/Light Mode", expanded=False):
        st.markdown(
            """
            ### Customizing the Theme

            AutoML Studio Pro supports both dark and light themes for comfortable viewing:

            **How to Toggle Theme:**
            1. Look at the **sidebar** on the left
            2. Find the **🎨 Theme** section
            3. Click **🌙 Dark** or **☀️ Light** button
            4. The theme changes instantly

            ### Theme Features

            | Theme | Best For | Colors |
            |-------|----------|--------|
            | **Dark Mode** | Low-light environments, reduced eye strain | Deep blues and teals |
            | **Light Mode** | Bright environments, printing | Clean whites and grays |

            ### Theme Persistence

            - Theme choice is saved during your session
            - Refreshing the page resets to dark mode (default)
            - Each user can choose their preferred theme
            """
        )

    with st.expander("📊 **Train & Learn Tab** - Model Training Workflow", expanded=False):
        st.markdown(
            """
            ### Upload Dataset

            1. **Select CSV File**: Click the upload area to choose your dataset
            2. **OR Load Sample**: Use built-in datasets for quick testing
            3. **Auto-Detection**: Task type (Classification/Regression) is detected automatically
            4. **Preview**: First few rows are displayed for verification

            #### Data Requirements
            | Requirement | Details |
            |-------------|---------|
            | Format | CSV (Comma-separated values) |
            | Min Rows | 30 records |
            | Columns | At least 2 (features + target) |
            | Target | Single column with target variable |
            | Missing | Handle before upload or use imputation |

            ### Training Configuration Options

            #### Training Mode
            - 🚀 **Fast Mode**: 4 core models, quick training (15-60 sec)
            - ⚡ **High Accuracy Mode**: 8-10 models, thorough evaluation (60-300+ sec)

            #### Time Budget (15-300 seconds)
            - Limits total training time
            - Useful for large datasets or time constraints

            #### Optimization Metric
            - **Classification**: Accuracy, F1-Weighted, ROC-AUC
            - **Regression**: R² Score, MAE, RMSE

            ### 🔧 Missing Value Strategy

            Choose how to handle missing values in your data:

            | Strategy | Description | Best For |
            |----------|-------------|----------|
            | **median** | Uses median value | Numeric with outliers |
            | **mean** | Uses average value | Numeric, normal distribution |
            | **most_frequent** | Uses mode value | Categorical or numeric |
            | **constant** | Uses fixed value | When missing means something |

            **How to Configure:**
            1. Expand **🔧 Missing Value Strategy** section
            2. Select strategy for **Numeric Columns**
            3. Select strategy for **Categorical Columns**
            4. Preview shows which columns will be affected

            ### 🔍 Preprocessing Pipeline Preview

            Before training, see exactly what preprocessing will be applied:

            **Numeric Columns:**
            - Imputation method (your choice)
            - StandardScaler for normalization

            **Categorical Columns:**
            - Imputation method (your choice)
            - OneHotEncoder for encoding

            This preview helps you understand and trust the preprocessing steps.

            ### 📈 Cross-Validation Visualization

            After training, view detailed CV performance:

            **Bar Chart**: Compare CV scores across all models
            - Best model highlighted in green
            - Easy visual comparison

            **Distribution Histogram**: See score spread
            - Shows how consistent models are
            - Mean line indicates average performance

            ### Hyperparameter Tuning
            - Enable to optimize top 1-2 models
            - Budget-aware (uses 30% of time budget)

            ### Model Training Process

            1. **Data Splitting**: 80% train, 20% test
            2. **Cross-Validation**: 5-fold CV for robust evaluation
            3. **Multiple Algorithms**: Tested in parallel
            4. **Metric Evaluation**: Ranked by selected metric
            5. **Best Model Selected**: Automatically chosen based on CV + holdout score

            ### Understanding the Leaderboard

            | Column | Meaning |
            |--------|---------|
            | Rank | Position based on score |
            | Model | Algorithm name |
            | CV Score | Cross-validation performance |
            | Test Score | Holdout test set performance |
            | Training Time | Seconds to train model |
            | Accuracy/F1/ROC | Classification metrics |

            🏆 **Top model** (Rank 1) is automatically saved as final model

            ### Download Options

            **Save Model (.zip)**
            - Contains trained model artifact
            - Reloadable for future predictions
            - Includes schema and preprocessing info

            **Export Code (Python)**
            - Training pipeline as Python script
            - Reproducible and modifiable
            - No platform dependency

            **📄 Generate Report (HTML)**
            - Comprehensive training report
            - Model details and metrics
            - Dataset information
            - Print to PDF for sharing
            """
        )

    with st.expander("🔧 **Feature Engineering** - Create New Features", expanded=False):
        st.markdown(
            """
            ### Automated Feature Engineering

            Automatically create new features to improve model performance:

            #### Polynomial Features
            - Creates polynomial combinations of numeric features
            - Captures non-linear relationships
            - **Options**: Degree 2 or 3
            - **Example**: If you have `age`, creates `age²`, `age³`

            **When to Use:**
            - When relationships might be non-linear
            - For regression problems
            - When you suspect quadratic/cubic effects

            #### Interaction Features
            - Creates products of feature pairs
            - Captures combined effects
            - **Options**: 2-5 interaction features
            - **Example**: `age x income`, `height x weight`

            **When to Use:**
            - When features might interact
            - For classification with complex boundaries
            - Domain knowledge suggests interactions

            #### Statistical Aggregations
            - Creates row-wise statistics
            - **Features Created**: row_mean, row_std, row_max, row_min
            - Useful for datasets with many similar features

            **When to Use:**
            - Datasets with many numeric columns
            - When overall patterns matter
            - For anomaly detection

            ### How to Enable Feature Engineering

            1. Expand **🔧 Feature Engineering** section
            2. Check the features you want to create
            3. Configure options (degree, max interactions)
            4. Preview shows how many features will be added
            5. Train as normal - features are created automatically
            """
        )

    with st.expander("🧠 **Advanced AutoML** - Optuna Optimization", expanded=False):
        st.markdown(
            """
            ### Optuna Hyperparameter Optimization

            Use advanced optimization to find the best model parameters:

            #### What is Optuna?
            - State-of-the-art hyperparameter optimization library
            - Uses Bayesian optimization for efficient search
            - Automatically finds optimal parameters

            #### Configuration Options

            | Option | Description | Recommended |
            |--------|-------------|-------------|
            | **Number of Trials** | How many parameter combinations to try | 30 |
            | **Timeout** | Maximum time in seconds | 60 |

            #### How It Works

            1. Selects top models from initial training
            2. Runs Optuna optimization on each
            3. Finds best hyperparameters
            4. Retrains with optimized parameters
            5. Updates leaderboard with improved scores

            ### When to Use Optuna

            - ✅ When you need maximum accuracy
            - ✅ For production models
            - ✅ When you have time to wait
            - ❌ Skip for quick exploratory analysis

            ### How to Enable

            1. Expand **🧠 Advanced AutoML (Optuna)** section
            2. Check **Enable Optuna Optimization**
            3. Set number of trials (10-100)
            4. Set timeout (30-300 seconds)
            5. Train as normal - Optuna runs automatically
            """
        )

    with st.expander("🎛️ **Ensemble Model Builder**", expanded=False):
        st.markdown(
            """
            ### Combine Multiple Models

            Ensemble methods combine predictions from multiple models for better performance:

            #### Ensemble Types

            | Type | Description | Best For |
            |------|-------------|----------|
            | **Voting Hard** | Majority vote of predictions | Classification |
            | **Voting Soft** | Average of probabilities | Classification |
            | **Stacking** | Meta-learner combines predictions | Complex problems |

            #### How Ensembles Work

            **Voting (Hard/Soft):**
            - Each model votes for the prediction
            - Final prediction = majority vote (hard) or average probability (soft)
            - Reduces variance and improves stability

            **Stacking:**
            - Train multiple base models
            - Train a meta-model on base model predictions
            - Meta-model learns to combine models optimally

            ### When to Use Ensembles

            - ✅ When single models have similar performance
            - ✅ For production systems needing robustness
            - ✅ When you can afford extra training time
            - ❌ Skip for quick prototyping

            ### How to Enable

            1. Expand **🎛️ Ensemble Model Builder** section
            2. Check **Enable Ensemble**
            3. Select ensemble type
            4. Train as normal - ensemble is created automatically
            """
        )

    with st.expander("📝 **NLP/Text Classification**", expanded=False):
        st.markdown(
            """
            ### Text-Based Classification

            Handle text data with automatic TF-IDF preprocessing:

            #### What is TF-IDF?
            - **Term Frequency**: How often a word appears
            - **Inverse Document Frequency**: How unique a word is
            - Converts text to numeric features for ML

            #### Configuration Options

            | Option | Description | Default |
            |--------|-------------|---------|
            | **Text Columns** | Select which columns contain text | Auto-detected |
            | **Max Features** | Maximum TF-IDF features to create | 1000 |
            | **N-gram Range** | Word combinations to consider | (1,2) |

            #### N-gram Examples

            | N-gram | Example Input | Features Created |
            |--------|---------------|------------------|
            | (1,1) | "good product" | "good", "product" |
            | (1,2) | "good product" | "good", "product", "good product" |
            | (1,3) | "not very good" | "not", "very", "good", "not very", "very good", "not very good" |

            ### When to Use NLP Mode

            - ✅ Text classification tasks
            - ✅ Sentiment analysis
            - ✅ Document categorization
            - ✅ Spam detection
            - ❌ Skip for purely numeric data

            ### How to Enable

            1. Expand **📝 NLP/Text Classification** section
            2. Check **Enable Text Processing**
            3. Select text columns (auto-detected)
            4. Configure max features and n-gram range
            5. Train as normal - TF-IDF is applied automatically
            """
        )

    with st.expander("📈 **Time Series Forecasting**", expanded=False):
        st.markdown(
            """
            ### Forecast Future Values

            Predict future values based on historical time series data:

            #### Supported Models

            | Model | Description | Best For |
            |-------|-------------|----------|
            | **ARIMA** | AutoRegressive Integrated Moving Average | Trending data |
            | **Exponential Smoothing** | Weighted average of past values | Seasonal data |

            #### Configuration Options

            | Option | Description | Default |
            |--------|-------------|---------|
            | **Date Column** | Column containing dates | Auto-detected |
            | **Forecast Periods** | How many periods to predict | 7 |
            | **Model** | ARIMA or Exponential Smoothing | ARIMA |

            ### How Time Series Works

            1. Detects date column automatically
            2. Sorts data by date
            3. Fits selected model to historical data
            4. Generates forecasts for specified periods
            5. Displays forecast with confidence intervals

            ### When to Use Time Series

            - ✅ Sales forecasting
            - ✅ Stock price prediction
            - ✅ Demand planning
            - ✅ Resource allocation
            - ❌ Skip for non-temporal data

            ### How to Enable

            1. Expand **📈 Time Series Forecasting** section
            2. Check **Enable Time Series Mode**
            3. Select date column
            4. Set forecast periods
            5. Choose model type
            6. Train as normal - forecasting runs automatically
            """
        )

    with st.expander("📁 **Data Versioning**", expanded=False):
        st.markdown(
            """
            ### Track Dataset Versions

            Keep track of different versions of your datasets:

            #### What is Data Versioning?
            - Saves metadata about each dataset version
            - Tracks changes over time
            - Enables comparison between versions

            #### Tracked Information

            | Field | Description |
            |-------|-------------|
            | **Timestamp** | When version was saved |
            | **Name** | Version identifier |
            | **Rows** | Number of records |
            | **Columns** | Number of features |
            | **Hash** | MD5 checksum for uniqueness |
            | **Column Names** | List of all columns |

            ### How to Use Data Versioning

            **Save a Version:**
            1. Expand **📁 Data Versioning** section
            2. Click **💾 Save Current Dataset Version**
            3. Version is saved with timestamp and hash
            4. View saved versions in the list

            **Compare Versions:**
            - View row/column counts across versions
            - Check if columns changed
            - Identify data growth over time

            **Clear History:**
            - Click **Clear History** to remove all saved versions
            - Does not affect your actual data

            ### When to Use Data Versioning

            - ✅ Tracking data changes over time
            - ✅ Comparing model performance across datasets
            - ✅ Auditing data lineage
            - ✅ Collaborative environments
            """
        )

    with st.expander("🔍 **SHAP Explainable AI**", expanded=False):
        st.markdown(
            """
            ### Understand Model Predictions

            SHAP (SHapley Additive exPlanations) explains how your model makes predictions:

            #### What is SHAP?
            - Game theory approach to explain ML predictions
            - Shows contribution of each feature to predictions
            - Provides both local (single prediction) and global (overall) explanations

            #### SHAP Visualization

            **Feature Importance Bar Chart:**
            - Shows average impact of each feature
            - Longer bars = more important features
            - Color indicates feature value (high/low)

            ### How to Interpret SHAP Values

            | Value | Meaning |
            |-------|---------|
            | **Positive SHAP** | Feature pushes prediction higher |
            | **Negative SHAP** | Feature pushes prediction lower |
            | **Large magnitude** | Strong influence on prediction |
            | **Small magnitude** | Weak influence on prediction |

            ### When to Use SHAP

            - ✅ Understanding why model makes certain predictions
            - ✅ Building trust with stakeholders
            - ✅ Identifying biased or unfair predictions
            - ✅ Feature selection insights
            - ❌ Skip for quick exploratory analysis

            ### How to Enable

            1. After training completes, expand **🔍 SHAP Explainable AI** section
            2. Check **Enable SHAP Analysis**
            3. Wait for SHAP values to be calculated
            4. View the feature importance visualization

            ### Limitations

            - SHAP can be slow for large datasets
            - Requires tree-based models (RandomForest, XGBoost, etc.)
            - May not work with all model types
            """
        )

    with st.expander("📜 **Model History & Comparison**", expanded=False):
        st.markdown(
            """
            ### Track Trained Models

            Keep history of all models trained in your session:

            #### What is Tracked

            | Field | Description |
            |-------|-------------|
            | **Timestamp** | When model was trained |
            | **Model** | Algorithm name |
            | **Score** | Test set performance |
            | **CV Score** | Cross-validation score |
            | **Task** | Classification or Regression |
            | **Dataset Size** | Rows and columns |
            | **Training Mode** | Fast or High Accuracy |

            ### Model Comparison Dashboard

            When you train 2+ models, the comparison dashboard appears:

            **Bar Chart Comparison:**
            - Side-by-side model scores
            - Best model highlighted
            - Easy visual comparison

            **Score Trend Over Time:**
            - Line chart showing score progression
            - Helps identify if later models are better
            - Shows training improvement over time

            ### How to Use Model History

            **View History:**
            1. After training, expand **📜 Model History** section
            2. See table of all trained models
            3. Compare scores and metrics

            **Clear History:**
            - Click **Clear History** to remove all entries
            - Does not affect saved models

            ### When to Use Model History

            - ✅ Comparing different training configurations
            - ✅ Tracking improvement over iterations
            - ✅ Documenting model experiments
            - ✅ Selecting best model for production
            """
        )

    with st.expander("📈 **Data Analysis Tab** - Advanced EDA", expanded=False):
        st.markdown(
            """
            ### Basic Analysis Section

            **Dataset Health Metrics**
            - **Rows**: Total number of records
            - **Columns**: Feature count
            - **Missing Cells**: Count of null values
            - **Duplicate Rows**: Repeated records
            - **Memory**: Dataset size in MB

            **Column Profile**
            - Data type for each column
            - Missing value percentage
            - Unique value count
            - Sample values preview

            **Missing Values Diagnostics**
            - Heatmap showing missing patterns
            - Top 20 columns with missing data
            - Recommendations for handling

            **Numeric Analysis**
            - Descriptive statistics (mean, median, std)
            - Percentiles (1%, 25%, 50%, 75%, 99%)
            - Outlier summary using IQR method

            **Categorical Analysis**
            - Unique value count per column
            - Top N categories visualization
            - Class distribution charts

            **Correlation Analysis**
            - Pearson/Spearman correlation methods
            - Top-variance feature selection
            - Heatmap visualization

            ### Advanced EDA Analytics (6 Tabs)

            #### 📈 Statistics Tab
            - **Skewness**: Measure of distribution asymmetry
              - 0-0.5: Fairly symmetric ✅
              - 0.5-1: Moderately skewed ⚠️
              - >1: Highly skewed (may need transformation)
            - **Kurtosis**: Tail weight indicator
              - 0: Normal distribution
              - >0: Heavy tails (outliers present)
              - <0: Light tails (few outliers)

            #### 🎯 Target Analysis Tab
            - Class balance assessment
            - Dominance ratios
            - Imbalance warnings (>3:1 ratio)
            - Recommendations for SMOTE/class weights

            #### 🔗 Correlations Tab
            - Top feature correlations
            - Full correlation matrix heatmap
            - Identifies collinearity issues
            - Helps with feature selection

            #### 📊 Distributions Tab
            - Histogram with KDE
            - Q-Q plots for normality
            - Categorical value distributions
            - Helps detect non-normal data

            #### 🗂️ Data Quality Tab
            - **Completeness** (0-100%): Missing data ratio
            - **Uniqueness** (0-100%): Duplicate detection
            - **Consistency** (0-100%): Outlier assessment
            - **Overall Quality Score**: Weighted average
            - Actionable insights and recommendations

            #### 🔍 Variance Analysis Tab
            - Feature importance by variance
            - Pie chart of variance distribution
            - Identifies low-variance features
            - Guides feature selection decisions

            ### When to Use This Tab

            ✅ Before training to understand data quality
            ✅ Identify preprocessing needs
            ✅ Detect outliers and missing patterns
            ✅ Make feature engineering decisions
            ✅ Verify data integrity
            """
        )

    with st.expander("🔮 **Predictions Tab** - Model Deployment", expanded=False):
        st.markdown(
            """
            ### Single Prediction

            Use the form to enter values for one record:
            1. Fill in feature values for your input
            2. Click **Predict** button
            3. See predicted value with confidence

            **Output Includes**
            - Predicted value/class
            - Confidence score (0-1)
            - Prediction timestamp
            - Model version used

            ### Batch Prediction

            Upload CSV with multiple records:
            1. Prepare CSV matching training features (same column names/order)
            2. Click **Choose File** under Batch Mode
            3. Click **Predict Batch**
            4. Download results as CSV with predictions

            **Batch Processing**
            - Process 100s-1000s of records
            - Fast parallel processing
            - Returns results with original data + predictions
            - Perfect for production workflows

            ### Model Management

            **Current Model**
            - Uses model trained in this session
            - Shows training date and accuracy

            **Load Pre-trained Model**
            - Upload previously saved .zip file
            - Restores full model state
            - Useful for production environments

            ### Best Practices

            ⚠️ **Match Training Data Format**
            - Same column names and types
            - Same feature engineering applied automatically
            - Outliers will be clipped as per training

            📌 **Confidence Thresholds**
            - Classification: Use probability scores to set decision threshold
            - Regression: Check predictions for outliers

            ✅ **Validation**
            - Test with known samples first
            - Verify predictions align with domain knowledge
            """
        )

    with st.expander("🤖 **Model Algorithms Explained**", expanded=False):
        st.markdown(
            """
            ### Classification Models

            #### HistGradientBoosting 🌟
            - **Best For**: General classification, mixed tabular data
            - **Speed**: Medium (handles missing values natively)
            - **Accuracy**: Excellent (often top performer)
            - **When to Use**: Default choice for most problems
            - Advantages: Robust, fast, handles mixed types

            #### RandomForest / ExtraTrees
            - **Best For**: Complex non-linear patterns, noisy data
            - **Speed**: Medium
            - **Accuracy**: Very Good
            - **When to Use**: When feature interactions matter
            - Advantages: Interpretable, handles outliers well

            #### XGBoost
            - **Best For**: Structured/tabular data, competitions
            - **Speed**: Medium-Fast
            - **Accuracy**: Excellent
            - **When to Use**: When maximum accuracy needed
            - Advantages: Regularization, handles missing values

            #### LogisticRegression
            - **Best For**: Linear separable data, baseline models
            - **Speed**: ⚡ Fast
            - **Accuracy**: Good for simple patterns
            - **When to Use**: Interpretability needed, quick baseline
            - Advantages: Simple, fast, provides probabilities

            #### KNeighborsClassifier (KNN)
            - **Best For**: Small-medium datasets (<50k rows)
            - **Speed**: Slow on large data
            - **Accuracy**: Good for non-linear boundaries
            - **When to Use**: Flexible decision boundaries needed
            - Advantages: Non-parametric, intuitive

            ### Regression Models

            #### HistGradientBoosting (Regressor)
            - **Best For**: General regression, any sized dataset
            - **Speed**: Medium
            - **Accuracy**: Excellent predictions
            - **When to Use**: Default regression choice

            #### XGBoostRegressor
            - **Best For**: Structured data, high accuracy needed
            - **Speed**: Medium-Fast
            - **Accuracy**: Excellent
            - **When to Use**: Competition-grade predictions

            #### RandomForestRegressor
            - **Best For**: Non-linear relationships
            - **Speed**: Medium
            - **Accuracy**: Very Good
            - **When to Use**: Complex patterns

            #### Ridge / LinearRegression
            - **Best For**: Linear relationships
            - **Speed**: ⚡ Very Fast
            - **Accuracy**: Good for simple relationships
            - **When to Use**: Quick baseline

            ### Model Selection Strategy

            The app automatically:
            1. Trains multiple models in parallel
            2. Evaluates on cross-validation + holdout set
            3. Ranks by your chosen metric
            4. Selects top performer
            5. Optionally runs Optuna optimization
            6. Creates ensemble if enabled

            **You get the ensemble benefit of testing all models!**
            """
        )

    with st.expander("❓ **FAQ & Troubleshooting**", expanded=False):
        st.markdown(
            """
            ### FAQ

            **Q: What's the minimum dataset size?**
            A: 30+ records recommended. Smaller datasets may need careful validation to avoid overfitting.

            **Q: Can I use categorical data?**
            A: Yes! Both numeric and categorical columns are automatically handled.

            **Q: What's the maximum file size?**
            A: Depends on available memory. Typically 100MB+ CSVs work fine.

            **Q: How accurate will my model be?**
            A: Depends on data quality and complexity. The app optimizes for your metric choice.

            **Q: Can I compare models?**
            A: Yes! Leaderboard shows all models ranked by your optimization metric. Model History tracks all trained models.

            **Q: How do I reproduce training?**
            A: Download exported Python code for full reproducibility.

            **Q: What is SHAP and why should I use it?**
            A: SHAP explains how your model makes predictions by showing feature contributions. Use it to build trust and understand model behavior.

            **Q: Can I do text classification?**
            A: Yes! Enable NLP/Text Classification to use TF-IDF preprocessing for text data.

            **Q: Does it support time series forecasting?**
            A: Yes! Enable Time Series mode to use ARIMA or Exponential Smoothing for temporal data.

            **Q: What's the difference between Fast and High Accuracy mode?**
            A: Fast mode uses 4 models for quick results. High Accuracy uses 8-10 models with more thorough evaluation.

            ### Troubleshooting

            **❌ "Training Failed" Error**
            - Check CSV format and column names
            - Ensure target column is present
            - Look for NaN/Inf values
            - Try with smaller dataset first

            **❌ Very Low Accuracy**
            - Check if features are informative
            - Look for class imbalance warnings
            - Verify target column is correct
            - May need more/better data

            **❌ "Out of Memory" Error**
            - Use Fast mode instead of High Accuracy
            - Reduce dataset size
            - Drop unnecessary features
            - Try different approach

            **❌ Predictions Look Wrong**
            - Verify input data format matches training
            - Check for outlier values
            - Confirm model accuracy on test data
            - Send sample through batch mode

            **❌ File Upload Issues**
            - Ensure CSV is properly formatted
            - Check file size (<1GB)
            - Verify no special encoding issues
            - Try re-saving from Excel as CSV

            **❌ SHAP Analysis Failed**
            - SHAP requires tree-based models
            - Try with smaller dataset
            - Ensure model trained successfully

            ### Performance Tips

            ⚡ **Faster Training**
            - Use Fast mode (Quick Accuracy)
            - Reduce time budget (lower timeout)
            - Use smaller dataset sample
            - Disable hyperparameter tuning

            🎯 **Better Accuracy**
            - Use High Accuracy mode
            - Enable Optuna optimization
            - Enable hyperparameter tuning
            - Use longer time budget
            - Enable Feature Engineering
            - Clean/validate data first
            - Check advanced EDA before training

            📊 **Better Data Quality**
            - Handle missing values thoughtfully
            - Remove obvious duplicates
            - Check column types are correct
            - Use Data Analysis tab before training
            - Choose appropriate imputation strategy
            """
        )

    with st.expander("💡 **Best Practices & Tips**", expanded=False):
        st.markdown(
            """
            ### Data Preparation Best Practices

            ✅ **DO:**
            - Clean data before upload (remove NaN, handle duplicates)
            - Use consistent column naming
            - Verify target variable is correct
            - Check data types (numeric vs categorical)
            - Remove personally identifiable information (PII)
            - Use sample datasets to learn the platform

            ❌ **DON'T:**
            - Upload raw, unprocessed data
            - Use same data for multiple target columns
            - Include ID columns as features
            - Mix units (e.g., different currencies)

            ### Training Best Practices

            ✅ **DO:**
            - Use Advanced EDA before training
            - Start with Fast mode, then High Accuracy
            - Try both Accuracy and F1 as metrics
            - Examine leaderboard carefully
            - Export code for reproducibility
            - Enable Feature Engineering for complex data
            - Use SHAP to understand model decisions
            - Track experiments with Model History

            ❌ **DON'T:**
            - Ignore class imbalance warnings
            - Use test set in training
            - Expect 100% accuracy on all data
            - Ignore low metric scores
            - Skip data analysis before training

            ### Prediction Best Practices

            ✅ **DO:**
            - Test with known samples first
            - Use batch mode for production
            - Check confidence scores
            - Monitor prediction quality over time
            - Retrain with new data periodically
            - Use Data Versioning to track changes

            ❌ **DON'T:**
            - Use stale models without retraining
            - Send data outside training range
            - Ignore prediction confidence scores
            - Change feature meanings between versions

            ### Model Validation

            **Before using in production:**
            1. Check Leaderboard - Is accuracy reasonable?
            2. Review Export Code - Understand what model does
            3. Do Manual Testing - Verify with known samples
            4. Check Feature Types - Are they reasonable?
            5. Save Model - Download .zip for backup
            6. Generate Report - Document training results
            7. Use SHAP - Understand feature importance

            ### When to Retrain

            🔄 **Retrain when:**
            - New data becomes available (monthly/quarterly)
            - Model accuracy drops in production
            - Data distribution changes
            - Business requirements change
            - New features become available
            - Using Data Versioning to track changes
            """
        )

    with st.expander("🔧 **Technical Details & Architecture**", expanded=False):
        st.markdown(
            """
            ### Model Training Pipeline

            ```
            1. Data Upload / Sample Dataset Load
                ↓
            2. Task Detection (Classification/Regression)
                ↓
            3. Data Validation & Preprocessing
                ├─ Handle Missing Values (user-selected strategy)
                ├─ Remove High-Correlation Features
                ├─ Clip Outliers (1st-99th percentile)
                └─ Feature Engineering (if enabled)
                    ├─ Polynomial Features
                    ├─ Interaction Features
                    └─ Statistical Aggregations
                ↓
            4. Train-Test Split (80:20)
                ↓
            5. 5-Fold Cross-Validation
                ↓
            6. Parallel Model Training
                ├─ HistGradientBoosting
                ├─ RandomForest/ExtraTrees
                ├─ XGBoost
                ├─ LogisticRegression
                └─ KNeighborsClassifier
                ↓
            7. Metric Evaluation & Ranking
                ↓
            8. Optional Hyperparameter Tuning
                ├─ Lightweight Tuning
                └─ Optuna Optimization (if enabled)
                ↓
            9. Ensemble Creation (if enabled)
                ├─ Voting (Hard/Soft)
                └─ Stacking
                ↓
            10. SHAP Analysis (if enabled)
                ↓
            11. Best Model Selection
                ↓
            12. Artifact Packaging & Export
            ```

            ### Scoring Methodology

            **Classification Metrics**
            - Accuracy: Percentage of correct predictions
            - F1-Score: Harmonic mean of precision & recall
            - ROC-AUC: Area under ROC curve (0-1 scale)

            **Regression Metrics**
            - R² Score: Proportion of variance explained
            - MAE: Mean absolute error
            - RMSE: Root mean squared error

            **Ranking Formula**
            - Combined Score = 0.7*CV_Score + 0.3*Holdout_Score
            - Balances generalization and test performance

            ### Model Artifacts

            **Saved .zip Contains**
            - `final_pipeline.joblib` - Trained model
            - `app_schema.json` - Feature schema
            - `label_encoder.joblib` - Label encoder (if applicable)
            - Feature preprocessing information
            - Model metadata

            ### Preprocessing Details

            **Automatic Preprocessing**
            - Categorical: OneHotEncoder + grouping rare categories
            - Numeric: StandardScaler for normalization
            - Missing: User-selected imputation strategy
            - Outliers: Quantile-based clipping

            **Feature Engineering (Optional)**
            - Polynomial: sklearn PolynomialFeatures
            - Interactions: Manual feature products
            - Aggregations: Row-wise statistics

            **NLP Preprocessing (Optional)**
            - TF-IDF vectorization
            - Configurable n-gram ranges
            - Max features limit

            ### Tech Stack

            | Component | Technology |
            |-----------|------------|
            | Frontend | Streamlit |
            | ML Framework | Scikit-Learn |
            | Optimization | Optuna |
            | Explainability | SHAP |
            | Time Series | Statsmodels |
            | Data Processing | Pandas, NumPy |
            | Visualization | Matplotlib, Seaborn |
            """
        )

    # Footer with support info
    st.markdown(
        """
        ---

        ### 📞 Need Help?

        - 📖 Read relevant sections above using expandable items
        - 🔍 Check **Data Analysis** tab to understand your data
        - 🐞 Review error messages carefully - they guide next steps
        - 💾 Always save your trained model (.zip) for later use
        - 📄 Generate reports to document your training results

        ### 📚 Learn More

        - **Getting Started**: See "Getting Started" section
        - **Sample Data**: Check "Sample Datasets" for quick demos
        - **Training Options**: Review "Train & Learn Tab" section
        - **Feature Engineering**: See "Feature Engineering" section
        - **Advanced ML**: Check "Advanced AutoML" and "SHAP" sections
        - **Model Selection**: See "Model Algorithms Explained"
        - **Data Issues**: Check "FAQ & Troubleshooting"
        - **Production Use**: Review "Best Practices & Tips"
        - **Technical Setup**: See "Technical Details & Architecture"

        **Version**: 1.3.0 | **Last Updated**: March 2026 | **Status**: Production Ready ✅
        """
    )
