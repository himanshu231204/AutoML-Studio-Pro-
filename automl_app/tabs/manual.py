import streamlit as st


def render_manual_tab() -> None:
    # Overview
    st.markdown(
        """
        # 📘 AutoML Studio Pro - Complete User Manual
        
        AutoML Studio Pro is an enterprise-grade machine learning platform that automates model training, 
        evaluation, and deployment. This guide covers all features and workflows to help you build 
        production-ready ML models efficiently.
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
            
            **Step 2: Train Model**
            - Go to **Train & Learn** tab
            - Upload your CSV file
            - Click **Train Model** button
            - Wait for training to complete (1-5 minutes depending on data size)
            
            **Step 3: Make Predictions**
            - Go to **Predictions** tab
            - Upload new data for prediction
            - Download results as CSV
            
            ✅ **That's it!** You now have a trained production-ready model.
            """
        )
    
    with st.expander("📊 **Train & Learn Tab** - Model Training Workflow", expanded=False):
        st.markdown(
            """
            ### Upload Dataset
            
            1. **Select CSV File**: Click the upload area to choose your dataset
            2. **Auto-Detection**: Task type (Classification/Regression) is detected automatically
            3. **Preview**: First few rows are displayed for verification
            
            #### Data Requirements
            | Requirement | Details |
            |-------------|---------|
            | Format | CSV (Comma-separated values) |
            | Min Rows | 30 records |
            | Columns | At least 2 (features + target) |
            | Target | Single column with target variable |
            | Missing | Handle before upload or use imputation |
            
            ### Feature Engineering Options
            
            #### Automatic Preprocessing
            - **Correlation Filtering**: Removes highly correlated features (>0.98)
            - **Outlier Clipping**: Uses 1st-99th percentile bounds
            - **Categorical Encoding**: Automatic grouping of rare categories
            - **Missing Value Handling**: Smart imputation strategies
            
            #### Manual Controls
            
            **Training Mode**
            - 🚀 **Fast Mode**: 4 core models, quick training (15-60 sec)
            - ⚡ **High Accuracy Mode**: 8-10 models, thorough evaluation (60-300+ sec)
            
            **Time Budget** (15-300 seconds)
            - Limits total training time
            - Useful for large datasets or time constraints
            
            **Optimization Metric**
            - **Classification**: Accuracy, F1-Weighted, ROC-AUC
            - **Regression**: R² Score, MAE, RMSE
            
            **Hyperparameter Tuning**
            - Enable to optimize top 1-2 models
            - Budget-aware (uses 30% of time budget)
            
            ### Model Training Process
            
            1. **Data Splitting**: 70% train, 20% validation, 10% test
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
            A: Yes! Leaderboard shows all models ranked by your optimization metric.
            
            **Q: How do I reproduce training?**
            A: Download exported Python code for full reproducibility.
            
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
            
            ### Performance Tips
            
            ⚡ **Faster Training**
            - Use Fast mode (Quick Accuracy)
            - Reduce time budget (lower timeout)
            - Use smaller dataset sample
            - Disable hyperparameter tuning
            
            🎯 **Better Accuracy**
            - Use High Accuracy mode
            - Enable hyperparameter tuning
            - Use longer time budget
            - Clean/validate data first
            - Check advanced EDA before training
            
            📊 **Better Data Quality**
            - Handle missing values thoughtfully
            - Remove obvious duplicates
            - Check column types are correct
            - Use Data Analysis tab before training
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
            
            ❌ **DON'T:**
            - Ignore class imbalance warnings
            - Use test set in training
            - Expect 100% accuracy on all data
            - Ignore low metric scores
            
            ### Prediction Best Practices
            
            ✅ **DO:**
            - Test with known samples first
            - Use batch mode for production
            - Check confidence scores
            - Monitor prediction quality over time
            - Retrain with new data periodically
            
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
            
            ### When to Retrain
            
            🔄 **Retrain when:**
            - New data becomes available (monthly/quarterly)
            - Model accuracy drops in production
            - Data distribution changes
            - Business requirements change
            - New features become available
            """
        )
    
    with st.expander("🔧 **Technical Details & Architecture**", expanded=False):
        st.markdown(
            """
            ### Model Training Pipeline
            
            ```
            1. Data Upload
                ↓
            2. Task Detection (Classification/Regression)
                ↓
            3. Data Validation & Preprocessing
                ├─ Handle Missing Values
                ├─ Remove High-Correlation Features
                └─ Clip Outliers (1st-99th percentile)
                ↓
            4. Train-Validation-Test Split (70:20:10)
                ↓
            5. 5-Fold Cross-Validation
                ↓
            6. Parallel Model Training
                ├─ HistGradientBoosting
                ├─ RandomForest/ExtraTrees
                ├─ LogisticRegression
                └─ KNeighborsClassifier
                ↓
            7. Metric Evaluation & Ranking
                ↓
            8. Optional Hyperparameter Tuning
                ↓
            9. Best Model Selection
                ↓
            10. Artifact Packaging & Export
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
            - Combined Score = 0.7×CV_Score + 0.3×Holdout_Score
            - Balances generalization and test performance
            
            ### Model Artifacts
            
            **Saved .zip Contains**
            - `final_pipeline.joblib` - Trained model
            - `app_schema.json` - Feature schema
            - Feature preprocessing information
            - Model metadata
            
            ### Preprocessing Details
            
            **Automatic Preprocessing**
            - Categorical: Label encoding + grouping rare categories
            - Numeric: Standardization for linear models
            - Missing: Mean/median imputation
            - Outliers: Quantile-based clipping
            
            **No Manual Configuration Needed** - All handled automatically!
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
        
        ### 📚 Learn More
        
        - **Model Selection**: See "Model Algorithms Explained"
        - **Data Issues**: Check "FAQ & Troubleshooting"
        - **Production Use**: Review "Best Practices & Tips"
        - **Technical Setup**: See "Technical Details & Architecture"
        
        **Version**: 1.0 | **Last Updated**: 2025 | **Status**: Production Ready ✅
        """
    )

