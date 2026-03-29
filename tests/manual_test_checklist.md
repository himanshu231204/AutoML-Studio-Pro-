# Phase 1 Features - Manual Test Checklist

This document provides a checklist for manually verifying UI features that require browser interaction.

## Test Environment Setup

```bash
cd "C:\Users\himan\Desktop\ML app"
streamlit run app.py
```

---

## Feature 1: Sample Datasets

### Test Procedure:
1. Navigate to the **Train & Learn** tab
2. Expand **"Load Sample Dataset"** expander
3. Select each dataset from the dropdown:
   - [ ] Iris Flower
   - [ ] Wine Quality
   - [ ] Breast Cancer
   - [ ] Diabetes
4. Click **"Load Dataset"** button
5. Verify:
   - [ ] Dataset loads without errors
   - [ ] Data preview shows correctly (150, 178, 569, 442 rows respectively)
   - [ ] Target column selector is populated correctly
   - [ ] "Clear Dataset" button works

### Expected Results:
- Success message appears: "✅ Loaded: [Dataset Name]"
- Data preview shows correct number of rows and columns

---

## Feature 2: Dark/Light Theme Toggle

### Test Procedure:
1. Look at the sidebar (left side)
2. Find **"🎨 Theme"** section
3. Click **"🌙 Dark"** button
4. Click **"☀️ Light"** button
5. Observe color changes

### Expected Results:
- [ ] Dark theme: Dark blue background (#0d1722), light text
- [ ] Light theme: Light gray background (#f4f7fb), dark text
- [ ] Current theme indicator updates correctly
- [ ] Theme persists during session

---

## Feature 3: Missing Value Strategy Selector

### Test Procedure:
1. Load a sample dataset (Iris or create one with missing values)
2. Expand **"Missing Value Strategy"** expander
3. Try each option for Numeric Columns:
   - [ ] median
   - [ ] mean
   - [ ] most_frequent
   - [ ] constant
4. Try each option for Categorical Columns:
   - [ ] most_frequent
   - [ ] constant
5. Verify **Preprocessing Pipeline Preview** updates

### Expected Results:
- [ ] Dropdowns are selectable
- [ ] Preview shows selected imputation strategy
- [ ] Strategy is stored in session state

---

## Feature 4: Preprocessing Pipeline Preview

### Test Procedure:
1. Load a sample dataset (Iris)
2. Expand **"Missing Value Strategy"** expander
3. Scroll to **"Preprocessing Pipeline Preview"** section
4. Verify displayed information

### Expected Results:
- [ ] Numeric columns listed with imputation method
- [ ] Categorical columns listed with encoding method
- [ ] "✓ Imputation: [strategy]" displayed
- [ ] "✓ Scaling: StandardScaler" displayed
- [ ] "✓ Encoding: OneHotEncoder" displayed

---

## Feature 5: Cross-Validation Visualization

### Test Procedure:
1. Click **"🚀 Start Training"** button
2. Wait for training to complete
3. Scroll to **"Cross-Validation Performance"** section

### Expected Results:
- [ ] Bar chart shows CV scores for each model
- [ ] Histogram shows CV score distribution
- [ ] Best model highlighted in different color (#16b3a0)
- [ ] Mean line shown on histogram

---

## Feature 6: Model History

### Test Procedure:
1. Train a model (Iris dataset, fast training)
2. Look for **"📜 Model History"** expander after training
3. Expand it
4. Train another model
5. Verify history updates

### Expected Results:
- [ ] Model history expander appears after first training
- [ ] Shows timestamp, model name, score
- [ ] Keeps last 10 entries maximum
- [ ] "Clear History" button works

---

## Feature 7: Model Comparison Dashboard

### Test Procedure:
1. Train at least 2 models
2. Expand **"📜 Model History"** expander
3. Look for **"📊 Model Comparison Dashboard"** section

### Expected Results:
- [ ] Comparison dashboard appears after 2+ models
- [ ] Bar chart compares model scores
- [ ] Best model highlighted in different color (#ff6b6b)
- [ ] Score trend line chart over time

---

## Feature 8: Ensemble Model Builder

### Test Procedure:
1. Expand **"🎛️ Ensemble Model Builder"** expander
2. Check **"Enable Ensemble (Voting Classifier)"**
3. Try each ensemble type:
   - [ ] voting_hard
   - [ ] voting_soft
   - [ ] stacking
4. Train model with ensemble enabled

### Expected Results:
- [ ] Checkbox enables/disables ensemble section
- [ ] Ensemble type dropdown appears when enabled
- [ ] Success message shows "✓ Ensemble enabled: [type]"
- [ ] Ensemble config stored in session state

---

## Feature 9: PDF Report Export

### Test Procedure:
1. Complete a model training
2. Scroll to **"📄 Generate Report"** section
3. Click **"📄 Download Report (.html)"** button

### Expected Results:
- [ ] Download button is clickable
- [ ] File downloads as "automl_report.html"
- [ ] Report contains:
  - [ ] Model name
  - [ ] Task type
  - [ ] Training mode
  - [ ] Time budget
  - [ ] Test score
  - [ ] CV score
  - [ ] Dataset info
  - [ ] Preprocessing steps

---

## Common Issues & Fixes

### Issue: Sample dataset fails to load
- **Fix**: Check sklearn version compatibility in `_load_sample_dataset` function

### Issue: Theme toggle doesn't persist
- **Fix**: Verify `st.session_state["theme_mode"]` is being set correctly

### Issue: Missing Value Strategy not applied
- **Fix**: Ensure `impute_strategy` is passed to `build_preprocessor`

### Issue: Model history not showing
- **Fix**: Check that training completes without errors and `model_history` is initialized

### Issue: Report download fails
- **Fix**: Verify HTML is properly formatted and `st.download_button` parameters are correct

---

## Test Summary

| Feature | Test Type | Status |
|---------|-----------|--------|
| Sample Datasets | Unit + Manual | Pending |
| Dark/Light Theme | Manual | Pending |
| Missing Value Strategy | Unit + Manual | Pending |
| Preprocessing Pipeline Preview | Unit + Manual | Pending |
| Cross-Validation Visualization | Manual | Pending |
| Model History | Unit + Manual | Pending |
| Model Comparison Dashboard | Manual | Pending |
| Ensemble Model Builder | Manual | Pending |
| PDF Report Export | Unit + Manual | Pending |
