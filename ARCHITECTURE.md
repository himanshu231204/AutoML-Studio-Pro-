# Architecture

This document describes the high-level architecture of AutoML Studio Pro.

## System Overview

AutoML Studio Pro is a Streamlit application organized by feature modules. The UI orchestrates data upload, analysis, model training, and prediction workflows.

## Directory Responsibilities

- `app.py`: Entry point, page setup, and tab routing
- `automl_app/core`: Shared configuration and helper functions
- `automl_app/ui/tabs`: Feature modules (training, analysis, prediction, guide, developer)
- `automl_app/ui`: Reusable UI components such as footer and tab containers
- `artifacts`: Persisted outputs such as model pipelines and schema
- `tests`: Unit tests for utility and training-related logic

## Runtime Flow

1. User uploads data and selects target
2. App infers task type and prepares features/labels
3. Preprocessing and model pipeline are built
4. Training and validation metrics are computed
5. Pipeline and schema are persisted for inference
6. Prediction modules consume saved artifacts for single/batch scoring
7. Guide and Developer tabs provide product guidance and portfolio context

## Model Training Layer

- Preprocessing uses Scikit-Learn transformers and pipelines
- Estimation uses a candidate model pool with fast/high-accuracy modes
- Cross-validation aware ranking is combined with holdout evaluation
- For imbalanced classification datasets, oversampling is applied
- Artifacts are serialized with joblib for reproducible inference

## Inference Layer

- Single prediction: dynamic form based on dataset schema
- Batch prediction: CSV upload and scored output table
- Inputs are validated against stored schema to reduce mismatch risk

## Testing Strategy

- Unit tests cover helper functions and training utilities
- Suite includes edge-case and integration scenarios for preprocessing and training flow
- New functionality should include tests for edge cases and regressions

## Future Evolution

Potential next architecture improvements:

- Introduce a service layer to isolate business logic from Streamlit UI
- Add experiment tracking for model/version lineage
- Add stronger schema/version compatibility checks for artifacts
