<div align="center">

# AutoML Studio Pro

**A no-code, end-to-end automated machine learning platform for training, evaluating, and deploying ML models—right from your browser.**

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
- [Suggested Repository Topics](#suggested-repository-topics)
- [Contributing](#contributing)
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

| Category | Feature | Description |
|---|---|---|
| **AutoML Engine** | Automatic Task Detection | Determines whether the problem is classification or regression based on the target column. |
| | Automated Preprocessing | Handles missing values, encoding, and feature scaling via Scikit-Learn pipelines. |
| | Intelligent Model Selection | Trains a `HistGradientBoosting` model optimized for accuracy and speed. |
| | Imbalanced Data Handling | Applies SMOTE oversampling for skewed classification datasets. |
| **Data Insights** | Exploratory Data Analysis | Generates descriptive statistics and correlation heatmaps instantly. |
| | Feature Importance (XAI) | Uses permutation importance to explain model predictions transparently. |
| | Performance Metrics | Displays confusion matrices, accuracy scores, R² scores, and prediction plots. |
| **Export & Deploy** | Model Export | Download trained models as `.zip` archives for reuse. |
| | Python Code Export | Export a ready-to-run `train_model.py` script for learning and customization. |
| | Batch Predictions | Upload CSV files to generate predictions at scale. |
| | Dynamic Prediction Form | Auto-generated input form based on dataset schema for single predictions. |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend / UI | [Streamlit](https://streamlit.io/) |
| Machine Learning | [Scikit-Learn](https://scikit-learn.org/) — HistGradientBoosting, Pipelines |
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

---

## Project Structure

```
AutoML-Studio-Pro-/
├── app.py               # Main Streamlit application
├── artifacts/           # Auto-generated model files and schema
│   ├── final_pipeline.joblib
│   └── app_schema.json
├── requirements.txt     # Python dependencies
├── .gitignore
└── README.md            # Project documentation
```

---

## Usage

1. **Train & Learn** — Upload a CSV file, select the target column, and click **Start Training**. The platform preprocesses data, trains a model, and displays performance metrics along with feature importance charts. Download the trained model or export the equivalent Python code.

2. **Data Analysis** — Explore the uploaded dataset through descriptive statistics and correlation heatmaps before training.

3. **Production Engine** — Load a previously trained model or use the current session model. Make single predictions via a dynamically generated form, or upload a CSV for batch inference.

---

## Suggested Repository Topics

`automl` · `machine-learning` · `no-code` · `streamlit` · `scikit-learn` · `data-science` · `python` · `automated-machine-learning` · `eda` · `classification` · `regression` · `model-training` · `feature-importance` · `smote` · `gradient-boosting`

---

## Contributing

Contributions are welcome! To get started:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a Pull Request.

Please open an [issue](https://github.com/himanshu231204/AutoML-Studio-Pro-/issues) for bug reports or feature requests.

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
