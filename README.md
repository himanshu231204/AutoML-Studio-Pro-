# 🧠 AutoML Studio Pro  
### A No-Code Machine Learning & Learning-Oriented Platform

[![Live App](https://img.shields.io/badge/Live_App-Visit_Now-brightgreen)](https://automl-studio-pro.onrender.com/)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![License](https://img.shields.io/badge/License-MIT-green)


## 📖 Overview
**AutoML Studio Pro** is an end-to-end automated machine learning (AutoML) platform built to simplify the entire data science workflow.  
Users can upload datasets, automatically train optimized ML models, explore insights, run predictions, and export the underlying Python code for educational purposes.

Developed using **Streamlit** and **Scikit-Learn**, the platform streamlines everything—from EDA and preprocessing to model training, evaluation, and deployment-ready inference.

---

## ✨ Key Features

### 🧠 AutoML Intelligence
- **Automatic Task Detection:** Identifies whether the task is Classification or Regression.  
- **Automated Preprocessing:** Handles missing values, encoding, scaling, and data cleanup.  
- **Intelligent Model Selection:** Uses **HistGradientBoosting** for high accuracy and fast training.  
- **Imbalanced Data Support:** Applies **SMOTE** for balanced classification datasets.

### 📊 Data Insights & Explainability
- **Instant EDA:** Generates correlation heatmaps and descriptive statistics.  
- **Feature Importance (XAI):** Uses Permutation Importance for transparent decision-making.  
- **Comprehensive Evaluation:** Confusion matrices, accuracy scores, and regression visualizations included.

### ⚙️ Production & Developer Tools
- **Model Export:** Download trained models as `.zip` files and reload anytime.  
- **Python Code Export:** Download a clean `train_model.py` file for learning and customization.  
- **Bulk Prediction Support:** Upload CSV files to generate thousands of predictions at once.  
- **Dynamic Prediction UI:** Automatically generated input form based on dataset schema.

---

## 🚀 Quick Start (Local Setup)

### Prerequisites
- Python 3.8+
- pip

### Installation

1. **Clone the Repository**
    ```bash
    git clone https://github.com/himanshu231204/automl-studio-pro.git
    cd automl-studio-pro
    ```

2. **Create a Virtual Environment (Optional)**
    ```bash
    # Windows
    python -m venv venv
    venv\Scripts\activate

    # macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3. **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the Application**
    ```bash
    streamlit run app.py
    ```

---

## 🛠️ Tech Stack
- **Frontend:** Streamlit  
- **Machine Learning:** Scikit-Learn (HistGradientBoosting, Pipelines)  
- **Data Processing:** Pandas, NumPy  
- **Visualization:** Matplotlib, Seaborn  
- **Model Serialization:** Joblib  
- **Imbalance Handling:** Imbalanced-Learn (SMOTE)

---

## 📂 Project Structure
```
├── artifacts/          # Auto-generated models & schema files
├── app.py              # Main Streamlit application
├── requirements.txt    # Required Python packages
└── README.md           # Project documentation
```

---

## 🤝 Contributing
Contributions are welcome!  
Please submit an issue or a pull request for feature requests or bug reports.

---

## 📄 License
This project is licensed under the MIT License.  
See the **LICENSE** file for details.

---

## 📬 Connect with Me
- **GitHub:** https://github.com/himanshu231204  
- **LinkedIn:** https://www.linkedin.com/in/himanshu231204/  
- **X (Twitter):** https://x.com/himanshu231204
