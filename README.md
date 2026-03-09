# 🤖 Machine Learning Models Project

Welcome to the **Machine Learning Models** project! This repository is a curated collection of lessons and hands-on code examples used at **Yared Coding School** to teach students the fundamentals of Machine Learning using Python.

---

## 📚 Table of Contents

1. [Introduction](#-introduction)
2. [Prerequisites](#-prerequisites)
3. [Installation](#-installation)
4. [Curriculum Overview](#-curriculum-overview)
    - [Class 1: Simple Linear Regression](#class-1-simple-linear-regression)
    - [Class 2: Multivariate Regression & Polynomial Features](#class-2-multivariate-regression--polynomial-features)
    - [Class 3: Model Selection & Hyperparameter Tuning](#class-3-model-selection--hyperparameter-tuning)
5. [Project Structure](#-project-structure)
6. [How to Run](#-how-to-run)
7. [Dependencies](#-dependencies)

---

## 🌟 Introduction

This project aims to bridge the gap between theory and practice. Students will explore various regression techniques, learn how to handle multi-dimensional data, and eventually master the art of model selection and optimization.

## 🛠️ Prerequisites

- Python 3.8 or higher
- Basic understanding of Python syntax
- Familiarity with Excel/CSV data formats

## 📥 Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/Yared-Coding-School/machine-learning-models.git
    cd machine-learning-models
    ```

2. **Create a virtual environment (Optional but Recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## 📖 Curriculum Overview

### **Class 1: Simple Linear Regression**

- **Objective**: Learn the basics of predicting a continuous value (Price) based on a single feature (Area).
- **Key Concepts**: `LinearRegression`, Slopes, Intercepts, Scikit-learn basics, and Matplotlib visualization.
- **Output**: Generates `data_with_pridicted_prices.xlsx`.

### **Class 2: Multivariate Regression & Polynomial Features**

- **Objective**: Move beyond single features. Predict house prices using multiple variables (California Housing dataset).
- **Key Concepts**: `train_test_split`, `r2_score`, `PolynomialFeatures`, and Model Optimization.

### **Class 3: Model Selection & Hyperparameter Tuning**

- **Objective**: Compare multiple models and fine-tune them for peak performance.
- **Key Concepts**: `RandomForestRegressor`, `HistGradientBoostingRegressor`, `RandomizedSearchCV`, and Model Persistence (`joblib`).
- **Output**: Saves the best performing model as `my_best_model.joblib`.

---

## 📂 Project Structure

```text
machine-learning-models/
├── class1/          # Simple Linear Regression (Area vs Price)
│   ├── data.xlsx
│   ├── area_predict.xlsx
│   └── main.py
├── class2/          # Multivariate Regression (California Housing)
│   └── main.py
├── class3/          # Model Selection & Optimization
│   └── main.py
├── requirements.txt # Project dependencies
├── test.py          # Quick test script for environment verification
└── README.md        # This file!
```

---

## 🚀 How to Run

Navigate to any class folder and run the `main.py` script:

```bash
# Example for Class 1
python class1/main.py
```

---

## 📦 Dependencies

This project utilizes the following powerful libraries:

- **Pandas**: Data manipulation and analysis.
- **Scikit-Learn**: Machine learning algorithms and tools.
- **Matplotlib**: Data visualization.
- **Joblib**: Efficient serialization of large Python objects (models).
- **Openpyxl**: Engine for reading/writing Excel files.

---

### 🎓 Created with ❤️ for students at Yared Technology School.
