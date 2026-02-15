# Breast Cancer Model Comparison: A Comparative Analysis of Machine Learning Algorithms for Tumor Classification

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7.0-green.svg)](https://scikit-learn.org/)
[![pandas](https://img.shields.io/badge/pandas-1.5.3-blue.svg)](https://pandas.pydata.org/)
[![numpy](https://img.shields.io/badge/numpy-1.24.3-blue.svg)](https://numpy.org/)
[![matplotlib](https://img.shields.io/badge/matplotlib-3.7.1-blue.svg)](https://matplotlib.org/)
[![seaborn](https://img.shields.io/badge/seaborn-0.12.2-blue.svg)](https://seaborn.pydata.org/)
[![Dataset](https://img.shields.io/badge/Dataset-WDBC-purple.svg)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
[![License](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

**Author:** Arshia Samimi  
**Institution:** Department of Mathematics and Computer Science, Iran University of Science and Technology

## 📋 Overview

This repository contains a comprehensive comparative analysis of machine learning models for breast tumor classification using the Wisconsin Diagnostic Breast Cancer (WDBC) dataset. The study evaluates five different algorithms through rigorous methodology including 10-fold stratified cross-validation and hyperparameter tuning.

### What This Project Does

- 🔬 **Replicates** a published logistic regression study and extends it with four additional models
- 📊 **Compares** SVM (RBF), Random Forest, K-Nearest Neighbors, and a custom Linear Regression Classifier
- 📈 **Evaluates** using 7 metrics: Accuracy, Precision, Recall, F1-Score, AUC, RMSE, and cross-validation stability
- ⚙️ **Optimizes** all models using `RandomizedSearchCV` with custom scoring functions
- 📉 **Visualizes** results with publication-quality plots (precision/recall curves, tuning results, bar charts)

### Key Files in This Repository

| File | Description |
|------|-------------|
| `proj.ipynb` | Main Jupyter notebook with complete analysis code |
| `Breast Cancer Model Comparison.pdf` | Full 24-page research paper |
| `DATA/wdbc.data` | Wisconsin Diagnostic Breast Cancer dataset |
| `DATA/wdbc.names` | Feature descriptions and dataset information |

### Why This Matters

Breast cancer is the most frequently diagnosed cancer among women worldwide, with approximately 2.3 million new cases in 2022. Early and accurate diagnosis is critical for patient outcomes. This project explores how machine learning can assist in distinguishing between benign and malignant tumors, providing clinically-relevant model selection guidelines.

## 🎯 Key Contributions

| Contribution | Description |
|--------------|-------------|
| **✅ Replicated & Improved** | Successfully replicated a published logistic regression study, achieving **98.25% accuracy** compared to the original **96.5%** |
| **✅ Extended Analysis** | Added four additional models: SVM (RBF), Random Forest, KNN, and a custom Linear Regression Classifier |
| **✅ Custom Implementation** | Created a custom `LinearRegressionClassifier` class that wraps linear regression for binary classification with tunable thresholds |
| **✅ Rigorous Validation** | Implemented **10-fold stratified cross-validation** to ensure robust performance estimates |
| **✅ Hyperparameter Tuning** | Used `RandomizedSearchCV` to optimize all models, with custom scoring functions for the linear classifier |
| **✅ Stability Analysis** | Analyzed model consistency through cross-validation standard deviations |
| **✅ Clinical Guidelines** | Provided model selection recommendations based on different clinical priorities (diagnostic safety, interpretability, speed) |
| **✅ Reproducible Research** | Complete code, data, and 24-page paper for full transparency |

## 🤖 Models Evaluated

This project compares five machine learning models, including one custom implementation:

| Model | Type | Implementation | Key Characteristics |
|-------|------|----------------|---------------------|
| **Logistic Regression** | Linear | `sklearn.linear_model.LogisticRegression` | Interpretable, probabilistic output, L2 regularization |
| **SVM with RBF Kernel** | Non-linear | `sklearn.svm.SVC` (kernel='rbf', probability=True) | Captures complex patterns, Platt scaling for probabilities |
| **Random Forest** | Ensemble | `sklearn.ensemble.RandomForestClassifier` | Robust to overfitting, feature importance estimation |
| **K-Nearest Neighbors** | Instance-based | `sklearn.neighbors.KNeighborsClassifier` | Non-parametric, distance-based classification |
| **Linear Regression Classifier** | Linear (custom) | Custom `LinearRegressionClassifier` class | Wraps `LinearRegression` with tunable threshold for binary classification |

## 📊 Key Findings

### Test Set Performance

| Model | Accuracy | Precision | Recall | F1-Score | AUC | RMSE |
|-------|----------|-----------|--------|----------|-----|------|
| **SVM (RBF)** | **99.12%** | 100.00% | **97.62%** | **98.80%** | 99.64% | 0.0937 |
| Logistic Regression | 98.25% | 100.00% | 95.24% | 97.56% | **99.80%** | 0.1325 |
| Random Forest | 97.37% | 100.00% | 92.86% | 96.30% | 99.44% | 0.1622 |
| K-Nearest Neighbors | 96.49% | 97.50% | 92.86% | 95.12% | 98.56% | 0.1873 |
| Linear Regression* | 98.25% | 100.00% | 95.24% | 97.56% | 99.70% | 0.1325 |

*\*Included for comparison - not theoretically suitable for classification tasks*

### Model Stability (10-Fold Cross-Validation Standard Deviation)

| Model | Accuracy Std | Precision Std | Recall Std | F1 Std | AUC Std | RMSE Std |
|-------|--------------|---------------|------------|--------|---------|----------|
| **Logistic Regression** | **0.0296** | **0.0176** | 0.0798 | **0.0441** | 0.0150 | -0.1102 |
| SVM (RBF) | 0.0357 | 0.0375 | 0.0809 | 0.0516 | 0.0154 | -0.1233 |
| Random Forest | 0.0355 | 0.0398 | **0.0668** | 0.0491 | **0.0127** | -0.1262 |
| KNN | 0.0467 | 0.0429 | 0.0941 | 0.0680 | 0.0249 | -0.1324 |
| Linear Regression | 0.0350 | 0.0231 | 0.0789 | 0.0530 | **0.0086** | **-0.0902** |

### Key Insights

- **SVM (RBF)** achieved the highest overall performance with **99.12% accuracy** and **97.62% recall** (best at catching malignant cases)
- **Logistic Regression** was the most stable model (lowest standard deviations) with excellent AUC (**99.80%**)
- **Random Forest** showed the most consistent recall across folds (lowest recall std)
- **KNN** exhibited the highest variability, making it less reliable for clinical use
- **Linear Regression** appears competitive numerically but is theoretically unsuitable for classification

## 🏆 Model Selection Guidelines

Based on your clinical priorities:

| Priority | Recommended Model | Rationale |
|----------|-------------------|-----------|
| **Diagnostic Safety** (minimize false negatives) | **SVM (RBF)** | Highest recall (97.62%) - catches most malignant cases |
| **Balanced Performance** | **SVM (RBF)** | Best accuracy & F1-score |
| **Interpretability** (explain to doctors) | **Logistic Regression** | Transparent coefficients, clinically accepted |
| **Stability & Consistency** | **Logistic Regression** | Lowest variance across data samples |
| **Computational Efficiency** | **Logistic Regression** | Lightweight training & inference |
| **Rapid Prototyping** | **KNN / Linear Regression** | Simple to implement |

## 🚀 Quick Start

### Prerequisites
```bash
python >= 3.8
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/arshiasamimi/breast-cancer-analysis.git
cd breast-cancer-analysis
```

2. **Install dependencies**
```bash
pip install scikit-learn pandas numpy matplotlib seaborn jupyter scipy
```

3. **Run the analysis**
```bash
jupyter notebook proj.ipynb
```

### Dependencies
- scikit-learn==1.7.0
- pandas==1.5.3
- numpy==1.24.3
- matplotlib==3.7.1
- seaborn==0.12.2
- jupyter==1.0.0
- scipy==1.10.1

## 📂 Repository Structure

```
breast-cancer-analysis/
│
├── proj.ipynb                              # Main analysis notebook
├── Breast Cancer Model Comparison.pdf       # Full research paper
├── README.md                                # This file
│
├── DATA/
│   ├── wdbc.data                            # WDBC dataset
│   └── wdbc.names                            # Feature descriptions
│
└── (figures will be generated when you run the notebook)
```

## 📊 Dataset: Wisconsin Diagnostic Breast Cancer (WDBC)

### Overview
- **Instances:** 569
- **Features:** 30 real-valued features
- **Classes:** 2 (Benign: 357, Malignant: 212)
- **Missing Values:** None

### Features
For each cell nucleus, 10 characteristics were computed:
- Radius, Texture, Perimeter, Area, Smoothness
- Compactness, Concavity, Concave Points, Symmetry, Fractal Dimension

Each characteristic appears as: mean, standard error, and "worst" (mean of largest values)

### Source & License
> Wolberg, W., Street, W., & Mangasarian, O. (1993). Breast Cancer Wisconsin (Diagnostic) [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B

**License:** CC BY 4.0
