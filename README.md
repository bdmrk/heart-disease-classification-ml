# Benchmarking Machine Learning Classifiers for Heart Disease Prediction

**Course:** PMIT_6121 — Machine Learning
**Assignment:** Comparative Analysis of SVM, KNN, and Naive Bayes Classifiers

## Overview

This project benchmarks three classical machine learning classifiers —
**Support Vector Machines (SVM)**, **K-Nearest Neighbors (KNN)**, and
**Gaussian Naive Bayes** — on the
[Personal Key Indicators of Heart Disease](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease)
dataset (CDC BRFSS 2020 annual survey, ~320,000 records).

The pipeline covers:

- Exploratory data analysis (class distribution, missing values)
- Preprocessing: target/binary/ordinal encoding, one-hot encoding, 80/20
  stratified train-test split, and feature scaling (`StandardScaler`)
- Model training and hyperparameter exploration:
  - SVM with **linear** and **RBF** kernels
  - KNN with **K = 3, 5, 11**
  - Gaussian Naive Bayes (default configuration)
- Evaluation using **Accuracy, Precision, Recall, and F1-score**
- Confusion matrix visualizations for all six configurations

## Repository Structure

```
.
├── notebooks/
│   └── heart_disease_classification.ipynb   # Full, executed analysis notebook
├── reports/
│   ├── Heart_Disease_ML_Assignment_Report.pdf
│   └── Heart_Disease_ML_Assignment_Report.docx
├── figures/
│   ├── class_distribution.png
│   ├── confusion_matrices.png
│   └── confusion_matrices_all.png
├── results/
│   ├── all_results.json        # Metrics for all 6 model configurations
│   ├── summary_table.csv        # Best configuration per model family
│   ├── results_table.csv        # Full results table
│   └── eda_summary.json         # Dataset/EDA summary stats
├── requirements.txt
└── README.md
```

## Dataset

The dataset is **not included** in this repository due to its size (~24 MB).
Download `heart_2020_cleaned.csv` from Kaggle and place it in the project
root (or update the path in the notebook) before running:

> https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease

## Results Summary

| Model       | Hyperparameters                        | Accuracy | Precision | Recall | F1-Score |
|-------------|-----------------------------------------|----------|-----------|--------|----------|
| SVM         | Kernel=RBF, C=1.0, gamma=scale          | 0.916    | 0.692     | 0.035  | 0.067    |
| KNN         | K=3                                      | 0.897    | 0.266     | 0.113  | 0.159    |
| Naive Bayes | Default Gaussian (var_smoothing=1e-9)   | 0.828    | 0.265     | 0.564  | 0.360    |

> Note: due to the SVM's computational complexity on large datasets, all
> models were trained and evaluated on a stratified random sample of 15,000
> records (preserving the original ~91.4% / 8.6% class ratio). Full
> methodology and discussion are in the report (`reports/`).

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/heart_disease_classification.ipynb
```

## Key Findings

- SVM and KNN achieved the highest raw **accuracy (~91–92%)**, largely by
  defaulting toward the majority "No Disease" class.
- **Gaussian Naive Bayes** achieved the highest **recall (56.4%)** and
  **F1-score (0.360)** for the minority "Heart Disease" class — making it
  the most clinically useful model for a screening application, despite
  lower overall accuracy.
- Full discussion of the clinical trade-offs between false positives and
  false negatives, and recommendations for future work (SMOTE, ensemble
  methods), are provided in the technical report.

## Author

Kausar — PMIT_6121, Machine Learning
