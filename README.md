# Heart Disease Prediction — SVM vs KNN vs Naive Bayes

Course: **PMIT-6121 Machine Learning** · Author: **Md. Tarequl Islam Mizi (252008)**

Benchmarks three classical classifiers on the CDC BRFSS 2020 *Personal Key
Indicators of Heart Disease* dataset (319,795 records, 18 columns).

## Setup
```bash
pip install -r requirements.txt
```

## Data
Download `heart_2020_cleaned.csv` from Kaggle:
https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease

## Run
```bash
python heart_disease_pipeline.py --data heart_2020_cleaned.csv
```

## Pipeline
1. **EDA** — class distribution, age/comorbidity prevalence, BMI overlap
2. **Preprocess** — defensive imputation → one-hot encode (12 cat cols → 33 dims) →
   stratified 80/20 split → StandardScaler (SVM & KNN only)
3. **Models** — SVM (linear/rbf × C∈{0.1,1,10}), KNN (K∈{3,5,11}), GaussianNB
4. **Evaluate** — accuracy, precision, recall, F1 on the positive class + confusion matrices

## Key result
SVM-RBF (C=10) gives the best F1 (0.478); Gaussian NB gives the best recall (0.881),
making it the recommended first-pass clinical screening model.
