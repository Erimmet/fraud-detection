# 💳 Fraud Detection System (Machine Learning)

A machine learning project focused on detecting fraudulent financial transactions using structured transaction data. This project combines exploratory data analysis, feature engineering, and a classification pipeline to identify suspicious activity.

---

## 📊 Project Overview

Financial fraud detection is a highly imbalanced classification problem where fraudulent transactions represent a very small percentage of the data.  

This project aims to:
- Analyze transaction behavior patterns
- Identify key indicators of fraud
- Build a robust classification model to detect fraudulent transactions

---

## 📁 Dataset

- Transaction-level financial dataset
- Includes features such as:
  - Transaction type
  - Amount
  - Account balances (before and after transaction)
- Target variable:
  - `isFraud` (1 = Fraud, 0 = Legitimate)

> (https://www.kaggle.com/datasets/amanalisiddiqui/fraud-detection-dataset/data)

---

## 🔍 Exploratory Data Analysis (EDA)

Key insights discovered:

- Fraud cases are **extremely imbalanced**
- Most fraud occurs in:
  - `TRANSFER`
  - `CASH_OUT`
- Transaction amount distribution is **highly skewed**
- Certain balance inconsistencies are strong fraud indicators

### Visualizations included:
- Transaction type distribution
- Fraud rate by transaction type
- Log-transformed amount distribution
- Correlation heatmap

---

## 🛠 Feature Engineering

New features created:
- `balanceDiffOrig` = oldbalanceOrg − newbalanceOrig  
- `balanceDiffDest` = oldbalanceDest − newbalanceDest  

These features help capture abnormal transaction patterns.

---

## 🤖 Model

### Algorithm:
- Logistic Regression

### Why?
- Interpretable
- Performs well with proper preprocessing
- Works efficiently with imbalanced data when adjusted

### Techniques used:
- `class_weight='balanced'` to handle class imbalance
- Feature scaling (`StandardScaler`)
- One-hot encoding for categorical variables

---

## ⚙️ Pipeline

A complete ML pipeline was built using:

- `ColumnTransformer`
- `Pipeline`

This ensures:
- Clean preprocessing
- Reproducibility
- Easy deployment

---

## 📈 Evaluation

Model evaluated using:
- Classification Report (Precision, Recall, F1-score)
- Confusion Matrix

Focus was placed on:
- Detecting fraud (Recall for class 1)
- Minimizing false negatives

---

## 💾 Model Export

The trained model is saved using:

```python
joblib.dump(pipeline, 'fraud_detection_model.pkl')
