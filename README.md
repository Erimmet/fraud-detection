# 💳 Fraud Detection System (Machine Learning)

A production-ready fraud detection system using **XGBoost** to identify fraudulent financial transactions in real-time. This project combines feature engineering, class imbalance handling, and an interactive web interface for predictions.

---

## 🎯 Key Achievements

- **96% fraud detection rate** (Recall)
- **55% precision** at deployment threshold
- **99% accuracy** for legitimate transactions
- **Real-time predictions** via Streamlit web app
- **0.8 false alarms per real fraud** caught

---

## 📊 Project Overview

Financial fraud detection is a highly imbalanced classification problem where fraudulent transactions represent <0.2% of all transactions. This project delivers a robust XGBoost model that catches 96% of fraud while maintaining manageable false alarm rates.

### Business Impact
| Metric | Value |
|--------|-------|
| Frauds Caught | 96.1% |
| False Alarm Ratio | 0.8:1 |
| Model Precision | 55.3% |
| Production Threshold | 0.95 |

---

## 📁 Dataset

- Transaction-level financial dataset from Kaggle
- Features include:
  - Transaction type (PAYMENT, TRANSFER, CASH_OUT, DEPOSIT)
  - Transaction amount
  - Sender balance (before/after)
  - Receiver balance (before/after)
- Target variable: `isFraud` (1 = Fraud, 0 = Legitimate)

**Source:** [Fraud Detection Dataset](https://www.kaggle.com/datasets/amanalisiddiqui/fraud-detection-dataset/data)

---

## 🔍 Exploratory Data Analysis (EDA)

Key insights discovered:

- **Class Imbalance:** Only 0.13% of transactions are fraudulent
- **Fraud Patterns:** 99.9% of fraud occurs in `TRANSFER` and `CASH_OUT` types
- **Amount Skew:** Fraud amounts vary widely, requiring log transformation for analysis
- **Balance Manipulation:** Fraudsters often create balance inconsistencies

### Visualizations included:
- Transaction type distribution
- Fraud rate by transaction type
- Log-transformed amount distribution
- Correlation heatmap
- Precision-Recall curves

---

## 🛠 Feature Engineering

Engineered features that significantly improved model performance:

### Created Features:
```python
# Balance differences (captures account activity)
balanceDiffOrig = oldbalanceOrg - newbalanceOrig
balanceDiffDest = oldbalanceDest - newbalanceDest

# Balance error (catches destination account manipulation)
errorBalanceDest = oldbalanceDest + amount - newbalanceDest

# Account emptying indicator (common fraud pattern)
is_sender_emptied = 1 if newbalanceOrig == 0 else 0
