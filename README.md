# 💳 Loan Default Risk Engine (XGBoost + SHAP + Optuna)

An end-to-end machine learning system to predict loan default risk using real-world financial data. This project demonstrates strong skills in feature engineering, imbalanced learning, model optimization, explainability, and deployment.

---

## 🚀 Problem Statement

Financial institutions need to minimize risk while issuing loans. The key challenge is identifying high-risk applicants before approval.

This project builds a production-style ML system that:
- Predicts probability of default
- Handles imbalanced data (~8% defaults)
- Uses behavioral and financial signals
- Optimizes decision thresholds for business use
- Explains predictions using SHAP

---

## 📊 Dataset

Based on Home Credit Default Risk dataset.

Main tables used:
- application_train.csv → applicant data
- bureau.csv → external credit history
- previous_application.csv → past loans
- installments_payments.csv → repayment behavior

---

## ⚙️ Feature Engineering (Core Strength)

Aggregated multiple data sources into a single training table.

### Bureau Features
- Debt ratio
- Total credit exposure
- Overdue indicators

### Previous Applications
- Number of past loans
- Average credit amount

### Installments (Most Important)
- Late payment flag
- Payment delay
- Payment ratio

Key insight:
> Past repayment behavior is the strongest predictor of default risk.

---

## 🤖 Model

Algorithm: XGBoost (Gradient Boosting)

Why:
- Strong performance on tabular data
- Handles missing values well
- Effective for imbalanced datasets

---

## ⚖️ Handling Imbalanced Data

Default rate ≈ 8%

Used:
- scale_pos_weight ≈ 9

This ensures the model focuses on detecting defaulters.

---

## 🎯 Threshold Optimization (Business Logic)

Instead of default threshold (0.5), evaluated precision vs recall tradeoff.

Final choice:
- Threshold = 0.40

Reason:
- High recall (~72–77%)
- Minimizes missed defaulters (critical in finance)

---

## 📈 Model Performance

- ROC-AUC: ~0.77
- Recall (default class): ~72%
- Precision: ~17%

Note:
Accuracy is not meaningful due to class imbalance.

---

## 🔧 Hyperparameter Tuning (Optuna)

Used Optuna for automated tuning.

Optimized:
- n_estimators
- max_depth
- learning_rate
- subsample
- colsample_bytree
- scale_pos_weight

Result:
- Small improvement in AUC
- More stable model

Key insight:
> Feature engineering had a much larger impact than hyperparameter tuning.

---

## 🔍 Model Explainability (SHAP)

Used SHAP to interpret predictions.

Key findings:
- External credit scores are strongest predictors
- Late payments significantly increase risk
- High debt ratio increases default probability
- Behavioral features outperform static demographics

Example:
> Applicants with frequent late payments and high debt ratios show significantly higher default risk.

---

## 🌐 Streamlit App (Deployment)

Built an interactive app for real-time prediction.

Features:
- User inputs financial details
- Model predicts default probability
- Displays risk score
- Provides decision (approve / reject)

Run locally:streamlit run app.py
## 📁 Project Structure


loan-default-risk-engine/
│
├── src/
│ ├── data_loader.py
│ ├── feature_engineering.py
│ ├── train.py
│ ├── evaluate.py
│ └── config.py
│
├── models/
│ └── xgb_model.pkl
│
├── app.py
├── main.py
├── requirements.txt
└── README.md


---

## 🧠 Key Learnings

- Feature engineering is more important than model tuning
- Threshold selection is critical in business applications
- Behavioral features are highly predictive
- Explainability is essential for trust in financial models

---

## 📌 Future Improvements

- Full feature pipeline inside deployment
- API deployment (FastAPI)
- Model monitoring and drift detection
- Ensemble models

---

## 🧾 Resume Summary

Built an end-to-end loan default prediction system using XGBoost on multi-source financial data. Engineered behavioral risk features, handled class imbalance, optimized decision thresholds, performed hyperparameter tuning with Optuna, and used SHAP for model explainability. Deployed an interactive Streamlit application for real-time risk prediction.

---

## 👨‍💻 Author

sayand p
Machine Learning Enthusiast