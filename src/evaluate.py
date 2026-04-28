# src/evaluate.py

from sklearn.metrics import roc_auc_score, classification_report

def evaluate_model(model, X_val, y_val):

    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_pred = (y_pred_proba > 0.4).astype(int)  # chosen threshold

    print("ROC-AUC:", roc_auc_score(y_val, y_pred_proba))
    print("\nClassification Report:\n")
    print(classification_report(y_val, y_pred))