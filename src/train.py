# src/train.py

import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier
from .config import MODEL_PARAMS


def train_model(df):
    """
    Train XGBoost model and return trained model + validation data
    """

    # =========================
    # SPLIT
    # =========================
    X = df.drop(columns=["TARGET"])
    y = df["TARGET"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # =========================
    # ENCODE CATEGORICALS
    # =========================
    cat_cols = X_train.select_dtypes(include=["object", "string"]).columns

    X_train = X_train.copy()
    X_val = X_val.copy()

    X_train[cat_cols] = X_train[cat_cols].astype(str)
    X_val[cat_cols] = X_val[cat_cols].astype(str)

    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1
    )

    X_train[cat_cols] = encoder.fit_transform(X_train[cat_cols])
    X_val[cat_cols] = encoder.transform(X_val[cat_cols])

    # =========================
    # DROP ID COLUMN
    # =========================
    if "SK_ID_CURR" in X_train.columns:
        X_train = X_train.drop(columns=["SK_ID_CURR"])
        X_val = X_val.drop(columns=["SK_ID_CURR"])

    # =========================
    # TRAIN MODEL
    # =========================
    model = XGBClassifier(**MODEL_PARAMS)

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100
    )

    # =========================
    # SAVE MODEL (FIXED)
    # =========================
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/xgb_model.pkl")

    print("✅ Model saved to models/xgb_model.pkl")

    return model, X_val, y_val