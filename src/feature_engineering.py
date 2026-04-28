# src/feature_engineering.py

import pandas as pd


def build_features(app, bureau, prev, inst):
    """
    Build final feature table from all datasets.
    Returns one row per SK_ID_CURR.
    """

    df = app.copy()

    # =========================
    # 1. BUREAU FEATURES
    # =========================
    bureau["DEBT_RATIO"] = bureau["AMT_CREDIT_SUM_DEBT"] / (
        bureau["AMT_CREDIT_SUM"] + 1
    )
    bureau["HAS_OVERDUE"] = (
        bureau["AMT_CREDIT_SUM_OVERDUE"] > 0
    ).astype(int)

    bureau_agg = bureau.groupby("SK_ID_CURR").agg({
        "SK_ID_BUREAU": "count",
        "AMT_CREDIT_SUM": ["mean", "sum"],
        "DEBT_RATIO": "mean",
        "HAS_OVERDUE": "max"
    })

    bureau_agg.columns = [
        "_".join(col).strip() for col in bureau_agg.columns
    ]
    bureau_agg = bureau_agg.reset_index()

    df = df.merge(bureau_agg, on="SK_ID_CURR", how="left")


    # =========================
    # 2. PREVIOUS APPLICATION
    # =========================
    prev_agg = prev.groupby("SK_ID_CURR").agg({
        "SK_ID_PREV": "count",
        "AMT_APPLICATION": "mean",
        "AMT_CREDIT": "mean"
    }).reset_index()

    df = df.merge(prev_agg, on="SK_ID_CURR", how="left")


    # =========================
    # 3. INSTALLMENTS (CRITICAL)
    # =========================

    # Merge installments with previous apps to get SK_ID_CURR
    inst_merged = inst.merge(
        prev[["SK_ID_PREV", "SK_ID_CURR"]],
        on="SK_ID_PREV",
        how="left"
    )

    # --- FIX duplicate column issue (robust) ---
    if "SK_ID_CURR" not in inst_merged.columns:
        if "SK_ID_CURR_y" in inst_merged.columns:
            inst_merged["SK_ID_CURR"] = inst_merged["SK_ID_CURR_y"]
        elif "SK_ID_CURR_x" in inst_merged.columns:
            inst_merged["SK_ID_CURR"] = inst_merged["SK_ID_CURR_x"]

    # Drop duplicate columns safely
    drop_cols = [c for c in ["SK_ID_CURR_x", "SK_ID_CURR_y"] if c in inst_merged.columns]
    inst_merged = inst_merged.drop(columns=drop_cols)

    # --- Final safety check ---
    if "SK_ID_CURR" not in inst_merged.columns:
        raise ValueError("SK_ID_CURR not found after merge. Check data integrity.")

    # =========================
    # CREATE INSTALLMENT FEATURES
    # =========================
    inst_merged["LATE_PAYMENT"] = (
        inst_merged["DAYS_ENTRY_PAYMENT"] > inst_merged["DAYS_INSTALMENT"]
    ).astype(int)

    inst_merged["PAYMENT_DELAY"] = (
        inst_merged["DAYS_ENTRY_PAYMENT"] - inst_merged["DAYS_INSTALMENT"]
    )

    inst_merged["PAYMENT_RATIO"] = (
        inst_merged["AMT_PAYMENT"] / (inst_merged["AMT_INSTALMENT"] + 1)
    )

    # =========================
    # AGGREGATE INSTALLMENTS
    # =========================
    inst_agg = inst_merged.groupby("SK_ID_CURR").agg({
        "LATE_PAYMENT": "mean",
        "PAYMENT_DELAY": "mean",
        "PAYMENT_RATIO": "mean"
    }).reset_index()

    df = df.merge(inst_agg, on="SK_ID_CURR", how="left")


    # =========================
    # FINAL CLEANING
    # =========================
    df = df.fillna(-999)

    return df