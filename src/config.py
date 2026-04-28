# src/config.py

DATA_PATH = "data/raw"
RANDOM_STATE = 42

MODEL_PARAMS = {
    "n_estimators": 343,
    "max_depth": 5,
    "learning_rate": 0.0957,
    "subsample": 0.712,
    "colsample_bytree": 0.811,
    "scale_pos_weight": 9,
    "n_jobs": -1,
    "eval_metric": "auc",
    "random_state": 42
}