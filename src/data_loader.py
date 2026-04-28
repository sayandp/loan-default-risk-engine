# src/data_loader.py

import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw"

def load_data():
    app = pd.read_csv(DATA_PATH / "application_train.csv")
    bureau = pd.read_csv(DATA_PATH / "bureau.csv")
    prev = pd.read_csv(DATA_PATH / "previous_application.csv")
    inst = pd.read_csv(DATA_PATH / "installments_payments.csv")

    return app, bureau, prev, inst