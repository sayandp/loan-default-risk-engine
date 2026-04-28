# main.py

from src.data_loader import load_data
from src.feature_engineering import build_features
from src.train import train_model
from src.evaluate import evaluate_model

def main():
    app, bureau, prev, inst = load_data()

    df = build_features(app, bureau, prev, inst)

    model, X_val, y_val = train_model(df)

    evaluate_model(model, X_val, y_val)


if __name__ == "__main__":
    main()