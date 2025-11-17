# scripts/evaluate.py
import argparse, json
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from framework.models.rf_model import load_model
from framework.config import Config


def main():
    ap = argparse.ArgumentParser(description="Evaluate trained model on full dataset")
    ap.add_argument("--data", type=str, default=Config().raw_path)
    ap.add_argument("--model", type=str, default="models/rf_model.joblib")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    X = df[list(Config().feature_names)]
    y = df[Config().label_name]

    clf = load_model(args.model)
    y_pred = clf.predict(X)

    report = classification_report(y, y_pred, output_dict=True)
    cm = confusion_matrix(y, y_pred)

    with open("results/eval_report.json", "w") as f:
        json.dump(report, f, indent=2)
    pd.DataFrame(cm).to_csv(
        "results/eval_confusion_matrix.csv", index=False, header=False
    )

    print("[ok] wrote results/eval_report.json and results/eval_confusion_matrix.csv")


if __name__ == "__main__":
    main()
