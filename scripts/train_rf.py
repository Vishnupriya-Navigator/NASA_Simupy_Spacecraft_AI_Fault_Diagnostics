# scripts/train_rf.py
import argparse, os, json, time
import pandas as pd

from framework.models.rf_model import train, save_model
from framework.config import Config


def main():
    ap = argparse.ArgumentParser(description="Train Random Forest on telemetry CSV")
    ap.add_argument("--data", type=str, default=Config().raw_path)
    ap.add_argument("--model-out", type=str, default="models/rf_model.joblib")
    ap.add_argument(
        "--featimp-out", type=str, default="results/feature_importances.csv"
    )
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    clf, report, cm = train(df, Config())

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    save_model(clf, args.model_out)

    # Feature importances
    feat_names = list(Config().feature_names)
    imp = pd.DataFrame(
        {"feature": feat_names, "importance": clf.feature_importances_}
    ).sort_values("importance", ascending=False)
    os.makedirs(os.path.dirname(args.featimp_out), exist_ok=True)
    imp.to_csv(args.featimp_out, index=False)

    # Reports
    ts = int(time.time())
    rep_path = f"results/classification_report_{ts}.json"
    cm_path = f"results/confusion_matrix_{ts}.csv"
    pd.DataFrame(cm).to_csv(cm_path, index=False, header=False)
    with open(rep_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"[ok] model  -> {args.model_out}")
    print(f"[ok] import -> {args.featimp_out}")
    print(f"[ok] report -> {rep_path}")
    print(f"[ok] cm     -> {cm_path}")


if __name__ == "__main__":
    main()
