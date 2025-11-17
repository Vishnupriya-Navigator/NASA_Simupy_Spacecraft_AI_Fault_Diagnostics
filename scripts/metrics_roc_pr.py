# scripts/metrics_roc_pr.py
import argparse, json, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
)

from framework.config import Config
from framework.models.rf_model import load_model


def main():
    ap = argparse.ArgumentParser(description="ROC/PR curves + threshold at target FPR")
    ap.add_argument("--data", type=str, default="data/raw/telemetry_simupy.csv")
    ap.add_argument("--model", type=str, default="models/rf_model_simupy.joblib")
    ap.add_argument(
        "--target-fpr", type=float, default=0.01, help="Desired FPR for operating point"
    )
    ap.add_argument("--out-prefix", type=str, default="results/metrics")
    args = ap.parse_args()

    cfg = Config()
    df = pd.read_csv(args.data)
    X = df[list(cfg.feature_names)]
    y = df[cfg.label_name].astype(int).values

    clf = load_model(args.model)
    prob = clf.predict_proba(X)[:, 1]

    # ROC
    fpr, tpr, thr = roc_curve(y, prob)
    roc_auc = auc(fpr, tpr)

    # PR
    prec, rec, thr_pr = precision_recall_curve(y, prob)
    pr_auc = average_precision_score(y, prob)

    os.makedirs("figures", exist_ok=True)
    # Plot ROC
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("ROC Curve")
    plt.tight_layout()
    roc_path = "figures/roc_curve.png"
    plt.savefig(roc_path, dpi=200)
    plt.close()

    # Plot PR
    plt.figure()
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.tight_layout()
    pr_path = "figures/pr_curve.png"
    plt.savefig(pr_path, dpi=200)
    plt.close()

    # Choose threshold at / below target FPR (pick the highest TPR among those)
    valid = np.where(fpr <= args.target_fpr)[0]
    if len(valid) == 0:
        # fallback: smallest FPR available
        idx = int(np.argmin(fpr))
    else:
        # among valid, choose idx with max tpr; break ties with lowest fpr
        idx = valid[np.argmax(tpr[valid])]
    threshold = thr[idx] if idx < len(thr) else 0.5  # sklearn thr len = len(fpr)-1

    # Confusion at chosen threshold
    y_hat = (prob >= threshold).astype(int)
    cm = confusion_matrix(y, y_hat)
    tn, fp, fn, tp = cm.ravel()

    # Precision/Recall at chosen threshold
    precision_at_thr = tp / (tp + fp) if (tp + fp) else 0.0
    recall_at_thr = tp / (tp + fn) if (tp + fn) else 0.0
    fpr_at_thr = fp / (fp + tn) if (fp + tn) else 0.0

    summary = {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "target_fpr": args.target_fpr,
        "threshold": float(threshold),
        "tpr_at_threshold": float(recall_at_thr),
        "fpr_at_threshold": float(fpr_at_thr),
        "precision_at_threshold": float(precision_at_thr),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "plots": {"roc": roc_path, "pr": pr_path},
    }

    os.makedirs("results", exist_ok=True)
    out_json = f"{args.out_prefix}_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[ok] ROC AUC={roc_auc:.3f}  PR AUC={pr_auc:.3f}")
    print(f"[ok] threshold@FPR<={args.target_fpr:.3f} -> {threshold:.4f}")
    print(f"[ok] summary -> {out_json}")
    print(f"[ok] plots   -> {roc_path}, {pr_path}")


if __name__ == "__main__":
    main()
