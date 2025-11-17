# scripts/plot_prob.py
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser(
        description="Plot fault probability vs time from stream log"
    )
    ap.add_argument("--log", type=str, default="results/stream_log.csv")
    ap.add_argument("--out", type=str, default="figures/fault_prob_vs_time.png")
    ap.add_argument(
        "--title", type=str, default="Fault Probability vs Time (SimuPy-Flight)"
    )
    args = ap.parse_args()

    df = pd.read_csv(args.log)
    if not {"t", "fault_prob"}.issubset(df.columns):
        raise SystemExit("Log missing required columns 't' and 'fault_prob'")

    plt.figure()
    plt.plot(df["t"], df["fault_prob"])
    plt.xlabel("Time (s)")
    plt.ylabel("P(fault)")
    plt.title(args.title)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    plt.close()
    print(f"[ok] saved {args.out}")


if __name__ == "__main__":
    main()
