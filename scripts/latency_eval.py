# scripts/latency_eval.py
import argparse, json
import pandas as pd
import numpy as np


def find_latency(df, threshold: float, hold: int, fault_start: float, hz: float):
    # index of fault start
    start_idx = int(round(fault_start * hz))
    probs = df["fault_prob"].to_numpy()

    # First index >= start_idx where prob >= threshold for 'hold' consecutive frames
    for i in range(start_idx, len(probs) - hold + 1):
        if np.all(probs[i : i + hold] >= threshold):
            return (i - start_idx) / hz
    return None  # not detected


def main():
    ap = argparse.ArgumentParser(
        description="Compute detection latency from stream_log.csv"
    )
    ap.add_argument("--log", type=str, default="results/stream_log.csv")
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument(
        "--hold", type=int, default=3, help="consecutive frames to confirm detection"
    )
    ap.add_argument("--fault-start", type=float, default=4.0)
    ap.add_argument("--hz", type=float, default=10.0)
    ap.add_argument("--out", type=str, default="results/latency_summary.json")
    args = ap.parse_args()

    df = pd.read_csv(args.log)
    latency = find_latency(df, args.threshold, args.hold, args.fault_start, args.hz)

    summary = {
        "threshold": args.threshold,
        "hold": args.hold,
        "fault_start_s": args.fault_start,
        "hz": args.hz,
        "latency_s": None if latency is None else float(latency),
    }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)

    if latency is None:
        print("[warn] No detection under given threshold/hold.")
    else:
        print(f"[ok] latency = {latency:.3f} s")
    print(f"[ok] wrote {args.out}")


if __name__ == "__main__":
    main()
