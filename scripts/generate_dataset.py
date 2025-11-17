# scripts/generate_dataset.py
import argparse
from framework.generators.telemetry_generator import generate_dataset, save_csv
from framework.config import Config


def main():
    ap = argparse.ArgumentParser(description="Generate synthetic telemetry with faults")
    ap.add_argument("--n-samples", type=int, default=Config().default_samples)
    ap.add_argument("--fault-rate", type=float, default=Config().default_fault_rate)
    ap.add_argument("--out", type=str, default=Config().raw_path)
    args = ap.parse_args()

    cfg = Config()
    df = generate_dataset(args.n_samples, args.fault_rate, cfg)
    save_csv(df, args.out)
    print(f"[ok] wrote {len(df):,} rows to {args.out}")


if __name__ == "__main__":
    main()
