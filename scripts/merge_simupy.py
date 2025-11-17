# scripts/merge_simupy.py
import argparse
import pandas as pd


def main():
    ap = argparse.ArgumentParser(
        description="Merge multiple SimuPy CSVs into one telemetry file"
    )
    ap.add_argument("--out", type=str, default="data/raw/telemetry_simupy.csv")
    ap.add_argument(
        "inputs",
        nargs="+",
        help="Input CSV paths (e.g., sf_nominal.csv sf_bias.csv ...)",
    )
    args = ap.parse_args()

    dfs = [pd.read_csv(p) for p in args.inputs]
    df = pd.concat(dfs, ignore_index=True)

    # Optional: shuffle to mix nominal/fault rows (keeps class balance)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    df.to_csv(args.out, index=False)
    print(f"[ok] merged {len(args.inputs)} files -> {args.out} ({len(df):,} rows)")


if __name__ == "__main__":
    main()
