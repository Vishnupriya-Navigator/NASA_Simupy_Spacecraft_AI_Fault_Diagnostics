# scripts/generate_simupy_dataset.py
import argparse
import pandas as pd

from framework.adapters.simupy_flight_adapter import SimuPyFlightAdapter
from framework.config import Config


def main():
    ap = argparse.ArgumentParser(
        description="Generate NASA SimuPy-Flight telemetry CSV"
    )
    ap.add_argument("--seconds", type=int, default=180, help="Duration to simulate")
    ap.add_argument("--hz", type=int, default=50, help="Sampling rate")
    ap.add_argument(
        "--fault",
        type=str,
        default=None,
        help="Optional fault mode: bias|drift|spike|dropout|saturation",
    )
    ap.add_argument(
        "--fault-start", type=float, default=10.0, help="Fault start time (s)"
    )
    ap.add_argument("--fault-end", type=float, default=20.0, help="Fault end time (s)")
    ap.add_argument("--out", type=str, default="data/raw/simupy_telemetry.csv")
    args = ap.parse_args()

    cfg = Config()
    feats = list(cfg.feature_names)  # feat_0..feat_9
    rows = []

    adapter = SimuPyFlightAdapter(
        hz=args.hz,
        fault_mode=args.fault,
        fault_start_s=args.fault_start,
        fault_end_s=args.fault_end,
    )
    telem = adapter.reset()

    total_steps = args.seconds * args.hz
    dt = 1.0 / float(args.hz)
    for i in range(total_steps):
        t = i * dt
        telem = adapter.step()

        # Ensure all 10 features exist; pad missing with 0.0
        row = {f: float(telem.get(f, 0.0)) for f in feats}

        # Label: 1 only when fault mode is active AND within the window; else 0
        if args.fault and (args.fault_start <= t <= args.fault_end):
            row[cfg.label_name] = 1
        else:
            row[cfg.label_name] = 0

        rows.append(row)

    df = pd.DataFrame(rows, columns=feats + [cfg.label_name])
    df.to_csv(args.out, index=False)
    print(f"[ok] wrote {len(df):,} rows to {args.out}")


if __name__ == "__main__":
    main()
