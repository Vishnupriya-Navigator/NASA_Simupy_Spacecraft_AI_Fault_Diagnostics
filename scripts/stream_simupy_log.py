# scripts/stream_simupy_log.py
import argparse, time, csv
import pandas as pd
from framework.adapters.simupy_flight_adapter import SimuPyFlightAdapter
from framework.models.rf_model import load_model
from framework.config import Config


def main():
    ap = argparse.ArgumentParser(
        description="Log real-time scoring from SimuPy-Flight to CSV"
    )
    ap.add_argument("--model", type=str, default="models/rf_model_simupy.joblib")
    ap.add_argument("--hz", type=int, default=10)
    ap.add_argument("--seconds", type=int, default=10)
    ap.add_argument(
        "--fault", type=str, default=None, help="bias|drift|spike|dropout|saturation"
    )
    ap.add_argument("--fault-start", type=float, default=3.0)
    ap.add_argument("--fault-end", type=float, default=7.0)
    ap.add_argument("--out", type=str, default="results/stream_log.csv")
    args = ap.parse_args()

    cfg = Config()
    model = load_model(args.model)
    adapter = SimuPyFlightAdapter(
        hz=args.hz,
        fault_mode=args.fault,
        fault_start_s=args.fault_start,
        fault_end_s=args.fault_end,
    )
    telem = adapter.reset()

    steps = args.seconds * args.hz
    dt = 1.0 / float(args.hz)

    fields = ["t", "fault_pred", "fault_prob"] + list(cfg.feature_names)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for i in range(steps):
            telem = adapter.step()
            X = pd.DataFrame([{f: float(telem.get(f, 0.0)) for f in cfg.feature_names}])
            pred = int(model.predict(X)[0])
            prob = float(model.predict_proba(X)[0][1])
            row = {"t": i * dt, "fault_pred": pred, "fault_prob": prob}
            for ftr in cfg.feature_names:
                row[ftr] = float(telem.get(ftr, 0.0))
            writer.writerow(row)
            time.sleep(dt)

    print(f"[ok] wrote {steps} rows to {args.out}")


if __name__ == "__main__":
    main()
