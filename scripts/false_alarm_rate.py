# scripts/false_alarm_rate.py
import argparse, time, json
import pandas as pd
from framework.adapters.simupy_flight_adapter import SimuPyFlightAdapter
from framework.models.rf_model import load_model
from framework.config import Config


def main():
    ap = argparse.ArgumentParser(
        description="Estimate false alarm rate in nominal streaming"
    )
    ap.add_argument("--model", type=str, default="models/rf_model_simupy.joblib")
    ap.add_argument("--hz", type=int, default=50)
    ap.add_argument(
        "--seconds", type=int, default=600, help="duration to run (nominal)"
    )
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--hold", type=int, default=3)
    ap.add_argument("--out", type=str, default="results/far_summary.json")
    args = ap.parse_args()

    cfg = Config()
    model = load_model(args.model)
    adapter = SimuPyFlightAdapter(hz=args.hz, fault_mode=None)
    telem = adapter.reset()

    dt = 1.0 / float(args.hz)
    steps = args.seconds * args.hz

    consec = 0
    triggers = 0

    for _ in range(steps):
        telem = adapter.step()
        X = pd.DataFrame([{f: float(telem.get(f, 0.0)) for f in cfg.feature_names}])
        prob = float(model.predict_proba(X)[0][1])

        if prob >= args.threshold:
            consec += 1
            if consec == args.hold:
                triggers += 1
        else:
            consec = 0
        time.sleep(dt)

    hours = args.seconds / 3600.0
    far_per_hour = triggers / hours if hours > 0 else None

    summary = {
        "threshold": args.threshold,
        "hold": args.hold,
        "hz": args.hz,
        "seconds": args.seconds,
        "triggers": triggers,
        "false_alarms_per_hour": far_per_hour,
    }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)

    print(
        f"[ok] nominal triggers={triggers} over {args.seconds}s "
        f"-> FAR={far_per_hour:.3f} per hour at threshold={args.threshold}, hold={args.hold}"
    )
    print(f"[ok] wrote {args.out}")


if __name__ == "__main__":
    main()
