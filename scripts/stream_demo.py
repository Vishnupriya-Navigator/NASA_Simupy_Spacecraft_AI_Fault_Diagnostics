# scripts/stream_demo.py
import argparse, csv, time
from framework.runtime.monitor import RuntimeMonitor
from framework.config import Config


def row_iter_from_csv(path: str):
    cfg = Config()
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            yield {k: float(r[k]) for k in cfg.feature_names}


def main():
    ap = argparse.ArgumentParser(description="Stream-score telemetry rows in real time")
    ap.add_argument("--data", type=str, default=Config().raw_path)
    ap.add_argument("--model", type=str, default="models/rf_model.joblib")
    ap.add_argument("--hz", type=int, default=10)
    ap.add_argument("--seconds", type=int, default=None, help="Stop after N seconds")
    ap.add_argument("--max-rows", type=int, default=None, help="Stop after N rows")
    args = ap.parse_args()

    mon = RuntimeMonitor(model_path=args.model, hz=args.hz)

    start = time.time()
    count = 0
    for out in mon.score_stream(row_iter_from_csv(args.data)):
        f0, f1, f2 = Config().feature_names[:3]
        print(
            f"fault={out['fault_pred']} prob={out['fault_prob']:.3f} | "
            f"{f0}={out[f0]:.2f} {f1}={out[f1]:.2f} {f2}={out[f2]:.2f}"
        )
        count += 1
        if args.max_rows is not None and count >= args.max_rows:
            break
        if args.seconds is not None and (time.time() - start) >= args.seconds:
            break


if __name__ == "__main__":
    main()


# stop after 600 seconds (10 minutes)
# python -m scripts.stream_demo --hz 10 --seconds 600

# or stop after 3000 rows
# python -m scripts.stream_demo --hz 10 --max-rows 3000
