# scripts/stream_simupy.py
import argparse, time
import pandas as pd
from framework.adapters.simupy_flight_adapter import SimuPyFlightAdapter
from framework.models.rf_model import load_model
from framework.config import Config


def main():
    ap = argparse.ArgumentParser(
        description="Real-time scoring from SimuPy-Flight adapter"
    )
    ap.add_argument("--model", type=str, default="models/rf_model_simupy.joblib")
    ap.add_argument("--hz", type=int, default=10)
    ap.add_argument("--seconds", type=int, default=10)
    ap.add_argument(
        "--fault", type=str, default=None, help="bias|drift|spike|dropout|saturation"
    )
    ap.add_argument("--fault-start", type=float, default=3.0)
    ap.add_argument("--fault-end", type=float, default=7.0)
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

    for i in range(steps):
        telem = adapter.step()
        # make a 1-row dataframe in the same feature order used in training
        X = pd.DataFrame([{f: float(telem.get(f, 0.0)) for f in cfg.feature_names}])
        pred = int(model.predict(X)[0])
        prob = float(model.predict_proba(X)[0][1])  # P(fault)
        print(
            f"t={i*dt:5.2f}s  fault={pred}  prob={prob:.3f}  p={telem.get('feat_0',0):.4f} q={telem.get('feat_1',0):.4f} r={telem.get('feat_2',0):.4f}"
        )
        time.sleep(dt)


if __name__ == "__main__":
    main()
