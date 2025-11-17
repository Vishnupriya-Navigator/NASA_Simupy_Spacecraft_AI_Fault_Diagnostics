# scripts/probe_sf_stream.py
import argparse, time
from framework.adapters.simupy_flight_adapter import SimuPyFlightAdapter


def main():
    ap = argparse.ArgumentParser(description="Probe SimuPy-Flight telemetry stream")
    ap.add_argument("--hz", type=int, default=10)
    ap.add_argument("--seconds", type=int, default=3)
    ap.add_argument(
        "--fault", type=str, default=None, help="bias|drift|spike|dropout|saturation"
    )
    args = ap.parse_args()

    adapter = SimuPyFlightAdapter(
        hz=args.hz,
        fault_mode=args.fault,
        fault_start_s=1.0,
        fault_end_s=2.0,
    )
    telem = adapter.reset()

    steps = args.seconds * args.hz
    for _ in range(steps):
        telem = adapter.step()
        print(
            f"p={telem.get('feat_0',0.0): .4f} "
            f"q={telem.get('feat_1',0.0): .4f} "
            f"r={telem.get('feat_2',0.0): .4f}  "
            f"q0={telem.get('feat_3',0.0): .4f} "
            f"q1={telem.get('feat_4',0.0): .4f} "
            f"q2={telem.get('feat_5',0.0): .4f} "
            f"q3={telem.get('feat_6',0.0): .4f}"
        )
        time.sleep(1.0 / args.hz)


if __name__ == "__main__":
    main()
