# framework/runtime/monitor.py
import time
import pandas as pd
from typing import Iterable, Dict
from framework.models.rf_model import load_model
from framework.config import Config


class RuntimeMonitor:
    def __init__(self, model_path: str = "models/rf_model.joblib", hz: int = 10):
        self.cfg = Config()
        self.model = load_model(model_path)
        self.dt = 1.0 / float(hz)

    def score_stream(self, rows: Iterable[Dict[str, float]]):
        """
        rows: iterable of dicts with keys == cfg.feature_names
        yields: dict with prediction, probability, and raw features
        """
        for row in rows:
            df = pd.DataFrame([row], columns=self.cfg.feature_names)
            y = self.model.predict(df)[0]
            p = float(self.model.predict_proba(df)[0][1])
            out = {"fault_pred": int(y), "fault_prob": p}
            out.update(row)
            yield out
            time.sleep(self.dt)
