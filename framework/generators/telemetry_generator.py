# framework/generators/telemetry_generator.py
import numpy as np
import pandas as pd
from typing import Tuple
from framework.config import Config
from framework.generators import faults as F


def _make_nominal(n: int, cfg: Config, rng=None) -> np.ndarray:
    rng = rng or np.random.default_rng(cfg.seed)
    return rng.normal(
        cfg.nominal_mu, cfg.nominal_sigma, size=(n, len(cfg.feature_names))
    )


def _apply_random_faults(
    X: np.ndarray, cfg: Config, rng=None
) -> Tuple[np.ndarray, np.ndarray]:
    rng = rng or np.random.default_rng(cfg.seed + 1)
    n = X.shape[0]
    y = np.zeros(n, dtype=int)
    X_faulty = X.copy()

    n_faults = int(cfg.default_fault_rate * n)
    fault_idx = rng.choice(n, size=n_faults, replace=False)
    y[fault_idx] = 1

    fault_types = rng.choice(
        ["bias", "drift", "spike", "dropout", "saturation"], size=n_faults, replace=True
    )

    for i, idx in enumerate(fault_idx):
        row = X_faulty[idx, :]
        ftype = fault_types[i]
        if ftype == "bias":
            mag = rng.normal(cfg.bias_mu, 0.5, size=row.shape)
            row = row + mag
        elif ftype == "drift":
            row = F.inject_drift(row, cfg.drift_mu)
        elif ftype == "spike":
            row = F.inject_spike(row, cfg.spike_scale, prob=0.2, rng=rng)
        elif ftype == "dropout":
            row = F.inject_dropout(row, prob=0.2, rng=rng)
        elif ftype == "saturation":
            row = F.inject_saturation(row, cfg.saturation_min, cfg.saturation_max)
        X_faulty[idx, :] = row

    return X_faulty, y


def generate_dataset(
    n: int, fault_rate: float | None = None, cfg: Config = Config()
) -> pd.DataFrame:
    """Return DataFrame with columns cfg.feature_names + cfg.label_name."""
    if fault_rate is not None:
        cfg.default_fault_rate = float(fault_rate)
    X = _make_nominal(n, cfg)
    X2, y = _apply_random_faults(X, cfg)

    df = pd.DataFrame(X2, columns=cfg.feature_names)
    df[cfg.label_name] = y
    return df


def save_csv(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)
