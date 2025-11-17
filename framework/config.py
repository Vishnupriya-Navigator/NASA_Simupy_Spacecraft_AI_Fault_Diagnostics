# framework/config.py
from dataclasses import dataclass


@dataclass
class Config:
    # Reproducibility
    seed: int = 42

    # Telemetry schema (we'll map SimuPy signals to these later)
    n_features: int = 10
    feature_names: tuple = tuple([f"feat_{i}" for i in range(10)])
    label_name: str = "fault_label"  # 0=nominal, 1=fault

    # Nominal generation (synthetic only for this step)
    nominal_mu: float = 0.0
    nominal_sigma: float = 1.0

    # Fault parameters (used by our custom generator)
    bias_mu: float = 2.0
    drift_mu: float = 0.02
    spike_scale: float = 6.0
    dropout_prob: float = 0.2
    saturation_min: float = -3.0
    saturation_max: float = 3.0

    # Dataset defaults
    default_samples: int = 50000
    default_fault_rate: float = 0.12

    # Paths
    raw_path: str = "data/raw/telemetry.csv"
