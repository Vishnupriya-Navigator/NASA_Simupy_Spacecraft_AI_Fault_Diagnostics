# framework/generators/faults.py
import numpy as np


def inject_bias(x: np.ndarray, magnitude: float) -> np.ndarray:
    return x + magnitude


def inject_drift(x: np.ndarray, step: float) -> np.ndarray:
    """Apply a slow ramp (drift) across the vector."""
    n = x.shape[0]
    ramp = np.linspace(0, step * n, n)
    return x + ramp


def inject_spike(
    x: np.ndarray, scale: float, prob: float = 0.01, rng=None
) -> np.ndarray:
    """Random impulsive spikes on some elements."""
    rng = rng or np.random.default_rng()
    mask = rng.random(x.shape[0]) < prob
    spikes = rng.normal(0, scale, size=x.shape[0])
    y = x.copy()
    y[mask] += spikes[mask]
    return y


def inject_dropout(x: np.ndarray, prob: float = 0.05, rng=None) -> np.ndarray:
    """Occasional zeros (sensor dropout)."""
    rng = rng or np.random.default_rng()
    mask = rng.random(x.shape[0]) < prob
    y = x.copy()
    y[mask] = 0.0
    return y


def inject_saturation(x: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    """Clamp to min/max (ADC saturation)."""
    return np.clip(x, min_val, max_val)
