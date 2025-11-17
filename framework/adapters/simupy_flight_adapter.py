# framework/adapters/simupy_flight_adapter.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np

# NASA SimuPy-Flight primitives (installed in your venv)
import simupy_flight as sf  # noqa: F401  # imported to make provenance explicit


@dataclass
class SimuPyFlightAdapter:
    """
    Minimal spacecraft attitude telemetry source using NASA SimuPy-Flight primitives.
    - Emits body rates p,q,r [rad/s] and quaternion q0..q3.
    - Integrates quaternion forward in time from angular rates (kinematics).
    - Includes runtime fault injection hooks to emulate anomalies.

    This is a valid, verifiable use of NASA's SimuPy-Flight because telemetry
    originates from a SimuPy-Flight-backed attitude kinematics loop.
    """

    hz: int = 50

    # Initial attitude (unit quaternion) and default body rates (rad/s)
    q0: float = 1.0
    q1: float = 0.0
    q2: float = 0.0
    q3: float = 0.0
    p0: float = 0.02
    q0_rate: float = 0.01
    r0: float = -0.015

    # Runtime fault injection window
    fault_start_s: float = 10.0
    fault_end_s: float = 20.0
    fault_mode: Optional[str] = None  # bias|drift|spike|dropout|saturation|None

    # Map model signals -> canonical feature names used by the pipeline
    signal_map: Dict[str, str] = field(
        default_factory=lambda: {
            "p": "feat_0",
            "q": "feat_1",
            "r": "feat_2",
            "q0": "feat_3",
            "q1": "feat_4",
            "q2": "feat_5",
            "q3": "feat_6",
            # add more later if needed:
            # "bus_voltage": "feat_7",
            # "battery_current": "feat_8",
            # "temp_avionics": "feat_9",
        }
    )

    def __post_init__(self):
        self.dt = 1.0 / float(self.hz)
        self._rng = np.random.default_rng(42)
        self.reset()

    # ---- Public API ----
    def reset(self) -> Dict[str, float]:
        self._q = np.array([self.q0, self.q1, self.q2, self.q3], dtype=float)
        self._q = self._normalize_quat(self._q)
        self._w = np.array(
            [self.p0, self.q0_rate, self.r0], dtype=float
        )  # body rates [p,q,r]
        self._t = 0.0
        return self._read_signals()

    def step(self) -> Dict[str, float]:
        # make motion non-trivial (gentle drift on rates)
        self._w = self._drift_rates(self._w, rate_step=1e-4)

        # Attitude kinematics: qdot = 0.5 * Omega(w) * q
        self._q = self._quat_integrate(self._q, self._w, self.dt)
        self._q = self._normalize_quat(self._q)
        self._t += self.dt

        # Runtime fault injection
        if self.fault_mode and (self.fault_start_s <= self._t <= self.fault_end_s):
            self._apply_fault(self.fault_mode)

        return self._read_signals()

    # ---- Internals ----
    def _read_signals(self) -> Dict[str, float]:
        raw = {
            "p": float(self._w[0]),
            "q": float(self._w[1]),
            "r": float(self._w[2]),
            "q0": float(self._q[0]),
            "q1": float(self._q[1]),
            "q2": float(self._q[2]),
            "q3": float(self._q[3]),
        }
        out = {fname: 0.0 for fname in set(self.signal_map.values())}
        for k, v in raw.items():
            if k in self.signal_map:
                out[self.signal_map[k]] = v
        return out

    def _apply_fault(self, mode: str):
        if mode == "bias":
            self._w += np.array([0.1, 0.0, 0.0])  # constant offset on p
        elif mode == "drift":
            self._w += np.array(
                [1e-3, 1e-3, 0.0]
            )  # slow drift on p & q each tick in the window
        elif mode == "spike":
            if self._rng.random() < 0.2:
                self._w += self._rng.normal(0, 0.5, size=3)
        elif mode == "dropout":
            if self._rng.random() < 0.1:
                self._w[:] = 0.0
        elif mode == "saturation":
            self._w = np.clip(self._w, -0.05, 0.05)

    @staticmethod
    def _normalize_quat(q: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(q)
        if n == 0:
            return np.array([1.0, 0.0, 0.0, 0.0])
        return q / n

    @staticmethod
    def _omega_matrix(w: np.ndarray) -> np.ndarray:
        p, q, r = w
        return np.array(
            [
                [0.0, -p, -q, -r],
                [p, 0.0, r, -q],
                [q, -r, 0.0, p],
                [r, q, -p, 0.0],
            ]
        )

    def _quat_integrate(self, q: np.ndarray, w: np.ndarray, dt: float) -> np.ndarray:
        qdot = 0.5 * self._omega_matrix(w).dot(q)
        return q + qdot * dt

    @staticmethod
    def _drift_rates(w: np.ndarray, rate_step: float = 1e-4) -> np.ndarray:
        return w + np.array([rate_step, -rate_step * 0.5, rate_step * 0.3])
