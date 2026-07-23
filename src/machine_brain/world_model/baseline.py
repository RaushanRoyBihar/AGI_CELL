"""Static baseline predictor. Built before JEPA so "prediction error vs.
baseline" has a real denominator from day one — JEPA is not allowed to be
called a working predictive world model until it beats this on held-out
data (see reports produced by tests/benchmarks).
"""

from __future__ import annotations

import numpy as np


class LastValueBaseline:
    """Predicts the next state equals the current state. Trivial, but a
    real, honest baseline — many physical/robot state signals are
    slow-changing enough that this is a non-trivial bar to beat."""

    def predict(self, state: np.ndarray) -> np.ndarray:
        return state.copy()


class LinearVelocityBaseline:
    """Predicts the next state via constant first-difference extrapolation:
    next = state + (state - previous_state). Slightly stronger baseline for
    signals with momentum (position, velocity-like channels)."""

    def predict(self, state: np.ndarray, previous_state: np.ndarray | None) -> np.ndarray:
        if previous_state is None:
            return state.copy()
        return state + (state - previous_state)


def prediction_error(predicted: np.ndarray, actual: np.ndarray) -> float:
    return float(np.linalg.norm(predicted - actual))
