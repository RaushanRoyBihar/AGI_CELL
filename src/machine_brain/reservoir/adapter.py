"""Layer 13 — physical reservoir adapter. Interface for future SAW,
quartz, mechanical, or photonic reservoir hardware. Input: a normalized
temporal signal window. Output: a reservoir state vector, timestamp, and
device calibration record. Only the readout layer is trained; the
reservoir's dynamics themselves are never treated as a persistent
database — this module holds no memory across calls beyond the recurrent
state needed to compute the next output, which is transient by design.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from machine_brain.contracts import wall_time


@dataclass(frozen=True)
class ReservoirReading:
    state_vector: np.ndarray
    timestamp: float
    calibration: dict


class ReservoirAdapter(ABC):
    @abstractmethod
    def step(self, signal_window: np.ndarray) -> ReservoirReading: ...


@dataclass
class SimulatedReservoirConfig:
    input_dim: int
    reservoir_dim: int = 64
    spectral_radius: float = 0.9
    leak_rate: float = 0.3
    seed: int = 0


class SimulatedReservoir(ReservoirAdapter):
    """Random-projection leaky-integrator reservoir — the standard
    software stand-in for a physical reservoir's nonlinear dynamics until
    real SAW/quartz/photonic hardware is wired in behind this same
    interface. State is transient (kept only for the leaky-integration
    recurrence), never persisted as memory."""

    def __init__(self, config: SimulatedReservoirConfig) -> None:
        self.config = config
        rng = np.random.default_rng(config.seed)
        W_in = rng.normal(0, 1, size=(config.input_dim, config.reservoir_dim))
        W_res = rng.normal(0, 1, size=(config.reservoir_dim, config.reservoir_dim))
        radius = max(abs(np.linalg.eigvals(W_res)))
        self.W_in = W_in
        self.W_res = W_res * (config.spectral_radius / (radius + 1e-9))
        self._state = np.zeros(config.reservoir_dim)

    def step(self, signal_window: np.ndarray) -> ReservoirReading:
        x = np.asarray(signal_window, dtype=float)
        if x.shape[-1] != self.config.input_dim:
            x = np.resize(x, self.config.input_dim)
        pre_activation = x @ self.W_in + self._state @ self.W_res
        new_state = np.tanh(pre_activation)
        self._state = (1 - self.config.leak_rate) * self._state + self.config.leak_rate * new_state
        return ReservoirReading(
            state_vector=self._state.copy(),
            timestamp=wall_time(),
            calibration={"spectral_radius": self.config.spectral_radius, "leak_rate": self.config.leak_rate},
        )


class ReadoutLayer:
    """The only trained component. Simple ridge-regression readout mapping
    reservoir state -> target — small on purpose, per spec ("train only a
    small readout layer")."""

    def __init__(self, reservoir_dim: int, output_dim: int, ridge: float = 1e-3) -> None:
        self.W = np.zeros((reservoir_dim, output_dim))
        self.ridge = ridge

    def fit(self, states: np.ndarray, targets: np.ndarray) -> None:
        reg = self.ridge * np.eye(states.shape[1])
        self.W = np.linalg.solve(states.T @ states + reg, states.T @ targets)

    def predict(self, state: np.ndarray) -> np.ndarray:
        return state @ self.W
