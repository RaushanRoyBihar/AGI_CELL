"""Action-conditioned forward dynamics model — the second, complementary
half of Layer 9's predictive world-model memory.

`world_model/jepa.py` answers "is this transition surprising" in an
unsupervised latent space — good for anomaly detection, useless for
planning because its latent dimensions aren't interpretable and it never
sees what action was taken.

This module answers a different, planning-shaped question: "if I take
action A from this state, what state do I expect next" — in the *same*
interpretable state space the rest of the system already uses (nearest
human distance, nearest obstacle distance, etc.), conditioned on an action
embedding. That's what makes imagination-based planning possible: score
candidate actions by their predicted consequences before committing to
one, without ever having actually taken them.

Small on purpose: one hidden layer, manual NumPy backprop, trained online
from real (state, action, next_state) transitions as they're observed —
consistent with the spec's "train only a small readout layer" ethos
elsewhere (Layer 13) and the "bounded numeric state prediction" scope for
Layer 9 generally.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _init_linear(in_dim: int, out_dim: int, rng: np.random.Generator) -> np.ndarray:
    scale = np.sqrt(2.0 / in_dim)
    return rng.normal(0, scale, size=(in_dim, out_dim))


@dataclass
class DynamicsConfig:
    state_dim: int
    action_dim: int
    hidden_dim: int = 16
    lr: float = 5e-3
    grad_clip_norm: float = 5.0
    seed: int = 0


class ActionConditionedDynamics:
    def __init__(self, config: DynamicsConfig) -> None:
        self.config = config
        rng = np.random.default_rng(config.seed)
        in_dim = config.state_dim + config.action_dim
        self.W1 = _init_linear(in_dim, config.hidden_dim, rng)
        self.b1 = np.zeros(config.hidden_dim)
        self.W2 = _init_linear(config.hidden_dim, config.state_dim, rng)
        self.b2 = np.zeros(config.state_dim)
        self.train_steps = 0

    def _forward(self, state: np.ndarray, action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.concatenate([state, action], axis=-1)
        h_pre = x @ self.W1 + self.b1
        h = np.tanh(h_pre)
        out = h @ self.W2 + self.b2
        return h, out

    def predict(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Imagine: predicted next state, no training, no side effects."""
        _, out = self._forward(np.atleast_1d(state), np.atleast_1d(action))
        return out

    def train_step(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray) -> float:
        state = np.atleast_2d(state).astype(float)
        action = np.atleast_2d(action).astype(float)
        next_state = np.atleast_2d(next_state).astype(float)
        batch = state.shape[0]

        x = np.concatenate([state, action], axis=-1)
        h_pre = x @ self.W1 + self.b1
        h = np.tanh(h_pre)
        pred = h @ self.W2 + self.b2

        diff = pred - next_state
        loss = float(np.mean(np.sum(diff ** 2, axis=1)))

        d_pred = 2 * diff / batch
        d_W2 = h.T @ d_pred
        d_b2 = d_pred.sum(axis=0)

        d_h = d_pred @ self.W2.T
        d_h_pre = d_h * (1 - h ** 2)  # tanh'
        d_W1 = x.T @ d_h_pre
        d_b1 = d_h_pre.sum(axis=0)

        d_W1, d_b1, d_W2, d_b2 = self._clip_grads([d_W1, d_b1, d_W2, d_b2], self.config.grad_clip_norm)

        self.W1 -= self.config.lr * d_W1
        self.b1 -= self.config.lr * d_b1
        self.W2 -= self.config.lr * d_W2
        self.b2 -= self.config.lr * d_b2

        self.train_steps += 1
        return loss

    @staticmethod
    def _clip_grads(grads: list[np.ndarray], max_norm: float) -> list[np.ndarray]:
        total_norm = float(np.sqrt(sum(float(np.sum(g ** 2)) for g in grads)))
        if not np.isfinite(total_norm) or total_norm <= max_norm:
            return grads
        scale = max_norm / (total_norm + 1e-9)
        return [g * scale for g in grads]
