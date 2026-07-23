"""Layer 9 — predictive world-model memory. Bounded numeric state
prediction only (per spec: this is not video-scale JEPA and must not be
described as such). Pure NumPy, adapted from the jepa_world_engine donor
(EMA target encoder, VICReg-style regularization, `surprise()` scoring).

Architecture: online encoder (linear) -> predictor (linear) compared
against an EMA target encoder's output of the *actual* next state. This is
the joint-embedding-predictive-architecture idea in its smallest honest
form: predict in latent space, not raw pixel/signal space, and use a
momentum-updated target to avoid representational collapse.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _init_linear(in_dim: int, out_dim: int, rng: np.random.Generator) -> np.ndarray:
    scale = np.sqrt(2.0 / in_dim)
    return rng.normal(0, scale, size=(in_dim, out_dim))


@dataclass
class JepaConfig:
    state_dim: int
    latent_dim: int = 16
    lr: float = 1e-3
    grad_clip_norm: float = 5.0
    ema_momentum: float = 0.99
    variance_weight: float = 1.0
    covariance_weight: float = 0.1
    variance_target: float = 1.0
    seed: int = 0


class JepaWorldEngine:
    """Bounded numeric-state JEPA. `train_step(state, next_state)` performs
    one online gradient step; `surprise(state, next_state)` scores how
    unexpected a transition is, in latent space, without training."""

    def __init__(self, config: JepaConfig) -> None:
        self.config = config
        rng = np.random.default_rng(config.seed)
        d, k = config.state_dim, config.latent_dim
        # online encoder + predictor
        self.W_enc = _init_linear(d, k, rng)
        self.b_enc = np.zeros(k)
        self.W_pred = _init_linear(k, k, rng)
        self.b_pred = np.zeros(k)
        # EMA target encoder (never trained by gradient descent directly)
        self.W_enc_ema = self.W_enc.copy()
        self.b_enc_ema = self.b_enc.copy()
        self.train_steps = 0

    def _encode(self, state: np.ndarray) -> np.ndarray:
        return state @ self.W_enc + self.b_enc

    def _encode_target(self, state: np.ndarray) -> np.ndarray:
        return state @ self.W_enc_ema + self.b_enc_ema

    def _predict_latent(self, z: np.ndarray) -> np.ndarray:
        return z @ self.W_pred + self.b_pred

    def predict_next_state_latent(self, state: np.ndarray) -> np.ndarray:
        return self._predict_latent(self._encode(state))

    def surprise(self, state: np.ndarray, next_state: np.ndarray) -> float:
        """Prediction error in latent space between predicted-next and the
        EMA target encoding of the actually-observed next state. This is
        the signal Phase 3/9 surprise-detection and anomaly benchmarks
        consume."""
        pred = self.predict_next_state_latent(state)
        target = self._encode_target(next_state)
        return float(np.linalg.norm(pred - target))

    @staticmethod
    def _clip_grads(grads: list[np.ndarray], max_norm: float) -> list[np.ndarray]:
        total_norm = float(np.sqrt(sum(float(np.sum(g ** 2)) for g in grads)))
        if not np.isfinite(total_norm) or total_norm <= max_norm:
            return grads
        scale = max_norm / (total_norm + 1e-9)
        return [g * scale for g in grads]

    def train_step(self, state: np.ndarray, next_state: np.ndarray) -> dict[str, float]:
        cfg = self.config
        state = np.atleast_2d(state).astype(float)
        next_state = np.atleast_2d(next_state).astype(float)
        batch = state.shape[0]

        z = state @ self.W_enc + self.b_enc          # online latent of current state
        target = next_state @ self.W_enc_ema + self.b_enc_ema  # EMA latent of next state (no grad)
        pred = z @ self.W_pred + self.b_pred          # predicted next latent

        diff = pred - target
        invariance_loss = float(np.mean(np.sum(diff ** 2, axis=1)))

        # VICReg-style variance term: penalize latent dims collapsing to a point
        std = np.sqrt(z.var(axis=0) + 1e-4)
        variance_loss = float(np.mean(np.maximum(0.0, cfg.variance_target - std)))

        # covariance term: decorrelate latent dimensions
        z_centered = z - z.mean(axis=0, keepdims=True)
        cov = (z_centered.T @ z_centered) / max(batch - 1, 1)
        off_diag = cov - np.diag(np.diag(cov))
        covariance_loss = float(np.sum(off_diag ** 2) / cfg.latent_dim)

        total_loss = invariance_loss + cfg.variance_weight * variance_loss + cfg.covariance_weight * covariance_loss

        # --- manual gradients (linear layers only, kept small and inspectable) ---
        d_pred = 2 * diff / batch                      # dL/dpred
        d_Wpred = z.T @ d_pred
        d_bpred = d_pred.sum(axis=0)
        d_z_from_pred = d_pred @ self.W_pred.T

        # crude variance-loss gradient push: nudge z away from its mean when std is below target
        below = (std < cfg.variance_target).astype(float)
        d_z_var = -cfg.variance_weight * below * (z - z.mean(axis=0, keepdims=True)) / (std + 1e-4) / batch

        d_z = d_z_from_pred + d_z_var
        d_Wenc = state.T @ d_z
        d_benc = d_z.sum(axis=0)

        d_Wpred, d_bpred, d_Wenc, d_benc = self._clip_grads(
            [d_Wpred, d_bpred, d_Wenc, d_benc], cfg.grad_clip_norm
        )

        self.W_pred -= cfg.lr * d_Wpred
        self.b_pred -= cfg.lr * d_bpred
        self.W_enc -= cfg.lr * d_Wenc
        self.b_enc -= cfg.lr * d_benc

        # EMA update of target encoder — never receives gradients directly
        m = cfg.ema_momentum
        self.W_enc_ema = m * self.W_enc_ema + (1 - m) * self.W_enc
        self.b_enc_ema = m * self.b_enc_ema + (1 - m) * self.b_enc

        self.train_steps += 1
        return {
            "total_loss": total_loss, "invariance_loss": invariance_loss,
            "variance_loss": variance_loss, "covariance_loss": covariance_loss,
        }
