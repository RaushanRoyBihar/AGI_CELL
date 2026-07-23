"""Layer 12 — acoustic/resonance memory. Compact resonance fingerprints,
tie-break authority only. This module cannot be asked to decide anything
on its own — its one entry point, `break_tie`, only ever chooses between
candidates that an upstream caller has already established as otherwise
equivalent (same canonical identity resolution, no contradiction, safety
already cleared). It has no code path that can be invoked before those
checks, by API shape: `break_tie` takes `equivalent_candidates`, plural,
never a single unchecked candidate to "approve".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ResonanceFingerprint:
    source_id: str
    vector: np.ndarray


def fingerprint(signal: np.ndarray, n_bands: int = 8) -> ResonanceFingerprint:
    """Compact spectral-band energy fingerprint — a stand-in for the
    donor's NADA/MFCC-based fingerprints, kept dependency-free (no scipy)
    for this prototype. Real deployments swap the feature extraction, not
    the tie-break contract below."""
    signal = np.asarray(signal, dtype=float)
    bands = np.array_split(np.abs(np.fft.rfft(signal)), n_bands) if signal.size > 1 else [signal] * n_bands
    energies = np.array([float(np.mean(b)) if b.size else 0.0 for b in bands])
    norm = np.linalg.norm(energies) + 1e-9
    return ResonanceFingerprint(source_id="", vector=energies / norm)


def similarity(a: ResonanceFingerprint, b: ResonanceFingerprint) -> float:
    return float(np.dot(a.vector, b.vector))


def break_tie(query: ResonanceFingerprint, equivalent_candidates: list[ResonanceFingerprint]) -> str | None:
    """Only called once identity/contradiction/safety have already reduced
    the field to candidates considered equivalent by canonical logic.
    Returns the source_id of the closest resonance match, or None if the
    input list is empty — never fabricates a winner."""
    if not equivalent_candidates:
        return None
    scored = [(c.source_id, similarity(query, c)) for c in equivalent_candidates]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[0][0]
