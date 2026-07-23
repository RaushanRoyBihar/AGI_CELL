"""Attention selection: active world state -> a bounded set of entities the
rest of the pipeline (episodic recall, association, planning) actually
considers this cycle. Prevents O(n) blow-up over "everything ever
perceived" by scoring salience and keeping only the top-N.
"""

from __future__ import annotations

from dataclasses import dataclass

from machine_brain.contracts import WorldEntity, monotonic_ns


@dataclass
class AttentionConfig:
    top_n: int = 8
    recency_half_life_ns: int = 2_000_000_000  # 2 seconds


def salience(entity: WorldEntity, now_ns: int, config: AttentionConfig) -> float:
    age_ns = max(0, now_ns - entity.last_seen_ns)
    recency = 0.5 ** (age_ns / config.recency_half_life_ns)
    return entity.confidence * recency


class AttentionSelector:
    def __init__(self, config: AttentionConfig | None = None) -> None:
        self.config = config or AttentionConfig()

    def select(self, entities: list[WorldEntity], now_ns: int | None = None) -> list[tuple[WorldEntity, float]]:
        now_ns = now_ns if now_ns is not None else monotonic_ns()
        scored = [(e, salience(e, now_ns, self.config)) for e in entities]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[: self.config.top_n]
