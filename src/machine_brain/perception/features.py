"""Perception and feature extraction: ring buffer -> active world state.
Also where cheap, local anomaly detection lives (stuck sensor, delayed
timestamp) since it has direct access to the raw window.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from machine_brain.contracts import ObservationFrame, WorldEntity, monotonic_ns, new_id


@dataclass
class PerceptionConfig:
    stuck_variance_eps: float = 1e-9
    stuck_window: int = 5
    max_timestamp_delay_seconds: float = 1.0


@dataclass
class PerceptionFlags:
    stale_topics: set[str] = field(default_factory=set)
    stuck_topics: set[str] = field(default_factory=set)
    delayed_frame_ids: set[str] = field(default_factory=set)


class PerceptionEngine:
    def __init__(self, config: PerceptionConfig | None = None) -> None:
        self.config = config or PerceptionConfig()

    def check_delay(self, frame: ObservationFrame) -> bool:
        """True if the frame's embedded timestamp is implausibly far from
        arrival time (delayed-timestamp adversarial case)."""
        now = monotonic_ns()
        delay_s = (now - frame.monotonic_ns) / 1e9
        return delay_s > self.config.max_timestamp_delay_seconds

    def check_stuck(self, window: list[ObservationFrame], value_key: str) -> bool:
        if len(window) < self.config.stuck_window:
            return False
        recent = window[-self.config.stuck_window :]
        raw_values = [f.payload.get(value_key) for f in recent]
        if any(v is None for v in raw_values):
            return False
        values: list[float] = [v for v in raw_values if v is not None]  # narrows for the type checker, matches the check above
        variance = sum((v - values[0]) ** 2 for v in values) / len(values)
        return variance < self.config.stuck_variance_eps

    def extract_entity(self, frame: ObservationFrame, confidence: float = 0.8) -> WorldEntity:
        return WorldEntity(
            entity_id=frame.payload.get("entity_id", frame.sensor_id),
            kind=frame.payload.get("kind", frame.topic),
            attributes=dict(frame.payload),
            last_seen_ns=frame.monotonic_ns,
            confidence=confidence,
        )
