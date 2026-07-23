"""Layer 1 — sensory memory. Fixed-capacity ring buffer. Never performs a
database write on this path — that is the hard real-time control-path rule.
Keeps only a recent, configurable time window; older frames are simply
overwritten in place.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from machine_brain.contracts import ObservationFrame


@dataclass
class RingBufferConfig:
    capacity: int = 2048
    window_seconds: float = 5.0


class RingBuffer:
    """Per-topic fixed-capacity buffer. Backed by `collections.deque` with a
    maxlen — O(1) append, automatic eviction of the oldest frame, no heap
    growth. A NumPy structured-array variant is a drop-in swap if a topic's
    payload has a fixed numeric shape and needs vectorized reads; the
    interface below is what perception code depends on, not the backing
    store.
    """

    def __init__(self, config: RingBufferConfig | None = None) -> None:
        self.config = config or RingBufferConfig()
        self._buffers: dict[str, deque[ObservationFrame]] = {}
        self._seen_keys: dict[str, set[str]] = {}
        self.dropped_stale = 0
        self.dropped_duplicate = 0

    def _buffer_for(self, topic: str) -> deque[ObservationFrame]:
        if topic not in self._buffers:
            self._buffers[topic] = deque(maxlen=self.config.capacity)
            self._seen_keys[topic] = set()
        return self._buffers[topic]

    def push(self, frame: ObservationFrame) -> bool:
        """Returns False if the frame was rejected (duplicate)."""
        buf = self._buffer_for(frame.topic)
        seen = self._seen_keys[frame.topic]
        key = frame.idempotency_key()
        if key in seen:
            self.dropped_duplicate += 1
            return False
        if len(buf) == buf.maxlen and buf[0] is not None:
            seen.discard(buf[0].idempotency_key())
        buf.append(frame)
        seen.add(key)
        return True

    def window(self, topic: str, now_ns: int) -> list[ObservationFrame]:
        buf = self._buffer_for(topic)
        cutoff = now_ns - int(self.config.window_seconds * 1e9)
        return [f for f in buf if f.monotonic_ns >= cutoff]

    def latest(self, topic: str) -> ObservationFrame | None:
        buf = self._buffer_for(topic)
        return buf[-1] if buf else None

    def is_stale(self, topic: str, now_ns: int, max_age_seconds: float) -> bool:
        f = self.latest(topic)
        if f is None:
            return True
        return (now_ns - f.monotonic_ns) > int(max_age_seconds * 1e9)
