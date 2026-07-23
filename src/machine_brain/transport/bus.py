"""Layer 0 — signal transport. Not a database: carries ObservationFrames and
commands between nodes. Real deployments swap this for ROS 2 DDS; the
interface (`Transport`) is what every other layer depends on, never the
implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Callable

from machine_brain.contracts import ObservationFrame

Handler = Callable[[ObservationFrame], None]


class Transport(ABC):
    @abstractmethod
    def publish(self, frame: ObservationFrame) -> None: ...

    @abstractmethod
    def subscribe(self, topic: str, handler: Handler) -> None: ...


class InProcessBus(Transport):
    """Default transport: in-process pub/sub. Zero external dependency —
    this is what keeps the system runnable on the RAM+SQLite+MCAP floor.
    A ROS2DDSTransport implementing the same interface is a drop-in swap
    for real hardware deployments; it is intentionally not built here
    since it requires a ROS 2 environment this prototype does not assume.
    """

    def __init__(self) -> None:
        self._subscribers: dict[str, list[Handler]] = defaultdict(list)
        self.published_count = 0

    def publish(self, frame: ObservationFrame) -> None:
        self.published_count += 1
        for handler in self._subscribers.get(frame.topic, ()):
            handler(frame)
        for handler in self._subscribers.get("*", ()):
            handler(frame)

    def subscribe(self, topic: str, handler: Handler) -> None:
        self._subscribers[topic].append(handler)
