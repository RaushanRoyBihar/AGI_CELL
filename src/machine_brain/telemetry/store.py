"""Layer 7 — temporal telemetry memory. No separate database by default:
MCAP (raw) + SQLite standing in for PostgreSQL (aggregates), per spec.
ClickHouse is an optional adapter, added only if a benchmark proves
PostgreSQL insufficient for ingestion or time-window aggregation — not
built speculatively here, only the interface it would fill.
"""

from __future__ import annotations

import sqlite3
import time
from abc import ABC, abstractmethod

SCHEMA = """
CREATE TABLE IF NOT EXISTS telemetry (
    robot_id TEXT NOT NULL,
    metric TEXT NOT NULL,
    value REAL NOT NULL,
    event_time REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_telemetry_metric_time ON telemetry(metric, event_time);
"""


class TelemetryStore(ABC):
    @abstractmethod
    def record(self, robot_id: str, metric: str, value: float) -> None: ...

    @abstractmethod
    def window_aggregate(self, metric: str, seconds: float, agg: str = "avg") -> float | None: ...


class SQLiteTelemetryStore(TelemetryStore):
    def __init__(self, db_path: str) -> None:
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def record(self, robot_id: str, metric: str, value: float) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO telemetry(robot_id, metric, value, event_time) VALUES (?,?,?,?)",
                (robot_id, metric, value, time.time()),
            )

    def window_aggregate(self, metric: str, seconds: float, agg: str = "avg") -> float | None:
        cutoff = time.time() - seconds
        sql_agg = {"avg": "AVG", "max": "MAX", "min": "MIN", "sum": "SUM", "count": "COUNT"}[agg]
        row = self.conn.execute(
            f"SELECT {sql_agg}(value) as v FROM telemetry WHERE metric=? AND event_time >= ?", (metric, cutoff)
        ).fetchone()
        return row["v"]


class ClickHouseTelemetryStore(TelemetryStore):
    """Optional adapter — activated via config/adapters.yaml `telemetry:
    clickhouse`, only after a documented benchmark shows SQLiteTelemetryStore
    (or PostgreSQL at fleet scale) can't keep up."""

    def __init__(self, dsn: str) -> None:
        try:
            import clickhouse_connect  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "ClickHouseTelemetryStore requires 'clickhouse-connect'. This adapter is optional and "
                "gated on a benchmark — the system runs on SQLiteTelemetryStore without it."
            ) from e
        self._client = clickhouse_connect.get_client(dsn=dsn)

    def record(self, robot_id, metric, value):
        raise NotImplementedError("wire real ClickHouse insert when benchmark justifies activation")

    def window_aggregate(self, metric, seconds, agg="avg"):
        raise NotImplementedError
