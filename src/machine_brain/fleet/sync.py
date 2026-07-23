"""Layer 6 fleet-sync — robot-local SQLite -> fleet-central PostgreSQL.
Append-only, idempotent, robot -> fleet direction only (see
DATA_CONSISTENCY_STRATEGY.md). `LocalFleetSync` simulates the fleet side
with a second SQLite database standing in for PostgreSQL — same
partition-by-(robot_id, event_time) shape, same idempotent-upsert
contract, so swapping in real PostgreSQL is a connection-string change,
not a redesign.
"""

from __future__ import annotations

import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass

SCHEMA = """
CREATE TABLE IF NOT EXISTS fleet_episodes (
    robot_id TEXT NOT NULL,
    local_row_id TEXT NOT NULL,
    skill_id TEXT NOT NULL,
    event_time REAL NOT NULL,
    payload_json TEXT NOT NULL,
    synced_at REAL NOT NULL,
    PRIMARY KEY (robot_id, local_row_id)
);
CREATE INDEX IF NOT EXISTS idx_fleet_robot_time ON fleet_episodes(robot_id, event_time);
"""


@dataclass(frozen=True)
class SyncBatchResult:
    accepted: int
    duplicates: int


class FleetSyncClient(ABC):
    @abstractmethod
    def push_batch(self, robot_id: str, rows: list[dict]) -> SyncBatchResult: ...

    @abstractmethod
    def last_synced_row_id(self, robot_id: str) -> str | None: ...


class LocalFleetSync(FleetSyncClient):
    def __init__(self, db_path: str) -> None:
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def push_batch(self, robot_id: str, rows: list[dict]) -> SyncBatchResult:
        import json
        import time

        accepted = 0
        duplicates = 0
        with self.conn:
            for row in rows:
                try:
                    self.conn.execute(
                        "INSERT INTO fleet_episodes(robot_id, local_row_id, skill_id, event_time, payload_json, synced_at) "
                        "VALUES (?,?,?,?,?,?)",
                        (robot_id, row["local_row_id"], row["skill_id"], row["event_time"],
                         json.dumps(row.get("payload", {})), time.time()),
                    )
                    accepted += 1
                except sqlite3.IntegrityError:
                    duplicates += 1  # already synced — idempotent replay after reconnect
        return SyncBatchResult(accepted=accepted, duplicates=duplicates)

    def last_synced_row_id(self, robot_id: str) -> str | None:
        row = self.conn.execute(
            "SELECT local_row_id FROM fleet_episodes WHERE robot_id=? ORDER BY event_time DESC LIMIT 1", (robot_id,)
        ).fetchone()
        return row["local_row_id"] if row else None

    def count_for_robot(self, robot_id: str) -> int:
        (n,) = self.conn.execute("SELECT COUNT(*) FROM fleet_episodes WHERE robot_id=?", (robot_id,)).fetchone()
        return n
