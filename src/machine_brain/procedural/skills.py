"""Layer 8 — procedural memory. Skill definitions, versions, preconditions,
permissions, and outcome statistics. SQLite by default (schema is
PostgreSQL-portable — same columns, same types); executable artifacts
would live in object storage (Layer 10) and are never executed directly
from this registry — a skill's `handler` here is a trusted, in-process
Python callable registered at startup, not code deserialized from
storage. Loading and running code fetched from an untrusted memory layer
is out of scope by design (see IMPLEMENTATION_CHECKLIST guardrails).
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Callable

SCHEMA = """
CREATE TABLE IF NOT EXISTS skills (
    skill_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    preconditions_json TEXT NOT NULL,
    permissions_json TEXT NOT NULL,
    success_count INTEGER NOT NULL DEFAULT 0,
    failure_count INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    PRIMARY KEY (skill_id, version)
);
"""


@dataclass(frozen=True)
class SkillDefinition:
    skill_id: str
    version: int
    preconditions: dict           # required working-memory facts, checked by orchestrator
    permissions: tuple[str, ...]  # e.g. ("actuate.motion",)
    handler: Callable[[dict], dict] = field(compare=False)  # in-process only, never deserialized


class SkillRegistry:
    def __init__(self, db_path: str) -> None:
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()
        self._handlers: dict[tuple[str, int], Callable[[dict], dict]] = {}

    def register(self, skill: SkillDefinition) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT OR IGNORE INTO skills(skill_id, version, preconditions_json, permissions_json, "
                "success_count, failure_count, created_at) VALUES (?,?,?,?,0,0,?)",
                (skill.skill_id, skill.version, json.dumps(skill.preconditions),
                 json.dumps(skill.permissions), time.time()),
            )
        self._handlers[(skill.skill_id, skill.version)] = skill.handler

    def latest_version(self, skill_id: str) -> int | None:
        row = self.conn.execute(
            "SELECT MAX(version) as v FROM skills WHERE skill_id=?", (skill_id,)
        ).fetchone()
        return row["v"] if row and row["v"] is not None else None

    def preconditions(self, skill_id: str, version: int | None = None) -> dict:
        resolved_version = version if version is not None else self.latest_version(skill_id)
        if resolved_version is None:
            return {}  # skill was never registered — nothing to report, not an error
        row = self.conn.execute(
            "SELECT preconditions_json FROM skills WHERE skill_id=? AND version=?", (skill_id, resolved_version)
        ).fetchone()
        return json.loads(row["preconditions_json"]) if row else {}

    def handler_for(self, skill_id: str, version: int | None = None) -> Callable[[dict], dict] | None:
        resolved_version = version if version is not None else self.latest_version(skill_id)
        if resolved_version is None:
            return None  # skill was never registered
        return self._handlers.get((skill_id, resolved_version))

    def record_outcome(self, skill_id: str, version: int, succeeded: bool) -> None:
        col = "success_count" if succeeded else "failure_count"
        with self.conn:
            self.conn.execute(f"UPDATE skills SET {col} = {col} + 1 WHERE skill_id=? AND version=?",
                               (skill_id, version))

    def success_rate(self, skill_id: str, version: int | None = None) -> float | None:
        version = version if version is not None else self.latest_version(skill_id)
        row = self.conn.execute(
            "SELECT success_count, failure_count FROM skills WHERE skill_id=? AND version=?",
            (skill_id, version),
        ).fetchone()
        if not row:
            return None
        total = row["success_count"] + row["failure_count"]
        return (row["success_count"] / total) if total else None
