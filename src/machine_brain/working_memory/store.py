"""Layer 3 — working memory. SQLite WAL, per robot. Canonical (local).
Active task, current entities, attention state, pending decisions,
contradictions, execution receipts. TTL + capacity limits + transactional
writes. This table set is also where restart recovery reads from on boot.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
import time
from dataclasses import dataclass

from machine_brain.contracts import Contradiction, Goal, GoalStatus, WorldEntity, monotonic_ns

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS tasks (
    task_id TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    status TEXT NOT NULL,       -- active | interrupted | done
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS entities (
    entity_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    attributes_json TEXT NOT NULL,
    last_seen_ns INTEGER NOT NULL,
    confidence REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS attention_state (
    entity_id TEXT PRIMARY KEY,
    salience REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS pending_decisions (
    proposal_id TEXT PRIMARY KEY,
    skill_id TEXT NOT NULL,
    args_json TEXT NOT NULL,
    status TEXT NOT NULL,       -- pending | allowed | refused | held | executed
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS contradictions (
    contradiction_id TEXT PRIMARY KEY,
    subject TEXT NOT NULL,
    claim_a_evidence TEXT NOT NULL,
    claim_b_evidence TEXT NOT NULL,
    detail TEXT NOT NULL,
    created_at_ns INTEGER NOT NULL,
    resolved INTEGER NOT NULL DEFAULT 0,
    resolution TEXT
);

CREATE TABLE IF NOT EXISTS goals (
    goal_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    target_json TEXT NOT NULL,
    created_at_ns INTEGER NOT NULL,
    status TEXT NOT NULL       -- active | completed | abandoned
);

CREATE TABLE IF NOT EXISTS execution_receipts (
    receipt_id TEXT PRIMARY KEY,
    proposal_id TEXT NOT NULL,
    outcome_id TEXT,
    succeeded INTEGER,
    detail_json TEXT NOT NULL,
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_entities_last_seen ON entities(last_seen_ns);
CREATE INDEX IF NOT EXISTS idx_contradictions_resolved ON contradictions(resolved);
"""


@dataclass
class WorkingMemoryConfig:
    max_entities: int = 5000
    entity_ttl_seconds: float = 3600.0


class WorkingMemoryStore:
    def __init__(self, db_path: str, config: WorkingMemoryConfig | None = None) -> None:
        self.db_path = db_path
        self.config = config or WorkingMemoryConfig()
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    @contextlib.contextmanager
    def transaction(self):
        try:
            yield self.conn
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    # --- entities ---------------------------------------------------

    def upsert_entity(self, entity: WorldEntity) -> None:
        with self.transaction() as c:
            c.execute(
                """INSERT INTO entities(entity_id, kind, attributes_json, last_seen_ns, confidence, updated_at)
                   VALUES (?,?,?,?,?,?)
                   ON CONFLICT(entity_id) DO UPDATE SET
                     kind=excluded.kind, attributes_json=excluded.attributes_json,
                     last_seen_ns=excluded.last_seen_ns, confidence=excluded.confidence,
                     updated_at=excluded.updated_at""",
                (entity.entity_id, entity.kind, json.dumps(entity.attributes),
                 entity.last_seen_ns, entity.confidence, time.time()),
            )
            self._enforce_entity_capacity(c)

    def _enforce_entity_capacity(self, c) -> None:
        (count,) = c.execute("SELECT COUNT(*) FROM entities").fetchone()
        if count > self.config.max_entities:
            overflow = count - self.config.max_entities
            c.execute(
                "DELETE FROM entities WHERE entity_id IN "
                "(SELECT entity_id FROM entities ORDER BY last_seen_ns ASC LIMIT ?)",
                (overflow,),
            )

    def purge_expired_entities(self, now_ns: int | None = None) -> int:
        now_ns = now_ns if now_ns is not None else monotonic_ns()
        cutoff = now_ns - int(self.config.entity_ttl_seconds * 1e9)
        with self.transaction() as c:
            cur = c.execute("DELETE FROM entities WHERE last_seen_ns < ?", (cutoff,))
            return cur.rowcount

    def get_entity(self, entity_id: str) -> WorldEntity | None:
        row = self.conn.execute("SELECT * FROM entities WHERE entity_id=?", (entity_id,)).fetchone()
        if not row:
            return None
        return WorldEntity(
            entity_id=row["entity_id"], kind=row["kind"],
            attributes=json.loads(row["attributes_json"]),
            last_seen_ns=row["last_seen_ns"], confidence=row["confidence"],
        )

    def all_entities(self) -> list[WorldEntity]:
        rows = self.conn.execute("SELECT * FROM entities").fetchall()
        return [WorldEntity(entity_id=r["entity_id"], kind=r["kind"],
                             attributes=json.loads(r["attributes_json"]),
                             last_seen_ns=r["last_seen_ns"], confidence=r["confidence"]) for r in rows]

    # --- attention ----------------------------------------------------

    def set_attention(self, entity_id: str, salience: float) -> None:
        with self.transaction() as c:
            c.execute(
                """INSERT INTO attention_state(entity_id, salience, updated_at) VALUES (?,?,?)
                   ON CONFLICT(entity_id) DO UPDATE SET salience=excluded.salience, updated_at=excluded.updated_at""",
                (entity_id, salience, time.time()),
            )

    def top_attention(self, n: int = 5) -> list[tuple[str, float]]:
        rows = self.conn.execute(
            "SELECT entity_id, salience FROM attention_state ORDER BY salience DESC LIMIT ?", (n,)
        ).fetchall()
        return [(r["entity_id"], r["salience"]) for r in rows]

    # --- pending decisions / execution receipts ------------------------

    def record_pending_decision(self, proposal_id: str, skill_id: str, args: dict) -> None:
        with self.transaction() as c:
            c.execute(
                "INSERT OR REPLACE INTO pending_decisions(proposal_id, skill_id, args_json, status, created_at) "
                "VALUES (?,?,?,?,?)",
                (proposal_id, skill_id, json.dumps(args), "pending", time.time()),
            )

    def update_decision_status(self, proposal_id: str, status: str) -> None:
        with self.transaction() as c:
            c.execute("UPDATE pending_decisions SET status=? WHERE proposal_id=?", (status, proposal_id))

    def interrupted_decisions(self) -> list[sqlite3.Row]:
        """Used on restart: anything still 'pending' at last shutdown was
        in-flight, not committed — must not be silently resumed as if it
        completed."""
        return self.conn.execute(
            "SELECT * FROM pending_decisions WHERE status='pending'"
        ).fetchall()

    def record_execution_receipt(self, receipt_id: str, proposal_id: str, outcome_id: str | None,
                                   succeeded: bool | None, detail: dict) -> None:
        with self.transaction() as c:
            c.execute(
                "INSERT INTO execution_receipts(receipt_id, proposal_id, outcome_id, succeeded, detail_json, created_at) "
                "VALUES (?,?,?,?,?,?)",
                (receipt_id, proposal_id, outcome_id, int(bool(succeeded)) if succeeded is not None else None,
                 json.dumps(detail), time.time()),
            )

    # --- contradictions -------------------------------------------------

    def record_contradiction(self, contradiction: Contradiction) -> None:
        with self.transaction() as c:
            c.execute(
                "INSERT INTO contradictions(contradiction_id, subject, claim_a_evidence, claim_b_evidence, "
                "detail, created_at_ns, resolved, resolution) VALUES (?,?,?,?,?,?,?,?)",
                (contradiction.contradiction_id, contradiction.subject, contradiction.claim_a_evidence,
                 contradiction.claim_b_evidence, contradiction.detail, contradiction.created_at_ns,
                 int(contradiction.resolved), contradiction.resolution),
            )

    def unresolved_contradictions(self) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM contradictions WHERE resolved=0").fetchall()

    def resolve_contradiction(self, contradiction_id: str, resolution: str) -> None:
        with self.transaction() as c:
            c.execute(
                "UPDATE contradictions SET resolved=1, resolution=? WHERE contradiction_id=?",
                (resolution, contradiction_id),
            )

    # --- goals ------------------------------------------------------------

    def add_goal(self, goal: Goal) -> None:
        with self.transaction() as c:
            c.execute(
                "INSERT INTO goals(goal_id, kind, target_json, created_at_ns, status) VALUES (?,?,?,?,?)",
                (goal.goal_id, goal.kind, json.dumps(goal.target), goal.created_at_ns, goal.status.value),
            )

    def active_goal(self) -> Goal | None:
        """Single-focus by design: the oldest still-active goal governs.
        A bounded agent should be pursuing one thing at a time, not an
        unbounded goal stack it never finishes."""
        row = self.conn.execute(
            "SELECT * FROM goals WHERE status='active' ORDER BY created_at_ns ASC LIMIT 1"
        ).fetchone()
        if not row:
            return None
        return Goal(goal_id=row["goal_id"], kind=row["kind"], target=json.loads(row["target_json"]),
                     created_at_ns=row["created_at_ns"], status=GoalStatus(row["status"]))

    def complete_goal(self, goal_id: str) -> None:
        with self.transaction() as c:
            c.execute("UPDATE goals SET status='completed' WHERE goal_id=?", (goal_id,))

    def abandon_goal(self, goal_id: str) -> None:
        with self.transaction() as c:
            c.execute("UPDATE goals SET status='abandoned' WHERE goal_id=?", (goal_id,))

    def close(self) -> None:
        self.conn.close()
