"""Layer 11 — audit and safety memory. Append-only, hash-chained decision
ledger, SQLite-backed locally. Every guarded decision (allow AND refuse) is
written here unconditionally, before the action is allowed to proceed.
Nothing in learning/consolidation/decay has a code path that can delete or
rewrite a row here — there is no UPDATE or DELETE statement anywhere in
this module, by construction, not just convention.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass

GENESIS_HASH = "0" * 64

SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS ledger (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    decision_id TEXT NOT NULL,
    proposal_id TEXT NOT NULL,
    verdict TEXT NOT NULL,
    reasons_json TEXT NOT NULL,
    rule_ids_json TEXT NOT NULL,
    source TEXT NOT NULL,           -- 'sutraflow' | 'safety_governor'
    recorded_at REAL NOT NULL,
    prev_hash TEXT NOT NULL,
    row_hash TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class LedgerEntry:
    seq: int
    decision_id: str
    proposal_id: str
    verdict: str
    reasons: list[str]
    rule_ids: list[str]
    source: str
    recorded_at: float
    prev_hash: str
    row_hash: str


def _compute_hash(prev_hash: str, decision_id: str, proposal_id: str, verdict: str,
                    reasons: list[str], rule_ids: list[str], source: str, recorded_at: float) -> str:
    payload = json.dumps(
        {"prev_hash": prev_hash, "decision_id": decision_id, "proposal_id": proposal_id,
         "verdict": verdict, "reasons": reasons, "rule_ids": rule_ids, "source": source,
         "recorded_at": recorded_at},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


class AuditLedger:
    def __init__(self, db_path: str) -> None:
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def _last_hash(self) -> str:
        row = self.conn.execute("SELECT row_hash FROM ledger ORDER BY seq DESC LIMIT 1").fetchone()
        return row["row_hash"] if row else GENESIS_HASH

    def record(self, decision_id: str, proposal_id: str, verdict: str, reasons: list[str],
                rule_ids: list[str], source: str) -> LedgerEntry:
        prev_hash = self._last_hash()
        recorded_at = time.time()
        row_hash = _compute_hash(prev_hash, decision_id, proposal_id, verdict, reasons, rule_ids, source, recorded_at)
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO ledger(decision_id, proposal_id, verdict, reasons_json, rule_ids_json, source, "
                "recorded_at, prev_hash, row_hash) VALUES (?,?,?,?,?,?,?,?,?)",
                (decision_id, proposal_id, verdict, json.dumps(reasons), json.dumps(rule_ids), source,
                 recorded_at, prev_hash, row_hash),
            )
        assert cur.lastrowid is not None, "lastrowid is only None when the prior statement wasn't an INSERT"
        return LedgerEntry(seq=cur.lastrowid, decision_id=decision_id, proposal_id=proposal_id, verdict=verdict,
                            reasons=reasons, rule_ids=rule_ids, source=source, recorded_at=recorded_at,
                            prev_hash=prev_hash, row_hash=row_hash)

    def verify_chain(self) -> tuple[bool, int | None]:
        """Returns (ok, first_broken_seq). Recomputes every row's hash from
        its stored fields and prev_hash and checks linkage — this is what
        the audit-tampering adversarial test exercises."""
        prev = GENESIS_HASH
        for row in self.conn.execute("SELECT * FROM ledger ORDER BY seq ASC"):
            expected = _compute_hash(
                prev, row["decision_id"], row["proposal_id"], row["verdict"],
                json.loads(row["reasons_json"]), json.loads(row["rule_ids_json"]), row["source"], row["recorded_at"],
            )
            if row["prev_hash"] != prev or row["row_hash"] != expected:
                return False, row["seq"]
            prev = row["row_hash"]
        return True, None

    def count(self) -> int:
        (n,) = self.conn.execute("SELECT COUNT(*) FROM ledger").fetchone()
        return n

    def entries(self, limit: int = 100) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM ledger ORDER BY seq DESC LIMIT ?", (limit,)).fetchall()
