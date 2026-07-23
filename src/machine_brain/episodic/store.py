"""Layer 4 — episodic memory. Single-robot: SQLite referencing MCAP offsets.
Fleet-scale swap target: PostgreSQL partitioned by (robot_id, event_time) —
same schema shape, same dedupe_key uniqueness constraint, so the migration
is a row copy, not a redesign. Repeated execution over the same
(skill, precondition, mcap offset range) must not create a false new
occurrence — enforced here via a UNIQUE constraint on dedupe_key.
"""

from __future__ import annotations

import sqlite3

from machine_brain.contracts import Episode

SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS episodes (
    episode_id TEXT PRIMARY KEY,
    robot_id TEXT NOT NULL,
    skill_id TEXT NOT NULL,
    precondition_hash TEXT NOT NULL,
    mcap_file_id TEXT NOT NULL,
    mcap_offset_start INTEGER NOT NULL,
    mcap_offset_end INTEGER NOT NULL,
    proposal_id TEXT NOT NULL,
    outcome_id TEXT,
    event_time REAL NOT NULL,
    dedupe_key TEXT NOT NULL UNIQUE
);

CREATE INDEX IF NOT EXISTS idx_episodes_robot_time ON episodes(robot_id, event_time);
CREATE INDEX IF NOT EXISTS idx_episodes_skill ON episodes(skill_id);
"""


class EpisodicStore:
    def __init__(self, db_path: str) -> None:
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()
        self.duplicate_rejections = 0

    def record(self, episode: Episode) -> bool:
        """Returns True if a new row was inserted, False if this exact
        (skill, precondition, mcap-range) episode already existed —
        idempotent replay guarantee."""
        try:
            with self.conn:
                self.conn.execute(
                    """INSERT INTO episodes(episode_id, robot_id, skill_id, precondition_hash,
                        mcap_file_id, mcap_offset_start, mcap_offset_end, proposal_id, outcome_id,
                        event_time, dedupe_key) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                    (episode.episode_id, episode.robot_id, episode.skill_id, episode.precondition_hash,
                     episode.mcap_file_id, episode.mcap_offset_start, episode.mcap_offset_end,
                     episode.proposal_id, episode.outcome_id, episode.event_time, episode.dedupe_key),
                )
            return True
        except sqlite3.IntegrityError:
            self.duplicate_rejections += 1
            return False

    def by_skill(self, skill_id: str, limit: int = 50) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM episodes WHERE skill_id=? ORDER BY event_time DESC LIMIT ?", (skill_id, limit)
        ).fetchall()

    def get(self, episode_id: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM episodes WHERE episode_id=?", (episode_id,)).fetchone()

    def count(self) -> int:
        (n,) = self.conn.execute("SELECT COUNT(*) FROM episodes").fetchone()
        return n

    def recent(self, limit: int = 100) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM episodes ORDER BY event_time DESC LIMIT ?", (limit,)).fetchall()
