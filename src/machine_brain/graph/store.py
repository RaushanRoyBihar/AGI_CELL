"""Layer 6 — semantic and causal memory. Compact typed graph: only the six
edge types in EdgeType are ever created — no all-to-all co-occurrence
edges. Graph conclusions are derived claims; every edge must carry
evidence_ids pointing back to canonical episodic/audit rows.

Default implementation is in-process + SQLite-backed (no external graph DB
required). Neo4jGraphStore is an optional adapter behind the same
GraphStore interface, activated only by config — never a hard dependency.
"""

from __future__ import annotations

import sqlite3
from abc import ABC, abstractmethod

from machine_brain.contracts import EdgeType, GraphEdge

SCHEMA = """
CREATE TABLE IF NOT EXISTS edges (
    edge_id TEXT PRIMARY KEY,
    src TEXT NOT NULL,
    edge_type TEXT NOT NULL,
    dst TEXT NOT NULL,
    evidence_ids TEXT NOT NULL,
    weight REAL NOT NULL,
    created_at_ns INTEGER NOT NULL,
    verified_count INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src);
CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst);
CREATE INDEX IF NOT EXISTS idx_edges_type ON edges(edge_type);
"""


class GraphStore(ABC):
    @abstractmethod
    def add_edge(self, edge: GraphEdge) -> None: ...

    @abstractmethod
    def add_edges(self, edges: list[GraphEdge]) -> None:
        """Batch insert in a single transaction — used on the hot path
        (one execution can create several edges) to avoid one fsync per
        edge under SQLite's default per-statement commit behavior."""

    @abstractmethod
    def edges_from(self, node: str, edge_type: EdgeType | None = None) -> list[GraphEdge]: ...

    @abstractmethod
    def strengthen(self, edge_id: str, amount: float = 0.1) -> None: ...

    @abstractmethod
    def strengthen_many(self, edge_ids: list[str], amount: float = 0.1) -> None:
        """Batch version — one transaction for a set of edges strengthened
        by the same learning event, instead of one commit per edge.
        Profiling this project's own cognitive loop found `strengthen`
        alone accounting for ~1s of a 3.3s run at ~3,300 individual
        commits; this is the fix, and it's *more* consistent than the
        per-call version, not less — a crash mid-update under the old
        version could leave some edges of one learning event strengthened
        and others not, which is a worse inconsistency than losing the
        whole update atomically."""

    @abstractmethod
    def weaken_many(self, edge_ids: list[str], amount: float = 0.1) -> None: ...

    @abstractmethod
    def weaken(self, edge_id: str, amount: float = 0.1) -> None: ...

    @abstractmethod
    def decay_weak_edges(self, threshold: float, min_verified: int = 1) -> int:
        """Remove edges below `threshold` weight that have never been
        verified — returns count removed. Verified edges are never decayed
        by this call."""

    @abstractmethod
    def edge_count(self) -> int: ...


def _row_to_edge(row: sqlite3.Row) -> GraphEdge:
    return GraphEdge(
        edge_id=row["edge_id"], src=row["src"], edge_type=EdgeType(row["edge_type"]), dst=row["dst"],
        evidence_ids=tuple(row["evidence_ids"].split(",")) if row["evidence_ids"] else (),
        weight=row["weight"], created_at_ns=row["created_at_ns"], verified_count=row["verified_count"],
    )


class SQLiteGraphStore(GraphStore):
    def __init__(self, db_path: str) -> None:
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def add_edge(self, edge: GraphEdge) -> None:
        if not edge.evidence_ids:
            raise ValueError("graph edges must carry at least one evidence id — no conclusion without provenance")
        with self.conn:
            self.conn.execute(
                "INSERT INTO edges(edge_id, src, edge_type, dst, evidence_ids, weight, created_at_ns, verified_count) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (edge.edge_id, edge.src, edge.edge_type.value, edge.dst,
                 ",".join(edge.evidence_ids), edge.weight, edge.created_at_ns, edge.verified_count),
            )

    def add_edges(self, edges: list[GraphEdge]) -> None:
        if not edges:
            return
        for edge in edges:
            if not edge.evidence_ids:
                raise ValueError("graph edges must carry at least one evidence id — no conclusion without provenance")
        with self.conn:
            self.conn.executemany(
                "INSERT INTO edges(edge_id, src, edge_type, dst, evidence_ids, weight, created_at_ns, verified_count) "
                "VALUES (?,?,?,?,?,?,?,?)",
                [(e.edge_id, e.src, e.edge_type.value, e.dst, ",".join(e.evidence_ids), e.weight,
                  e.created_at_ns, e.verified_count) for e in edges],
            )

    def edges_from(self, node: str, edge_type: EdgeType | None = None) -> list[GraphEdge]:
        if edge_type:
            rows = self.conn.execute(
                "SELECT * FROM edges WHERE src=? AND edge_type=?", (node, edge_type.value)
            ).fetchall()
        else:
            rows = self.conn.execute("SELECT * FROM edges WHERE src=?", (node,)).fetchall()
        return [_row_to_edge(r) for r in rows]

    def strengthen(self, edge_id: str, amount: float = 0.1) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE edges SET weight=MIN(1.0, weight+?), verified_count=verified_count+1 WHERE edge_id=?",
                (amount, edge_id),
            )

    def weaken(self, edge_id: str, amount: float = 0.1) -> None:
        with self.conn:
            self.conn.execute("UPDATE edges SET weight=MAX(0.0, weight-?) WHERE edge_id=?", (amount, edge_id))

    def strengthen_many(self, edge_ids: list[str], amount: float = 0.1) -> None:
        if not edge_ids:
            return
        with self.conn:
            self.conn.executemany(
                "UPDATE edges SET weight=MIN(1.0, weight+?), verified_count=verified_count+1 WHERE edge_id=?",
                [(amount, edge_id) for edge_id in edge_ids],
            )

    def weaken_many(self, edge_ids: list[str], amount: float = 0.1) -> None:
        if not edge_ids:
            return
        with self.conn:
            self.conn.executemany(
                "UPDATE edges SET weight=MAX(0.0, weight-?) WHERE edge_id=?",
                [(amount, edge_id) for edge_id in edge_ids],
            )

    def decay_weak_edges(self, threshold: float, min_verified: int = 1) -> int:
        with self.conn:
            cur = self.conn.execute(
                "DELETE FROM edges WHERE weight < ? AND verified_count < ?", (threshold, min_verified)
            )
            return cur.rowcount

    def edge_count(self) -> int:
        (n,) = self.conn.execute("SELECT COUNT(*) FROM edges").fetchone()
        return n


class Neo4jGraphStore(GraphStore):
    """Optional adapter — same GraphStore interface, activated only via
    config/adapters.yaml `graph: neo4j`. Requires the `neo4j` extra
    (`pip install machine_brain[neo4j]`) and a running Neo4j instance; not
    used by default and not required for the system to function.
    """

    def __init__(self, uri: str, user: str, password: str) -> None:
        try:
            from neo4j import GraphDatabase  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "Neo4jGraphStore requires the 'neo4j' package (pip install machine_brain[neo4j]). "
                "This adapter is optional — the system runs on SQLiteGraphStore without it."
            ) from e
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def add_edge(self, edge: GraphEdge) -> None:
        with self._driver.session() as session:
            session.run(
                "MERGE (a {id:$src}) MERGE (b {id:$dst}) "
                "CREATE (a)-[r:%s {edge_id:$edge_id, evidence_ids:$evidence, weight:$weight}]->(b)"
                % edge.edge_type.value,
                src=edge.src, dst=edge.dst, edge_id=edge.edge_id,
                evidence=list(edge.evidence_ids), weight=edge.weight,
            )

    def add_edges(self, edges: list[GraphEdge]) -> None:
        for edge in edges:
            self.add_edge(edge)

    def edges_from(self, node, edge_type=None):
        raise NotImplementedError("wire real Cypher query when Neo4j is actually deployed")

    def strengthen(self, edge_id, amount=0.1):
        raise NotImplementedError

    def weaken(self, edge_id, amount=0.1):
        raise NotImplementedError

    def strengthen_many(self, edge_ids, amount=0.1):
        raise NotImplementedError

    def weaken_many(self, edge_ids, amount=0.1):
        raise NotImplementedError

    def decay_weak_edges(self, threshold, min_verified=1):
        raise NotImplementedError

    def edge_count(self):
        raise NotImplementedError
