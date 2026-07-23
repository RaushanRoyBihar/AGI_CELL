# Data Consistency and Synchronization Strategy

## Principle

Consistency is enforced by **ownership + one-directional promotion**, not by distributed
transactions across heterogeneous stores. Each layer owns its writes (see
`MEMORY_OWNERSHIP_MATRIX.md`); cross-layer consistency is achieved by defining, for every pair of
adjacent layers, which direction data is allowed to flow and what the flow must carry as proof of
origin.

## Local (single-robot) consistency

- **SQLite WAL mode** for Layer 3 (working memory) and the local half of Layer 4 (episodic).
  WAL gives us: readers never block writers, crash-safe commits, and a natural point for
  transactional TTL/eviction. Every working-memory mutation is a single transaction; partial
  writes cannot be observed.
- **MCAP offsets as provenance, not copies.** Episodic rows never duplicate raw sensor data —
  they store `(mcap_file_id, topic, offset, timestamp)`. This is what prevents "repeated
  execution creates a false new occurrence" (an explicit requirement): the episode consolidator
  checks for an existing row with matching `(skill_id, precondition_hash, mcap_offset_range)`
  before inserting a new episode, so replaying or reprocessing the same MCAP segment is
  idempotent.
- **Ring buffer -> SQLite boundary.** The ring buffer (Layer 1) never itself performs a database
  write. Perception/feature-extraction reads from the ring buffer and writes derived features to
  Layer 3; the ring buffer's own contents are never treated as durable.

## Fleet (multi-robot) consistency

- **Ownership boundary = robot ID.** PostgreSQL episodic and telemetry tables are partitioned by
  `(robot_id, event_time)`. A robot only ever writes rows it owns; cross-robot reads are allowed,
  cross-robot writes are not.
- **Sync direction is always robot -> fleet, append-only.** A robot's local SQLite is the source
  of truth for its own recent history; PostgreSQL is the source of truth for fleet-wide queries
  once a row has synced. Sync is a batched, idempotent upsert keyed by
  `(robot_id, local_row_id)` — safe to replay after a dropped connection.
- **No fleet-to-robot write-back of canonical state.** A robot never rewrites its local episodic
  history based on what PostgreSQL says other robots did. Cross-robot learning happens at the
  derived layers (Layer 5 associative candidates, Layer 6 graph edges), never by mutating another
  robot's canonical episodes.
- **Conflict handling:** if two robots report contradictory observations about the same external
  entity, this is not resolved by "last write wins." It creates an explicit unresolved
  contradiction record (see Learning Design) that a reviewer or a designated resolution rule must
  clear.

## Derived-index consistency (Qdrant, graph, ClickHouse)

- **Derived indexes are eventually consistent, by design, and that is acceptable** because they
  are never used as sole authority. The consistency requirement on them is narrower: **every
  candidate they return must carry the canonical row ID it came from**, and the caller resolves
  that ID against SQLite/PostgreSQL before acting. A stale or missing vector-index entry produces
  a missed candidate (a recall problem, tolerable and measured), never a false action (a
  correctness problem, not tolerable).
- **Rebuild, don't reconcile.** Because derived indexes are pure functions of canonical data, the
  standard recovery path for a corrupted or diverged index is to rebuild it from canonical storage
  wholesale, not to attempt row-by-row reconciliation.
- **Graph edges carry source evidence pointers.** Every typed edge (`CAUSED_BY`, `PRECEDES`, etc.)
  stores the episodic/audit row(s) that justified it. A graph conclusion without a resolvable
  evidence pointer is invalid and is dropped during consolidation, not trusted.

## Audit ledger consistency

- **Hash-chained, append-only, local-first.** Every guarded decision is written to the local
  SQLite ledger synchronously, before the action is allowed to proceed — this is on the safety
  path, so it uses SQLite (fast, local), not PostgreSQL (network-dependent).
- **Sync to PostgreSQL is asynchronous and additive** — it never rewrites or reorders the local
  chain. PostgreSQL's copy is verified against the local hash chain on sync; a mismatch is a
  tamper/corruption signal, not something auto-resolved.
- **Periodic sealing to object storage** takes a contiguous, verified batch and signs it. Sealed
  batches are immutable; nothing in the system — including reviewed learning and memory decay —
  has a code path capable of deleting or rewriting a sealed batch or the local hash chain.

## Cross-store timestamp and clock discipline

- All canonical writes carry both a monotonic local clock reading (for ordering within a robot)
  and a wall-clock timestamp (for fleet correlation). Clock drift between robots is expected and
  handled at the fleet-sync layer by trusting monotonic order within a robot and treating
  wall-clock timestamps as approximate for cross-robot correlation — this is what the clock-drift
  adversarial test (see test plan) verifies.

## What is explicitly NOT done

- No two-phase commit across SQLite/PostgreSQL/Qdrant/graph.
- No synchronous fan-out write to derived indexes on the hot path — indexing is always
  asynchronous relative to the canonical write that triggered it.
- No derived index is ever queried as the last step before an unguarded action; the
  SutraFlow validator and safety governor always sit between any retrieval (however sourced) and
  actuation.
