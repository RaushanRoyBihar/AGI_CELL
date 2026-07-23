# Memory Ownership Matrix

Rule: **one writer of record per layer.** Every other component that touches a layer's data does
so through that layer's adapter interface, never by reaching into its storage directly. This is
what keeps "RAM + SQLite + MCAP" a valid minimal deployment — every layer past that is an optional
adapter behind an interface, never a hard dependency.

| Layer | Storage | Writer of record | Readers | Truth status | Retention |
|---|---|---|---|---|---|
| 0. Signal transport | ROS 2 DDS topics | Sensor/driver nodes, planner/governor nodes | All nodes subscribed to a topic | Not stored — in-flight only | N/A (transport, not memory) |
| 1. Sensory memory | Fixed-capacity shared-memory / NumPy ring buffer | Perception intake process | Feature extraction, attention selection | Ephemeral, never authoritative | Fixed time window (config, e.g. 2–10s), overwritten in place |
| 2. Raw experience | rosbag2 + MCAP files, bounded rotation | Recording node (fast-write path, off the control loop) | Episodic-memory indexer, offline training, incident replay | Archive authority (immutable once sealed) | Bounded rotation (config: N files / N hours), older files roll to Layer 10 or are pruned |
| 3. Working memory | SQLite (WAL mode), per robot | Task executor / active-state manager on that robot | Planner, attention selection, SutraFlow validator | Canonical (local) | TTL + capacity limits per table; transactional |
| 4. Episodic memory | SQLite (single robot) referencing MCAP offsets; PostgreSQL partitioned by robot+event time (fleet) | Episode consolidator (writes after outcome is observed, not before) | Recall/retrieval, reviewed-learning process | Canonical | Partitioned, no TTL by default — pruned only via explicit retention policy |
| 5. Associative memory | Qdrant Edge (local) / Qdrant server (fleet) | Embedding indexer, fed only from Layer 3/4 canonical writes | Retrieval candidate generator | **Derived index only — never truth.** Every hit must resolve back to its canonical row before use | Rebuildable from canonical storage at any time; safe to wipe and reindex |
| 6. Semantic/causal graph | PAIL compact typed graph (later: optional Neo4j adapter) | Association/consolidation process, writes only vetted typed edges (CAUSED_BY, PRECEDES, OBSERVED_AT, EXECUTED_BY, FAILED_AFTER, RESOLVED_BY) | Predictive world model, planner, reviewer | Derived claims with source evidence pointers — not truth on their own | Decays weak/unverified edges on schedule; verified edges consolidate |
| 7. Temporal telemetry | MCAP + PostgreSQL (ClickHouse only if benchmarked as necessary) | Telemetry writer | Fleet analytics, dashboards | Derived/archive hybrid — PostgreSQL rows are canonical for aggregates, MCAP is canonical for raw signal | Time-windowed aggregation; raw MCAP follows Layer 2/10 retention |
| 8. Procedural memory | PostgreSQL (skill metadata, versions, preconditions, permissions, outcome stats) + object storage (executable artifacts) | Skill registry service | Planner, executor | Canonical for metadata; artifacts require signature verification before execution | Versioned, never silently overwritten — new version = new row |
| 9. Predictive world model | Parquet (training episodes), safetensors/ONNX (weights, signed metadata) | Training pipeline (offline, batch) | Planner (inference only, read-only at runtime) | Not truth — a prediction, always checked against observed outcome | Model versions retained until explicitly retired |
| 10. Long-term artifacts | Local files (prototype) / MinIO-S3 (fleet); hashes/ownership/retention in PostgreSQL | Archival service | Anyone needing raw video/audio/model/MCAP/capsule | Archive authority (content) + canonical (metadata in PostgreSQL) | Governed by explicit retention policy per artifact class |
| 11. Audit and safety | Append-only hash-chained ledger, SQLite local -> synced to PostgreSQL -> periodically sealed to immutable object storage | Safety governor and decision-logging middleware (every guarded decision, unconditionally) | Reviewer, compliance/audit tooling, incident investigation | Canonical, tamper-evident | **Never deleted by learning or decay processes.** Sealed batches immutable |
| 12. Acoustic/resonance | Qdrant (fingerprints), object storage (signed capsules) | Acoustic indexer | Tie-break candidate resolver only | Derived, tie-break authority only — cannot override identity/contradiction/safety | Rebuildable; capsules follow Layer 10 retention |
| 13. Physical reservoir adapter | MCAP + Parquet (input/output experiment logs); readout weights in safetensors | Reservoir adapter driver | Predictive world model (as an optional feature source) | Not truth — transient computation, state itself is never persisted | Experiment logs follow Layer 2/7 retention; readout layer versioned like Layer 9 |

## Canonical vs. derived, restated

- **Canonical authority:** SQLite (robot-local), PostgreSQL (fleet-central). These are the only
  stores that may be treated as ground truth for entities, episodes, procedural definitions, and
  audit decisions.
- **Derived indexes:** Qdrant, the semantic/causal graph (PAIL graph, later optional Neo4j),
  ClickHouse (if introduced), acoustic fingerprints. These may propose, rank, and accelerate
  retrieval but may never be the sole basis for an action — every derived hit resolves back to a
  canonical row before the planner or governor acts on it.
- **Archive authority:** Signed MCAP, Parquet, and object-storage artifacts. Authoritative for raw
  content (what a sensor actually recorded, what a model's weights actually are) but not for the
  system's beliefs about that content — those beliefs live in canonical storage as pointers/derived
  rows referencing archive offsets.

## Enforced boundaries

- No derived index (Qdrant/graph/ClickHouse/acoustic) ever writes to canonical storage.
  Consolidation/promotion is one-directional: canonical -> derived index, never the reverse.
- No layer performs a database write inside the hard real-time control path (Layers 0 and 1 are
  transport/ring-buffer only, by construction).
- Layer 11 (audit) is the only layer immune to decay/eviction from Layer 12's or Layer 5's
  learning cycle — safety refusals, e-stops, and conflict decisions are permanent regardless of
  what any other layer later "learns."
- Every layer's adapter is swappable/optional except Layers 0, 1, 3 (RAM/SQLite) and Layer 2 (MCAP)
  — these three plus a filesystem are the minimum viable deployment.
