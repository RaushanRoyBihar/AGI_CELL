# Database Introduction Plan by Project Stage

Every database beyond RAM + SQLite + MCAP is introduced **only when the phase that needs it
starts**, and always behind an adapter interface defined a phase earlier than the database itself.
Nothing in Phase N+1 blocks on infrastructure Phase N didn't need — the system must run and pass
its tests on `RAM + SQLite + MCAP` alone at every phase gate up to Phase 4.

| Phase | New storage introduced | Why now, not earlier | Adapter interface defined |
|---|---|---|---|
| **1. Foundations** | RAM ring buffer, SQLite (WAL), MCAP | These three are the floor of the whole system — nothing else is real without canonical contracts and a place to observe/replay data | `ObservationFrame`, `RingBuffer`, `WorkingMemoryStore` |
| **2. Working/episodic core** | (none new — SQLite schema extended for episodic tables + provenance columns referencing MCAP offsets) | Episodic memory is still single-robot; PostgreSQL isn't needed until there's a fleet | `EpisodicStore`, `ProvenanceRef`, compact typed graph interface (`GraphStore`) using PAIL's in-process graph, no external graph DB yet |
| **3. Prediction** | Parquet (training episodes), safetensors/ONNX (weights) | JEPA training needs a columnar dataset format and a signed model artifact format; still no new *query-serving* database | `WorldModelStore` (train/infer split), baseline-comparison harness |
| **4. Planning & safety** | (none new) | SutraFlow validation and the safety governor are deterministic logic, not storage — this phase proves the guard layer works against Phase 1–3 storage only | `Planner`, `SutraFlowValidator`, `SafetyGovernor`, `AuditLedger` (still backed by SQLite hash-chain) |
| **5. Associative memory** | Qdrant Edge (local, embedded/single-process) | First point embeddings and resonance fingerprints exist in volume large enough that brute-force SQLite scans stop being adequate for candidate generation | `AssociativeIndex` — Qdrant sits behind it; every candidate resolves back to `EpisodicStore`/`WorkingMemoryStore` before use |
| **6. Fleet sync** | PostgreSQL (central), Qdrant server (fleet-scale, replacing Qdrant Edge) | Only once there is more than one robot to reconcile does a central canonical store and a shared vector index make sense | `FleetSyncClient`, ownership-boundary rules (per-robot partition key) |
| **7. Long-term artifacts** | MinIO/S3 (object storage) | Local files are fine for a prototype; fleet scale needs shared artifact storage with hash/ownership/retention rows in PostgreSQL | `ArtifactStore`, `CapsuleSigner` |
| **8. Optional graph/telemetry scale-out** | Neo4j (optional), ClickHouse (optional) | **Only activated by configuration, and only after a benchmark shows the compact graph or PostgreSQL is insufficient.** Never introduced by default | `GraphStore` gains a Neo4j implementation; `TelemetryStore` gains a ClickHouse implementation. Both are drop-in behind existing interfaces from Phase 2/7 |
| **9. Physical reservoir + acoustic ablation** | (none new — reuses MCAP/Parquet from Phase 1/3) | Proves the reservoir adapter and acoustic tie-break logic against existing storage before any new database is considered | `ReservoirAdapter` |

## Non-negotiable ordering rules

1. **SQLite and MCAP exist before anything else is coded.** No phase after Phase 1 may introduce a
   database that bypasses them.
2. **An adapter interface is always defined in the phase before its first non-trivial
   implementation.** E.g., `GraphStore` exists in Phase 2 (backed by the in-process PAIL graph);
   Neo4j only ever fills that same interface in Phase 8, never a new one.
3. **Qdrant, Neo4j, ClickHouse, and MinIO are all optional adapters, gated by configuration.**
   Deleting their config block must leave the system fully functional on RAM + SQLite + MCAP,
   with associative recall falling back to canonical SQLite scans, graph queries running against
   the in-process PAIL graph, and telemetry aggregation running against PostgreSQL/MCAP.
4. **No database is introduced speculatively.** Phase 8's Neo4j/ClickHouse adapters are built only
   once a documented benchmark (Layer 6 graph query latency, Layer 7 telemetry ingestion/query
   latency) shows the existing store is the bottleneck — this mirrors the user's explicit
   instruction not to add ClickHouse "until a benchmark proves PostgreSQL insufficient."
5. **Donor code is pulled in per-phase, not all at once.** E.g., `pail_sutraflow` (from
   `sutraflow_chainflow_pro` / `vajra-v0.39-UNIFIED`) is pulled in at Phase 4, not Phase 1, even
   though it exists today — importing it early would violate "build order" and re-introduce the
   donor's own SQL/vector/graph assumptions before the layer that needs them exists.
