# Exact Implementation Checklist

All work happens under `machine_brain/`. The existing RAG product and website are never touched.
Nothing is committed or pushed until explicitly requested. No database beyond RAM/SQLite/MCAP is
introduced before its phase (see `DATABASE_INTRODUCTION_PLAN.md`).

## Proposed `machine_brain/` layout

```
machine_brain/
  00_ARCHIVE_ANALYSIS/          # done — donor component map
  01_ARCHITECTURE/              # done — this document set
  src/
    contracts/                  # canonical frame/message schemas (Phase 1)
    transport/                  # ROS2/DDS glue (Phase 1) — salvage panani_pckge/transport
    sensory/                    # ring buffer (Phase 1)
    working_memory/             # SQLite WAL store (Phase 1-2)
    episodic/                   # episodic store + MCAP provenance (Phase 2)
    graph/                      # PAIL compact typed graph interface + impl (Phase 2, Neo4j adapter Phase 8)
    world_model/                # JEPA predictive engine (Phase 3) — salvage jepa_world_engine
    planner/                    # planning proposals (Phase 4) — salvage pail_robotics/planning
    sutraflow/                  # DSL parser/compiler/runtime (Phase 4) — salvage pail_sutraflow wholesale
    safety/                     # deterministic safety governor (Phase 4) — salvage dharma_gov_india + robot_safety.py
    audit/                      # hash-chained ledger (Phase 4/11) — salvage vajra/audit + dharma/sakshi.py
    associative/                # Qdrant adapter (Phase 5) — salvage resonance/ (already folded in vajra-v0.39)
    procedural/                 # skills registry (Phase 8 in spec numbering, built early as needed by planner) — salvage pail_robotics/skills
    fleet_sync/                 # PostgreSQL sync client (Phase 6)
    artifacts/                  # MinIO/S3 adapter (Phase 7)
    acoustic/                   # resonance fingerprints + tie-break (Phase 9) — salvage panani_platform/nada, middleware/sound_match.py
    reservoir/                  # physical reservoir adapter interface (Phase 9)
  tests/
    unit/
    adversarial/                # dropout, delay, duplicate, clock drift, stuck sensor, contradictions, injection, etc.
    benchmarks/                 # 100k / 1M frame campaigns
    ablations/
  config/
    adapters.yaml                # which optional adapters (Qdrant/Neo4j/ClickHouse/MinIO) are enabled
  docs/
    provenance/                  # per-module "salvaged from <archive>:<path>, modified how" notes
```

## Phase 1 — Canonical contracts, simulation, RAM buffer, SQLite, MCAP

- [ ] Define `ObservationFrame` canonical schema (sensor id, timestamp pair [monotonic+wall],
      payload, sequence id) — used by every later layer, so gets written once and frozen.
- [ ] Stand up ROS 2 DDS topics for observation frames and commands (transport only, no storage).
- [ ] Implement fixed-capacity ring buffer (shared-memory or NumPy) with configurable time window;
      no DB writes on this path.
- [ ] Implement SQLite WAL working-memory store: active task, current entities, attention state,
      pending decisions, contradictions, execution receipts tables. TTL + capacity + transactional
      writes.
- [ ] Wire rosbag2/MCAP recording for camera/lidar/audio/joints/odometry/diagnostics with bounded
      file rotation.
- [ ] Build a minimal simulation harness that can generate synthetic observation frames (needed
      for the adversarial test campaigns later; build it now so every subsequent phase has
      something to run against without hardware).
- [ ] Verify: system runs end-to-end on RAM + SQLite + MCAP only, zero optional adapters enabled.

## Phase 2 — Working memory, episodic ledger, compact graph, provenance

- [ ] Extend SQLite schema: episodic tables with `(mcap_file_id, topic, offset, timestamp)`
      provenance columns.
- [ ] Implement idempotency check so repeated execution over the same MCAP offset range does not
      create a duplicate episode.
- [ ] Port PAIL's compact typed graph (donor: `vajra-v0.39-UNIFIED/vajra/grammar_dag/` and
      relation modules) — implement only the required typed edges: `CAUSED_BY`, `PRECEDES`,
      `OBSERVED_AT`, `EXECUTED_BY`, `FAILED_AFTER`, `RESOLVED_BY`. Reject all-to-all co-occurrence
      edge generation explicitly (add a test asserting edge count stays linear in episode count,
      not quadratic).
- [ ] Every graph edge carries a source-evidence pointer back to an episodic/audit row.
- [ ] Verify: idempotent replay test (feed the same MCAP segment twice, assert one episode row).

## Phase 3 — JEPA prediction, surprise detection, baseline comparison

- [ ] Port `jepa_engine.py` (donor: `jepa_world_engine.tar.gz`) — EMA target encoder, VICReg loss,
      `surprise()` scoring.
- [ ] Define Parquet schema for training episodes; safetensors/ONNX for signed model weights.
- [ ] Build the static baseline (e.g. last-value or linear predictor) required for comparison —
      do this before touching JEPA so "prediction error vs. baseline" has a denominator from day
      one.
- [ ] Verify: report prediction error for both baseline and JEPA on the same held-out episodes;
      do not claim "predictive world model" value until JEPA beats baseline on this report.

## Phase 4 — Planner, SutraFlow, deterministic safety governor

- [ ] Port `pail_sutraflow` wholesale (donor: `sutraflow_chainflow_pro` /
      `vajra-v0.39-UNIFIED/pail_sutraflow/`) — parser, compiler, registry, cell types.
- [ ] Port planner scaffolding (donor: `neurobot_brain_pro/pail_robotics/planning/planner.py`).
- [ ] Port safety governor (donor: `robotics_safety_runtime.tar.gz`'s `robot_safety.py`
      SafetyEnvelope/CommandGovernor, plus `dharma_gov_india`'s `kavaca.py`/`niti.py`) —
      deterministic, no learned component, no code path from the learning loop can write to it.
- [ ] Port audit ledger (donor: `dharma_gov_india/sakshi.py`, `vajra-v0.39-UNIFIED/vajra/audit/`)
      as the append-only hash-chained SQLite ledger. Every governor decision (allow AND refuse)
      writes here unconditionally.
- [ ] Verify: adversarial test — unsafe command is refused, refusal appears in ledger, refusal
      cannot be produced by any learning-loop code path (static check: no import of the
      learning/consolidation module inside `safety/` or `audit/`).

## Phase 5 — Qdrant associative-memory adapter

- [ ] Add Qdrant Edge (local) behind an `AssociativeIndex` interface defined in Phase 2.
- [ ] Port resonance modules (donor: `vajra-v0.39-UNIFIED/resonance/`, originally
      `resonance_engine_full`) as the embedding/fingerprint source feeding Qdrant.
- [ ] Enforce: every Qdrant hit resolves to a canonical SQLite row before being handed to the
      planner; add a test that a stale/deleted canonical row makes its Qdrant hit unusable rather
      than silently trusted.
- [ ] Verify: system still passes all Phase 1-4 tests with the Qdrant config block removed
      entirely (fallback to canonical scan).

## Phase 6 — PostgreSQL synchronization, multi-robot ownership boundaries

- [ ] Stand up PostgreSQL partitioned episodic/telemetry tables `(robot_id, event_time)`.
- [ ] Implement robot -> fleet append-only idempotent sync client keyed by
      `(robot_id, local_row_id)`.
- [ ] Enforce ownership boundary: reject any write attempt to another robot's partition at the
      application layer (defense in depth beyond DB permissions).
- [ ] Verify: partition/reconnect adversarial test — kill fleet connectivity mid-sync, confirm
      robot keeps operating locally and catches up cleanly on reconnect with no duplicate rows.

## Phase 7 — MinIO artifact storage, signed offline capsules

- [ ] Add `ArtifactStore` (MinIO/S3) with PostgreSQL rows for hash/ownership/timestamp/retention.
- [ ] Port capsule-signing pattern (donor: `panani_pckge/transport/capsule.py`,
      `panani_pckge/security/`, from `PAIL_RCA_ELECTRIC_BRIDGE_HANDOFF`).
- [ ] Verify: local-file fallback still works with MinIO config removed (prototype mode).

## Phase 8 — Optional Neo4j and ClickHouse adapters

- [ ] Do not start this phase until a benchmark from Phase 2/7 shows the compact graph or
      PostgreSQL is the bottleneck. If no benchmark demands it, skip and document why in
      `docs/provenance/`.
- [ ] If triggered: implement Neo4j as a second `GraphStore` implementation (same interface as
      Phase 2), ClickHouse as a second `TelemetryStore` implementation (same interface as Phase 1
      MCAP+PostgreSQL). Both gated by `config/adapters.yaml`.

## Phase 9 — Physical-reservoir adapter, acoustic ablation benchmark

- [ ] Define `ReservoirAdapter` interface: input normalized temporal signal window, output
      reservoir state vector + timestamp + calibration.
- [ ] Implement a simulated reservoir (no real hardware yet) and train only a small readout layer
      against it.
- [ ] Port acoustic/resonance modules (donor: `panani_platform/nada/*`,
      `panani_pckge/acoustic/*`, `middleware/sound_match.py`) as tie-break-only candidates per the
      ownership matrix — add a test proving acoustic resonance cannot override an exact-identity
      match, a contradiction record, or a safety rule.
- [ ] Log all reservoir/acoustic experiments to MCAP + Parquet.

## Cross-cutting, start in parallel with Phase 1

- [ ] Set up the adversarial test harness (sensor dropout, delayed timestamps, duplicate frames,
      clock drift, stuck sensors, impossible transitions, contradictory observations, unsafe
      commands, low confidence, prompt injection, restart recovery, audit tampering, memory
      eviction, model regression) — even before there's much to test, so every phase adds cases
      to a growing suite rather than writing tests as an afterthought.
- [ ] Set up the benchmark harness (100,000 frames, then 1,000,000 streaming frames) reporting
      throughput, p50/p95/p99 latency, peak RAM, disk use per frame, prediction error, anomaly
      precision/recall/F1, false safety refusals, missed unsafe actions, recovery time.
- [ ] Set up the ablation harness (no learning / reviewed learning; no vector memory / vector
      candidate memory; no resonance / resonance tie-break only; static world model / JEPA world
      model) with negative results preserved and reported, not discarded.
- [ ] For every donor module ported, write one line in `docs/provenance/` naming the source
      archive+path and what was changed — this is what keeps the donor lineage auditable instead
      of turning into another unlabeled "final" bundle like the ones we just spent time
      untangling.

## Guardrails for every phase

- [ ] Never commit or push without being explicitly asked.
- [ ] Never touch the existing RAG product or website (including the `PAIL_Website_20260627/`
      folder embedded inside `PAIL_RCA_ELECTRIC_BRIDGE_HANDOFF` — leave that archive's website
      content alone even when salvaging its transport/acoustic code).
- [ ] Never execute code extracted from untrusted donor archives directly — read, understand,
      re-implement or copy-with-review into `machine_brain/src/`, don't `import` straight from an
      extracted donor tree into production code paths.
- [ ] Confirm before any destructive action on the original archives (they are the user's source
      material — copy from them, never modify/delete originals without being asked).
