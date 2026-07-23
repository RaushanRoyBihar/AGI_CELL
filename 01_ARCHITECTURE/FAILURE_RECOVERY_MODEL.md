# Failure and Recovery Model

Organized by the required adversarial test campaign, so every failure mode below has a
corresponding test and a defined recovery behavior — not just a hope.

## Sensor/transport failures

| Failure | Detection | Recovery |
|---|---|---|
| Sensor dropout | Ring buffer staleness check (no new frame within expected period) | Attention selection marks the source stale; planner treats it as reduced confidence, not absence-of-danger; safety governor may force a conservative default if a required sensor is stale beyond a hard threshold |
| Delayed timestamps | Compare arrival time vs. embedded timestamp against a bound | Frame is accepted but flagged low-confidence if within tolerance; rejected (logged, not silently dropped) if outside tolerance |
| Duplicate frames | Idempotency key `(topic, sequence_id)` at ring-buffer ingest | Duplicate is dropped at ingest, never reaches Layer 3 — this is also what prevents "repeated execution creates a false episode" at the source |
| Clock drift | Monotonic-vs-wall-clock divergence check per robot (see Data Consistency Strategy) | Local ordering trusts monotonic clock; fleet correlation flags high-drift robots for wall-clock resync, does not halt local operation |
| Stuck sensor (frozen value) | Variance-over-window check in perception | Treated as dropout once variance stays at zero past threshold — same downstream handling as dropout |

## Data/logic failures

| Failure | Detection | Recovery |
|---|---|---|
| Impossible state transition | SutraFlow validator's precondition/postcondition rules | Transition rejected before it reaches the safety governor; logged as a validation failure, not executed |
| Contradictory observations | Association/consolidation step comparing new evidence to existing graph edges or episodic rows | Creates an explicit unresolved-contradiction record (Layer 6/11); the contradicting action is not silently picked — it is held for reviewer/rule resolution |
| Unsafe command (from planner or external operator input) | Safety governor, deterministic and independent of the planner that proposed the action | Refused. Refusal is logged to the audit ledger unconditionally. Refusal logic itself is never modified by learning |
| Low confidence retrieval/prediction | Confidence score attached to every retrieval and prediction | Below-threshold results either widen the candidate set (ask for more evidence) or fall back to the deterministic default, never silently proceed as if confident |
| Prompt injection / adversarial input in operator or sensor-derived text | Guard layer (donor: `dharma_gov_india`'s `astra.py` pattern, `aiguard`'s injection module) validates before any input reaches planning | Input sanitized/rejected at the boundary; never reaches the planner un-vetted |

## Storage/process failures

| Failure | Detection | Recovery |
|---|---|---|
| Process crash mid-write (Layer 3/4 SQLite) | WAL mode guarantees the DB itself never corrupts; on restart, SQLite replays or discards the incomplete WAL frame automatically | Application-level: any multi-step operation that isn't a single SQLite transaction is designed to be safely re-run (idempotent by construction, per Data Consistency Strategy) |
| MCAP file corruption / truncated write (e.g. power loss during recording) | MCAP's chunked format allows reading up to the last valid chunk; recording node checksums on rotation | Truncated tail is dropped, prior chunks remain readable; episode rows referencing the lost tail are flagged incomplete-provenance rather than deleted |
| Restart / full process recovery | Working memory (Layer 3) reload on boot: replay pending-decision and contradiction tables, resume from last committed task state | No re-execution of already-committed actions (idempotency keys prevent duplicate execution); any task left "in-flight" at crash time is marked interrupted, not silently resumed as if nothing happened |
| Audit ledger tampering (local hash chain broken) | Hash-chain verification on every append and on PostgreSQL sync | A broken chain is a hard fault: the robot is not permitted to continue autonomous operation past a broken audit chain — this is a safety-relevant failure, escalated, not auto-repaired |
| Memory eviction under pressure (Layer 3 TTL/capacity limits, Layer 1 ring buffer overwrite) | Capacity checks on write | Eviction always follows explicit policy (oldest-first within TTL, or lowest-confidence-first where defined) — never evicts Layer 11 audit data, never evicts an episode still referenced by an unresolved contradiction |
| Model regression (Layer 9 JEPA / procedural success stats drift) | Continuous comparison against the static baseline required by the spec | Regression beyond threshold blocks promotion of the new model version; the previous signed model version remains active until a human/reviewer approves promotion |

## Network/fleet failures

| Failure | Detection | Recovery |
|---|---|---|
| Robot-to-fleet sync interruption | Sync client tracks last-acked local_row_id | Robot continues operating fully on local SQLite/MCAP (this is the point of the RAM+SQLite+MCAP floor); sync resumes and catches up via idempotent upsert when connectivity returns — no blocking on fleet availability |
| Qdrant/graph/ClickHouse adapter unavailable | Adapter-level health check, timeout-bounded calls | System falls back to canonical-store-only retrieval (brute-force SQLite/PostgreSQL scan for the same query); degraded latency, not degraded correctness, because these layers were never sole authority |
| Partial fleet partition (subset of robots unreachable from center) | Standard sync-lag detection | Each robot remains independently safe and correct (ownership boundary is per-robot); only cross-robot analytics/consolidation degrades, not any single robot's safety behavior |

## Explicit non-recovery (by design)

- A broken audit hash chain does not self-heal — it requires investigation.
- A safety refusal, e-stop, or conflict decision is never reversed by any automated recovery
  path — only by explicit reviewed action outside the learning loop.
- Learning never modifies safety limits automatically, under any failure or recovery scenario —
  this is enforced structurally (safety governor rules are not a target the learning/consolidation
  process is permitted to write to), not just as a convention.
