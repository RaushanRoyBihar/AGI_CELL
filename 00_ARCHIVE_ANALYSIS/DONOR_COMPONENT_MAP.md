# Donor Component Map — machine_brain

Inspection method: `tar -tzvf`/`tar -tvf` listings for every archive (no full extraction of
the large ones), targeted `tar -xOzf archive path/to/file` reads of READMEs/manifests/entry
points, `file`/`md5sum`/`sqlite3` checks on loose root files. All scratch work happened under
a temp dir outside the project and was deleted afterward. Nothing in the project directory was
modified by the inspection. No `RAG`/`website`-named top-level items exist in the project root;
one archive (`PAIL_RCA_ELECTRIC_BRIDGE_HANDOFF`) contains an internal `PAIL_Website_20260627/`
folder — noted, not touched, not analyzed further, per instructions to leave the existing RAG
product/website alone.

## 1. Archive Table

| Archive | Approx. real LOC (prod, excl. dupes/tests/dist) | Primary purpose | Category |
|---|---|---|---|
| aiguard_community.tar.gz | ~2,500 | "aiguard" guardrail lib wrapped as open-source community starter kit | guardrail/safety |
| langchain_guardrails.tar.gz | ~2,600 | Same "aiguard" core + a LangChain adapter shim | guardrail/safety |
| nvidia_guardrail_runtime.tar.gz | ~2,600 | Same "aiguard" core + `guardrail_runtime.py` NeMo-style shim | guardrail/safety |
| robotics_safety_runtime.tar.gz | ~2,600 | Same "aiguard" core + `robot_safety.py` (SafetyEnvelope/CommandGovernor) | guardrail/safety |
| dharma_gov_india.tar.gz | ~2,900 | **DHARMA** — Sanskrit-named origin of the aiguard core; air-gapped govt AI-governance gateway (PII/DPDP, injection, audit) | guardrail/safety |
| jepa_world_engine.tar.gz / "jepa_world_engine (1).tar.gz" | ~330 | Pure-NumPy JEPA latent predictive engine (EMA encoder, VICReg, `surprise()`) | cognitive-core (JEPA) |
| neurobot_brain_pro.tar.gz | ~5,800 | `pail_robotics` package: skills, planner, world_model, ROS bridge, mission/SutraFlow executor, droid commander | robotics/droid-prototype |
| sutraflow_chainflow_pro.tar.gz | ~5,000 | `pail_sutraflow` — the `.sutra` DSL parser/compiler/runtime + cell types (evidence/memory/reasoning/agent) | SutraFlow guard/validation |
| vajra_sna_rca_pro.tar.gz | ~7,900 | `vajra_sna` domain-pack RCA engine (banking/insurance packs, learning, memory, recovery) | cognitive-core "vajra" |
| resonance_engine_full (5).tar.gz | ~6,300 | Pure-Python Panini-sound resonance associative memory (no numpy/GPU) | acoustic/resonance |
| darshana-ai-full (2).tar.gz | ~11,300 (TypeScript) | Next.js app: 6 Darshana philosophy-school reasoning modules + brain layers L1–L11 in TS | other (web app / UI reference) |
| panani_om_spine_v2_final_8of10.tar.gz | ~3,800 (+287MB non-source runtime jsonl, discard) | 20-component "living body" (nerve bus, hormone policy, karma ledger, immune layer, sleep/dream cycle) wrapping a cognitive core | cognitive-core "vajra" (auxiliary) |
| Panani_V80_NADA_ACOUSTIC_BRAIN_SCAFFOLD_20260626.tar | ~24,000 | Origin of `panani_platform`: RAG-adjacent evidence guard, storage adapters (DuckDB/Postgres/pgvector), NADA acoustic codec, relation/sutra guard | sanskrit/panini-linguistic + acoustic |
| PAIL_RCA_ELECTRIC_BRIDGE_HANDOFF_20260630 (1).tar.gz | ~13,700 (`panini-smriti/panani_pckge`) | "Pāṇini-LLM-Vector Middleware Brain" v3.2: transport (CRDT/ECC/modulation), 280 executable sutras, SoundMatchEngine, sub-50ms RobotReflex; also bundles a 40MB standalone website (`PAIL_Website_20260627/`, left untouched) | PAIL-handoff-bundle |
| vajra-v0.26-MICRO-CELLS-NEURAL-SYMBOLIC.tar | (lineage) | Early vajra: micro-cells + neural-symbolic core | cognitive-core "vajra" |
| vajra-v0.35-SELF-TRAINING-COGNITIVE.tar | (lineage) | vajra + self-training, brain organs, darshanas, vedas | cognitive-core "vajra" |
| vajra-v0.38-HARDWARE-AGNOSTIC.tar(2) | (lineage) | vajra + `vajra.hardware` (sensor adapters), `vajra.quartz` (6-layer physical evidence runtime), SNA integration | cognitive-core "vajra" |
| vajra-v0.39-UNIFIED.tar.gz (+ loose dir in project root) | ~96,800 | **Most current vajra**: v0.38 + sutraflow_chainflow_pro + vajra_sna_rca_pro + resonance_engine_full + aiguard, fully merged | cognitive-core "vajra" |
| PAIL-MACHINE-BRAIN-FINAL-1.1.0-20260714.tar.gz | ~78,100 | Parallel "grand merge": vajra v0.25–v0.37 + PAIL Robot v1 + V8 Robot Mind + panani_platform + pail_sutraflow + 10-product catalog docs; has its own `MERGE_PROVENANCE.md` | PAIL-handoff-bundle (most documented) |
| PAIL_COGNITIVE_MIDDLEWARE_LAUNCH_READY_v1.4.0.tar.gz | ~33,900 | Curated "launch-ready" packaging (Dockerfile, START_HERE.md, LAUNCH_READINESS.md) of aiguard+pail+sutraflow+panani+vajra subset | PAIL-handoff-bundle (deployment reference) |
| PAIL_Robotics_Starship_Droid_Prototype_20260714.tar.gz | ~4,900 | Same `pail_robotics` lineage as neurobot_brain_pro, packaged as a droid prototype with build artifacts | robotics/droid-prototype |
| sanskrit_brain_v0.2.tar.gz | ~5,500 real / 339MB decompressed (mostly scratch bloat) | Sanskrit trainer scaffold: syllabus, cerebrum trainer, thin glue over 4+ archives | sanskrit/panini-linguistic |
| sanskrit_brain_v0.4.tar.gz | ~8,000 real / 339MB decompressed (mostly scratch bloat) | v0.2 + audio_generator, firewall, RCA sutras, Postgres/pgvector schema, docker-compose; `brain.py` explicitly documents wiring 9+ donor archives together | sanskrit/panini-linguistic |
| Machine_Neural_Brain_v0.6_full.tar | ~9,000 new real / 344MB decompressed (mostly scratch bloat) | Newest in this lineage: wraps sanskrit_brain_v0.4 + adds model_registry, durable_memory, reliability, observability, payments, security, unified_cognition — a productization facade | sanskrit/panini-linguistic (most current) |

## 2. Duplicate / Version Families

**A. "aiguard" guardrail core (byte-identical across 4 archives)**
`aiguard_community`, `langchain_guardrails`, `nvidia_guardrail_runtime`, `robotics_safety_runtime`
all wrap the *exact same* `aiguard/` package (identical module sizes throughout), differing only
in a thin wrapper/README per audience. `dharma_gov_india` is the Sanskrit-named original this was
derived from (raksha=PII, astra=injection, kavaca=safety, niti=policy, viveka=abstention,
pramana=grounding, sakshi=audit, nyaya=fairness, pariksha=redteam) and is the most complete of the
five. **Keep `dharma_gov_india` only; discard the 4 derivatives.**

**B. `jepa_world_engine.tar.gz` and `jepa_world_engine (1).tar.gz`**
Byte-identical (md5 `9b5fe6d9eff876a4505dc352747d6af1` for both). Delete one copy.

**C. The "vajra" cognitive-core lineage (strictly cumulative)**
`vajra-v0.26 → v0.35 → v0.38 → v0.39-UNIFIED` is a clean sequential lineage. v0.39-UNIFIED
absorbs `sutraflow_chainflow_pro`, `vajra_sna_rca_pro`, `resonance_engine_full`, and the `aiguard`
family directly into its tree. The **loose extracted directory `vajra-v0.39-UNIFIED/` in the
project root is further ahead of its own `.tar.gz`** — it has `droid_runtime.py`
(`AsyncDroidCore`, imported live by `run_test.py`), `dashboard.py`/`dashboard/`,
`moksha_runtime.py`, `planning/`, and `ltm_memory.db`. **Treat the loose directory as the actual
current head, not the tar.gz.**

**D. PAIL_MACHINE_BRAIN_FINAL-1.1.0 vs vajra-v0.39-UNIFIED — parallel, not sequential**
Its own `MERGE_PROVENANCE.md` states it merges vajra v0.25–v0.37 + PAIL Robot v1 + V8 Robot Mind +
panani_platform + pail_sutraflow, built 14 Jul (before v0.38/v0.39 existed). Near-identical breadth
to v0.39 (89 vs 91 vajra/ subdirs) but v0.39 lacks FINAL's `products/` (10-product catalog docs)
and `provenance/donor_sources/` (audited legacy robot adapters, reference only). **Use v0.39-UNIFIED
as the primary source tree; pull `products/` and `provenance/donor_sources/` from
PAIL-MACHINE-BRAIN-FINAL as reference material only.**

**E. `PAIL_COGNITIVE_MIDDLEWARE_LAUNCH_READY_v1.4.0`**
A slimmer curated subset of the same lineage, dated 17 Jul, packaged with Dockerfile/START_HERE/
LAUNCH_READINESS docs. Treat as a **"how to package for deployment" reference**, not a separate
code source.

**F. `neurobot_brain_pro` and `PAIL_Robotics_Starship_Droid_Prototype`**
Both ship the same `pail_robotics/` package. Starship adds `build/`, `build_artifacts/`,
`.egg-info` (discard). **Keep `neurobot_brain_pro` as the cleaner copy.**

**G. "Panini/Panani" linguistic-platform lineage**
`Panani_V80_NADA_ACOUSTIC_BRAIN_SCAFFOLD` (26 Jun) is the origin of `panani_platform` /
`panani_core_layers` / `panani_pckge`, subsequently embedded verbatim into every vajra archive
from v0.26 onward. `PAIL_RCA_ELECTRIC_BRIDGE_HANDOFF` (30 Jun, `panini-smriti/panani_pckge`) is a
**separate, earlier "electric bridge" branch** of the same package name with different, richer
content (transport/CRDT/ECC/modulation, 280 sutras, SoundMatchEngine, RobotReflex) that does not
look fully folded into the vajra tree — worth a direct diff against `vajra/panini*` and
`panani_pckge/` in v0.39 before assuming full coverage.

**H. `sanskrit_brain_v0.2 → sanskrit_brain_v0.4 → Machine_Neural_Brain_v0.6_full`**
Clean sequential lineage. v0.4 adds `audio_generator.py`, `firewall.py`, `rca_sutras.py`,
Postgres/pgvector schema, docker-compose/prometheus configs over v0.2.
`Machine_Neural_Brain_v0.6` contains a full copy of sanskrit_brain_v0.4 nested inside it
(`src/sanskrit_brain/`) and layers a new `machine_neural_brain/` package on top (model_registry,
durable_memory, reliability, observability, security, payments, unified_cognition) — the **most
current member of this family**. All three carry an essentially identical ~300MB
`workspace/extracted/` (or `sanskrit_brain/extracted/`) scratch directory of nested duplicate
archives — pure noise, not original source.

**I. `resonance_engine_full`**
Fully absorbed into `vajra-v0.39-UNIFIED/resonance/` (same file set). Keep only for its clean
standalone design docs (`DESIGN.md`, `HETU_CHAIN.md`); the code itself is redundant with v0.39.

## 3. Notable Donor Components (salvage map by target layer)

| Component | Archive + path | Target layer |
|---|---|---|
| `AsyncDroidCore` robot runtime, live-tested | loose `vajra-v0.39-UNIFIED/droid_runtime.py` | Layer 8 procedural/skills + robot control loop |
| `panani_pckge/transport/` (sync, capsule, crdt, security, channel, ecc, keys, pipeline, modulation) | PAIL_RCA_ELECTRIC_BRIDGE_HANDOFF `.../panani_pckge/transport/` | Layer 0 transport |
| `vajra/persistence/`, `vajra/storage/`, `vajra/spine/` | vajra-v0.39-UNIFIED / PAIL-MACHINE-BRAIN-FINAL | Layer 3 working memory (SQLite) |
| `panani_om_spine/core/karma_ledger.py`, `memory_guard.py`, `query_cache.py` | panani_om_spine_v2_final_8of10.tar.gz | Layer 3/11 working memory + audit ledger |
| `ltm_memory.db` (table `ltm_events`), `.vajra_droid_memory` (SQLite) | project root loose files | Layer 4 episodic — already a live working example |
| `panani_pckge/storage/neural_memory.py`, `varnamala_store.py` | PAIL_RCA_ELECTRIC_BRIDGE_HANDOFF | Layer 4/5 episodic/associative |
| `panani_platform/storage/{duckdb_adapter,postgres_adapter,postgres_pgvector_adapter,hybrid_router}.py` | Panani_V80_NADA_ACOUSTIC_BRAIN_SCAFFOLD | Layer 5 Qdrant/associative + PostgreSQL |
| `resonance/resonance_net.py`, `resonance_embed.py`, `lsh_index.py` | resonance_engine_full (also in vajra-v0.39/resonance/) | Layer 5 associative memory |
| `vajra/grammar_dag/`, `vajra/panini/`, `vajra/panini_sutras/`, `vajra/panini_catalog/` | vajra-v0.39-UNIFIED, PAIL-MACHINE-BRAIN-FINAL | Layer 6 semantic/causal graph |
| `panani_platform/relations/schema_registry.py`, `relational_guard.py` | Panani_V80_NADA / PAIL-MACHINE-BRAIN-FINAL | Layer 6 semantic/causal graph |
| `pail_robotics/skills/registry.py`, `planning/planner.py` | neurobot_brain_pro | Layer 8 procedural/skills |
| `vajra_sna/domain_runtime.py`, `learning.py`, `memory.py` | vajra_sna_rca_pro (now in vajra-v0.39/vajra_sna/) | Layer 8 procedural/skills (domain RCA) |
| `jepa_engine.py` (EMA target encoder, VICReg, `surprise()`) | jepa_world_engine.tar.gz (also copied into neurobot_brain_pro) | Layer 9 JEPA predictive |
| `vajra/diff_physics/`, `vajra/physics/`, `vajra/world_model/` | vajra-v0.39-UNIFIED / PAIL-MACHINE-BRAIN-FINAL | Layer 9 JEPA predictive / world model |
| `dharma/sakshi.py` (audit), `dharma/kavaca.py`, `dharma/niti.py` | dharma_gov_india.tar.gz | Layer 11 audit/decision ledger + safety governor |
| `vajra/audit/`, `vajra/firewall/`, `vajra/hardening/`, `vajra/no_fail_open.py` | vajra-v0.39-UNIFIED / PAIL-MACHINE-BRAIN-FINAL | Layer 11 audit/safety ledger |
| `panani_platform/audit/immutable_export.py` | Panani_V80_NADA | Layer 11 audit ledger |
| `robot_safety.py` (SafetyEnvelope, CommandGovernor) | robotics_safety_runtime.tar.gz | Layer 11 / safety governor |
| `panani_platform/nada/{signal_layer,codec,memory_layer,grammar_symbolic,bidirectional,robot_adapter}.py` | Panani_V80_NADA_ACOUSTIC_BRAIN_SCAFFOLD | Layer 12 acoustic/resonance |
| `panani_pckge/acoustic/{dsp,model,psnd,psnd2}.py`, `spectral/{prosody,memory,ann}.py` | PAIL_RCA_ELECTRIC_BRIDGE_HANDOFF | Layer 12 acoustic/resonance |
| `middleware/sound_match.py` (phonetic-route + MFCC + 8-band spectral + prosody) | PAIL_RCA_ELECTRIC_BRIDGE_HANDOFF | Layer 12 acoustic/resonance |
| `pail_sutraflow/` (parser, compiler, registry, cells/{evidence,memory,reasoning,agent,attention,output}) | sutraflow_chainflow_pro / vajra-v0.39-UNIFIED / PAIL-MACHINE-BRAIN-FINAL | **SutraFlow guard/validation layer** — build directly on this |
| `panani_platform/sutra_guard.py`, `entrypoint_guard.py`, `math_guard/invariants.py` | Panani_V80_NADA / PAIL-MACHINE-BRAIN-FINAL | SutraFlow guard/validation layer |
| `sutras/catalog.py` + `extended.py` (280 executable sutras) | PAIL_RCA_ELECTRIC_BRIDGE_HANDOFF | SutraFlow guard/validation layer (rule content) |
| `products/catalog-10/*.md`, `provenance/donor_sources/` | PAIL-MACHINE-BRAIN-FINAL only | Reference docs |
| `run_cli.py`, `panani_pckge/service/{api.py,dashboard.*}` | PAIL_RCA_ELECTRIC_BRIDGE_HANDOFF | Reference only — dashboard UI pattern |
| `darshana-ai-full/src/lib/brain/{l1-sound,l2-grammar,l3-rules,l7-semantic,l8-memory,l11-compression}.ts` | darshana-ai-full | Reference only — wrong language for this stack, useful for layer-naming inspiration |

## 4. What to Discard

| Item | Where | Approx. size |
|---|---|---|
| `PAIL_Website_20260627/` (Cloudflare site + hero PNGs) | inside PAIL_RCA_ELECTRIC_BRIDGE_HANDOFF | ~40MB of the archive's 41.5MB |
| `workspace/extracted/` (or `sanskrit_brain/extracted/`) scratch copies of other donor archives | sanskrit_brain_v0.2, sanskrit_brain_v0.4, Machine_Neural_Brain_v0.6 (all three, near-identical) | ~300MB decompressed **each** (dominant single file: `panani_om_spine/runtime/cold_cells.jsonl`, 287MB) |
| `runtime/cold_cells.jsonl` (accumulated runtime cache, not source) | panani_om_spine_v2_final_8of10.tar.gz | 287MB decompressed |
| `legacy_snapshots/` (old vajra versions re-embedded in newer vajra archives) | vajra-v0.35, v0.38, v0.39-UNIFIED | several MB each, confirmed duplicate of v0.26–v0.29 content |
| `dist/` wheels/sdists, `*.egg-info`, `build/`, `build_artifacts/` | PAIL-MACHINE-BRAIN-FINAL, vajra-v0.38/v0.39, PAIL_Robotics_Starship | low MB, pure build output |
| `.pytest_cache/`, `__pycache__/`, `*.pyc` | dharma_gov_india, panani_om_spine_v2, loose vajra-v0.39-UNIFIED/ dir | low MB but numerous |
| `TEST_REPORT_*.md`, `V0##_*_REPORT.md`, `AUDIT_REPORT_*.md`, benchmark result JSON | panani_om_spine_v2, vajra-v0.35/v0.38/v0.39, Panani_V80_NADA | historical narrative only; keep 1–2 for changelog context, discard rest |
| Duplicated `aiguard/` package copies (4x) | aiguard_community, langchain_guardrails, nvidia_guardrail_runtime, robotics_safety_runtime | ~180KB compressed combined, 100% redundant with dharma_gov_india |
| Duplicate `jepa_world_engine` archive | `jepa_world_engine (1).tar.gz` | 4,951 bytes, byte-identical |
| Benchmark fixture data (`.panani`, `.docx`, `.pdf`, `.sqlite3` test pools) | Panani_V80_NADA, sanskrit_brain workspace | several MB, test fixtures not runtime code |

Rough reclaimable volume if all scratch/duplicate/report material is dropped: **~350–400MB out of
the ~830MB total decompressed footprint** of the three largest offenders alone
(sanskrit_brain_v0.2/v0.4/Machine_Neural_Brain_v0.6), plus the ~40MB website bundle, plus the
low-value redundant guardrail copies.

## 5. Total LOC / Size Summary

- **Total on-disk archive size (25 archives, compressed):** ~501MB, dominated by
  `Machine_Neural_Brain_v0.6_full.tar` (329MB), `sanskrit_brain_v0.2`/`v0.4` (~32MB each),
  `panani_om_spine_v2` (25MB), `Panani_V80_NADA` (13MB), `PAIL_RCA_ELECTRIC_BRIDGE_HANDOFF` (40MB)
  — five archives account for ~440MB of the 501MB.
- **Real, de-duplicated source worth keeping, estimated:**
  - vajra-v0.39-UNIFIED (primary cognitive core, current head): ~97,000 LOC
  - PAIL-MACHINE-BRAIN-FINAL unique additions (docs only, code redundant with v0.39): negligible extra code
  - Panani_V80_NADA `panani_platform` (origin platform, partially distinct from what's folded into vajra): ~24,000 LOC
  - PAIL_RCA_ELECTRIC_BRIDGE `panini-smriti/panani_pckge` (distinct "electric bridge" branch — transport/RCA/acoustic): ~13,700 LOC
  - dharma_gov_india (guardrail origin): ~2,900 LOC
  - Machine_Neural_Brain_v0.6 new productization layer: ~9,000 LOC
  - sanskrit_brain_v0.4 own package (beyond what Machine_Neural_Brain_v0.6 already carries): ~2,500 LOC net-new
  - panani_om_spine (living-body organs, distinct from vajra's own spine dir): ~3,800 LOC
  - resonance_engine_full / darshana-ai-full: redundant with vajra-v0.39 / reference-only respectively
  - **Approximate real, non-duplicated, worth-reviewing total: ~150,000–160,000 LOC**, against the
    donor's claimed ~200,000 — consistent with the user's own note that most of the raw total is
    duplicated across versions, and matches what `PAIL-MACHINE-BRAIN-FINAL/MERGE_PROVENANCE.md`
    itself states about nested legacy snapshots, build copies, and repeated packages inflating the
    apparent total.
- **Practical takeaway:** the effective foundation to build `machine_brain` on is small:
  **vajra-v0.39-UNIFIED (primary) + PAIL-MACHINE-BRAIN-FINAL's `products/` + `provenance/` docs +
  Panani_V80_NADA's `panani_platform` (for anything not already folded in) + dharma_gov_india + the
  panini-smriti transport/acoustic branch from PAIL_RCA_ELECTRIC_BRIDGE**. Everything else in the
  25 archives is either an intermediate snapshot already absorbed into v0.39, a thin repackaging of
  the same guardrail core, or non-code bulk (scratch extractions, media, reports).

## Key file paths for follow-up

- `vajra-v0.39-UNIFIED/` (loose dir, project root — current live head, ahead of its own `.tar.gz`)
- `vajra-v0.39-UNIFIED.tar.gz`
- `PAIL-MACHINE-BRAIN-FINAL-1.1.0-20260714.tar.gz` (`MERGE_PROVENANCE.md` inside is worth reading directly)
- `Panani_V80_NADA_ACOUSTIC_BRAIN_SCAFFOLD_20260626.tar`
- `PAIL_RCA_ELECTRIC_BRIDGE_HANDOFF_20260630 (1).tar.gz` (contains embedded website — leave `PAIL_Website_20260627/` untouched)
- `dharma_gov_india.tar.gz`
- `ltm_memory.db`, `.vajra_droid_memory` (live SQLite examples of episodic/working memory already in use)
