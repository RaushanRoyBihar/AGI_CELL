# AGI_CELL

### A safety-first "brain" for robots and AI agents — free and open source to run yourself; paid only if you want it built into your own product.

[![License: AGPL v3](https://img.shields.io/badge/license-AGPLv3-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

## In plain terms

If you're building a robot, or any AI system that takes real actions in the real world, the hard
problem isn't making it smart — it's making sure it never does anything dangerous, and being able
to prove, after the fact, exactly why it did what it did. AGI_CELL is a **decision-making core**
that sits between "the AI wants to do X" and "X actually happens": it checks every single proposed
action against hard safety rules that nothing — not even its own learning — can override, and it
keeps a permanent, tamper-evident log of every decision, allowed or refused.

Think of it as a flight recorder plus a safety brake, for robots and AI agents, that you can
actually read and verify yourself instead of taking on faith.

## What you get for free

- **The entire thing.** Full source code, nothing hidden, nothing crippled. Run it on your own
  machine, forever, for free — that's what open source (AGPLv3, see [Licensing](#licensing) below)
  means here.
- A working robot "brain": perception, memory, planning, and a safety layer that's been
  deliberately attacked (property-based fuzzing) to try to break it — and did, a few times. Those
  bugs and fixes are documented, not hidden — see [`docs/provenance/NOTES.md`](docs/provenance/NOTES.md).
- Real 3D physics simulation (Google DeepMind's MuJoCo), and an optional small local AI assistant
  that understands plain-English instructions like *"watch human-1 from about 2 meters away."*
- Everything is tested (100+ automated tests) and benchmarked against real numbers, not
  marketing claims — see [Measured, not claimed](#measured-not-claimed).

## What's paid

- **Custom integration** — connecting this into your actual robot, product, or business systems.
- **Enterprise features** (fleet management for many robots at once, compliance dashboards,
  managed hosting) — built as a separate add-on, not part of the open-source core.
- **Support contracts** for teams that need a guaranteed response time, not best-effort.

Interested in any of the above? Open an issue on this repo, or reach out directly:

- **Email:** raushanraj1112@gmail.com
- **Phone:** +91-9241386853 · +91-7004478802
- **Other work:** [PAIL Evidence Runtime](https://ancient-intelligence-lab.kakarotvira06.workers.dev/)
  — a deterministic evidence-verification checkpoint for RAG pipelines (checks retrieved evidence
  actually supports an answer before it reaches an LLM), also built by the same author.

## Try it in a few minutes

**With Docker** (nothing to install locally — build once, run):
```bash
docker build -t agi_cell .
docker run agi_cell
```
*(Wraps the exact same install-and-run steps below in a standard Python container.)*

**Without Docker:**
```bash
python3 -m pip install -e .
python3 demo_run.py --frames 2000 --fresh
```
You'll see real output: how fast it processed simulated sensor data, how many decisions it made,
and confirmation that its audit trail is intact and untampered.

**Talk to it in plain English** (optional, downloads a small ~470MB AI model, runs entirely on
your own machine — no API key, no cloud):
```bash
python3 -m pip install -e ".[llm]"
mkdir -p models && curl -L -o models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf
python3 agi_demo.py --cycles 900 --fresh --instruction "Watch human-1 from about 2 meters away"
```

## Licensing

AGI_CELL is licensed under the **GNU Affero General Public License v3.0 (AGPLv3)** — see
[`LICENSE`](LICENSE). In plain terms: you can use, run, and modify this freely, including
commercially. The one thing AGPLv3 requires that a plainer license wouldn't: if you take this,
modify it, and offer it to others as a hosted service over a network, you must also publish your
modifications under AGPLv3. It does not let someone take this code, host it, and sell it as a
closed competing product without contributing back — that protection is the whole reason for this
specific license (the same reason Grafana, MongoDB, and several other open-source companies use a
similar approach).

---

## For engineers: full technical detail

Everything below is unchanged, exact, and verifiable against the code and tests in this repo — the
section above is the front door; this is the actual documentation.

### What actually runs today

The full cognitive data flow is wired end to end and tested:

```
sensors/operator input -> ObservationFrame -> ring buffer (Layer 1) -> perception/feature
extraction -> working memory (Layer 3, SQLite WAL) -> attention selection -> episodic recall
(Layer 4) + associative candidates (Layer 5) + graph hints (Layer 6) -> JEPA surprise score
(Layer 9) -> planner proposal -> SutraFlow validation -> safety governor -> simulated action
-> observed outcome -> reviewed (Samskara) learning -> graph/procedural consolidation
```

Every guarded decision (SutraFlow AND safety governor, allow or refuse) is written to a
hash-chained audit ledger (Layer 11) before anything executes. Duplicate frames are rejected at
ingest. Repeated execution of the same skill over the same MCAP offset range does not create a
duplicate episode. Contradictory observations of the same entity create an unresolved
contradiction rather than silently picking a side.

### Full command reference

```bash
python3 -m pip install -e .
python3 demo_run.py --frames 3000 --fresh              # full Phase 1-9 loop on synthetic sensor data
python3 agi_demo.py --cycles 900 --fresh                # goal-directed, model-based agent (see below)
python3 -m pytest -q                                    # unit + adversarial + ablation + smoke tests
python3 tests/benchmarks/bench_throughput.py --frames 100000   # the required 100k benchmark

# real 3D physics (Google DeepMind's MuJoCo) instead of the toy 2D kinematics:
python3 -m pip install -e ".[physics]"
python3 agi_demo.py --cycles 900 --fresh --physics mujoco

# natural-language goal-setting via a small local LLM (CPU-only, no API key):
python3 -m pip install -e ".[llm]"
mkdir -p models && curl -L -o models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf
python3 agi_demo.py --cycles 900 --fresh --instruction "Watch human-1 from about 2 meters away"

# regenerate the OKF knowledge bundle after changing skills/goals/safety limits, and see
# the measured effect of curated context on the LLM's behavior:
python3 scripts/generate_okf_bundle.py
python3 examples/okf_context_experiment.py
```

### World model / "basic AGI" layer

On top of the Phase 1-9 loop, `planner/imagination.py` + `world_model/dynamics.py` turn the
reactive planner into a model-based agent:

- **Action-conditioned dynamics model** (`world_model/dynamics.py`): a small NumPy MLP that
  predicts the next *interpretable* state vector (nearest human distance, nearest obstacle
  distance, focused-entity count, velocity, distance-to-active-goal-target) given the current
  state and a candidate action. Trained online, only on real observed transitions — never on
  imagined ones. This is deliberately separate from `world_model/jepa.py`, which stays an
  unsupervised, action-blind latent-space anomaly detector; the two serve different jobs.
- **Imagination-based planning** (`planner/imagination.py`): before proposing an action, the
  planner asks the dynamics model "what happens if I do each of my 6 candidate actions", scores
  each imagined outcome by predicted safety risk, progress toward the active goal, and the skill's
  real historical success rate, and proposes the best one. Untrusted until the dynamics model has
  learned from `min_train_steps` (20) real transitions — before that, `CognitiveBrain` falls back
  to the original reactive rule-based `Planner`. Imagined proposals go through SutraFlow and the
  safety governor exactly like reactive ones; imagination changes what gets *proposed*, never what
  gets *allowed*.
- **Goals** (`Goal` in `contracts/`, `goals` table in working memory): a single persistent
  standing intent (e.g. "hold ~1.5m distance from obstacle-2") instead of pure frame-by-frame
  reaction. `CognitiveBrain.set_goal(kind, target)`.
- **Epsilon-greedy exploration**: pure utility-maximizing selection creates a lock-in loop — the
  first action that wins accumulates competence, which makes it win again, while the model never
  collects training data on the alternatives it keeps skipping. `ImaginationConfig.exploration_epsilon`
  (default 0.15) fixes this by occasionally proposing a non-optimal candidate — still fully guarded,
  never a bypass.
- **Real 2D spatial world** (`simulate/world.py`): the robot and every entity have actual (x, y)
  positions; sensed "distance" is genuine Euclidean distance, not an asserted scalar. Executing
  `approach_target` really does turn the robot toward its target and close the distance;
  `avoid_obstacle`/`yield_to_human` really do turn it away — a simplified holonomic point-robot
  model (no acceleration/inertia/differential-drive constraints), documented as a deliberate scope
  cut, not an oversight.
- **Multi-step imagination rollout** (`ImaginationConfig.horizon`, default 4): instead of scoring
  candidates by one imagined step, the planner rolls the same candidate action forward `horizon`
  steps (open-loop "shooting" — the standard cheap MPC simplification, not a full tree search) and
  sums discounted per-step utility, so sustained progress toward a goal outweighs one-step noise.
  The dynamics model itself still only ever trains on real, single-step transitions.

**Real bugs found and fixed while building this, worth knowing about:**

1. Exploratory proposals initially had their confidence deliberately dampened (`×0.7`) to mark them
   as "less sure." That pushed them below SutraFlow's low-confidence-hold threshold (0.6), which
   *held* every exploratory proposal — permanently starving the untested action of the executed
   outcome it needed to ever build up competence. Fixed by not conflating "was this the optimal
   pick" with "how safe/well-understood is this action" — confidence now reflects the latter only.
2. `SensorSimulator.apply_action` originally nudged *every* entity ever seen when the agent chose
   an avoidance action. Entities not perceived in many cycles accumulated drift between sightings,
   and the next real sighting looked like a >2m jump — which the contradiction detector correctly
   flagged, holding almost everything. Fixed (in the pre-spatial-model version) by only nudging
   entities currently within a plausible "nearby" range, then superseded entirely by the real 2D
   world model below.
3. After adding real 2D kinematics, entity wander was unbounded — a true random walk with no
   leash, so distance could drift arbitrarily far between infrequent sightings of the same entity,
   again reading as a sensor contradiction. Fixed with a bounded "leash" radius per entity kind
   (entities wander within a room-sized area, not to infinity).
4. Even bounded, a robot that genuinely moves several meters between sparse sightings of the same
   entity produces real (not spurious) distance changes larger than the original 2.0m contradiction
   threshold, which was tuned for a world where nothing really moved. Raised to 4.5m — comfortably
   above plausible real motion, comfortably below the ~7.7m gap the deliberate
   `contradictory_pair()` adversarial test injects, so genuine sensor contradictions are still
   caught.

**Known open limitation, reported rather than hidden:** even with real spatial kinematics and
multi-step rollout, `agi_demo.py`'s default run reaches its goal and spreads competence evenly
across all 6 skills, but the log still doesn't always cleanly show `approach_target` as the
decisive action right before completion — other skills are often active at that moment too, and
the target's own wander plausibly does some of the work. A 16-unit MLP trained on a few hundred
real transitions (only ~15 touching `approach_target`) may simply not have converged a strong
action→goal-distance signal yet. Worth a closer look (more training data, a larger network, or
curriculum-style targeted data collection) before trusting this planner beyond a demo.

### Real Pāṇinian phonology (first literal donor-code integration)

`sutraflow/panini_phonology.py` is the first place actual donor **files** were used, not just
donor *concepts*. Two JSON files were copied verbatim from `vajra-v0.39-UNIFIED` — the real 14
Māheśvara/Śiva Sūtras (the phoneme-classification sutras that open the Aṣṭādhyāyī) and a set of
sandhi/definitional rules carrying real, checkable canonical sutra numbers (6.1.87, 6.1.88,
6.1.89, 6.1.101, 1.1.1, 1.1.2). The pratyāhāra-construction algorithm and sandhi logic are freshly
written against the classical definition and checked against independently-verifiable facts, not
just internal consistency: `ac_vowels()` must equal the real 9-vowel inventory, `hal_consonants()`
must equal the real 33-consonant inventory (with "ha" — the one phoneme that appears in two
different sutras — correctly deduplicated), and `apply_vowel_sandhi("a", "i")` must produce the
textbook guṇa-sandhi result `"e"` under sutra 6.1.87. Two other donor sutra JSON files were
inspected and deliberately **not** used — they use Paninian terminology but aren't real sutra
citations, and using them would have meant presenting synthetic content as authentic grammar. Full
accounting in `docs/provenance/NOTES.md`. Not yet wired into the active SutraFlow guard rule set —
a real, tested capability sitting next to the guard layer, not inside it yet.

### SutraFlow's guard logic is literally built from two Aṣṭādhyāyī sutras

`src/machine_brain/sutraflow/rules.py` resolves conflicts between matching guard rules using:

1. **अपवादः उत्सर्गं बाधते** (apavādaḥ utsargaṃ bādhate) — the exception overrides the general rule.
2. **विप्रतिषेधे परं कार्यम्** — Aṣṭādhyāyī 1.4.2 (vipratiṣedhe paraṃ kāryam) — among rules of equal
   standing, the one declared later governs.

And the ordering between SutraFlow (task-grammar validity) and the safety governor (hard limits)
follows the antaraṅga/bahiraṅga principle behind the asiddhavat paribhāṣās (e.g. 8.2.1
pūrvatrāsiddham): the safety governor is treated as the "inner" rule, already decided, and its
REFUSE unconditionally overrides a SutraFlow ALLOW — never the reverse. See `safety/governor.py`'s
docstring.

### Layer -> module map

| Spec layer | Module | Default backing | Optional adapter |
|---|---|---|---|
| 0. Transport | `transport/bus.py` | In-process pub/sub | ROS2 DDS (not wired — no ROS2 env assumed) |
| 1. Sensory | `sensory/ring_buffer.py` | `collections.deque` ring buffer | — |
| 2. Raw experience | `raw_experience/mcap_log.py` | "mcap_lite" (chunked, checksummed, bounded rotation) | real `mcap` package |
| 3. Working memory | `working_memory/store.py` | SQLite WAL | — |
| 4. Episodic | `episodic/store.py` | SQLite | PostgreSQL (fleet) |
| 5. Associative | `associative/index.py` | `LocalVectorIndex` (NumPy cosine) | `QdrantIndex` |
| 6. Semantic/causal graph | `graph/store.py` | `SQLiteGraphStore` | `Neo4jGraphStore` |
| 7. Telemetry | `telemetry/store.py` | SQLite | `ClickHouseTelemetryStore` |
| 8. Procedural | `procedural/skills.py` | SQLite | — |
| 9. Predictive world model | `world_model/jepa.py`, `baseline.py` | NumPy JEPA + static baseline | — |
| 10. Artifacts | `artifacts/store.py` | `LocalFileArtifactStore` | `MinioArtifactStore` |
| 11. Audit/safety | `audit/ledger.py`, `safety/governor.py` | SQLite hash chain + deterministic rules | — |
| 12. Acoustic/resonance | `acoustic/resonance.py` | FFT-band fingerprint, tie-break only | — |
| 13. Reservoir | `reservoir/adapter.py` | `SimulatedReservoir` | real SAW/quartz/photonic hardware |
| Fleet sync | `fleet/sync.py` | `LocalFleetSync` (SQLite stand-in) | real PostgreSQL |

Every "optional adapter" column entry requires an explicit `pip install machine_brain[extra]` and
raises a clear `RuntimeError` (not a silent failure) if selected without the dependency installed.
`config/adapters.yaml` defaults every adapter to its local implementation — delete the file and
the system still runs.

### Measured, not claimed

- **100,000-frame benchmark** (`tests/benchmarks/bench_throughput.py`): 1,303 frames/sec,
  p50 3.4ms / p95 4.8ms / p99 16.4ms cycle latency, ~48MB peak RSS, ~675 bytes/frame on disk,
  audit chain verified valid across all 100k frames. See `reports/` for the full run output.
- **Ablations** (`tests/ablations/test_ablations.py` -> `reports/ABLATION_REPORT.md`): includes a
  genuine negative result — on the demo's small structured synthetic signal, the trained JEPA
  world model did *not* beat the trivial static (last-value) baseline after 150 training steps.
  Preserved and reported rather than hidden, per spec.

### What's simulated, not real hardware/services

- No ROS2/DDS — `InProcessBus` stands in.
- No physical reservoir — `SimulatedReservoir` (random-projection leaky integrator) stands in.
- No real robot actuation — skill handlers are deterministic simulated functions.
- Qdrant/Neo4j/ClickHouse/MinIO/PostgreSQL are not installed in this environment; their adapters
  are written and interface-complete but untested against real running services.
- No real hardware at all — `simulate/mujoco_world.py` is real physics, but a simulation.
- The `Dockerfile` in this repo wraps the exact install/run steps verified throughout development,
  but the Docker build itself was not run in the environment this was built in (no Docker daemon
  available there) — verify it locally before relying on it, and open an issue if it doesn't work.

### Real 3D physics (MuJoCo) and a small local LLM

Two optional adapters, both genuinely real, both CPU-only:

**`simulate/mujoco_world.py`** replaces the hand-rolled 2D holonomic kinematics with actual
rigid-body physics via Google DeepMind's MuJoCo — a velocity-actuated planar robot base with real
mass and joint damping (commanded velocity produces realistic ramp-up, not an instant position
change), and human/obstacle entities as MuJoCo "mocap" bodies (externally-scripted position, real
collision geometry). `SensorSimulator` takes either backend behind a shared `WorldBackend`
`Protocol` (`simulate/world.py`) with zero changes to its own logic — `--physics mujoco` on
`agi_demo.py` swaps it in. No GPU: single-robot MuJoCo stepping is a CPU workload; GPU/TPU
parallelization (MJX) is for training thousands of simultaneous environments, not this.

**`planner/llm_goal_interpreter.py`** lets a small local LLM (Qwen2.5-0.5B-Instruct, Q4_K_M GGUF,
Apache 2.0, ~470MB, via `llama-cpp-python`) turn a natural-language instruction into a `Goal`.
Deliberately scoped to a role CPU inference can actually serve: called once per instruction, never
inside the reactive loop (measured on this machine: ~0.7s per LLM call vs. 3-5ms p50 for a guarded
decision cycle — those two numbers cannot share a loop). Tested against the real downloaded model,
not a mock. What that testing found, and why validation is the actual safety net here, not model
compliance: given a clear instruction ("watch human-1 from 2 meters"), it reliably produced correct
JSON. Given an off-topic one ("what's your favorite color?"), it did **not** follow its own explicit
"respond null" instruction — it confidently fabricated a plausible-looking fake entity
(`"color-0"`) wrapped in otherwise well-formed JSON. `LLMGoalInterpreter._validate` is what catches
this (checking the entity_id against a real pattern and the working-memory's actual known entities),
not the model's own restraint — and every validated goal still only ever influences the *planner's
target*, never bypasses SutraFlow or the safety governor, which see the resulting proposal exactly
like any other. Run `python agi_demo.py --instruction "..."` to see both the success and rejection
paths live.

### Google's Open Knowledge Format (OKF) — measured, not just wired in

Google Cloud launched OKF (a vendor-neutral spec for packaging curated knowledge as markdown +
YAML frontmatter, explicit relationships instead of vector-similarity guessing — spec fetched
directly from `GoogleCloudPlatform/knowledge-catalog` before any code was written here, not
inferred from blog summaries) in June 2026. `knowledge/okf_loader.py` is a real, spec-conformant
parser: it implements the conformance rules exactly (a concept requires a non-empty `type` field;
`index.md`/`log.md` are reserved; malformed files are skipped and tolerated, never crash the load;
unrecognized frontmatter keys are preserved, not dropped).

`scripts/generate_okf_bundle.py` builds an actual bundle (`okf/`) describing this robot's own
skills, supported goal schema, and safety envelope — critically, the safety numbers are read
**directly from the live `SafetyEnvelope` dataclass**, not hand-typed into a markdown file that
could silently drift out of sync with what's actually enforced. That sync mechanism is the point:
a hand-maintained duplicate of a safety limit is exactly the kind of hazard the rest of this
project's effort (the audit ledger, the provable safety invariants) has gone toward avoiding.

**Does it help? Measured, not assumed** (`examples/okf_context_experiment.py`, real model, real
numbers): giving `LLMGoalInterpreter` this bundle as system-prompt context instead of a hardcoded
string dropped the hallucination rate on off-topic instructions ("what's the weather like?", "tell
me a joke") from **4/4 (100%) to 1/4 (25%)** — the model started correctly responding `null`
instead of confidently inventing a fake entity, in 3 of 4 cases. It did **not** fix a separate,
unrelated weakness: the model still mis-parsed "a meter and a half" as `0.5` instead of `1.5`
meters, with or without OKF context. Both results are reported because the negative one is just as
real as the positive one. Note also: system-level safety was already 100% correct in *both*
conditions — `_validate` catches every hallucination regardless of whether the model produced it —
so this measures a genuine improvement in model behavior and prompt efficiency, not a safety gap
that only OKF closes.
