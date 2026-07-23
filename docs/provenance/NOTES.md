# Provenance Notes

Honest accounting of what in `machine_brain/src` came from where.

## What this build actually is

Every file under `src/machine_brain/` is **freshly written for this build**, not copy-pasted from
the donor `.tar.gz` archives. The donor archives were inspected (see
`00_ARCHIVE_ANALYSIS/DONOR_COMPONENT_MAP.md`) to identify *which concepts and interfaces* were
worth having — but literally extracting and wiring ~97,000 LOC of `vajra-v0.39-UNIFIED` (which
assumes ROS2, PyTorch, and specific donor-internal conventions) directly into a from-scratch,
dependency-light architecture in one pass would have violated the brief's own instruction not to
use the archives wholesale, and would have re-imported whatever inconsistency, dead code, and
implicit coupling made the donor map necessary in the first place.

What was carried over is **the architectural idea**, reimplemented cleanly against this project's
own contracts:

| Concept | Donor origin (per DONOR_COMPONENT_MAP.md) | What's actually in this build |
|---|---|---|
| EMA target encoder + VICReg-style regularization + `surprise()` | `jepa_world_engine.tar.gz` | Reimplemented from scratch in `world_model/jepa.py`, same architecture (online encoder, EMA target, invariance/variance/covariance loss terms), original code, added gradient clipping the donor didn't have (numerical instability found during testing — see below) |
| Hash-chained append-only audit ledger | `dharma_gov_india` (`sakshi.py`), `vajra/audit/` | Reimplemented from scratch in `audit/ledger.py` — SQLite, SHA-256 chained rows, `verify_chain()` |
| `.sutra` DSL / guard rule engine | `sutraflow_chainflow_pro`, `pail_sutraflow` | **Not ported.** `sutraflow/rules.py` is an original, much smaller rule engine whose conflict-resolution algorithm is built directly from two real Aṣṭādhyāyī paribhāṣā sutras (vipratiṣedhe paraṃ kāryam 1.4.2, apavādaḥ utsargaṃ bādhate) — see that file's docstring. This is a deliberate reimplementation, not an attempt to reproduce the donor DSL's parser/compiler. |
| Resonance tie-break, never overriding identity/safety | `resonance_engine_full`, `panani_platform/nada` | Reimplemented from scratch in `acoustic/resonance.py` as a narrow tie-break-only function; donor's MFCC/NADA codec not used — a dependency-free FFT-band fingerprint stands in |
| SafetyEnvelope / CommandGovernor shape | `robotics_safety_runtime.tar.gz` | Reimplemented from scratch in `safety/governor.py`, same envelope concept (velocity/proximity/zone limits), added the injection-guard pattern from `dharma_gov_india`'s `astra.py` concept |
| Physical reservoir adapter interface | Spec-only (no direct donor) | Original — `reservoir/adapter.py`, random-projection leaky-integrator stand-in per spec instructions |

## Real donor code integration (first actual slice, not just concepts)

Everything above is a clean-room reimplementation of a *concept* the donor archives suggested.
`sutraflow/panini_phonology.py` is different: it's the first place actual donor **files** were
copied in, verbatim, and used.

- **Copied verbatim, unmodified:** `vajra-v0.39-UNIFIED/vajra/sutras/maheshvara_shiva_sutras.json`
  and `.../ashtadhyayi_legacy_executable_seed.json`, now living at `sutraflow/data/`. These two
  files were chosen deliberately out of the donor's four sutra JSON files because they're
  genuinely authentic: the 14 Māheśvara/Śiva Sūtras (the real phoneme-classification sutras that
  open the Aṣṭādhyāyī) and a set of vowel-sandhi/definitional rules carrying real, checkable
  canonical sutra numbers (6.1.87, 6.1.88, 6.1.89, 6.1.101, 1.1.1, 1.1.2, etc.). The other two
  donor sutra files (`v82_paribhasha_karaka_electric_sutras.json`,
  `production_guard_sutras.json`) were inspected and deliberately **not** used — they're
  donor-authored policy rules *named* using Paninian terminology (paribhāṣā, kāraka) but aren't
  actual sutra citations, and their `condition`/`operation` fields just reference an unseen
  `PananiSutraGuard.evaluate` method rather than containing real logic. Using them would have
  meant presenting synthetic content as authentic grammar, which this project treats as a real
  line not to cross.
- **Not copied from the donor, written fresh:** the pratyāhāra-construction algorithm and the
  sandhi transformation logic in `panini_phonology.py`. The donor's own Python implementation of
  these (if any — not inspected) was not used; the logic here was built directly against the
  classical definition and verified against independently-checkable facts (the "ac" pratyāhāra
  must equal the 9-vowel inventory; "hal" must equal the 33-consonant inventory; 6.1.87 must
  produce guṇa sandhi a+i→e) — see `tests/unit/test_panini_phonology.py`, which fails loudly if
  the construction is wrong, not just internally inconsistent.
- **Not yet wired into the guard pipeline.** This module is a real, tested, standalone phonology
  capability, but `sutraflow/validator.py`'s active rule set doesn't call into it yet — that's a
  natural next step (e.g., using `hal_consonants()`/`ac_vowels()` to validate phonetic content in
  the acoustic/resonance layer) rather than something claimed as done here.

## What was NOT carried over at all

- ROS 2 / DDS transport (`transport/bus.py` is an in-process pub/sub stand-in — real ROS2 wiring
  is future work, flagged explicitly in that file's docstring)
- `panani_pckge/transport` (CRDT/ECC/modulation) — not used; this build's transport layer is
  intentionally minimal until a real ROS2 environment exists to test against
- Any PyTorch-based model code — everything numeric here is NumPy, per spec ("bounded numeric
  state prediction", not "video-scale JEPA")
- The website bundle inside `PAIL_RCA_ELECTRIC_BRIDGE_HANDOFF` — untouched, per instructions

## Bugs found and fixed during this build (for the record)

- `world_model/jepa.py`: initial learning rate (1e-2) caused gradient overflow (`RuntimeWarning:
  overflow encountered in square`) within ~100 training steps on the demo's synthetic data. Fixed
  with global-norm gradient clipping and a lower default learning rate (1e-3).
- `orchestrator/cognitive_loop.py`: episode dedupe initially hardcoded `mcap_offset_start/end=0`
  for every execution, which made the idempotency-by-MCAP-offset mechanism (correctly, by design)
  treat almost every real execution as a duplicate of the first. Fixed by logging each execution
  to MCAP for a real, unique offset before building the `Episode`.
- `simulate/sensors.py`: the synthetic sensor generator drew an independent random distance every
  time it emitted the same `entity_id`, which made the contradiction detector fire on nearly every
  repeat sighting (a simulator realism bug, not a guard-logic bug). Fixed with a per-entity random
  walk so distances evolve smoothly and the deliberate `contradictory_pair()` injector is what
  actually exercises the contradiction path in tests.

## Adversarial hardening pass (property-based fuzzing + static analysis)

Prompted by a direct request to find and close real gaps rather than declare the system finished.
`hypothesis` (property-based fuzzing) and `mypy` (static type checking) were installed and run
against the guard boundary specifically, since that's the component every other claim in this
project depends on being trustworthy. Four real, exploitable bugs were found and fixed — not
edge-case nitpicks:

1. **NaN-velocity bypass.** `float('nan') > max_velocity` evaluates `False` in IEEE-754 — a NaN
   velocity sailed straight through `SafetyGovernor.check` as `ALLOW`. Fixed with an explicit
   `math.isfinite` check that fails closed (refuses) on any non-finite number.
2. **Zone-name case/whitespace bypass.** `'Restricted'`, `'RESTRICTED'`, `' restricted'`, and
   `'restricted '` all bypassed the exact-string `in` check against the forbidden-zones tuple,
   while only exact-lowercase `'restricted'` was actually blocked. Fixed with normalized
   (trimmed, case-folded) comparison for both zones and forbidden skills.
3. **Injection-pattern whitespace/zero-width bypass.** The regexes used literal single spaces
   (`"ignore (all )?(previous|prior) instructions"`), so doubled spaces, tabs, newlines, or a
   zero-width Unicode character interleaved between words all defeated the match. Fixed with
   `\s+` patterns and an explicit normalization pass that replaces (not deletes — deleting first
   merged words into one unmatched token, a second bug caught while fixing the first) zero-width
   characters with real spaces before scanning.
4. **Unhandled-exception fail-open risk.** A non-numeric `velocity` (e.g. an empty string from a
   malformed or malicious upstream policy) crashed `math.isfinite` with a bare `TypeError` instead
   of producing a verdict. In a component built specifically to wrap *untrusted* policy sources,
   an unhandled exception is worse than a wrong verdict — it can propagate past a caller that
   doesn't expect it and skip the safety check entirely. Fixed with an explicit type check that
   fails closed (refuses) on any non-numeric type, instead of trusting the input's shape.

The property-based tests that found these (`tests/adversarial/test_fuzz_guard.py`) state actual
safety invariants — "for all finite velocities exceeding the envelope, the verdict is REFUSE",
not "for these three examples I thought of" — and stay in the suite to catch regressions.

`mypy` (run clean, zero findings after fixes) additionally caught a latent crash: `horizon = 0` in
`ImaginationConfig` would have left the rollout loop never executing, leaving a `None` where an
`ndarray` was expected several calls later. Fixed with `__post_init__` validation that fails at
construction time instead of deep inside a rollout. Two more mypy findings pointed at *real*
cross-attribute invariants (`_last_state`/`_last_action_vec` always being set together,
`cur.lastrowid` always being non-None immediately after an INSERT) that were true by construction
but unenforced — both are now explicit, checked `assert` statements at the point they're relied on,
not just an implicit assumption.

Also added: `tests/unit/test_safety_invariants.py`, which parses the actual source via `ast`
(not text search — a naive substring check false-positived on this exact module's own docstring
explaining the invariant, caught while writing the test) to mechanically verify that
`learning/reviewed_learning.py` never imports or references `SafetyGovernor`/`SafetyEnvelope`, and
that no method outside construction ever reassigns a `SafetyEnvelope` field. This turns "learning
can never touch safety" from a documented convention into something the test suite actually
disproves the build on.

Also added: `tests/unit/test_real_physics_dynamics.py`, training the JEPA world model against a
real damped-pendulum trajectory (textbook RK4-integrated equations of motion, not an arbitrary
self-authored sine wave) instead of only synthetic signals. The simulator itself is checked against
a real physical invariant (mechanical energy must not increase under damping) before being trusted
as test data. Reported honestly: on this real-physics signal, the small JEPA model's mean
prediction error (0.051) was still *worse* than the trivial last-value baseline's (0.008) — the
same negative result the ablation report already surfaced on synthetic data, now confirmed on
genuine physics rather than dismissible as an artifact of a convenient made-up signal. This is a
real, current limitation of the JEPA implementation's size/training regime, not fixed here.

## Suggested next step if more literal donor code integration is wanted

The phonology slice above is deliberately small and low-risk: pure JSON data plus ~130 lines of
fresh, independently-verified logic, zero new dependencies. If the goal becomes "actually merge
`vajra-v0.39-UNIFIED`'s 97k LOC of *application* code in" (not just its data), that is a distinct,
much larger follow-up: it requires resolving donor-internal dependencies (PyTorch, ROS2, Postgres,
Qdrant clients), reconciling its own internal package layout against this project's layer
boundaries, and re-running the donor's own test suite to establish a baseline before any merge.
Worth doing deliberately, phase by phase, per `IMPLEMENTATION_CHECKLIST.md` — not in the same pass
as building the clean-room interfaces this prototype needed to be tested and benchmarked at all.
