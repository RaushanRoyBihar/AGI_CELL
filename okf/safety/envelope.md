---
type: Safety Envelope
title: Robot Safety Envelope
description: The hard velocity, proximity, and zone limits SafetyGovernor enforces unconditionally.
tags: [safety, hard-limits]
generated_from: machine_brain.safety.governor.SafetyEnvelope
---

# Hard Limits

These are not policy suggestions. They are enforced unconditionally by `SafetyGovernor.check`,
and no learning or planning process in this system has a code path that can modify them —
see `docs/00_ARCHIVE_ANALYSIS/../01_ARCHITECTURE` and `tests/unit/test_safety_invariants.py`,
which mechanically verifies this rather than just documenting it.

| Limit | Value |
|---|---|
| Max velocity (magnitude, either direction) | 1.5 m/s |
| Minimum distance to any human | 0.5 m |
| Forbidden zones | restricted, stairwell_edge |
| Forbidden skills (fleet policy) | (none currently disabled) |

# Also Enforced

- Non-finite velocity or distance readings (NaN, Inf) are refused outright, not compared past.
- Zone and skill blocklist checks are case- and whitespace-normalized.
- Proposal text is scanned for prompt-injection patterns (whitespace- and zero-width-character-robust).

These specific hardening details exist because adversarial fuzzing found each of them missing at
some point during this project's development — see `docs/provenance/NOTES.md` for the exact bugs.
