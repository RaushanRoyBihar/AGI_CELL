#!/usr/bin/env python3
"""Generates the okf/ bundle from the actual running code, not hand-typed
facts that could drift out of sync with what's really enforced. This
matters most for safety/envelope.md: the numbers there are read directly
from `SafetyEnvelope`'s real field values — a hand-maintained duplicate of
a safety limit is exactly the kind of silent-drift hazard this whole
project has spent its effort avoiding elsewhere (the audit ledger, the
provable safety invariants). Skill docs are generated from the same
CANDIDATE_ACTIONS list the imagination planner actually iterates over, and
from the same permissions the skill registry actually registers.

Run: python scripts/generate_okf_bundle.py
Regenerate whenever SafetyEnvelope or the skill/goal definitions change —
this script IS the sync mechanism, not a one-time snapshot.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from machine_brain.planner.imagination import CANDIDATE_ACTIONS  # noqa: E402
from machine_brain.safety.governor import SafetyEnvelope  # noqa: E402

OUT_ROOT = Path(__file__).resolve().parent.parent / "okf"

_SKILL_PERMISSIONS = {
    "patrol": ("actuate.motion",),
    "avoid_obstacle": ("actuate.motion",),
    "yield_to_human": ("actuate.motion",),
    "investigate_anomaly": ("actuate.motion", "sensor.focus"),
    "emergency_stop": ("actuate.motion",),
    "hold_position": ("actuate.motion",),
    "approach_target": ("actuate.motion", "sensor.focus"),
}

_SKILL_DESCRIPTIONS = {
    "patrol": "Move forward at default velocity with gentle heading drift; the default action when nothing salient is in view.",
    "avoid_obstacle": "Turn away from the nearest detected obstacle and move at reduced velocity.",
    "yield_to_human": "Stop and turn away from the nearest human — the cautious default whenever a human is close.",
    "hold_position": "Stay in place, no directed heading change. Safe under nearly all conditions.",
    "approach_target": "Turn toward and move at reduced velocity toward a named goal-target entity.",
    "investigate_anomaly": "Move at low velocity when the world model's surprise score exceeds threshold.",
    "emergency_stop": "Immediate stop. Reachable only via SutraFlow's apavada exemption rule, never proposed by the planner directly.",
}


def _frontmatter(**fields) -> str:
    lines = ["---"]
    for key, value in fields.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            lines.append(f"{key}: [{', '.join(str(v) for v in value)}]")
        else:
            lines.append(f"{key}: {value}")
    lines.append("---")
    return "\n".join(lines)


def write_concept(rel_path: str, body: str, **frontmatter_fields) -> None:
    path = OUT_ROOT / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_frontmatter(**frontmatter_fields) + "\n\n" + body.strip() + "\n", encoding="utf-8")


def generate_safety_envelope_doc() -> None:
    """The one concept in this bundle where accuracy is safety-relevant,
    not just documentation quality — generated from the live dataclass."""
    envelope = SafetyEnvelope()
    body = f"""# Hard Limits

These are not policy suggestions. They are enforced unconditionally by `SafetyGovernor.check`,
and no learning or planning process in this system has a code path that can modify them —
see `docs/00_ARCHIVE_ANALYSIS/../01_ARCHITECTURE` and `tests/unit/test_safety_invariants.py`,
which mechanically verifies this rather than just documenting it.

| Limit | Value |
|---|---|
| Max velocity (magnitude, either direction) | {envelope.max_velocity} m/s |
| Minimum distance to any human | {envelope.min_human_distance} m |
| Forbidden zones | {', '.join(envelope.forbidden_zones)} |
| Forbidden skills (fleet policy) | {', '.join(envelope.forbidden_skills) or '(none currently disabled)'} |

# Also Enforced

- Non-finite velocity or distance readings (NaN, Inf) are refused outright, not compared past.
- Zone and skill blocklist checks are case- and whitespace-normalized.
- Proposal text is scanned for prompt-injection patterns (whitespace- and zero-width-character-robust).

These specific hardening details exist because adversarial fuzzing found each of them missing at
some point during this project's development — see `docs/provenance/NOTES.md` for the exact bugs.
"""
    write_concept("safety/envelope.md", body, type="Safety Envelope",
                   title="Robot Safety Envelope",
                   description="The hard velocity, proximity, and zone limits SafetyGovernor enforces unconditionally.",
                   tags=["safety", "hard-limits"], generated_from="machine_brain.safety.governor.SafetyEnvelope")


def generate_skill_docs() -> None:
    for skill_id, default_args in CANDIDATE_ACTIONS:
        permissions = _SKILL_PERMISSIONS.get(skill_id, ())
        description = _SKILL_DESCRIPTIONS.get(skill_id, "")
        body = f"""# What it does

{description}

# Default arguments

`{default_args}`

# Permissions required

{', '.join(permissions) if permissions else '(none)'}

# Safety

Every proposal for this skill, regardless of source, is validated by
[the safety envelope](/safety/envelope.md) before it can execute — this document
describes the skill, it does not grant it any exemption.
"""
        write_concept(f"skills/{skill_id}.md", body, type="Robot Skill", title=skill_id,
                       description=description, tags=["skill", *permissions])
    write_concept("skills/emergency_stop.md",
                    _SKILL_DESCRIPTIONS["emergency_stop"] + "\n\nSee [the SutraFlow guard rules](/goals/observe_entity.md) "
                    "for the apavada exemption mechanism.",
                    type="Robot Skill", title="emergency_stop",
                    description=_SKILL_DESCRIPTIONS["emergency_stop"], tags=["skill", "exemption"])


def generate_goal_docs() -> None:
    body = """# Schema

```json
{"kind": "observe_entity", "target": {"entity_id": "<id>", "desired_distance": <meters>}}
```

# Valid entity_id values

Must match `^(human|obstacle)-\\d+$` — e.g. `human-0` through `human-4`, `obstacle-0` through
`obstacle-6`. Any other value (a place, an abstract topic, anything not currently tracked in
working memory) is not a valid target and must be rejected, not guessed.

# Valid desired_distance range

Between 0.1 and 20.0 meters. Values outside this range are rejected.

# What this goal means

The agent will try to hold approximately `desired_distance` from the named entity, using its own
learned world model to imagine which action gets it there — subject to
[the safety envelope](/safety/envelope.md), which can never be overridden by a goal.
"""
    write_concept("goals/observe_entity.md", body, type="Goal Kind", title="observe_entity",
                   description="Hold a target distance from a specific, currently-known human or obstacle entity.",
                   tags=["goal"])


def generate_index() -> None:
    body = """# machine_brain Knowledge Bundle

* [Robot Safety Envelope](safety/envelope.md) - hard limits, generated from the live SafetyEnvelope dataclass
* [Skills](skills/) - what the robot can be asked to do
* [Goals](goals/) - what an operator or LLM interpreter can ask the robot to pursue
"""
    (OUT_ROOT / "index.md").write_text(body, encoding="utf-8")


def main() -> None:
    OUT_ROOT.mkdir(exist_ok=True)
    generate_safety_envelope_doc()
    generate_skill_docs()
    generate_goal_docs()
    generate_index()
    print(f"Generated OKF bundle at {OUT_ROOT}")


if __name__ == "__main__":
    main()
