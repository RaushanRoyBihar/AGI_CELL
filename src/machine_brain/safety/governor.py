"""Deterministic safety governor. No learned component anywhere in this
module — nothing in the learning/consolidation loop is permitted to import
or write to this file's rules, and that is enforced structurally
(tests/adversarial checks that `learning/` never imports `safety/` for
anything but read-only decision logging).

Ordering principle (borrowed from the Aṣṭādhyāyī's antaraṅga/bahiraṅga
treatment of rule interaction — most explicit in the asiddhavat
paribhāṣās, e.g. 8.2.1 pūrvatrāsiddham): an antaraṅga ("inner", closer to
the operation itself) rule is treated as already in effect before a
bahiraṅga ("outer") rule gets to act. Safety is the antaraṅga concern here
— closer to the actual actuation than task-grammar validity is — so a
safety REFUSE is treated as already decided, and unconditionally overrides
whatever SutraFlow's (bahiraṅga, task-grammar) validation concluded. This
is why the orchestrator always applies the safety governor's verdict last
and lets it override an ALLOW, never the reverse.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from machine_brain.contracts import ActionProposal, GuardDecision, GuardVerdict

# \s+ (any run of whitespace), not a literal single space — the original
# literal-space patterns were trivially bypassed by extra spaces, tabs,
# newlines, or a zero-width character, none of which \s+ matches but all
# of which broke a literal " " (found by adversarial fuzzing: "IGNORE
# PREVIOUS INSTRUCTIONS" with doubled spaces sailed straight through).
_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior)\s+instructions", re.I),
    re.compile(r"disregard\s+(the\s+)?(safety|guard)", re.I),
    re.compile(r"system\s+prompt", re.I),
    re.compile(r"you\s+are\s+now", re.I),
    re.compile(r"\bsudo\b", re.I),
]

_ZERO_WIDTH_CHARS = ("\u200b", "\u200c", "\u200d", "\ufeff")  # ZWSP, ZWNJ, ZWJ, BOM


def _normalize_for_injection_scan(text: str) -> str:
    """Replace (not delete) zero-width characters an attacker could
    interleave to split a keyword while keeping it visually intact —
    deleting them merges 'ignore' + ZWSP + 'previous' into
    'ignoreprevious' with no separator at all, which then fails the
    \\s+ patterns just as badly as the original bypass did (found while
    testing the first fix, not a hypothetical second bug)."""
    for zw in _ZERO_WIDTH_CHARS:
        text = text.replace(zw, " ")
    return text


@dataclass(frozen=True)
class SafetyEnvelope:
    max_velocity: float = 1.5           # m/s, hard limit
    min_human_distance: float = 0.5     # meters
    forbidden_zones: tuple[str, ...] = ("restricted", "stairwell_edge")
    forbidden_skills: tuple[str, ...] = ()  # e.g. skills disabled fleet-wide by policy


@dataclass
class SafetyGovernor:
    envelope: SafetyEnvelope = field(default_factory=SafetyEnvelope)

    def check(self, proposal: ActionProposal, context: dict) -> GuardDecision:
        reasons: list[str] = []
        rule_ids: list[str] = []

        # Fail closed on the wrong *type*, not just the wrong value. This
        # class exists specifically to wrap policies this codebase doesn't
        # control — a malformed `velocity` (a string, a list, whatever an
        # untrusted or buggy upstream sends) must never raise an unhandled
        # exception here. An exception is worse than a wrong verdict: it
        # can propagate past a caller that doesn't expect it and end up
        # skipping the safety check entirely, an accidental fail-open.
        # (Found by fuzzing: `{"velocity": ""}` crashed `math.isfinite`
        # with a bare TypeError before this check existed.)
        velocity = proposal.args.get("velocity")
        if velocity is not None and not isinstance(velocity, (int, float)):
            reasons.append(f"velocity {velocity!r} is not a number (type {type(velocity).__name__})")
            rule_ids.append("safety.invalid_velocity_type")
        # Fail closed on non-finite numbers too. NaN comparisons are always
        # False in Python/IEEE-754 — `float('nan') > max_velocity` silently
        # evaluates False, which let a NaN velocity sail through as ALLOW
        # (found by adversarial fuzzing, not a hypothetical). A sensor or
        # model producing NaN/Inf has malfunctioned; that is itself grounds
        # for refusal, not something to compare past.
        elif velocity is not None and not math.isfinite(velocity):
            reasons.append(f"velocity {velocity} is not a finite number")
            rule_ids.append("safety.non_finite_velocity")
        # Magnitude, not just the forward direction: a large negative
        # (reverse) velocity is exactly as dangerous as a large positive
        # one, and the original `velocity > max_velocity` check never
        # caught it.
        elif velocity is not None and abs(velocity) > self.envelope.max_velocity:
            reasons.append(f"velocity {velocity} exceeds max_velocity {self.envelope.max_velocity} in magnitude")
            rule_ids.append("safety.max_velocity")

        human_distance = context.get("nearest_human_distance")
        if human_distance is not None and not isinstance(human_distance, (int, float)):
            reasons.append(f"nearest_human_distance {human_distance!r} is not a number "
                             f"(type {type(human_distance).__name__})")
            rule_ids.append("safety.invalid_human_distance_type")
        elif human_distance is not None and not math.isfinite(human_distance):
            reasons.append(f"nearest_human_distance {human_distance} is not a finite number")
            rule_ids.append("safety.non_finite_human_distance")
        elif human_distance is not None and human_distance < self.envelope.min_human_distance:
            reasons.append(
                f"nearest human at {human_distance}m is closer than min_human_distance {self.envelope.min_human_distance}m"
            )
            rule_ids.append("safety.min_human_distance")

        # Normalized (trimmed, case-folded) comparison. Exact-match `in`
        # checks against a blocklist are trivially bypassed by
        # 'Restricted' / 'RESTRICTED' / ' restricted' — found by adversarial
        # fuzzing: all four passed straight through the original check.
        zone = proposal.args.get("zone")
        if isinstance(zone, str) and zone.strip().lower() in {z.lower() for z in self.envelope.forbidden_zones}:
            reasons.append(f"zone '{zone}' is forbidden")
            rule_ids.append("safety.forbidden_zone")

        skill_id = proposal.skill_id
        if isinstance(skill_id, str) and skill_id.strip().lower() in {s.lower() for s in self.envelope.forbidden_skills}:
            reasons.append(f"skill '{skill_id}' is disabled by policy")
            rule_ids.append("safety.forbidden_skill")

        injection_hit = self._check_injection(proposal)
        if injection_hit:
            reasons.append(f"possible prompt injection in proposal args: matched pattern '{injection_hit}'")
            rule_ids.append("safety.injection_guard")

        if reasons:
            return GuardDecision.make(proposal.proposal_id, GuardVerdict.REFUSE, reasons, rule_ids)
        return GuardDecision.make(proposal.proposal_id, GuardVerdict.ALLOW, [], [])

    def _check_injection(self, proposal: ActionProposal) -> str | None:
        text_fields = [str(v) for v in proposal.args.values() if isinstance(v, str)]
        text_fields.append(proposal.justification)
        blob = _normalize_for_injection_scan(" ".join(text_fields))
        for pattern in _INJECTION_PATTERNS:
            if pattern.search(blob):
                return pattern.pattern
        return None
