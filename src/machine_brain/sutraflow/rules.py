"""SutraFlow guard/validation layer — rule engine.

The conflict-resolution logic is not invented for this project; it is two
real Aṣṭādhyāyī paribhāṣā (meta-)sutras, used here for exactly the job they
do in the grammar: deciding which of several applicable rules governs when
more than one matches the same case.

1. अपवादः उत्सर्गं बाधते (apavādaḥ utsargaṃ bādhate)
   "The exception (apavāda) overrules the general rule (utsarga)."
   A general Paninian interpretive principle: a rule of narrower, more
   specific scope always wins over a rule of broader scope, independent of
   which was declared first. Here: a `Rule` tagged `apavada` beats any
   matching `Rule` tagged `utsarga`, regardless of registration order.

2. विप्रतिषेधे परं कार्यम् — Aṣṭādhyāyī 1.4.2 (vipratiṣedhe paraṃ kāryam)
   "In case of conflict [between rules of equal standing], the one that
   comes later [in the text] applies." Here: among matching rules of the
   *same* kind (two utsarga rules, or two apavada rules) that still
   disagree, the one registered later (higher `priority`) governs.

These two sutras together give a total order over any set of matching
rules: apavada beats utsarga; ties within a kind are broken by declaration
order. That total order is exactly what a guard/validation layer needs and
is not a decorative touch — it is the actual conflict-resolution algorithm
below.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

from machine_brain.contracts import ActionProposal, GuardVerdict


class RuleKind(str, Enum):
    UTSARGA = "utsarga"   # general rule
    APAVADA = "apavada"   # exception rule — overrides a matching utsarga


Predicate = Callable[[ActionProposal, dict], bool]


@dataclass(frozen=True)
class Rule:
    rule_id: str
    name: str                 # e.g. "no-human-proximity-at-speed"
    sutra_note: str            # which grammatical principle motivates this rule's placement, if any
    kind: RuleKind
    priority: int              # declaration order — later registered = higher priority within same kind
    predicate: Predicate       # returns True if this rule matches/applies to the proposal
    verdict: GuardVerdict
    reason: str


@dataclass
class RuleEvaluation:
    matched_rules: list[Rule]
    governing_rule: Rule | None
    verdict: GuardVerdict
    reasons: list[str]


class SutraRuleEngine:
    """Holds an ordered rule set and resolves conflicts using the two
    paribhasha sutras above. `evaluate` never mutates state — it is pure
    given (proposal, context)."""

    def __init__(self) -> None:
        self._rules: list[Rule] = []
        self._next_priority = 0

    def register(self, name: str, kind: RuleKind, predicate: Predicate, verdict: GuardVerdict,
                  reason: str, sutra_note: str = "") -> Rule:
        rule = Rule(
            rule_id=f"rule-{len(self._rules)}-{name}",
            name=name, sutra_note=sutra_note, kind=kind,
            priority=self._next_priority, predicate=predicate, verdict=verdict, reason=reason,
        )
        self._next_priority += 1
        self._rules.append(rule)
        return rule

    def evaluate(self, proposal: ActionProposal, context: dict) -> RuleEvaluation:
        matched = [r for r in self._rules if r.predicate(proposal, context)]
        if not matched:
            return RuleEvaluation(matched_rules=[], governing_rule=None, verdict=GuardVerdict.ALLOW, reasons=[])

        # Sutra 1: apavādaḥ utsargaṃ bādhate — apavada rules outrank utsarga rules outright.
        apavada_matches = [r for r in matched if r.kind is RuleKind.APAVADA]
        pool = apavada_matches if apavada_matches else matched

        # Sutra 2 (A. 1.4.2): vipratiṣedhe paraṃ kāryam — among the remaining pool
        # (all same kind by construction), the later-declared rule governs.
        governing = max(pool, key=lambda r: r.priority)

        # A REFUSE from any matched rule is never silently dropped even if it
        # doesn't "win" governance — refusals are collected as reasons, but the
        # governing rule's verdict is what the caller acts on.
        reasons = [f"{r.name}: {r.reason}" for r in matched]
        return RuleEvaluation(matched_rules=matched, governing_rule=governing,
                               verdict=governing.verdict, reasons=reasons)
