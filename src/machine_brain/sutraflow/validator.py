"""SutraFlow validator: task-grammar validity. Checks preconditions,
impossible transitions, unresolved contradictions, and confidence — using
the SutraRuleEngine's apavada/utsarga + vipratishedha conflict resolution.
This is deliberately separate from the SafetyGovernor (safety/governor.py):
SutraFlow asks "does this action make grammatical/procedural sense given
what we believe", the safety governor asks "is this action allowed to
happen at all, regardless of belief". See safety/governor.py docstring for
why the safety governor's refusal is always final even over a SutraFlow
ALLOW.
"""

from __future__ import annotations

from machine_brain.contracts import ActionProposal, Confidence, GuardDecision, GuardVerdict
from machine_brain.sutraflow.rules import RuleKind, SutraRuleEngine


def default_rule_engine() -> SutraRuleEngine:
    engine = SutraRuleEngine()

    engine.register(
        name="low-confidence-hold",
        kind=RuleKind.UTSARGA,
        predicate=lambda p, ctx: p.predicted_confidence < Confidence.MEDIUM.value,
        verdict=GuardVerdict.HOLD,
        reason="predicted confidence below MEDIUM threshold — held for more evidence, not silently executed",
    )

    engine.register(
        name="unresolved-contradiction-hold",
        kind=RuleKind.UTSARGA,
        predicate=lambda p, ctx: bool(ctx.get("unresolved_contradictions_for_entities")),
        verdict=GuardVerdict.HOLD,
        reason="an unresolved contradiction touches an entity this proposal depends on",
    )

    engine.register(
        name="unmet-precondition-refuse",
        kind=RuleKind.UTSARGA,
        predicate=lambda p, ctx: ctx.get("preconditions_met") is False,
        verdict=GuardVerdict.REFUSE,
        reason="skill preconditions are not satisfied by current working-memory state — impossible transition",
    )

    # apavādaḥ utsargaṃ bādhate: emergency_stop is a narrow exception that
    # overrides any of the above general holds/refusals.
    engine.register(
        name="emergency-stop-exempt",
        kind=RuleKind.APAVADA,
        sutra_note="apavādaḥ utsargaṃ bādhate — exception overrides the general rule",
        predicate=lambda p, ctx: p.skill_id == "emergency_stop",
        verdict=GuardVerdict.ALLOW,
        reason="emergency_stop is exempt from ordinary preconditions/confidence gating by design",
    )

    return engine


class SutraFlowValidator:
    def __init__(self, rule_engine: SutraRuleEngine | None = None) -> None:
        self.engine = rule_engine or default_rule_engine()

    def validate(self, proposal: ActionProposal, context: dict) -> GuardDecision:
        evaluation = self.engine.evaluate(proposal, context)
        rule_ids = [r.rule_id for r in evaluation.matched_rules]
        return GuardDecision.make(proposal.proposal_id, evaluation.verdict, evaluation.reasons, rule_ids)
