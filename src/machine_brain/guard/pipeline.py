"""GuardedPolicy — the standalone, model-agnostic safety boundary.

Everything upstream of this module (a hand-written rule planner, the
imagination-based planner elsewhere in this package, or — this is the
point — a completely different, much larger system: a frontier-scale
world model such as Nvidia's Cosmos or Meta's V-JEPA driving a real robot)
has exactly one job: propose an `ActionProposal`. It has no authority to
act on its own proposal.

`GuardedPolicy` is the only thing in the loop with that authority. It
takes any proposal from any source, runs it through SutraFlow (task-
grammar validity) and the safety governor (hard limits), combines their
verdicts with the safety governor's refusal always winning, writes every
verdict — allowed or refused — to the hash-chained audit ledger
unconditionally, and returns a verdict the caller must obey.

This is deliberately extracted from `orchestrator/cognitive_loop.py`
rather than duplicated: `CognitiveBrain` uses this exact class for its own
guard step (see `cognitive_loop.py`), so this isn't a demo-only wrapper
sitting next to the real pipeline — it IS the real pipeline, packaged so
anything else can use it too. `examples/wrap_external_policy.py` shows an
external, unrelated proposal source (standing in for a third-party world
model) plugged into the same class with zero changes.
"""

from __future__ import annotations

from dataclasses import dataclass

from machine_brain.audit.ledger import AuditLedger
from machine_brain.contracts import ActionProposal, GuardDecision, GuardVerdict
from machine_brain.safety.governor import SafetyGovernor
from machine_brain.sutraflow.validator import SutraFlowValidator


@dataclass(frozen=True)
class GuardOutcome:
    verdict: GuardVerdict
    sutraflow_decision: GuardDecision
    safety_decision: GuardDecision

    @property
    def reasons(self) -> list[str]:
        return list(self.sutraflow_decision.reasons) + list(self.safety_decision.reasons)


class GuardedPolicy:
    """Construct once per robot/session; call `evaluate()` for every
    proposal any policy produces, regardless of what generated it."""

    def __init__(self, sutraflow: SutraFlowValidator | None = None, safety: SafetyGovernor | None = None,
                  audit: AuditLedger | None = None, audit_db_path: str = "guard_audit.sqlite") -> None:
        self.sutraflow = sutraflow or SutraFlowValidator()
        self.safety = safety or SafetyGovernor()
        self.audit = audit or AuditLedger(audit_db_path)

    def evaluate(self, proposal: ActionProposal, context: dict) -> GuardOutcome:
        sutra_decision = self.sutraflow.validate(proposal, context)
        self.audit.record(sutra_decision.decision_id, proposal.proposal_id, sutra_decision.verdict.value,
                            list(sutra_decision.reasons), list(sutra_decision.rule_ids), source="sutraflow")

        safety_decision = self.safety.check(proposal, context)
        self.audit.record(safety_decision.decision_id, proposal.proposal_id, safety_decision.verdict.value,
                            list(safety_decision.reasons), list(safety_decision.rule_ids), source="safety_governor")

        verdict = self._combine(sutra_decision.verdict, safety_decision.verdict)
        return GuardOutcome(verdict=verdict, sutraflow_decision=sutra_decision, safety_decision=safety_decision)

    @staticmethod
    def _combine(sutra_verdict: GuardVerdict, safety_verdict: GuardVerdict) -> GuardVerdict:
        # Safety (antaranga) is treated as already decided and overrides
        # SutraFlow (bahiranga) unconditionally — see safety/governor.py
        # for the Pāṇinian reasoning behind this ordering.
        if safety_verdict is GuardVerdict.REFUSE:
            return GuardVerdict.REFUSE
        if sutra_verdict in (GuardVerdict.REFUSE, GuardVerdict.HOLD):
            return sutra_verdict
        return GuardVerdict.ALLOW

    def verify_audit_chain(self) -> tuple[bool, int | None]:
        return self.audit.verify_chain()
