#!/usr/bin/env python3
"""Proof that GuardedPolicy is genuinely model-agnostic: an "external
policy" here — standing in for a third-party world model like Nvidia
Cosmos, Meta's V-JEPA, or any other planner an integrator brings — that
has never seen this codebase's ImaginationPlanner, WorldEntity, or state
vector conventions. It only needs to produce an ActionProposal. The guard
step is the exact same GuardedPolicy class CognitiveBrain itself uses
(see orchestrator/cognitive_loop.py) — not a lookalike built for this demo.

Run: python examples/wrap_external_policy.py
"""

from __future__ import annotations

import tempfile

from machine_brain.contracts import ActionProposal
from machine_brain.guard.pipeline import GuardedPolicy


class ThirdPartyWorldModelStub:
    """A stand-in for an external, unrelated policy source. Represents:
    'you already have a world model (large, small, learned, hand-coded —
    doesn't matter) that decides what to do next; you want a safety
    boundary around it without rewriting it.' This class knows nothing
    about SutraFlow, ActionProposal internals beyond the constructor, or
    any other machine_brain module."""

    def decide(self, sensor_reading: dict) -> ActionProposal:
        # Whatever internal logic a real external model would use — here,
        # a deliberately naive one-liner, to make the point that GuardedPolicy
        # doesn't care how the proposal was produced.
        if sensor_reading.get("obstacle_ahead"):
            skill, velocity, note = "brake", 0.0, "obstacle detected by external model"
        else:
            skill, velocity, note = "cruise", 1.2, "external model sees a clear path"
        return ActionProposal.make(skill_id=skill, args={"velocity": velocity}, justification=note,
                                     predicted_confidence=0.9)


def main() -> None:
    external_model = ThirdPartyWorldModelStub()

    with tempfile.TemporaryDirectory() as tmp:
        guard = GuardedPolicy(audit_db_path=f"{tmp}/external_policy_audit.sqlite")

        print("Scenario 1: clear path, external model proposes a normal cruise.")
        proposal = external_model.decide({"obstacle_ahead": False})
        outcome = guard.evaluate(proposal, context={})
        print(f"  proposed: {proposal.skill_id} @ velocity={proposal.args['velocity']}")
        print(f"  verdict:  {outcome.verdict.value}\n")

        print("Scenario 2: external model malfunctions and proposes an unsafe velocity.")
        bad_proposal = ActionProposal.make(skill_id="cruise", args={"velocity": 50.0},
                                              justification="external model output, unvetted", predicted_confidence=0.95)
        outcome = guard.evaluate(bad_proposal, context={})
        print(f"  proposed: {bad_proposal.skill_id} @ velocity={bad_proposal.args['velocity']}")
        print(f"  verdict:  {outcome.verdict.value}")
        print(f"  reasons:  {outcome.reasons}\n")

        ok, _ = guard.verify_audit_chain()
        print(f"Audit chain (both scenarios, same ledger, same class the robot's own loop uses): valid = {ok}")


if __name__ == "__main__":
    main()
