"""GuardedPolicy is the standalone product: it must work identically for
a proposal from any source, be genuinely the same code path
CognitiveBrain uses (not a parallel demo-only implementation), and its
audit trail must be real and verifiable.
"""

from machine_brain.contracts import ActionProposal, GuardVerdict
from machine_brain.guard.pipeline import GuardedPolicy
from machine_brain.orchestrator.cognitive_loop import CognitiveBrain


def test_guarded_policy_allows_a_safe_unrelated_proposal(tmp_path):
    guard = GuardedPolicy(audit_db_path=str(tmp_path / "audit.sqlite"))
    proposal = ActionProposal.make("cruise", {"velocity": 1.0}, "external model output", 0.9)
    outcome = guard.evaluate(proposal, context={})
    assert outcome.verdict is GuardVerdict.ALLOW


def test_guarded_policy_refuses_unsafe_velocity_regardless_of_source(tmp_path):
    guard = GuardedPolicy(audit_db_path=str(tmp_path / "audit.sqlite"))
    proposal = ActionProposal.make("cruise", {"velocity": 50.0}, "external model output", 0.95)
    outcome = guard.evaluate(proposal, context={})
    assert outcome.verdict is GuardVerdict.REFUSE
    assert any("max_velocity" in r for r in outcome.reasons)


def test_guarded_policy_writes_both_decisions_to_a_verifiable_audit_chain(tmp_path):
    guard = GuardedPolicy(audit_db_path=str(tmp_path / "audit.sqlite"))
    proposal = ActionProposal.make("cruise", {"velocity": 1.0}, "ok", 0.9)
    guard.evaluate(proposal, context={})
    unsafe = ActionProposal.make("cruise", {"velocity": 50.0}, "bad", 0.9)
    guard.evaluate(unsafe, context={})

    assert guard.audit.count() == 4  # sutraflow + safety, twice
    ok, _ = guard.verify_audit_chain()
    assert ok is True


def test_cognitive_brain_uses_the_same_guardedpolicy_class(tmp_path):
    """Not a coincidence, not a lookalike — CognitiveBrain.guard IS a
    GuardedPolicy instance, so the product claim ("wrap any policy,
    including the one already driving this robot") is literally true."""
    brain = CognitiveBrain(data_dir=str(tmp_path / "data"))
    assert isinstance(brain.guard, GuardedPolicy)
    assert brain.guard.audit is brain.audit_ledger
