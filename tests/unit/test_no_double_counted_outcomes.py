"""Regression test for a real bug found via profiling (not deliberate
fuzzing this time — an anomalously high call count for record_outcome
relative to the number of executions was the tell): `_execute()` used to
call `skill_registry.record_outcome()` directly AND `reviewed_learning`
called it again internally for the same execution, silently doubling
every CORRECT/INCORRECT outcome's count and — worse — recording a stat
update for UNCERTAIN outcomes too, which the system's own documented
bounded-learning principle says must not happen. success_rate() alone
wouldn't have caught this (doubling numerator and denominator together
preserves the ratio) — this test checks the raw counts.
"""

from machine_brain.orchestrator.cognitive_loop import CognitiveBrain
from machine_brain.simulate.sensors import SensorSimulator, SimConfig


def _raw_counts(brain: CognitiveBrain, skill_id: str, version: int = 1):
    row = brain.skill_registry.conn.execute(
        "SELECT success_count, failure_count FROM skills WHERE skill_id=? AND version=?", (skill_id, version)
    ).fetchone()
    return row["success_count"], row["failure_count"]


def test_a_single_execution_increments_the_skill_count_exactly_once(tmp_path):
    brain = CognitiveBrain(data_dir=str(tmp_path / "data"))
    sim = SensorSimulator(SimConfig(seed=0))

    executed_skill = None
    for i in range(200):
        brain.perceive(sim.next_frame())
        if i % 3 == 0:
            result = brain.cycle()
            if result.outcome is not None:
                executed_skill = result.proposal.skill_id
                break

    assert executed_skill is not None, "test setup: expected at least one execution in 200 frames"
    success, failure = _raw_counts(brain, executed_skill)
    total = success + failure
    assert total == 1, f"expected exactly 1 recorded outcome after 1 execution, got {total} (success={success}, failure={failure})"


def test_total_recorded_outcomes_never_exceeds_total_executions(tmp_path):
    """Broader property over a longer run: summed across every skill, the
    number of recorded outcomes must never exceed the number of times
    something actually executed — catches the doubling bug (or any future
    reintroduction of it) without hardcoding which skill runs when."""
    brain = CognitiveBrain(data_dir=str(tmp_path / "data"))
    sim = SensorSimulator(SimConfig(seed=1))

    executions = 0
    for i in range(600):
        brain.perceive(sim.next_frame())
        if i % 3 == 0:
            result = brain.cycle()
            if result.outcome is not None:
                executions += 1

    rows = brain.skill_registry.conn.execute("SELECT success_count, failure_count FROM skills").fetchall()
    total_recorded = sum(r["success_count"] + r["failure_count"] for r in rows)
    assert total_recorded <= executions, (
        f"{total_recorded} outcomes recorded across all skills but only {executions} executions happened — "
        f"some execution's outcome was counted more than once"
    )
