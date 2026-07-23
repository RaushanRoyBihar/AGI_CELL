import numpy as np

from machine_brain.contracts import Goal
from machine_brain.orchestrator.cognitive_loop import CognitiveBrain
from machine_brain.planner.imagination import (
    CANDIDATE_ACTIONS, ImaginationConfig, ImaginationPlanner, encode_action,
)
from machine_brain.procedural.skills import SkillDefinition, SkillRegistry
from machine_brain.world_model.dynamics import ActionConditionedDynamics, DynamicsConfig


def _noop_handler(args):
    return {"succeeded": True}


def _fresh_registry(tmp_path, name="skills.sqlite"):
    reg = SkillRegistry(str(tmp_path / name))
    for skill_id, _ in CANDIDATE_ACTIONS:
        reg.register(SkillDefinition(skill_id, 1, {}, ("actuate.motion",), _noop_handler))
    return reg


def test_dynamics_model_learns_a_simple_transition(tmp_path):
    state_dim, action_dim = 5, 7
    dynamics = ActionConditionedDynamics(DynamicsConfig(state_dim=state_dim, action_dim=action_dim, lr=1e-2, seed=0))
    rng = np.random.default_rng(0)

    # deterministic synthetic rule: next_state = state + action_effect (first 2 dims move toward 0)
    def true_transition(state, action):
        return state * 0.9 + action[:state_dim] * 0.1

    state = rng.normal(size=state_dim)
    action = rng.normal(size=action_dim)
    before_error = float(np.linalg.norm(dynamics.predict(state, action) - true_transition(state, action)))

    for _ in range(300):
        s = rng.normal(size=state_dim)
        a = rng.normal(size=action_dim)
        dynamics.train_step(s, a, true_transition(s, a))

    after_error = float(np.linalg.norm(dynamics.predict(state, action) - true_transition(state, action)))
    assert after_error < before_error


def test_imagination_planner_avoids_predicted_risk(tmp_path):
    dynamics = ActionConditionedDynamics(DynamicsConfig(state_dim=5, action_dim=len(CANDIDATE_ACTIONS) + 1, seed=0))

    # monkeypatch predict: yield_to_human (velocity=0) imagines a SAFE human distance,
    # every other action imagines a dangerously close human.
    def fake_predict(state, action):
        velocity = action[-1] * 2.0
        human_dist = 2.0 if velocity == 0.0 else 0.1
        return np.array([human_dist, 5.0, 1.0, velocity, 5.0])

    dynamics.predict = fake_predict
    planner = ImaginationPlanner(dynamics, ImaginationConfig(exploration_epsilon=0.0))
    registry = _fresh_registry(tmp_path)

    proposal, candidates = planner.propose(np.array([2.0, 5.0, 1.0, 1.0, 5.0]), goal=None, skill_registry=registry)
    # yield_to_human and hold_position both predict velocity 0 -> safe; either is an
    # acceptable "avoid risk" choice, patrol (velocity>0, risky) must lose.
    assert proposal.skill_id != "patrol"
    assert proposal.args.get("velocity", 1.0) == 0.0


def test_imagination_planner_pursues_observe_entity_goal(tmp_path):
    dynamics = ActionConditionedDynamics(DynamicsConfig(state_dim=5, action_dim=len(CANDIDATE_ACTIONS) + 1, seed=0))

    # approach_target imagines landing exactly on the desired distance; everything else overshoots far away.
    def fake_predict(state, action):
        velocity = action[-1] * 2.0
        target_dist = 1.0 if velocity == 0.4 else 4.5  # 0.4 is approach_target's velocity
        return np.array([5.0, 5.0, 1.0, velocity, target_dist])

    dynamics.predict = fake_predict
    planner = ImaginationPlanner(dynamics, ImaginationConfig(exploration_epsilon=0.0))
    registry = _fresh_registry(tmp_path)
    goal = Goal.make("observe_entity", {"entity_id": "human-2", "desired_distance": 1.0})

    proposal, candidates = planner.propose(np.array([5.0, 5.0, 1.0, 1.0, 4.0]), goal=goal, skill_registry=registry)
    assert proposal.skill_id == "approach_target"


def test_cognitive_brain_falls_back_to_reactive_planner_before_dynamics_trained(tmp_path):
    brain = CognitiveBrain(data_dir=str(tmp_path / "data"))
    assert brain.imagination_planner.ready() is False
    result = brain.cycle()
    assert result.used_imagination is False


def test_cognitive_brain_uses_imagination_once_dynamics_is_trained(tmp_path):
    from machine_brain.simulate.sensors import SensorSimulator, SimConfig

    brain = CognitiveBrain(data_dir=str(tmp_path / "data"))
    sim = SensorSimulator(SimConfig(seed=42))
    used_imagination_at_some_point = False
    for i in range(200):
        brain.perceive(sim.next_frame())
        if i % 3 == 0:
            result = brain.cycle()
            if result.used_imagination:
                used_imagination_at_some_point = True
    assert used_imagination_at_some_point is True
    assert brain.dynamics.train_steps >= brain.imagination_planner.config.min_train_steps


def test_goal_store_crud_and_completion(tmp_path):
    brain = CognitiveBrain(data_dir=str(tmp_path / "data"))
    assert brain.working_memory.active_goal() is None
    goal = brain.set_goal("observe_entity", {"entity_id": "human-0", "desired_distance": 1.0})
    active = brain.working_memory.active_goal()
    assert active is not None and active.goal_id == goal.goal_id
    brain.working_memory.complete_goal(goal.goal_id)
    assert brain.working_memory.active_goal() is None


def test_guarded_execution_never_exceeds_safety_envelope_even_with_imagination(tmp_path):
    """End-to-end: run enough cycles for imagination to kick in, and confirm
    every *executed* episode still respects the safety envelope — imagination
    changes what gets proposed, never what gets allowed through."""
    from machine_brain.simulate.sensors import SensorSimulator, SimConfig

    brain = CognitiveBrain(data_dir=str(tmp_path / "data"))
    brain.set_goal("observe_entity", {"entity_id": "human-0", "desired_distance": 1.0})
    sim = SensorSimulator(SimConfig(seed=7))

    executed_velocities = []
    for i in range(400):
        brain.perceive(sim.next_frame())
        if i % 3 == 0:
            result = brain.cycle()
            if result.outcome is not None:
                executed_velocities.append(result.proposal.args.get("velocity", 0.0))

    assert all(v <= brain.safety_governor.envelope.max_velocity for v in executed_velocities)
    ok, _ = brain.audit_ledger.verify_chain()
    assert ok is True
