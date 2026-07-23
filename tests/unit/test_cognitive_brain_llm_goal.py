from machine_brain.contracts import WorldEntity
from machine_brain.orchestrator.cognitive_loop import CognitiveBrain
from machine_brain.planner.llm_goal_interpreter import LLMGoalInterpreter


def test_no_interpreter_configured_fails_closed(tmp_path):
    brain = CognitiveBrain(data_dir=str(tmp_path / "data"))
    goal, reason = brain.set_goal_from_instruction("watch human-0")
    assert goal is None
    assert reason == "no LLM interpreter configured"
    assert brain.working_memory.active_goal() is None


def test_valid_instruction_sets_a_real_goal(tmp_path):
    interp = LLMGoalInterpreter(
        llm_call=lambda i: '{"kind": "observe_entity", "target": {"entity_id": "human-0", "desired_distance": 1.5}}'
    )
    brain = CognitiveBrain(data_dir=str(tmp_path / "data"), llm_interpreter=interp)
    brain.working_memory.upsert_entity(WorldEntity("human-0", "human", {"distance": 3.0}, 0, 0.9))

    goal, reason = brain.set_goal_from_instruction("watch human-0 from 1.5 meters")
    assert reason is None
    assert goal is not None
    active = brain.working_memory.active_goal()
    assert active is not None and active.target["entity_id"] == "human-0"


def test_hallucinated_entity_never_becomes_a_real_goal(tmp_path):
    """The system-level version of the unit-level hallucination test:
    confirms a fabricated entity name never reaches working memory as an
    active goal, end to end through CognitiveBrain."""
    interp = LLMGoalInterpreter(
        llm_call=lambda i: '{"kind": "observe_entity", "target": {"entity_id": "weather", "desired_distance": 0.0}}'
    )
    brain = CognitiveBrain(data_dir=str(tmp_path / "data"), llm_interpreter=interp)

    goal, reason = brain.set_goal_from_instruction("what's the weather like?")
    assert goal is None
    assert reason is not None
    assert brain.working_memory.active_goal() is None
