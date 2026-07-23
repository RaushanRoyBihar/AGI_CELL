"""Fast, deterministic tests using an injected stub instead of the real
469MB model (see test_llm_goal_interpreter_real_model.py for that) — these
exercise the validation logic itself, including the exact hallucination
failure mode observed from the real model during development: confident,
well-formed JSON naming an entity that doesn't exist.
"""

from __future__ import annotations

from machine_brain.planner.llm_goal_interpreter import LLMGoalInterpreter


def _stub(response: str) -> LLMGoalInterpreter:
    return LLMGoalInterpreter(llm_call=lambda instruction: response)


def test_valid_instruction_produces_a_goal():
    interp = _stub('{"kind": "observe_entity", "target": {"entity_id": "human-1", "desired_distance": 2.0}}')
    result = interp.interpret("watch human-1 from 2 meters")
    assert result.goal is not None
    assert result.goal.target["entity_id"] == "human-1"
    assert result.goal.target["desired_distance"] == 2.0
    assert result.rejected_reason is None


def test_hallucinated_entity_id_is_rejected_not_trusted():
    """The real observed failure mode: the model ignored the 'respond
    null' instruction for off-topic input and instead fabricated a
    plausible-looking entity_id wrapped in valid JSON. Validation, not
    model compliance, is what must catch this."""
    interp = _stub('{"kind": "observe_entity", "target": {"entity_id": "weather", "desired_distance": 0.0}}')
    result = interp.interpret("what's the weather like today?")
    assert result.goal is None
    assert "does not match" in result.rejected_reason


def test_entity_matching_pattern_but_not_currently_known_is_rejected():
    interp = _stub('{"kind": "observe_entity", "target": {"entity_id": "human-99", "desired_distance": 1.0}}')
    result = interp.interpret("watch human-99", known_entity_ids={"human-0", "human-1"})
    assert result.goal is None
    assert "isn't a currently known entity" in result.rejected_reason


def test_model_saying_null_is_respected():
    interp = _stub("null")
    result = interp.interpret("what time is it?")
    assert result.goal is None
    assert result.rejected_reason == "model declined (null)"


def test_malformed_json_fails_closed_not_crashes():
    interp = _stub("I think the answer is human-1 at 2 meters")
    result = interp.interpret("watch human-1")
    assert result.goal is None
    assert "not valid JSON" in result.rejected_reason


def test_json_wrapped_in_markdown_fence_is_still_parsed():
    interp = _stub('```json\n{"kind": "observe_entity", "target": {"entity_id": "obstacle-2", "desired_distance": 1.5}}\n```')
    result = interp.interpret("watch obstacle-2 from 1.5m")
    assert result.goal is not None
    assert result.goal.target["entity_id"] == "obstacle-2"


def test_out_of_bounds_distance_is_rejected():
    interp = _stub('{"kind": "observe_entity", "target": {"entity_id": "human-0", "desired_distance": 500.0}}')
    result = interp.interpret("stay 500 meters from human-0")
    assert result.goal is None
    assert "outside the sane bound" in result.rejected_reason


def test_non_numeric_distance_is_rejected_not_crashed():
    interp = _stub('{"kind": "observe_entity", "target": {"entity_id": "human-0", "desired_distance": "far away"}}')
    result = interp.interpret("stay far from human-0")
    assert result.goal is None
    assert "is not a number" in result.rejected_reason


def test_unsupported_goal_kind_is_rejected():
    interp = _stub('{"kind": "conquer_the_world", "target": {}}')
    result = interp.interpret("take over")
    assert result.goal is None
    assert "unsupported goal kind" in result.rejected_reason


def test_system_prompt_without_bundle_is_the_fallback():
    from machine_brain.planner.llm_goal_interpreter import _FALLBACK_SYSTEM_PROMPT, build_system_prompt
    assert build_system_prompt(None) == _FALLBACK_SYSTEM_PROMPT


def test_system_prompt_with_bundle_includes_real_safety_numbers(tmp_path):
    from machine_brain.knowledge.okf_loader import OKFBundle
    from machine_brain.planner.llm_goal_interpreter import build_system_prompt
    from machine_brain.safety.governor import SafetyEnvelope

    (tmp_path / "safety").mkdir()
    (tmp_path / "safety" / "envelope.md").write_text(
        "---\ntype: Safety Envelope\ntitle: Robot Safety Envelope\n---\n\nmax velocity 1.5 m/s\n"
    )
    (tmp_path / "goals").mkdir()
    (tmp_path / "goals" / "observe_entity.md").write_text(
        "---\ntype: Goal Kind\ntitle: observe_entity\n---\n\nschema details here\n"
    )
    bundle = OKFBundle(tmp_path)
    prompt = build_system_prompt(bundle)
    assert "max velocity 1.5 m/s" in prompt  # the actual generated content, not a hardcoded duplicate
    assert "schema details here" in prompt
    envelope = SafetyEnvelope()
    assert str(envelope.max_velocity) in prompt
