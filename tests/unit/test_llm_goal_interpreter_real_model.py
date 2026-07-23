"""Integration test against the real downloaded model — slow (~10s model
load), skipped automatically if the 469MB GGUF file or llama-cpp-python
isn't present, since neither is a hard dependency of this project."""

from pathlib import Path

import pytest

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "qwen2.5-0.5b-instruct-q4_k_m.gguf"

pytest.importorskip("llama_cpp")
if not MODEL_PATH.exists():
    pytest.skip(f"model not downloaded at {MODEL_PATH}", allow_module_level=True)

from machine_brain.planner.llm_goal_interpreter import LLMGoalInterpreter  # noqa: E402


@pytest.fixture(scope="module")
def interpreter() -> LLMGoalInterpreter:
    return LLMGoalInterpreter(model_path=str(MODEL_PATH))


def test_real_model_parses_a_clear_instruction(interpreter):
    result = interpreter.interpret("Watch human-1 but stay respectfully back, like 2 meters.")
    assert result.goal is not None
    assert result.goal.target["entity_id"] == "human-1"


def test_real_model_output_is_validated_regardless_of_content(interpreter):
    """Not asserting the model behaves well here — asserting that whatever
    it says, the interpreter never returns an invalid Goal. On an
    off-topic instruction during development, this exact model ignored
    its 'respond null' instruction and hallucinated a fake entity; this
    test's job is to confirm validation still catches that today, not to
    demand the model itself improve."""
    result = interpreter.interpret("What's the weather like today?")
    if result.goal is not None:
        assert result.goal.target["entity_id"] in {f"human-{i}" for i in range(5)} | {f"obstacle-{i}" for i in range(7)}
