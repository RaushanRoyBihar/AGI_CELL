"""Optional adapter: a small local LLM translates a natural-language
instruction into a `Goal`, called once per instruction — never inside the
tight reactive loop. See `README.md`'s "small LLM" section for why: on
this machine, the guard/planning loop runs at 3-5ms p50; a 0.5B-parameter
CPU LLM call takes hundreds of milliseconds. Those two facts are why the
LLM sits above the loop, interpreting intent occasionally, while the
existing dynamics-model-based imagination stays in the per-cycle path.

Validation here is not a formality — it's the actual safety net. Tested
against the real downloaded model (Qwen2.5-0.5B-Instruct, Q4_K_M, Apache
2.0), it reliably parses clear instructions ("watch human-1 from 2
meters" -> correct JSON) but on off-topic input it did NOT follow an
explicit "respond null" instruction — it hallucinated a fake entity_id
("weather", "building") wrapped in otherwise well-formed JSON instead.
That's the real failure mode small models produce: not silence, confident
fabrication. `_validate` is what actually catches it, by checking the
entity_id against a real pattern (and optionally a known-entities set)
rather than trusting the model followed the "say null" instruction.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable

from machine_brain.contracts import Goal
from machine_brain.knowledge.okf_loader import OKFBundle

_ENTITY_ID_PATTERN = re.compile(r"^(human|obstacle)-\d+$")
_MIN_DESIRED_DISTANCE = 0.1
_MAX_DESIRED_DISTANCE = 20.0

_FALLBACK_SYSTEM_PROMPT = """You are a goal interpreter for a small mobile robot. Convert the instruction into JSON with this exact schema:
{"kind": "observe_entity", "target": {"entity_id": "<id>", "desired_distance": <meters as float>}}
Known entity ids look like "human-0".."human-4" or "obstacle-0".."obstacle-6".
If the instruction doesn't clearly map to observing one specific known entity, respond with exactly: null
Respond with ONLY the JSON object or the word null, nothing else."""

_RESPONSE_INSTRUCTIONS = """
Convert the instruction into JSON with this exact schema:
{"kind": "observe_entity", "target": {"entity_id": "<id>", "desired_distance": <meters as float>}}
If the instruction doesn't clearly map to observing one specific known entity, respond with exactly: null
Respond with ONLY the JSON object or the word null, nothing else."""


def build_system_prompt(okf_bundle: OKFBundle | None) -> str:
    """Without a bundle: the original hardcoded prompt. With one: the
    model gets the *actual* curated knowledge — the real goal schema and
    the real safety envelope, generated from the live SafetyEnvelope
    dataclass (see scripts/generate_okf_bundle.py) — instead of a fixed
    string only this one function knew about."""
    if okf_bundle is None:
        return _FALLBACK_SYSTEM_PROMPT

    parts = ["You are a goal interpreter for a small mobile robot. "
              "Use the following curated knowledge about this specific robot to answer accurately."]

    goal_concept = okf_bundle.get("goals/observe_entity")
    if goal_concept is not None:
        parts.append(f"## Goal schema: {goal_concept.title}\n{goal_concept.body}")

    safety_concept = okf_bundle.get("safety/envelope")
    if safety_concept is not None:
        parts.append(f"## Safety limits (informational — you are not enforcing these, just aware of them)\n{safety_concept.body}")

    parts.append(_RESPONSE_INSTRUCTIONS)
    return "\n\n".join(parts)


@dataclass
class InterpretationResult:
    goal: Goal | None
    raw_model_output: str
    rejected_reason: str | None = None


def _extract_json_text(raw: str) -> str:
    """Small models often wrap JSON in markdown code fences even when
    told not to — strip that before parsing rather than failing on it."""
    text = raw.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()
    return text


def _validate(parsed: dict, known_entity_ids: set[str] | None) -> tuple[Goal | None, str | None]:
    if parsed.get("kind") != "observe_entity":
        return None, f"unsupported goal kind {parsed.get('kind')!r}"
    target = parsed.get("target")
    if not isinstance(target, dict):
        return None, "target is not an object"

    entity_id = target.get("entity_id")
    if not isinstance(entity_id, str) or not _ENTITY_ID_PATTERN.match(entity_id):
        return None, f"entity_id {entity_id!r} does not match the known id pattern — likely a hallucinated entity"
    if known_entity_ids is not None and entity_id not in known_entity_ids:
        return None, f"entity_id {entity_id!r} matches the id pattern but isn't a currently known entity"

    desired = target.get("desired_distance")
    if not isinstance(desired, (int, float)) or isinstance(desired, bool):
        return None, f"desired_distance {desired!r} is not a number"
    if not (_MIN_DESIRED_DISTANCE <= desired <= _MAX_DESIRED_DISTANCE):
        return None, f"desired_distance {desired} is outside the sane bound [{_MIN_DESIRED_DISTANCE}, {_MAX_DESIRED_DISTANCE}]"

    return Goal.make("observe_entity", {"entity_id": entity_id, "desired_distance": float(desired)}), None


class LLMGoalInterpreter:
    def __init__(self, model_path: str | None = None,
                  llm_call: Callable[[str], str] | None = None, n_ctx: int = 1024,
                  okf_bundle: OKFBundle | None = None) -> None:
        """`llm_call` is injectable so tests don't need to load a real
        469MB model — pass a stub callable for fast, deterministic unit
        tests, or leave it None with a real `model_path` for the actual
        Qwen2.5-0.5B-Instruct GGUF. `okf_bundle`, if provided, replaces the
        hardcoded system prompt with one built from the real OKF-formatted
        knowledge bundle (see scripts/generate_okf_bundle.py) — the
        skill/goal/safety documentation the robot actually has, not a
        second, hand-maintained copy of the same facts."""
        self.system_prompt = build_system_prompt(okf_bundle)
        if llm_call is not None:
            self._llm_call = llm_call
        elif model_path is not None:
            self._llm_call = self._build_real_llm_call(model_path, n_ctx, self.system_prompt)
        else:
            raise ValueError("either model_path or llm_call must be provided")

    @staticmethod
    def _build_real_llm_call(model_path: str, n_ctx: int, system_prompt: str) -> Callable[[str], str]:
        try:
            from llama_cpp import Llama  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "LLMGoalInterpreter with a real model requires 'llama-cpp-python' "
                "(pip install machine_brain[llm]). This adapter is optional — goals can "
                "always be set directly via CognitiveBrain.set_goal() without it."
            ) from e
        llm = Llama(model_path=model_path, n_ctx=n_ctx, verbose=False)

        def call(instruction: str) -> str:
            # create_chat_completion's return type is a union of the
            # streaming and non-streaming response shapes; stream=False is
            # explicit here so the non-streaming (dict, indexable) shape is
            # the only one actually possible at runtime, and the isinstance
            # check makes that fact visible to the type checker instead of
            # asserting past it.
            out = llm.create_chat_completion(
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": instruction}],
                max_tokens=100, temperature=0.1, stream=False,
            )
            assert isinstance(out, dict), "stream=False must return the non-streaming response dict"
            content = out["choices"][0]["message"]["content"]
            return content if content is not None else ""
        return call

    def interpret(self, instruction: str, known_entity_ids: set[str] | None = None) -> InterpretationResult:
        raw = self._llm_call(instruction)
        text = _extract_json_text(raw)

        if text.strip().lower() == "null":
            return InterpretationResult(goal=None, raw_model_output=raw, rejected_reason="model declined (null)")

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return InterpretationResult(goal=None, raw_model_output=raw,
                                          rejected_reason="model output was not valid JSON — failing closed, not guessing")

        if not isinstance(parsed, dict):
            return InterpretationResult(goal=None, raw_model_output=raw, rejected_reason="parsed JSON was not an object")

        goal, rejected_reason = _validate(parsed, known_entity_ids)
        return InterpretationResult(goal=goal, raw_model_output=raw, rejected_reason=rejected_reason)
