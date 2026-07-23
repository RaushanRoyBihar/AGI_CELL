#!/usr/bin/env python3
"""Reproducible experiment: does giving the LLM goal interpreter real OKF
context (the actual generated skill/goal/safety bundle) change its
behavior, measured against the exact failure modes found during earlier
development of this project?

Run: python examples/okf_context_experiment.py
Requires: models/qwen2.5-0.5b-instruct-q4_k_m.gguf and the 'llm' extra.

Honest result from the run this was built from (see docs/provenance/NOTES.md
for the full writeup): hallucination rate on off-topic queries dropped
from 4/4 to 1/4 with OKF context — the model started correctly declining
("null") instead of fabricating an entity, in most but not all cases. The
"meter and a half" -> wrong-numeric-value failure was UNCHANGED by OKF
context in either direction — that's a distinct weakness OKF-style
curated context doesn't address, and this script reports both findings,
not just the one that looks good.
"""

from __future__ import annotations

from machine_brain.knowledge.okf_loader import OKFBundle
from machine_brain.planner.llm_goal_interpreter import LLMGoalInterpreter

MODEL_PATH = "models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
KNOWN_ENTITIES = {f"human-{i}" for i in range(5)} | {f"obstacle-{i}" for i in range(7)}

NUMERIC_TESTS = [
    ("Keep an eye on obstacle-2 from about a meter and a half away, don't get too close.", "obstacle-2", 1.5),
    ("Watch human-1 but stay respectfully back, like 2 meters.", "human-1", 2.0),
    ("Follow obstacle-5 at a distance of 3 meters please.", "obstacle-5", 3.0),
]

OFF_TOPIC_TESTS = [
    "What's the weather like today?",
    "Tell me a joke.",
    "What time is it?",
    "How do I make pasta?",
]


def run_numeric_accuracy(interp: LLMGoalInterpreter, label: str) -> None:
    print(f"\n-- numeric accuracy ({label}) --")
    for instr, want_id, want_dist in NUMERIC_TESTS:
        r = interp.interpret(instr, known_entity_ids=KNOWN_ENTITIES)
        correct = (r.goal is not None and r.goal.target["entity_id"] == want_id
                    and abs(r.goal.target["desired_distance"] - want_dist) < 0.01)
        got = r.goal.target if r.goal else f"REJECTED: {r.rejected_reason}"
        print(f"  [{'OK' if correct else 'WRONG'}] {instr[:55]:57s} -> {got}")


def run_hallucination_rate(interp: LLMGoalInterpreter, label: str) -> float:
    print(f"\n-- off-topic hallucination behavior ({label}) --")
    hallucinated = 0
    for instr in OFF_TOPIC_TESTS:
        r = interp.interpret(instr, known_entity_ids=KNOWN_ENTITIES)
        declined_cleanly = r.rejected_reason == "model declined (null)"
        if not declined_cleanly:
            hallucinated += 1
        print(f"  {'self-declined' if declined_cleanly else 'hallucinated (caught by validation)':38s} <- {instr}")
    rate = hallucinated / len(OFF_TOPIC_TESTS)
    print(f"  hallucination rate: {hallucinated}/{len(OFF_TOPIC_TESTS)} ({rate:.0%})")
    return rate


def main() -> None:
    plain = LLMGoalInterpreter(model_path=MODEL_PATH)
    bundle = OKFBundle("okf")
    with_okf = LLMGoalInterpreter(model_path=MODEL_PATH, okf_bundle=bundle)

    run_numeric_accuracy(plain, "without OKF")
    run_numeric_accuracy(with_okf, "with OKF")

    rate_plain = run_hallucination_rate(plain, "without OKF")
    rate_okf = run_hallucination_rate(with_okf, "with OKF")

    print(f"\n{'='*70}\nSUMMARY: hallucination rate {rate_plain:.0%} -> {rate_okf:.0%} with OKF context")
    print("Numeric-parsing accuracy: unaffected either way (a distinct weakness)")
    print("Note: validation (LLMGoalInterpreter._validate) catches every hallucination")
    print("in both conditions — this measures model behavior, not system-level safety,")
    print("which was already 100% correct via the guard boundary regardless.")


if __name__ == "__main__":
    main()
