#!/usr/bin/env python3
"""Narrative demo of the "basic AGI" loop: perceive -> imagine -> plan ->
act -> learn, bounded throughout by SutraFlow + the safety governor.

Honest framing: "basic AGI" here means a bounded, model-based agent with
standing goals and imagination-based planning — not a claim of general
intelligence. What makes it worth the name at all: it (1) learns a forward
model of its world from its own experience, (2) uses that model to imagine
consequences of actions it hasn't taken yet, (3) pursues a persistent goal
rather than reacting frame-by-frame, (4) tracks its own competence per
skill and factors that into planning, and (5) never lets any of the above
override a hard-coded safety rule.

Usage: python agi_demo.py --cycles 400 --data-dir ./agi_runtime
       python agi_demo.py --cycles 400 --physics mujoco   # real 3D rigid-body physics
"""

from __future__ import annotations

import argparse
import shutil

from machine_brain.orchestrator.cognitive_loop import CognitiveBrain
from machine_brain.simulate.sensors import SensorSimulator, SimConfig


def narrate(i: int, brain: CognitiveBrain, result) -> None:
    tag = "IMAGINED" if result.used_imagination else "reactive"
    verdict = result.final_verdict.value if result.final_verdict else "?"
    outcome = "ok" if (result.outcome and result.outcome.succeeded) else ("FAIL" if result.outcome else "-")
    print(f"[cycle {i:4d}] ({tag}, dyn_steps={result.dynamics_train_steps:3d}) "
          f"proposed={result.proposal.skill_id:<18s} verdict={verdict:<7s} outcome={outcome:<4s} "
          f"conf={result.proposal.predicted_confidence:.2f}")
    if i % 50 == 0:
        print(f"           justification: {result.proposal.justification}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=400)
    parser.add_argument("--data-dir", type=str, default="./agi_runtime")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--quiet", action="store_true", help="only print summary, not per-cycle narration")
    parser.add_argument("--physics", choices=["toy", "mujoco"], default="toy",
                          help="toy = hand-rolled 2D holonomic kinematics (default, no extra deps); "
                               "mujoco = real 3D rigid-body physics (requires the 'physics' extra)")
    parser.add_argument("--instruction", type=str, default=None,
                          help="set the goal from natural language via the local LLM instead of the "
                               "hardcoded default (requires the 'llm' extra and models/qwen2.5-0.5b-instruct-q4_k_m.gguf)")
    args = parser.parse_args()

    if args.fresh:
        shutil.rmtree(args.data_dir, ignore_errors=True)

    world = None
    if args.physics == "mujoco":
        from machine_brain.simulate.mujoco_world import MuJoCoWorld
        world = MuJoCoWorld(seed=123)

    llm_interpreter = None
    if args.instruction:
        from pathlib import Path
        from machine_brain.planner.llm_goal_interpreter import LLMGoalInterpreter
        model_path = Path(__file__).parent / "models" / "qwen2.5-0.5b-instruct-q4_k_m.gguf"
        print(f"Loading local LLM ({model_path.name})...")
        llm_interpreter = LLMGoalInterpreter(model_path=str(model_path))

    brain = CognitiveBrain(data_dir=args.data_dir, llm_interpreter=llm_interpreter)
    sim = SensorSimulator(SimConfig(seed=123), world=world)

    print("=" * 78)
    print(f"machine_brain — basic AGI demo: goal-directed, model-based, guarded ({args.physics} physics)")
    print("=" * 78)

    if args.instruction:
        # Prime working memory with a few sensor frames so the interpreter
        # has real known_entity_ids to validate against, not an empty set.
        for _ in range(30):
            brain.perceive(sim.next_frame())
        print(f'Interpreting instruction: "{args.instruction}"')
        goal, reason = brain.set_goal_from_instruction(args.instruction)
        if goal is None:
            print(f"LLM interpretation REJECTED by validation: {reason}")
            print("No goal set by the LLM — falling back to the hardcoded default.\n")
        else:
            print(f"LLM produced a validated goal: {goal.kind}({goal.target})\n")

    if brain.working_memory.active_goal() is None:
        print("Setting goal: observe_entity(obstacle-2, desired_distance=1.5m)")
        print("This means: the agent will try to hold ~1.5m distance from obstacle-2,")
        print("using its own learned forward model to imagine which action gets it")
        print("there without violating the safety governor's hard limits. 1.5m is well")
        print("clear of the 0.5m obstacle safety margin, so pursuing this goal never")
        print("has to fight the safety governor — a fair test of whether imagination-")
        print("based planning actually chases the goal instead of defaulting to habit.\n")
        goal = brain.set_goal("observe_entity", {"entity_id": "obstacle-2", "desired_distance": 1.5})
    else:
        goal = brain.working_memory.active_goal()

    reactive_cycles = 0
    imagined_cycles = 0
    refused = 0
    held = 0
    goal_completed = False

    for i in range(args.cycles):
        frame = sim.next_frame()
        brain.perceive(frame)
        if i % 3 != 0:
            continue
        result = brain.cycle()
        if result.outcome is not None:
            # Close the loop: the executed action actually changes the
            # simulated world the next perceive() call will sense. See
            # SensorSimulator.apply_action's docstring for why this matters.
            sim.apply_action(result.proposal.skill_id, result.proposal.args.get("velocity", 0.0),
                               target_entity_id=goal.target.get("entity_id"))
        if not args.quiet:
            narrate(i, brain, result)

        if result.used_imagination:
            imagined_cycles += 1
        else:
            reactive_cycles += 1
        if result.final_verdict and result.final_verdict.value == "refuse":
            refused += 1
        if result.final_verdict and result.final_verdict.value == "hold":
            held += 1
        if brain.working_memory.active_goal() is None and not goal_completed:
            goal_completed = True
            target = goal.target.get("entity_id")
            desired = goal.target.get("desired_distance")
            print(f"\n*** goal reached at cycle {i} — active_goal cleared, agent held ~{desired}m from {target} ***\n")

    ok, _ = brain.audit_ledger.verify_chain()

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"cycles run:                 {reactive_cycles + imagined_cycles}")
    print(f"  reactive-planner cycles:  {reactive_cycles}  (before dynamics model was trusted)")
    print(f"  imagination-based cycles: {imagined_cycles}  (after {brain.imagination_planner.config.min_train_steps} real transitions learned)")
    print(f"guard refusals:             {refused}")
    print(f"guard holds:                {held}")
    print(f"goal reached:               {goal_completed}")
    print(f"episodes recorded:          {brain.episodic_store.count()}")
    print(f"graph edges:                {brain.graph_store.edge_count()}")
    print(f"audit ledger entries:       {brain.audit_ledger.count()} | chain valid: {ok}")
    for skill_id in ("patrol", "avoid_obstacle", "yield_to_human", "hold_position", "approach_target", "investigate_anomaly"):
        rate = brain.skill_registry.success_rate(skill_id, 1)
        print(f"  skill competence[{skill_id:<18s}]: {rate}")

    brain.close()


if __name__ == "__main__":
    main()
