#!/usr/bin/env python3
"""Runnable end-to-end demo of the full cognitive data flow, Phases 1-9,
on synthetic sensor data. No hardware, no external services required —
this is the RAM+SQLite+MCAP floor exercising every layer.

Usage: python demo_run.py --frames 2000 --data-dir ./runtime_data
"""

from __future__ import annotations

import argparse
import shutil
import time
from collections import Counter

from machine_brain.orchestrator.cognitive_loop import CognitiveBrain
from machine_brain.simulate.sensors import SensorSimulator, SimConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=2000)
    parser.add_argument("--data-dir", type=str, default="./runtime_data")
    parser.add_argument("--fresh", action="store_true", help="wipe data-dir before running")
    args = parser.parse_args()

    if args.fresh:
        shutil.rmtree(args.data_dir, ignore_errors=True)

    brain = CognitiveBrain(data_dir=args.data_dir)
    sim = SensorSimulator(SimConfig())

    verdict_counts = Counter()
    outcomes = Counter()
    t0 = time.perf_counter()

    for i in range(args.frames):
        frame = sim.next_frame()
        brain.perceive(frame)
        if i % 5 == 0:  # cycle less often than raw perception, like a real control loop
            result = brain.cycle()
            verdict_counts[result.final_verdict.value] += 1
            if result.outcome is not None:
                outcomes["succeeded" if result.outcome.succeeded else "failed"] += 1

    elapsed = time.perf_counter() - t0
    ok, broken_seq = brain.audit_ledger.verify_chain()

    print(f"Processed {args.frames} frames in {elapsed:.3f}s ({args.frames/elapsed:.0f} frames/sec)")
    print(f"Ring buffer: {brain.ring_buffer.dropped_duplicate} duplicates dropped")
    print(f"Episodes recorded: {brain.episodic_store.count()} (dup-rejected: {brain.episodic_store.duplicate_rejections})")
    print(f"Audit ledger entries: {brain.audit_ledger.count()} | chain valid: {ok}")
    print(f"Graph edges: {brain.graph_store.edge_count()}")
    print(f"Guard verdicts: {dict(verdict_counts)}")
    print(f"Execution outcomes: {dict(outcomes)}")
    print(f"JEPA train steps: {brain.jepa.train_steps}")

    brain.close()


if __name__ == "__main__":
    main()
