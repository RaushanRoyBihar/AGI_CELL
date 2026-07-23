#!/usr/bin/env python3
"""Standalone benchmark script (not part of default `pytest` run — it's
slow by design). Run directly:

    python tests/benchmarks/bench_throughput.py --frames 100000
    python tests/benchmarks/bench_throughput.py --frames 1000000

Reports throughput, p50/p95/p99 per-cycle latency, peak RSS, disk use per
frame, and audit-chain validity at the end. This is the harness required
by the spec's "Benchmark 100,000 frames first and then 1,000,000
streaming frames" — run manually since 1M frames takes minutes, which is
too slow for a default test-suite run.
"""

from __future__ import annotations

import argparse
import os
import resource
import shutil
import statistics
import time

from machine_brain.orchestrator.cognitive_loop import CognitiveBrain
from machine_brain.simulate.sensors import SensorSimulator, SimConfig


def dir_size_bytes(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total


def run(frames: int, data_dir: str, cycle_every: int = 5) -> dict:
    shutil.rmtree(data_dir, ignore_errors=True)
    brain = CognitiveBrain(data_dir=data_dir)
    sim = SensorSimulator(SimConfig())

    cycle_latencies_ms = []
    t0 = time.perf_counter()
    for i in range(frames):
        frame = sim.next_frame()
        brain.perceive(frame)
        if i % cycle_every == 0:
            c0 = time.perf_counter()
            brain.cycle()
            cycle_latencies_ms.append((time.perf_counter() - c0) * 1000)
    elapsed = time.perf_counter() - t0

    ok, broken = brain.audit_ledger.verify_chain()
    # ru_maxrss is KB on Linux, bytes on macOS/BSD.
    divisor = 1024.0 if os.uname().sysname == "Linux" else 1024.0 * 1024.0
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / divisor
    disk_bytes = dir_size_bytes(data_dir)
    brain.close()

    cycle_latencies_ms.sort()
    def pct(p):
        if not cycle_latencies_ms:
            return None
        idx = min(len(cycle_latencies_ms) - 1, int(p * len(cycle_latencies_ms)))
        return cycle_latencies_ms[idx]

    return {
        "frames": frames,
        "elapsed_seconds": elapsed,
        "throughput_fps": frames / elapsed,
        "p50_cycle_ms": pct(0.50),
        "p95_cycle_ms": pct(0.95),
        "p99_cycle_ms": pct(0.99),
        "peak_rss_mb": peak_rss_mb,
        "disk_bytes_per_frame": disk_bytes / frames,
        "audit_chain_valid": ok,
        "episodes_recorded": brain.episodic_store.count(),
        "graph_edges": brain.graph_store.edge_count(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=100_000)
    parser.add_argument("--data-dir", type=str, default="./bench_data")
    args = parser.parse_args()

    report = run(args.frames, args.data_dir)
    for k, v in report.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
