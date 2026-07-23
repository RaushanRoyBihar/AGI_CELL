"""Fast smoke test that the benchmark harness itself works end to end at
small scale. The real 100k/1M-frame runs are invoked manually via
bench_throughput.py's __main__ — too slow for a default pytest run.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from bench_throughput import run  # noqa: E402


def test_benchmark_harness_runs_and_reports_sane_metrics(tmp_path):
    report = run(frames=300, data_dir=str(tmp_path / "bench_data"), cycle_every=5)
    assert report["frames"] == 300
    assert report["throughput_fps"] > 0
    assert report["audit_chain_valid"] is True
    assert report["p50_cycle_ms"] is not None
    assert report["disk_bytes_per_frame"] > 0
