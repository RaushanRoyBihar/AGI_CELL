"""Ablation harness: no learning / reviewed learning; no vector memory /
vector candidate memory; no resonance / resonance tie-break only; static
world model / JEPA predictive world model. Negative results are written to
machine_brain/reports/ABLATION_REPORT.md rather than discarded — a
"learning made things worse" or "JEPA lost to baseline" result is exactly
as reportable as a positive one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from machine_brain.acoustic.resonance import ResonanceFingerprint, break_tie, fingerprint
from machine_brain.associative.index import LocalVectorIndex
from machine_brain.contracts import EdgeType, GraphEdge, ObservedOutcome
from machine_brain.episodic.store import EpisodicStore
from machine_brain.graph.store import SQLiteGraphStore
from machine_brain.learning.reviewed_learning import ReviewedLearning
from machine_brain.procedural.skills import SkillDefinition, SkillRegistry
from machine_brain.world_model.baseline import LastValueBaseline, prediction_error
from machine_brain.world_model.jepa import JepaConfig, JepaWorldEngine

REPORT_PATH = Path(__file__).resolve().parents[2] / "reports" / "ABLATION_REPORT.md"


def _noop_handler(args):
    return {"succeeded": True}


def ablation_learning(tmp_path) -> dict:
    graph = SQLiteGraphStore(str(tmp_path / "g.sqlite"))
    skills = SkillRegistry(str(tmp_path / "s.sqlite"))
    episodic = EpisodicStore(str(tmp_path / "e.sqlite"))
    skills.register(SkillDefinition("patrol", 1, {}, ("actuate.motion",), _noop_handler))

    edge = GraphEdge.make("a", EdgeType.CAUSED_BY, "b", evidence_ids=["ev1"], weight=0.5)
    graph.add_edge(edge)

    learning = ReviewedLearning(graph, skills, episodic)
    for _ in range(5):
        outcome = ObservedOutcome.make("p1", succeeded=True, detail={})
        learning.process("patrol", 1, predicted_confidence=0.9, outcome=outcome, related_edge_ids=[edge.edge_id])

    with_learning_weight = graph.edges_from("a")[0].weight
    with_learning_success_rate = skills.success_rate("patrol", 1)

    # "no learning" ablation: same starting state, but reviewed_learning.process is never called
    graph2 = SQLiteGraphStore(str(tmp_path / "g2.sqlite"))
    skills2 = SkillRegistry(str(tmp_path / "s2.sqlite"))
    skills2.register(SkillDefinition("patrol", 1, {}, ("actuate.motion",), _noop_handler))
    edge2 = GraphEdge.make("a", EdgeType.CAUSED_BY, "b", evidence_ids=["ev1"], weight=0.5)
    graph2.add_edge(edge2)
    no_learning_weight = graph2.edges_from("a")[0].weight
    no_learning_success_rate = skills2.success_rate("patrol", 1)

    return {
        "with_learning_edge_weight": with_learning_weight,
        "no_learning_edge_weight": no_learning_weight,
        "with_learning_success_rate": with_learning_success_rate,
        "no_learning_success_rate": no_learning_success_rate,
        "learning_strengthened_edge": with_learning_weight > no_learning_weight,
    }


def ablation_vector_memory() -> dict:
    dim = 5
    index = LocalVectorIndex(dim=dim)
    rng = np.random.default_rng(0)
    items = [(f"ep-{i}", rng.normal(size=dim)) for i in range(50)]
    for cid, vec in items:
        index.upsert(cid, vec)

    query = items[10][1] + rng.normal(0, 0.01, dim)
    with_vector_candidates = [c.canonical_id for c in index.search(query, top_k=5)]
    no_vector_candidates: list[str] = []  # "no vector memory" ablation: candidate generation simply disabled

    return {
        "with_vector_top1_is_nearest_ground_truth": with_vector_candidates[0] == "ep-10",
        "with_vector_candidate_count": len(with_vector_candidates),
        "no_vector_candidate_count": len(no_vector_candidates),
    }


def ablation_resonance() -> dict:
    rng = np.random.default_rng(0)
    signal_a = np.sin(np.linspace(0, 10, 64)) + rng.normal(0, 0.01, 64)
    signal_b = np.sin(np.linspace(0, 10, 64) + 3.0) + rng.normal(0, 0.01, 64)  # phase-shifted, less similar
    query_fp = fingerprint(signal_a)

    candidate_a = ResonanceFingerprint(source_id="a", vector=fingerprint(signal_a).vector)
    candidate_b = ResonanceFingerprint(source_id="b", vector=fingerprint(signal_b).vector)

    resonance_tie_break_winner = break_tie(query_fp, [candidate_a, candidate_b])
    no_resonance_winner = None  # "no resonance" ablation: tie-break step skipped entirely, no decision made

    return {
        "resonance_tie_break_winner": resonance_tie_break_winner,
        "resonance_picked_closer_match": resonance_tie_break_winner == "a",
        "no_resonance_winner": no_resonance_winner,
    }


def ablation_world_model() -> dict:
    dim = 3
    jepa = JepaWorldEngine(JepaConfig(state_dim=dim, latent_dim=6, seed=0))
    baseline = LastValueBaseline()
    rng = np.random.default_rng(0)

    def make_state(t):
        return np.array([np.sin(t * 0.3), np.cos(t * 0.3), t * 0.01]) + rng.normal(0, 0.01, dim)

    states = [make_state(t) for t in range(200)]
    for t in range(150):
        jepa.train_step(states[t], states[t + 1])

    jepa_errors, baseline_errors = [], []
    for t in range(150, 199):
        jepa_errors.append(jepa.surprise(states[t], states[t + 1]))
        baseline_errors.append(prediction_error(baseline.predict(states[t]), states[t + 1]))

    return {
        "jepa_mean_error": float(np.mean(jepa_errors)),
        "static_baseline_mean_error": float(np.mean(baseline_errors)),
        "jepa_beat_baseline": float(np.mean(jepa_errors)) < float(np.mean(baseline_errors)),
    }


def test_ablation_suite_runs_and_preserves_all_results(tmp_path):
    results = {
        "learning": ablation_learning(tmp_path),
        "vector_memory": ablation_vector_memory(),
        "resonance": ablation_resonance(),
        "world_model": ablation_world_model(),
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Ablation Report", "", "Auto-generated by tests/ablations/test_ablations.py. "
             "Negative results are kept, not filtered out.", ""]
    for name, data in results.items():
        lines.append(f"## {name}")
        for k, v in data.items():
            lines.append(f"- `{k}`: {v}")
        lines.append("")
    REPORT_PATH.write_text("\n".join(lines))

    # Structural assertions only — the *values* (e.g. whether JEPA beat the
    # baseline) are reported, not asserted as must-pass, since a negative
    # result here is informative, not a bug.
    assert "learning_strengthened_edge" in results["learning"]
    assert "resonance_picked_closer_match" in results["resonance"]
    assert "jepa_beat_baseline" in results["world_model"]
    assert REPORT_PATH.exists()
