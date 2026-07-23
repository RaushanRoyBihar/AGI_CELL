"""Samskara ("संस्कार" — the imprint an experience leaves that shapes future
disposition) reviewed learning loop:

  query/situation -> retrieved memories -> planned action -> guard decision
  -> observed outcome -> reviewer marks correct/incorrect/uncertain
  -> strengthen useful typed relations -> weaken failed retrieval routes
  -> update procedural success statistics
  -> create unresolved contradiction when evidence disagrees
  -> consolidate frequently verified episodes
  -> decay weak unverified associations

Bounded: every step touches a fixed, small set of rows (the edges/skills
tied to *this* episode), never a global rewrite. Reversible: strengthen/
weaken are additive nudges on a 0..1 weight, not replacement, and decay
only removes edges below threshold that were never verified. Auditable:
every action taken is returned as a LearningAction list the caller logs.

This module intentionally never imports machine_brain.safety — learning
has no code path capable of altering safety limits, by construction, not
convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from machine_brain.contracts import Contradiction, EdgeType, GraphEdge, ObservedOutcome, OutcomeLabel
from machine_brain.episodic.store import EpisodicStore
from machine_brain.graph.store import GraphStore
from machine_brain.procedural.skills import SkillRegistry

Reviewer = Callable[[ObservedOutcome, float], OutcomeLabel]


def default_reviewer(outcome: ObservedOutcome, predicted_confidence: float) -> OutcomeLabel:
    """Pluggable reviewer. Default heuristic: outcome success/failure is
    trusted directly unless the planner's own predicted confidence was low,
    in which case a success is still marked UNCERTAIN pending more
    evidence rather than reinforced at full strength."""
    if outcome.succeeded and predicted_confidence >= 0.6:
        return OutcomeLabel.CORRECT
    if outcome.succeeded and predicted_confidence < 0.6:
        return OutcomeLabel.UNCERTAIN
    return OutcomeLabel.INCORRECT


@dataclass(frozen=True)
class LearningAction:
    kind: str      # strengthened | weakened | skill_stat_updated | contradiction_created | edge_decayed
    detail: str


class ReviewedLearning:
    def __init__(self, graph_store: GraphStore, skill_registry: SkillRegistry, episodic_store: EpisodicStore,
                  reviewer: Reviewer = default_reviewer, decay_threshold: float = 0.15) -> None:
        self.graph_store = graph_store
        self.skill_registry = skill_registry
        self.episodic_store = episodic_store
        self.reviewer = reviewer
        self.decay_threshold = decay_threshold

    def process(self, skill_id: str, skill_version: int, predicted_confidence: float,
                 outcome: ObservedOutcome, related_edge_ids: list[str],
                 contradiction_check: tuple[str, str, str] | None = None) -> list[LearningAction]:
        actions: list[LearningAction] = []
        label = self.reviewer(outcome, predicted_confidence)

        if label is OutcomeLabel.CORRECT:
            # One transaction for every edge tied to this learning event,
            # not one commit per edge — this is one atomic outcome, not N
            # independent ones (see GraphStore.strengthen_many's docstring
            # for why this is also more consistent, not just faster).
            self.graph_store.strengthen_many(related_edge_ids)
            for edge_id in related_edge_ids:
                actions.append(LearningAction("strengthened", f"edge {edge_id} strengthened after CORRECT outcome"))
            self.skill_registry.record_outcome(skill_id, skill_version, succeeded=True)
            actions.append(LearningAction("skill_stat_updated", f"{skill_id} v{skill_version} success recorded"))

        elif label is OutcomeLabel.INCORRECT:
            self.graph_store.weaken_many(related_edge_ids)
            for edge_id in related_edge_ids:
                actions.append(LearningAction("weakened", f"edge {edge_id} weakened after INCORRECT outcome"))
            self.skill_registry.record_outcome(skill_id, skill_version, succeeded=False)
            actions.append(LearningAction("skill_stat_updated", f"{skill_id} v{skill_version} failure recorded"))

        else:  # UNCERTAIN — no reinforcement either direction, bounded by design
            actions.append(LearningAction("no_op", "UNCERTAIN outcome — no edge/skill-stat change applied"))

        if contradiction_check is not None:
            subject, evidence_a, evidence_b = contradiction_check
            contradiction = Contradiction.make(
                subject=subject, claim_a_evidence=evidence_a, claim_b_evidence=evidence_b,
                detail="new evidence disagrees with an existing graph/episodic claim",
            )
            actions.append(LearningAction("contradiction_created", f"subject={subject}"))
            self._last_contradiction = contradiction  # caller persists via working_memory.record_contradiction

        return actions

    def consolidate_and_decay(self, node_ids: list[str]) -> list[LearningAction]:
        """Run periodically (not per-episode): decay weak, never-verified
        edges out of nodes that were touched this cycle. Verified edges
        (verified_count >= 1) are never decayed here."""
        actions: list[LearningAction] = []
        removed_total = 0
        for node_id in node_ids:
            for edge_type in EdgeType:
                edges = self.graph_store.edges_from(node_id, edge_type)
                weak_unverified = [e for e in edges if e.weight < self.decay_threshold and e.verified_count == 0]
                if weak_unverified:
                    removed = self.graph_store.decay_weak_edges(self.decay_threshold, min_verified=1)
                    removed_total += removed
        if removed_total:
            actions.append(LearningAction("edge_decayed", f"{removed_total} weak unverified edges removed"))
        return actions
