"""Planner: turns attended world state + retrieved memory + a world-model
surprise score into a single ActionProposal. Deliberately simple
(reactive, rule-based utility scoring) — the planner is allowed to be
wrong; SutraFlow and the safety governor are what keep the system correct.
Sophistication belongs in retrieval/prediction quality, not in trusting the
planner more.
"""

from __future__ import annotations

from dataclasses import dataclass

from machine_brain.contracts import ActionProposal, WorldEntity


@dataclass
class PlannerContext:
    focused_entities: list[tuple[WorldEntity, float]]  # (entity, salience)
    surprise_score: float
    retrieved_episode_ids: list[str]
    graph_hint_ids: list[str]


class Planner:
    def __init__(self, surprise_threshold: float = 1.5) -> None:
        self.surprise_threshold = surprise_threshold

    def propose(self, ctx: PlannerContext) -> ActionProposal:
        humans = [(e, s) for e, s in ctx.focused_entities if e.kind == "human"]
        obstacles = [(e, s) for e, s in ctx.focused_entities if e.kind == "obstacle"]

        if humans:
            entity, sal = max(humans, key=lambda pair: pair[1])
            distance = entity.attributes.get("distance", 5.0)
            if distance < 1.0:
                return ActionProposal.make(
                    skill_id="yield_to_human",
                    args={"velocity": 0.0, "zone": entity.attributes.get("zone")},
                    justification=f"human {entity.entity_id} within {distance}m",
                    predicted_confidence=min(0.95, entity.confidence),
                    source_episode_ids=ctx.retrieved_episode_ids,
                )

        if obstacles:
            entity, sal = max(obstacles, key=lambda pair: pair[1])
            return ActionProposal.make(
                skill_id="avoid_obstacle",
                args={"velocity": 0.5, "zone": entity.attributes.get("zone")},
                justification=f"obstacle {entity.entity_id} detected, salience {sal:.2f}",
                predicted_confidence=min(0.9, entity.confidence),
                source_episode_ids=ctx.retrieved_episode_ids,
                source_frame_ids=(),
            )

        if ctx.surprise_score > self.surprise_threshold:
            return ActionProposal.make(
                skill_id="investigate_anomaly",
                args={"velocity": 0.2},
                justification=f"world-model surprise {ctx.surprise_score:.3f} exceeds threshold",
                predicted_confidence=0.55,
                source_episode_ids=ctx.retrieved_episode_ids,
            )

        return ActionProposal.make(
            skill_id="patrol",
            args={"velocity": 1.0},
            justification="no salient entities or anomalies this cycle",
            predicted_confidence=0.8,
            source_episode_ids=ctx.retrieved_episode_ids,
        )
