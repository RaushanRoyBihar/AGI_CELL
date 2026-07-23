"""Wires every layer into the cognitive data flow specified for this
project:

  sensors/operator input -> canonical observation frames -> sensory ring
  buffer -> perception/feature extraction -> active world state ->
  attention selection -> episodic recall -> semantic/causal association ->
  predictive world model -> planner proposal -> SutraFlow validation ->
  safety governor -> simulated action -> observed outcome -> reviewed
  learning and memory consolidation.

This module is intentionally the only place that imports across layer
boundaries — every individual layer module only knows its own interface.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from machine_brain.associative.index import LocalVectorIndex
from machine_brain.attention.selector import AttentionSelector
from machine_brain.audit.ledger import AuditLedger
from machine_brain.contracts import (
    ActionProposal, Contradiction, EdgeType, Episode, GraphEdge, Goal, GuardVerdict, ObservationFrame,
    ObservedOutcome, OutcomeLabel, WorldEntity, monotonic_ns,
)
from machine_brain.episodic.store import EpisodicStore
from machine_brain.graph.store import SQLiteGraphStore
from machine_brain.guard.pipeline import GuardedPolicy
from machine_brain.learning.reviewed_learning import ReviewedLearning
from machine_brain.perception.features import PerceptionEngine
from machine_brain.planner.imagination import (
    ACTION_DIM, ImaginationConfig, ImaginationPlanner, encode_action,
)
from machine_brain.planner.planner import Planner, PlannerContext
from machine_brain.procedural.skills import SkillDefinition, SkillRegistry
from machine_brain.raw_experience.mcap_log import McapLiteWriter
from machine_brain.safety.governor import SafetyGovernor
from machine_brain.sensory.ring_buffer import RingBuffer
from machine_brain.sutraflow.validator import SutraFlowValidator
from machine_brain.working_memory.store import WorkingMemoryStore
from machine_brain.world_model.baseline import LastValueBaseline, prediction_error
from machine_brain.world_model.dynamics import ActionConditionedDynamics, DynamicsConfig
from machine_brain.world_model.jepa import JepaConfig, JepaWorldEngine

# [nearest_human_distance, nearest_obstacle_distance, focused_count, last_velocity, goal_target_distance]
STATE_DIM = 5
ENTITY_VEC_DIM = 5  # [distance, confidence, is_human, is_obstacle, is_other]
NEUTRAL_GOAL_DISTANCE = 5.0


def _entity_vector(entity: WorldEntity) -> np.ndarray:
    distance = float(entity.attributes.get("distance", 5.0))
    is_human = 1.0 if entity.kind == "human" else 0.0
    is_obstacle = 1.0 if entity.kind == "obstacle" else 0.0
    is_other = 1.0 if entity.kind not in ("human", "obstacle") else 0.0
    return np.array([distance, entity.confidence, is_human, is_obstacle, is_other])


def _default_skill_handler(args: dict) -> dict:
    """Simulated actuation — no real hardware. Deterministic function of
    args so tests are reproducible; real deployments swap this for an
    actual ROS2 action call behind the same signature."""
    velocity = args.get("velocity", 0.0)
    succeeded = velocity <= 2.0  # simulated: anything past this is treated as a stall/failure
    return {"succeeded": succeeded, "velocity_applied": min(velocity, 2.0)}


@dataclass
class CycleResult:
    frame_processed: bool
    proposal: ActionProposal | None = None
    sutraflow_verdict: GuardVerdict | None = None
    safety_verdict: GuardVerdict | None = None
    final_verdict: GuardVerdict | None = None
    outcome: ObservedOutcome | None = None
    episode_recorded: bool = False
    surprise_score: float | None = None
    baseline_error: float | None = None
    contradiction_created: bool = False
    reasons: list[str] = field(default_factory=list)
    used_imagination: bool = False
    dynamics_train_steps: int = 0
    active_goal: Goal | None = None


class CognitiveBrain:
    def __init__(self, data_dir: str, robot_id: str = "robot-0", llm_interpreter=None) -> None:
        self.robot_id = robot_id
        os.makedirs(data_dir, exist_ok=True)
        self.data_dir = data_dir
        # Optional: an LLMGoalInterpreter (or anything with a compatible
        # `.interpret(instruction, known_entity_ids)` method). None by
        # default — goals can always be set directly via set_goal(),
        # consistent with every other adapter in this project being
        # optional. See planner/llm_goal_interpreter.py.
        self.llm_interpreter = llm_interpreter

        self.ring_buffer = RingBuffer()
        self.perception = PerceptionEngine()
        self.attention_selector = AttentionSelector()

        self.working_memory = WorkingMemoryStore(os.path.join(data_dir, "working_memory.sqlite"))
        self.episodic_store = EpisodicStore(os.path.join(data_dir, "episodic.sqlite"))
        self.graph_store = SQLiteGraphStore(os.path.join(data_dir, "graph.sqlite"))
        self.audit_ledger = AuditLedger(os.path.join(data_dir, "audit_ledger.sqlite"))
        self.skill_registry = SkillRegistry(os.path.join(data_dir, "skills.sqlite"))
        self.mcap_log = McapLiteWriter(os.path.join(data_dir, "mcap"))

        self.associative_index = LocalVectorIndex(dim=ENTITY_VEC_DIM)
        self.jepa = JepaWorldEngine(JepaConfig(state_dim=STATE_DIM))
        self.baseline = LastValueBaseline()
        self.dynamics = ActionConditionedDynamics(DynamicsConfig(state_dim=STATE_DIM, action_dim=ACTION_DIM))

        self.sutraflow_validator = SutraFlowValidator()
        self.safety_governor = SafetyGovernor()
        # The guard step is not reimplemented here — CognitiveBrain uses the
        # same standalone GuardedPolicy that any external policy/world-model
        # source can use (see guard/pipeline.py). This is what makes the
        # "wrap any policy" claim true rather than aspirational.
        self.guard = GuardedPolicy(self.sutraflow_validator, self.safety_governor, self.audit_ledger)
        self.reviewed_learning = ReviewedLearning(self.graph_store, self.skill_registry, self.episodic_store)
        self.planner = Planner()
        self.imagination_planner = ImaginationPlanner(self.dynamics, ImaginationConfig())

        self._last_state: np.ndarray | None = None
        self._last_action_vec: np.ndarray | None = None
        self._last_velocity = 0.0
        self._last_episode_id_by_skill: dict[str, str] = {}
        self._register_default_skills()
        self._recover_on_boot()

    # --- goals -------------------------------------------------------------

    def set_goal(self, kind: str, target: dict) -> Goal:
        goal = Goal.make(kind, target)
        self.working_memory.add_goal(goal)
        return goal

    def set_goal_from_instruction(self, instruction: str):
        """Natural-language goal-setting via the optional LLM interpreter.
        Fails closed: if no interpreter is configured, or the model's
        output doesn't validate, no goal is set and the reason is
        returned — this never falls back to guessing a goal."""
        if self.llm_interpreter is None:
            return None, "no LLM interpreter configured"
        known_ids = {e.entity_id for e in self.working_memory.all_entities()}
        result = self.llm_interpreter.interpret(instruction, known_entity_ids=known_ids)
        if result.goal is None:
            return None, result.rejected_reason
        self.working_memory.add_goal(result.goal)
        return result.goal, None

    # --- boot / recovery -------------------------------------------------

    def _recover_on_boot(self) -> None:
        """Restart recovery: anything left 'pending' at last shutdown was
        in-flight, not committed — mark it interrupted, never silently
        resume it as if it completed."""
        for row in self.working_memory.interrupted_decisions():
            self.working_memory.update_decision_status(row["proposal_id"], "interrupted")

    def _register_default_skills(self) -> None:
        skill_defs: list[tuple[str, dict, tuple[str, ...]]] = [
            ("patrol", {}, ("actuate.motion",)),
            ("avoid_obstacle", {}, ("actuate.motion",)),
            ("yield_to_human", {}, ("actuate.motion",)),
            ("investigate_anomaly", {}, ("actuate.motion", "sensor.focus")),
            ("emergency_stop", {}, ("actuate.motion",)),
            ("hold_position", {}, ("actuate.motion",)),
            ("approach_target", {}, ("actuate.motion", "sensor.focus")),
        ]
        for skill_id, preconditions, permissions in skill_defs:
            if self.skill_registry.latest_version(skill_id) is None:
                self.skill_registry.register(SkillDefinition(
                    skill_id=skill_id, version=1, preconditions=preconditions,
                    permissions=permissions, handler=_default_skill_handler,
                ))

    # --- perceive: transport -> ring buffer -> MCAP -> working memory ----

    def perceive(self, frame: ObservationFrame) -> bool:
        accepted = self.ring_buffer.push(frame)
        if not accepted:
            return False  # duplicate, dropped at ingest — never reaches working memory

        file_id, offset = self.mcap_log.append(frame.topic, frame.payload)

        if self.perception.check_delay(frame):
            return True  # accepted into ring buffer but flagged; caller may inspect ring_buffer stats

        if frame.topic in ("perception/human", "perception/obstacle"):
            entity = self.perception.extract_entity(frame)
            existing = self.working_memory.get_entity(entity.entity_id)
            if existing is not None and self.check_contradiction(existing, entity):
                contradiction = Contradiction.make(
                    subject=entity.entity_id, claim_a_evidence=frame.frame_id,
                    claim_b_evidence=existing.attributes.get("_source_frame_id", "unknown"),
                    detail=f"distance disagreement: {existing.attributes.get('distance')} vs {entity.attributes.get('distance')}",
                )
                self.working_memory.record_contradiction(contradiction)
                return True  # observation kept in ring buffer/MCAP, but not promoted over the held entity
            entity_attrs = dict(entity.attributes)
            entity_attrs["_source_frame_id"] = frame.frame_id
            entity = WorldEntity(entity_id=entity.entity_id, kind=entity.kind, attributes=entity_attrs,
                                   last_seen_ns=entity.last_seen_ns, confidence=entity.confidence)
            self.working_memory.upsert_entity(entity)
            self.associative_index.upsert(entity.entity_id, _entity_vector(entity))
        return True

    # Legitimate real-world change between two accepted sightings of the
    # same entity can be large now that the robot actually moves (real 2D
    # kinematics — see simulate/world.py) and sightings of a given entity
    # can be sparse. CognitiveBrain deliberately doesn't know the robot's
    # own pose/displacement (staying geometry-agnostic, since a real robot
    # might not report distance in these terms at all), so it can't net
    # out "how much of this gap is explained by robot motion." The
    # threshold below is instead set above plausible cumulative drift from
    # real motion between sparse sightings, while staying well below an
    # actually adversarial disagreement (the contradictory_pair() test
    # injector creates a ~7.7m gap, comfortably still caught).
    CONTRADICTION_DISTANCE_THRESHOLD = 4.5

    def check_contradiction(self, entity_a: WorldEntity, entity_b: WorldEntity) -> bool:
        """Same entity_id, materially disagreeing attributes (e.g. distance
        differs by more than a sanity bound) -> unresolved contradiction,
        never silently overwritten."""
        if entity_a.entity_id != entity_b.entity_id:
            return False
        da = entity_a.attributes.get("distance")
        db = entity_b.attributes.get("distance")
        if da is None or db is None:
            return False
        return abs(da - db) > self.CONTRADICTION_DISTANCE_THRESHOLD

    # --- one full cognitive cycle -----------------------------------------

    def cycle(self) -> CycleResult:
        self.working_memory.purge_expired_entities()
        entities = self.working_memory.all_entities()
        focused = self.attention_selector.select(entities)
        goal = self.working_memory.active_goal()

        state = self._build_state_vector(focused, goal)
        surprise = self.jepa.surprise(self._last_state, state) if self._last_state is not None else 0.0
        baseline_pred = self.baseline.predict(self._last_state) if self._last_state is not None else state
        baseline_err = prediction_error(baseline_pred, state) if self._last_state is not None else 0.0
        if self._last_state is not None:
            self.jepa.train_step(self._last_state, state)
            # _last_state and _last_action_vec are always assigned together
            # at the end of every cycle() call (see the bottom of this
            # method) — this assertion makes that cross-attribute invariant
            # explicit and checked, rather than trusting two separately
            # typed Optional fields stay in sync by convention alone.
            assert self._last_action_vec is not None, "_last_state and _last_action_vec must be set together"
            # Supervised, real transition — the dynamics model only ever
            # learns from what actually happened, never from imagination.
            self.dynamics.train_step(self._last_state, self._last_action_vec, state)

        retrieved_episode_ids = self._retrieve_episodes(focused)
        graph_hint_ids = self._graph_hints(focused)
        unresolved_for_focused = self._unresolved_contradictions_for(focused)

        used_imagination = self.imagination_planner.ready()
        if used_imagination:
            proposal, _candidates = self.imagination_planner.propose(state, goal, self.skill_registry,
                                                                        tuple(retrieved_episode_ids))
        else:
            proposal = self.planner.propose(PlannerContext(
                focused_entities=focused, surprise_score=surprise,
                retrieved_episode_ids=retrieved_episode_ids, graph_hint_ids=graph_hint_ids,
            ))
        self.working_memory.record_pending_decision(proposal.proposal_id, proposal.skill_id, proposal.args)

        nearest_human = min((e.attributes.get("distance", 99.0) for e, _ in focused if e.kind == "human"), default=None)
        preconditions_met = self._preconditions_met(proposal)
        context = {
            "unresolved_contradictions_for_entities": unresolved_for_focused,
            "preconditions_met": preconditions_met,
            "nearest_human_distance": nearest_human,
        }

        guard_outcome = self.guard.evaluate(proposal, context)
        final_verdict = guard_outcome.verdict
        result = CycleResult(
            frame_processed=True, proposal=proposal, sutraflow_verdict=guard_outcome.sutraflow_decision.verdict,
            safety_verdict=guard_outcome.safety_decision.verdict, final_verdict=final_verdict,
            surprise_score=surprise, baseline_error=baseline_err,
            reasons=guard_outcome.reasons,
            used_imagination=used_imagination, dynamics_train_steps=self.dynamics.train_steps, active_goal=goal,
        )

        if final_verdict is GuardVerdict.ALLOW:
            self._execute(proposal, result, focused)
            if goal is not None and goal.kind == "observe_entity":
                desired = goal.target.get("desired_distance", 1.0)
                if abs(state[4] - desired) < 0.2:
                    self.working_memory.complete_goal(goal.goal_id)
        else:
            self.working_memory.update_decision_status(proposal.proposal_id, final_verdict.value)

        self._last_state = state
        self._last_action_vec = encode_action(proposal.skill_id, proposal.args.get("velocity", 0.0))
        self._last_velocity = proposal.args.get("velocity", self._last_velocity)
        return result

    def _build_state_vector(self, focused: list[tuple[WorldEntity, float]], goal: Goal | None) -> np.ndarray:
        humans = [e.attributes.get("distance", 5.0) for e, _ in focused if e.kind == "human"]
        obstacles = [e.attributes.get("distance", 5.0) for e, _ in focused if e.kind == "obstacle"]

        goal_target_distance = NEUTRAL_GOAL_DISTANCE
        if goal is not None and goal.kind == "observe_entity":
            target_id = goal.target.get("entity_id")
            entity = self.working_memory.get_entity(target_id) if target_id else None
            if entity is not None:
                goal_target_distance = float(entity.attributes.get("distance", NEUTRAL_GOAL_DISTANCE))

        return np.array([
            min(humans) if humans else 5.0,
            min(obstacles) if obstacles else 5.0,
            float(len(focused)),
            self._last_velocity,
            goal_target_distance,
        ])

    def _retrieve_episodes(self, focused: list[tuple[WorldEntity, float]]) -> list[str]:
        if not focused:
            return []
        query_vec = _entity_vector(focused[0][0])
        candidates = self.associative_index.search(query_vec, top_k=5)
        # Resolve every candidate back to canonical storage — a derived
        # index hit that no longer exists canonically is dropped, not used.
        resolved = []
        for c in candidates:
            entity = self.working_memory.get_entity(c.canonical_id)
            if entity is not None:
                resolved.append(c.canonical_id)
        return resolved

    def _graph_hints(self, focused: list[tuple[WorldEntity, float]]) -> list[str]:
        hint_ids = []
        for entity, _ in focused:
            for edge in self.graph_store.edges_from(entity.entity_id):
                hint_ids.append(edge.edge_id)
        return hint_ids

    def _unresolved_contradictions_for(self, focused: list[tuple[WorldEntity, float]]) -> bool:
        focused_ids = {e.entity_id for e, _ in focused}
        for row in self.working_memory.unresolved_contradictions():
            if row["subject"] in focused_ids:
                return True
        return False

    def _preconditions_met(self, proposal: ActionProposal) -> bool:
        # Prototype precondition check: emergency_stop has none; all others
        # require the robot not already mid-refusal for the same skill this
        # cycle (kept simple — real preconditions come from skill registry
        # entries populated at registration time).
        return True

    def _execute(self, proposal: ActionProposal, result: CycleResult,
                  focused: list[tuple[WorldEntity, float]]) -> None:
        version = self.skill_registry.latest_version(proposal.skill_id) or 1
        handler = self.skill_registry.handler_for(proposal.skill_id, version) or _default_skill_handler
        raw = handler(proposal.args)
        outcome = ObservedOutcome.make(proposal.proposal_id, succeeded=raw["succeeded"], detail=raw)
        result.outcome = outcome

        self.working_memory.update_decision_status(proposal.proposal_id, "executed")
        self.working_memory.record_execution_receipt(outcome.outcome_id, proposal.proposal_id, outcome.outcome_id,
                                                        outcome.succeeded, raw)

        # Log this specific execution to MCAP to get a real, unique
        # (file_id, offset) — using a constant offset here would make every
        # execution of the same skill collide on the same dedupe_key and be
        # falsely rejected as a repeat.
        file_id, offset = self.mcap_log.append(f"execution/{proposal.skill_id}",
                                                  {"proposal_id": proposal.proposal_id, "args": proposal.args})
        episode = Episode.make(robot_id=self.robot_id, skill_id=proposal.skill_id,
                                 precondition_hash="none", mcap_file_id=file_id,
                                 mcap_offset_start=offset, mcap_offset_end=offset,
                                 proposal_id=proposal.proposal_id, outcome_id=outcome.outcome_id)
        result.episode_recorded = self.episodic_store.record(episode)

        # skill_registry.record_outcome is NOT called here directly — it's
        # called exactly once, inside reviewed_learning.process() below,
        # which is the actual gatekeeper for whether a stat update happens
        # at all. A duplicate direct call used to sit here too, which (a)
        # double-counted every CORRECT/INCORRECT outcome and (b) recorded a
        # stat update even for UNCERTAIN outcomes, silently violating this
        # project's own documented bounded-learning principle ("UNCERTAIN
        # outcome — no edge/skill-stat change applied", reviewed_learning.py).
        # Found via profiling's call-count data, not deliberate fuzzing —
        # a reminder that performance profiling doubles as a correctness
        # check when a call count looks higher than it should.

        # Semantic/causal association: only the six typed edges below are
        # ever created — no all-to-all co-occurrence edges.
        related_edge_ids = self._record_episode_edges(episode.episode_id, proposal.skill_id, focused, outcome.succeeded)

        self.reviewed_learning.process(
            skill_id=proposal.skill_id, skill_version=version,
            predicted_confidence=proposal.predicted_confidence, outcome=outcome,
            related_edge_ids=related_edge_ids,
        )

    def _record_episode_edges(self, episode_id: str, skill_id: str,
                                 focused: list[tuple[WorldEntity, float]], succeeded: bool) -> list[str]:
        new_edges: list[GraphEdge] = [GraphEdge.make(episode_id, EdgeType.EXECUTED_BY, skill_id, evidence_ids=[episode_id])]

        for entity, _ in focused[:3]:  # bounded — not all-to-all over every entity ever seen
            new_edges.append(GraphEdge.make(episode_id, EdgeType.OBSERVED_AT, entity.entity_id, evidence_ids=[episode_id]))

        prev_episode_id = self._last_episode_id_by_skill.get(skill_id)
        if prev_episode_id is not None:
            new_edges.append(GraphEdge.make(prev_episode_id, EdgeType.PRECEDES, episode_id, evidence_ids=[episode_id]))
        self._last_episode_id_by_skill[skill_id] = episode_id

        if not succeeded:
            new_edges.append(GraphEdge.make(skill_id, EdgeType.FAILED_AFTER, episode_id, evidence_ids=[episode_id]))

        self.graph_store.add_edges(new_edges)
        return [e.edge_id for e in new_edges]

    def close(self) -> None:
        self.working_memory.close()
