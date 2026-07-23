"""Canonical schemas shared by every layer. Frozen once stable — every layer
downstream of Phase 1 depends on these shapes, so changes here ripple
everywhere. Prefer additive fields over breaking changes.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


def monotonic_ns() -> int:
    return time.monotonic_ns()


def wall_time() -> float:
    return time.time()


def new_id() -> str:
    return uuid.uuid4().hex


class Confidence(float, Enum):
    """Named thresholds so callers don't sprinkle magic numbers."""

    LOW = 0.35
    MEDIUM = 0.6
    HIGH = 0.85


@dataclass(frozen=True)
class ObservationFrame:
    """A single canonical sensor/operator observation.

    Layer 0 (transport) carries these; Layer 1 (ring buffer) holds a bounded
    recent window; Layer 2 (MCAP) archives them durably.
    """

    frame_id: str
    topic: str
    sensor_id: str
    sequence_id: int
    monotonic_ns: int
    wall_time: float
    payload: dict[str, Any]
    robot_id: str = "robot-0"

    def idempotency_key(self) -> str:
        # (topic, sensor, sequence) — duplicate frames on the wire collapse
        # to the same key so they never create two rows downstream.
        return f"{self.topic}:{self.sensor_id}:{self.sequence_id}"

    @staticmethod
    def make(topic: str, sensor_id: str, sequence_id: int, payload: dict[str, Any],
              robot_id: str = "robot-0") -> "ObservationFrame":
        return ObservationFrame(
            frame_id=new_id(),
            topic=topic,
            sensor_id=sensor_id,
            sequence_id=sequence_id,
            monotonic_ns=monotonic_ns(),
            wall_time=wall_time(),
            payload=payload,
            robot_id=robot_id,
        )


@dataclass(frozen=True)
class WorldEntity:
    """A perceived/tracked entity in active world state (Layer working-memory)."""

    entity_id: str
    kind: str
    attributes: dict[str, Any]
    last_seen_ns: int
    confidence: float


@dataclass(frozen=True)
class ActionProposal:
    """What the planner wants to do, before SutraFlow + safety governor see it."""

    proposal_id: str
    skill_id: str
    args: dict[str, Any]
    justification: str
    predicted_confidence: float
    source_episode_ids: tuple[str, ...] = ()
    source_frame_ids: tuple[str, ...] = ()

    @staticmethod
    def make(skill_id: str, args: dict[str, Any], justification: str,
              predicted_confidence: float, source_episode_ids=(), source_frame_ids=()) -> "ActionProposal":
        return ActionProposal(
            proposal_id=new_id(),
            skill_id=skill_id,
            args=args,
            justification=justification,
            predicted_confidence=predicted_confidence,
            source_episode_ids=tuple(source_episode_ids),
            source_frame_ids=tuple(source_frame_ids),
        )


class GuardVerdict(str, Enum):
    ALLOW = "allow"
    REFUSE = "refuse"
    HOLD = "hold"  # e.g. unresolved contradiction, needs review


@dataclass(frozen=True)
class GuardDecision:
    decision_id: str
    proposal_id: str
    verdict: GuardVerdict
    reasons: tuple[str, ...]
    rule_ids: tuple[str, ...]
    decided_at_ns: int

    @staticmethod
    def make(proposal_id: str, verdict: GuardVerdict, reasons, rule_ids) -> "GuardDecision":
        return GuardDecision(
            decision_id=new_id(),
            proposal_id=proposal_id,
            verdict=verdict,
            reasons=tuple(reasons),
            rule_ids=tuple(rule_ids),
            decided_at_ns=monotonic_ns(),
        )


class OutcomeLabel(str, Enum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    UNCERTAIN = "uncertain"
    PENDING = "pending"


@dataclass(frozen=True)
class ObservedOutcome:
    outcome_id: str
    proposal_id: str
    succeeded: bool
    detail: dict[str, Any]
    observed_at_ns: int
    label: OutcomeLabel = OutcomeLabel.PENDING

    @staticmethod
    def make(proposal_id: str, succeeded: bool, detail: dict[str, Any]) -> "ObservedOutcome":
        return ObservedOutcome(
            outcome_id=new_id(),
            proposal_id=proposal_id,
            succeeded=succeeded,
            detail=detail,
            observed_at_ns=monotonic_ns(),
        )


@dataclass(frozen=True)
class Episode:
    """Observation + action + outcome + provenance, the unit of episodic memory."""

    episode_id: str
    robot_id: str
    skill_id: str
    precondition_hash: str
    mcap_file_id: str
    mcap_offset_start: int
    mcap_offset_end: int
    proposal_id: str
    outcome_id: str | None
    event_time: float
    dedupe_key: str

    @staticmethod
    def make(robot_id: str, skill_id: str, precondition_hash: str, mcap_file_id: str,
              mcap_offset_start: int, mcap_offset_end: int, proposal_id: str,
              outcome_id: str | None) -> "Episode":
        dedupe_key = hashlib.sha256(
            f"{skill_id}:{precondition_hash}:{mcap_file_id}:{mcap_offset_start}:{mcap_offset_end}".encode()
        ).hexdigest()
        return Episode(
            episode_id=new_id(),
            robot_id=robot_id,
            skill_id=skill_id,
            precondition_hash=precondition_hash,
            mcap_file_id=mcap_file_id,
            mcap_offset_start=mcap_offset_start,
            mcap_offset_end=mcap_offset_end,
            proposal_id=proposal_id,
            outcome_id=outcome_id,
            event_time=wall_time(),
            dedupe_key=dedupe_key,
        )


class EdgeType(str, Enum):
    CAUSED_BY = "CAUSED_BY"
    PRECEDES = "PRECEDES"
    OBSERVED_AT = "OBSERVED_AT"
    EXECUTED_BY = "EXECUTED_BY"
    FAILED_AFTER = "FAILED_AFTER"
    RESOLVED_BY = "RESOLVED_BY"


@dataclass(frozen=True)
class GraphEdge:
    edge_id: str
    src: str
    edge_type: EdgeType
    dst: str
    evidence_ids: tuple[str, ...]
    weight: float
    created_at_ns: int
    verified_count: int = 0

    @staticmethod
    def make(src: str, edge_type: EdgeType, dst: str, evidence_ids, weight: float = 0.5) -> "GraphEdge":
        return GraphEdge(
            edge_id=new_id(),
            src=src,
            edge_type=edge_type,
            dst=dst,
            evidence_ids=tuple(evidence_ids),
            weight=weight,
            created_at_ns=monotonic_ns(),
        )


class GoalStatus(str, Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


@dataclass(frozen=True)
class Goal:
    """Something the agent is persistently trying to do, as opposed to a
    one-cycle reactive proposal. Deliberately minimal: a kind + a target
    dict the planner interprets. This is what turns a reflex agent into
    one with standing intent."""

    goal_id: str
    kind: str                 # e.g. "observe_entity", "patrol_safely"
    target: dict[str, Any]
    created_at_ns: int
    status: GoalStatus = GoalStatus.ACTIVE

    @staticmethod
    def make(kind: str, target: dict[str, Any]) -> "Goal":
        return Goal(goal_id=new_id(), kind=kind, target=target, created_at_ns=monotonic_ns())


@dataclass(frozen=True)
class Contradiction:
    contradiction_id: str
    subject: str
    claim_a_evidence: str
    claim_b_evidence: str
    detail: str
    created_at_ns: int
    resolved: bool = False
    resolution: str | None = None

    @staticmethod
    def make(subject: str, claim_a_evidence: str, claim_b_evidence: str, detail: str) -> "Contradiction":
        return Contradiction(
            contradiction_id=new_id(),
            subject=subject,
            claim_a_evidence=claim_a_evidence,
            claim_b_evidence=claim_b_evidence,
            detail=detail,
            created_at_ns=monotonic_ns(),
        )
