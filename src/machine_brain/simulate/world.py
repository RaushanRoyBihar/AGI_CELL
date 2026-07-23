"""A real (if simple) 2D simulated world, replacing the earlier ad hoc
"nudge a distance scalar per skill" hack in `sensors.py` with actual
geometry: the robot has a position and heading, entities have positions
and wander independently, and every sensed "distance" is the honest
Euclidean distance between the two — computed the same way regardless of
which skill produced the robot's current position, rather than a
special-cased bump per skill_id.

This is what "approach_target actually gets closer" and "avoid_obstacle
actually creates distance" mean now: real coordinates change, and distance
falls out of that, instead of being asserted directly. It's what a
simulator like Gazebo does for a real ROS2 stack, radically simplified
(a point robot, no acceleration/inertia, no differential-drive
constraints — holonomic motion: the robot instantaneously faces and moves
along the direction implied by its chosen action each step). That
simplification is a deliberate, documented scope cut for this prototype,
not an oversight.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Protocol

_HUMAN_SPAWN_RADIUS = (1.0, 5.0)
_OBSTACLE_SPAWN_RADIUS = (0.5, 4.0)
_WANDER_STEP = 0.12
_MAX_LEASH_RADIUS = {"human": 5.0, "obstacle": 4.0}


@dataclass
class Vec2:
    x: float
    y: float

    def dist_to(self, other: "Vec2") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)

    def angle_to(self, other: "Vec2") -> float:
        return math.atan2(other.y - self.y, other.x - self.x)


class WorldBackend(Protocol):
    """The contract `SensorSimulator` depends on — deliberately narrow, so
    any physics backend can implement it. `SimulatedWorld` below (2D
    holonomic kinematics, no dependencies) and `MuJoCoWorld`
    (`mujoco_world.py`, real 3D rigid-body physics, optional dependency)
    both satisfy this without either knowing the other exists."""

    robot_pos: Vec2

    def ensure_entity(self, entity_id: str, kind: str) -> Vec2: ...
    def wander_entity(self, entity_id: str) -> None: ...
    def distance_to(self, entity_id: str) -> float: ...
    def position_of(self, entity_id: str) -> Vec2: ...
    def apply_robot_action(self, skill_id: str, velocity: float, target_entity_id: str | None = None) -> None: ...


class SimulatedWorld:
    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)
        self.robot_pos = Vec2(0.0, 0.0)
        self.robot_heading = 0.0
        self._entity_pos: dict[str, Vec2] = {}
        self._entity_kind: dict[str, str] = {}

    def _spawn_radius(self, kind: str) -> tuple[float, float]:
        return _HUMAN_SPAWN_RADIUS if kind == "human" else _OBSTACLE_SPAWN_RADIUS

    def ensure_entity(self, entity_id: str, kind: str) -> Vec2:
        if entity_id not in self._entity_pos:
            lo, hi = self._spawn_radius(kind)
            angle = self._rng.uniform(0, 2 * math.pi)
            r = self._rng.uniform(lo, hi)
            self._entity_pos[entity_id] = Vec2(self.robot_pos.x + r * math.cos(angle),
                                                  self.robot_pos.y + r * math.sin(angle))
            self._entity_kind[entity_id] = kind
        return self._entity_pos[entity_id]

    def wander_entity(self, entity_id: str) -> None:
        """Bounded random walk relative to the robot's *current* position —
        like an NPC wandering within a room, not drifting to infinity. An
        unbounded walk would let real distance grow arbitrarily large
        between infrequent sightings of the same entity; the next sighting
        would then look like a multi-meter jump against the stale
        working-memory snapshot, which the contradiction detector would
        (correctly, given that reading) flag — but that's an artifact of
        an unrealistic unbounded environment, not a genuine sensor fault,
        so it's bounded here instead of relied on the guard to absorb."""
        pos = self._entity_pos[entity_id]
        pos.x += self._rng.uniform(-_WANDER_STEP, _WANDER_STEP)
        pos.y += self._rng.uniform(-_WANDER_STEP, _WANDER_STEP)

        max_radius = _MAX_LEASH_RADIUS[self._entity_kind[entity_id]]
        dist = self.robot_pos.dist_to(pos)
        if dist > max_radius:
            scale = max_radius / dist
            pos.x = self.robot_pos.x + (pos.x - self.robot_pos.x) * scale
            pos.y = self.robot_pos.y + (pos.y - self.robot_pos.y) * scale

    def distance_to(self, entity_id: str) -> float:
        return self.robot_pos.dist_to(self._entity_pos[entity_id])

    def position_of(self, entity_id: str) -> Vec2:
        return self._entity_pos[entity_id]

    def _nearest_threat(self) -> str | None:
        """Nearest currently-known human or obstacle — what an avoidance
        maneuver actually reacts to."""
        if not self._entity_pos:
            return None
        return min(self._entity_pos, key=lambda eid: self.distance_to(eid))

    def apply_robot_action(self, skill_id: str, velocity: float, target_entity_id: str | None = None) -> None:
        """One simulated time-step of robot kinematics in reaction to an
        executed skill. Called by the demo/benchmark harness after a
        proposal is actually allowed and executed — never by CognitiveBrain
        itself, which has no reference to this class and doesn't know
        whether its sensors are simulated or real."""
        if skill_id == "approach_target" and target_entity_id and target_entity_id in self._entity_pos:
            heading = self.robot_pos.angle_to(self._entity_pos[target_entity_id])
        elif skill_id in ("avoid_obstacle", "yield_to_human"):
            threat = self._nearest_threat()
            heading = (self._entity_pos[threat].angle_to(self.robot_pos)
                        if threat is not None else self.robot_heading)
        elif skill_id == "patrol":
            heading = self.robot_heading + self._rng.uniform(-0.3, 0.3)  # gentle drift, not a straight line forever
        else:  # hold_position, emergency_stop, investigate_anomaly: no directed heading change
            heading = self.robot_heading

        self.robot_heading = heading
        self.robot_pos.x += velocity * math.cos(heading)
        self.robot_pos.y += velocity * math.sin(heading)
