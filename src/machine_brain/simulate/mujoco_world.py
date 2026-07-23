"""A real 3D physics backend for the simulated world, using MuJoCo
(Google DeepMind's rigid-body physics engine) in place of the hand-rolled
holonomic kinematics in `world.py`. Same external interface as
`SimulatedWorld` — `ensure_entity`, `wander_entity`, `distance_to`,
`position_of`, `apply_robot_action`, `robot_pos` — so `SensorSimulator`
can use either backend interchangeably with zero changes to its own code.

What's different from `world.py`, and why it matters: the robot here is
a real rigid body with mass and joint damping, driven by MuJoCo velocity
actuators and integrated by `mj_step` — commanding a velocity produces a
realistic ramp-up under inertia, not an instantaneous position change.
Collision geometry is real (the human/obstacle bodies have actual
capsule/sphere shapes MuJoCo's contact solver can reason about), even
though their *positions* are externally driven via MuJoCo's "mocap body"
mechanism (the standard MuJoCo pattern for representing motion-captured or
externally-scripted objects: their kinematics are set from Python each
step rather than simulated, while they still participate fully in
collision detection against real physics bodies like the robot).

Scope, stated plainly: still a simplified holonomic base (independently
actuated x/y velocity, not a differential-drive or nonholonomic
constraint) — same simplification `world.py` already documented, now
running through a real physics integrator instead of hand-rolled Euler
steps. CPU-only; no GPU required for single-robot stepping at this scale.
"""

from __future__ import annotations

import math
import random

import mujoco
import numpy as np

from machine_brain.simulate.world import Vec2

_HUMAN_SPAWN_RADIUS = (1.0, 5.0)
_OBSTACLE_SPAWN_RADIUS = (0.5, 4.0)
_WANDER_STEP = 0.12
_MAX_LEASH_RADIUS = {"human": 5.0, "obstacle": 4.0}

_N_HUMAN_SLOTS = 5
_N_OBSTACLE_SLOTS = 7

_TIMESTEP = 0.002
_SUBSTEPS_PER_TICK = 25  # ~50ms of simulated time per tick() call


def _build_mjcf(n_humans: int, n_obstacles: int) -> str:
    mocap_bodies = []
    for i in range(n_humans):
        mocap_bodies.append(
            f'<body name="human_{i}" mocap="true" pos="{2 + i} 0 0.3">'
            f'<geom name="human_{i}_geom" type="capsule" size="0.15 0.3" '
            f'rgba="0.9 0.2 0.2 0.6" contype="2" conaffinity="1"/></body>'
        )
    for i in range(n_obstacles):
        mocap_bodies.append(
            f'<body name="obstacle_{i}" mocap="true" pos="{-2 - i} 0 0.2">'
            f'<geom name="obstacle_{i}_geom" type="sphere" size="0.2" '
            f'rgba="0.5 0.5 0.5 0.6" contype="2" conaffinity="1"/></body>'
        )
    mocap_xml = "\n    ".join(mocap_bodies)

    return f"""
<mujoco model="machine_brain_scene">
  <option timestep="{_TIMESTEP}" gravity="0 0 -9.81"/>
  <worldbody>
    <light diffuse="1 1 1" pos="0 0 5"/>
    <geom name="floor" type="plane" size="30 30 0.1" rgba="0.85 0.85 0.85 1"/>
    <body name="robot" pos="0 0 0.15">
      <joint name="robot_x" type="slide" axis="1 0 0" damping="4"/>
      <joint name="robot_y" type="slide" axis="0 1 0" damping="4"/>
      <joint name="robot_yaw" type="hinge" axis="0 0 1" damping="1"/>
      <geom name="robot_geom" type="box" size="0.2 0.2 0.15" mass="10"
            rgba="0.2 0.4 0.8 1" contype="1" conaffinity="2"/>
    </body>
    {mocap_xml}
  </worldbody>
  <actuator>
    <velocity name="act_x" joint="robot_x" kv="25" ctrlrange="-2 2"/>
    <velocity name="act_y" joint="robot_y" kv="25" ctrlrange="-2 2"/>
  </actuator>
</mujoco>
"""


class MuJoCoWorld:
    def __init__(self, seed: int = 0, n_human_slots: int = _N_HUMAN_SLOTS,
                  n_obstacle_slots: int = _N_OBSTACLE_SLOTS) -> None:
        self._rng = random.Random(seed)
        xml = _build_mjcf(n_human_slots, n_obstacle_slots)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

        self._robot_body_id = self.model.body("robot").id
        self._human_mocap_ids = [self.model.body(f"human_{i}").mocapid[0] for i in range(n_human_slots)]
        self._obstacle_mocap_ids = [self.model.body(f"obstacle_{i}").mocapid[0] for i in range(n_obstacle_slots)]
        self._entity_mocap_id: dict[str, int] = {}
        self._entity_kind: dict[str, str] = {}
        self._robot_heading = 0.0

    # --- entity management (fixed mocap-slot pool, matches the existing
    # human-{0..4}/obstacle-{0..6} entity_id convention SensorSimulator uses) ---

    def _slot_for(self, entity_id: str, kind: str) -> int:
        if kind == "human":
            idx = int(entity_id.split("-")[-1]) % len(self._human_mocap_ids)
            return self._human_mocap_ids[idx]
        idx = int(entity_id.split("-")[-1]) % len(self._obstacle_mocap_ids)
        return self._obstacle_mocap_ids[idx]

    def ensure_entity(self, entity_id: str, kind: str) -> Vec2:
        if entity_id not in self._entity_mocap_id:
            mocap_id = self._slot_for(entity_id, kind)
            self._entity_mocap_id[entity_id] = mocap_id
            self._entity_kind[entity_id] = kind
            lo, hi = _HUMAN_SPAWN_RADIUS if kind == "human" else _OBSTACLE_SPAWN_RADIUS
            angle = self._rng.uniform(0, 2 * math.pi)
            r = self._rng.uniform(lo, hi)
            rx, ry, _ = self.data.xpos[self._robot_body_id]
            self.data.mocap_pos[mocap_id] = [rx + r * math.cos(angle), ry + r * math.sin(angle), 0.3]
        return self.position_of(entity_id)

    def wander_entity(self, entity_id: str) -> None:
        mocap_id = self._entity_mocap_id[entity_id]
        kind = self._entity_kind[entity_id]
        pos = self.data.mocap_pos[mocap_id].copy()
        pos[0] += self._rng.uniform(-_WANDER_STEP, _WANDER_STEP)
        pos[1] += self._rng.uniform(-_WANDER_STEP, _WANDER_STEP)

        max_radius = _MAX_LEASH_RADIUS[kind]
        rx, ry, _ = self.data.xpos[self._robot_body_id]
        dist = math.hypot(pos[0] - rx, pos[1] - ry)
        if dist > max_radius:
            scale = max_radius / dist
            pos[0] = rx + (pos[0] - rx) * scale
            pos[1] = ry + (pos[1] - ry) * scale
        self.data.mocap_pos[mocap_id] = pos
        self.tick()

    def distance_to(self, entity_id: str) -> float:
        mocap_id = self._entity_mocap_id[entity_id]
        rx, ry, rz = self.data.xpos[self._robot_body_id]
        ex, ey, ez = self.data.mocap_pos[mocap_id]
        return float(math.hypot(ex - rx, ey - ry))

    def position_of(self, entity_id: str) -> Vec2:
        mocap_id = self._entity_mocap_id[entity_id]
        x, y, _ = self.data.mocap_pos[mocap_id]
        return Vec2(float(x), float(y))

    @property
    def robot_pos(self) -> Vec2:
        x, y, _ = self.data.xpos[self._robot_body_id]
        return Vec2(float(x), float(y))

    def _nearest_threat_pos(self) -> tuple[float, float] | None:
        if not self._entity_mocap_id:
            return None
        nearest_id = min(self._entity_mocap_id, key=self.distance_to)
        pos = self.position_of(nearest_id)
        return pos.x, pos.y

    def apply_robot_action(self, skill_id: str, velocity: float, target_entity_id: str | None = None) -> None:
        rx, ry, _ = self.data.xpos[self._robot_body_id]

        if skill_id == "approach_target" and target_entity_id and target_entity_id in self._entity_mocap_id:
            target = self.position_of(target_entity_id)
            heading = math.atan2(target.y - ry, target.x - rx)
        elif skill_id in ("avoid_obstacle", "yield_to_human"):
            threat = self._nearest_threat_pos()
            heading = math.atan2(ry - threat[1], rx - threat[0]) if threat is not None else self._robot_heading
        elif skill_id == "patrol":
            heading = self._robot_heading + self._rng.uniform(-0.3, 0.3)
        else:  # hold_position, emergency_stop, investigate_anomaly
            heading = self._robot_heading

        self._robot_heading = heading
        self.data.ctrl[0] = velocity * math.cos(heading)
        self.data.ctrl[1] = velocity * math.sin(heading)
        self.tick()

    def tick(self, substeps: int = _SUBSTEPS_PER_TICK) -> None:
        """Advance real physics integration. Called on every entity
        sighting and every executed action, so the simulated world keeps
        evolving under genuine dynamics as frames stream in — not just
        when the robot acts."""
        for _ in range(substeps):
            mujoco.mj_step(self.model, self.data)
