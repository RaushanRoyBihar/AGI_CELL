"""Tests against real MuJoCo physics — not a mock, the actual rigid-body
simulator. Skipped automatically if mujoco isn't installed, since it's an
optional dependency (the system must still work without it)."""

import math

import pytest

mujoco = pytest.importorskip("mujoco")

from machine_brain.orchestrator.cognitive_loop import CognitiveBrain
from machine_brain.simulate.mujoco_world import MuJoCoWorld
from machine_brain.simulate.sensors import SensorSimulator, SimConfig


def test_robot_starts_at_origin_and_stays_put_with_no_action():
    world = MuJoCoWorld(seed=0)
    start = world.robot_pos
    assert start.x == pytest.approx(0.0, abs=1e-6)
    assert start.y == pytest.approx(0.0, abs=1e-6)
    for _ in range(50):
        world.tick()
    assert world.robot_pos.x == pytest.approx(0.0, abs=1e-6)
    assert world.robot_pos.y == pytest.approx(0.0, abs=1e-6)


def test_commanded_velocity_moves_the_robot_under_real_dynamics():
    """Not a teleport — motion should show realistic ramp-up under mass
    and joint damping, not jump straight to the commanded distance."""
    world = MuJoCoWorld(seed=0)
    world.apply_robot_action("patrol", velocity=1.0)
    after_one_tick = world.robot_pos.x
    assert 0 < after_one_tick < 1.0 * (world.model.opt.timestep * 25)  # moved, but less than a naive v*t teleport would give

    for _ in range(20):
        world.apply_robot_action("patrol", velocity=1.0)
    assert world.robot_pos.x > after_one_tick  # kept moving with sustained commands


def test_approach_target_reduces_real_distance():
    world = MuJoCoWorld(seed=3)
    world.ensure_entity("human-0", "human")
    initial_distance = world.distance_to("human-0")

    for _ in range(80):
        world.apply_robot_action("approach_target", velocity=0.8, target_entity_id="human-0")
    final_distance = world.distance_to("human-0")

    assert final_distance < initial_distance


def test_avoid_obstacle_increases_real_distance_from_nearest_threat():
    world = MuJoCoWorld(seed=4)
    world.ensure_entity("obstacle-0", "obstacle")
    # Force the obstacle close so it's unambiguously "the nearest threat."
    mocap_id = world._entity_mocap_id["obstacle-0"]
    world.data.mocap_pos[mocap_id] = [0.6, 0.0, 0.2]
    initial_distance = world.distance_to("obstacle-0")

    for _ in range(60):
        world.apply_robot_action("avoid_obstacle", velocity=0.6)
    final_distance = world.distance_to("obstacle-0")

    assert final_distance > initial_distance


def test_entity_wander_stays_within_leash_radius():
    world = MuJoCoWorld(seed=5)
    world.ensure_entity("human-1", "human")
    for _ in range(200):
        world.wander_entity("human-1")
    assert world.distance_to("human-1") <= 5.0 + 1e-6  # _MAX_LEASH_RADIUS["human"]


def test_sensor_simulator_works_with_mujoco_backend():
    world = MuJoCoWorld(seed=6)
    sim = SensorSimulator(SimConfig(seed=6), world=world)
    frames = [sim.next_frame() for _ in range(80)]
    assert any(f.topic in ("perception/human", "perception/obstacle") for f in frames)
    for f in frames:
        if f.topic in ("perception/human", "perception/obstacle"):
            assert math.isfinite(f.payload["distance"])
            assert f.payload["distance"] >= 0.0


def test_full_cognitive_brain_loop_runs_against_real_physics(tmp_path):
    world = MuJoCoWorld(seed=7)
    sim = SensorSimulator(SimConfig(seed=7), world=world)
    brain = CognitiveBrain(data_dir=str(tmp_path / "data"))

    for i in range(150):
        brain.perceive(sim.next_frame())
        if i % 3 == 0:
            result = brain.cycle()
            if result.outcome is not None:
                sim.apply_action(result.proposal.skill_id, result.proposal.args.get("velocity", 0.0))

    ok, _ = brain.audit_ledger.verify_chain()
    assert ok is True
    # The robot must have actually moved through real physics at some point,
    # not stayed pinned at the origin the whole run.
    assert (world.robot_pos.x, world.robot_pos.y) != (0.0, 0.0)
