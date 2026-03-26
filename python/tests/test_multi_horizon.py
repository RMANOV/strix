"""Tests for the multi-horizon planner tactical hardening."""

import math

from strix.brain import DroneSnapshot, Vec3
from strix.temporal.multi_horizon import FleetState, MultiHorizonPlanner


def _drone(
    drone_id: int,
    x: float,
    y: float,
    z: float = 0.0,
    vx: float = 0.0,
    vy: float = 0.0,
    vz: float = 0.0,
) -> DroneSnapshot:
    return DroneSnapshot(
        drone_id=drone_id,
        position=Vec3(x, y, z),
        velocity=Vec3(vx, vy, vz),
        alive=True,
    )


def test_tactical_planner_deflects_around_forward_obstacle():
    planner = MultiHorizonPlanner()
    state = FleetState(
        drones=[_drone(1, 0.0, 0.0, vx=8.0, vy=0.0)],
        obstacles=[(Vec3(7.0, 0.0, 0.0), 3.0)],
    )
    state.recompute_metrics()

    plan = planner.plan_tactical(state)[1]

    assert plan.valid is True
    assert plan.obstacle_clearance_m > 2.0
    assert max(abs(wp.position.y) for wp in plan.waypoints) > 0.5


def test_tactical_plan_vetoes_when_starting_inside_obstacle():
    planner = MultiHorizonPlanner()
    state = FleetState(
        drones=[_drone(1, 0.0, 0.0, vx=4.0, vy=0.0)],
        obstacles=[(Vec3(0.0, 0.0, 0.0), 5.0)],
    )
    state.recompute_metrics()

    plan = planner.plan_tactical(state)[1]

    assert plan.valid is False
    assert "obstacle clearance" in plan.veto_reason


def test_operational_heading_tracks_fleet_motion():
    planner = MultiHorizonPlanner()
    state = FleetState(
        drones=[
            _drone(1, 0.0, 0.0, vx=0.0, vy=6.0),
            _drone(2, 10.0, 0.0, vx=0.0, vy=8.0),
        ]
    )
    state.recompute_metrics()

    plan = planner.plan_operational(state)

    assert math.isclose(plan.formation_heading_rad, math.pi / 2, rel_tol=1e-6, abs_tol=1e-6)
