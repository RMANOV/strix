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


# ── B1: APF rollout ──

def test_apf_rollout_curves_around_obstacle():
    """APF rollout should produce a curved path around an obstacle."""
    from strix.temporal.multi_horizon import Horizon
    planner = MultiHorizonPlanner()
    cfg = planner._configs[Horizon.TACTICAL]
    drone = _drone(1, 0.0, 0.0, vx=10.0, vy=0.0)
    obstacles = [(Vec3(15.0, 0.0, 0.0), 5.0)]
    wps = MultiHorizonPlanner._rollout_apf(
        drone=drone, direction=Vec3(1.0, 0.0, 0.0),
        speed_ms=10.0, cfg=cfg, steps=20, obstacles=obstacles, threats=[],
    )
    assert any(abs(wp.position.y) > 1.0 for wp in wps), "APF should curve around obstacle"
    assert wps[-1].position.x > drone.position.x, "should make forward progress"


def test_apf_rollout_no_obstacles_is_straight():
    """Without obstacles, APF degrades to near-constant-velocity."""
    from strix.temporal.multi_horizon import Horizon
    planner = MultiHorizonPlanner()
    cfg = planner._configs[Horizon.TACTICAL]
    drone = _drone(1, 0.0, 0.0, vx=10.0, vy=0.0)
    wps = MultiHorizonPlanner._rollout_apf(
        drone=drone, direction=Vec3(1.0, 0.0, 0.0),
        speed_ms=10.0, cfg=cfg, steps=20, obstacles=[], threats=[],
    )
    for wp in wps:
        assert abs(wp.position.y) < 0.1, f"no obstacle: y={wp.position.y} should be ~0"


# ── B3: Waypoint-to-Decision bridge ──

def test_tactical_to_decisions_conversion():
    """TacticalPlan waypoints should convert to a Decision sequence."""
    from strix.temporal.multi_horizon import tactical_to_decisions
    from strix.brain import DecisionKind
    planner = MultiHorizonPlanner()
    state = FleetState(
        drones=[_drone(1, 0.0, 0.0, vx=8.0, vy=0.0)],
        obstacles=[(Vec3(20.0, 0.0, 0.0), 3.0)],
    )
    state.recompute_metrics()
    plans = planner.plan_tactical(state)
    decisions = tactical_to_decisions(plans)
    assert len(decisions) >= 1
    d = decisions[0]
    assert d.drone_id == 1
    assert d.kind == DecisionKind.GOTO
    assert d.speed_ms > 0
