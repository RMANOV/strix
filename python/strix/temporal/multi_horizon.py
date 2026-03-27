"""Multi-Horizon Planner -- t-CoT (temporal Chain of Thought).

Military planning operates across three fundamentally different time scales,
each demanding different resolution, particle counts, and decision logic.
This module unifies them into a single coherent planning pipeline.

Horizon 1 (Tactical):   dt=0.1s,  100 particles  -- obstacle avoidance, collision
Horizon 2 (Operational): dt=5s,   500 particles  -- formation, coordination
Horizon 3 (Strategic):   dt=60s, 2000 particles  -- mission-level planning

The cascade works top-down: strategic decisions constrain operational plans,
which constrain tactical manoeuvres.  But feedback flows bottom-up: a tactical
impossibility (e.g., terrain collision) vetoes the operational plan that
requested it.

This mirrors how quantitative trading firms manage multiple time-horizon
strategies -- an HFT alpha signal at 100ms does not override the portfolio
risk budget set at the daily level.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from strix.brain import DroneSnapshot, MissionArea, MissionType, RegimeLabel, Vec3

logger = logging.getLogger("strix.temporal.multi_horizon")

# ---------------------------------------------------------------------------
# Horizon configuration
# ---------------------------------------------------------------------------


class Horizon(Enum):
    """Planning horizons with associated time scales."""

    TACTICAL = auto()      # 0.1s dt, ~10s lookahead
    OPERATIONAL = auto()   # 5.0s dt, ~5min lookahead
    STRATEGIC = auto()     # 60s dt,  ~1hr lookahead


@dataclass(frozen=True)
class HorizonConfig:
    """Parameters for a single planning horizon."""

    dt: float
    n_particles: int
    lookahead_s: float
    replan_interval_s: float

    @property
    def steps(self) -> int:
        return int(self.lookahead_s / self.dt)


DEFAULT_CONFIGS = {
    Horizon.TACTICAL: HorizonConfig(dt=0.1, n_particles=100, lookahead_s=10.0, replan_interval_s=0.5),
    Horizon.OPERATIONAL: HorizonConfig(dt=5.0, n_particles=500, lookahead_s=300.0, replan_interval_s=15.0),
    Horizon.STRATEGIC: HorizonConfig(dt=60.0, n_particles=2000, lookahead_s=3600.0, replan_interval_s=120.0),
}


# ---------------------------------------------------------------------------
# Plan data types
# ---------------------------------------------------------------------------


@dataclass
class Waypoint:
    """A single waypoint in a plan trajectory."""

    position: Vec3 = field(default_factory=Vec3)
    velocity: Vec3 = field(default_factory=Vec3)
    time_s: float = 0.0
    heading_rad: float = 0.0


@dataclass
class TacticalPlan:
    """Horizon 1: immediate obstacle avoidance and collision prevention.

    Runs at 10 Hz with 100 particles.  The plan is a short sequence of
    waypoints that avoid obstacles, maintain safe separation from other
    drones, and respect altitude limits.
    """

    drone_id: int = 0
    waypoints: list[Waypoint] = field(default_factory=list)
    obstacle_clearance_m: float = 0.0
    min_separation_m: float = 5.0
    valid: bool = True
    veto_reason: str = ""
    timestamp: float = 0.0


@dataclass
class FormationSlot:
    """A drone's assigned position within a formation."""

    drone_id: int = 0
    offset: Vec3 = field(default_factory=Vec3)
    role: str = "wingman"


@dataclass
class OperationalPlan:
    """Horizon 2: formation management, coordination, and task execution.

    Runs at 0.2 Hz with 500 particles.  Manages formation geometry,
    inter-drone coordination, sensor coverage optimization, and
    handoff scheduling.
    """

    formation: list[FormationSlot] = field(default_factory=list)
    formation_centroid: Vec3 = field(default_factory=Vec3)
    formation_heading_rad: float = 0.0
    speed_ms: float = 15.0
    sensor_coverage_pct: float = 0.0
    valid: bool = True
    timestamp: float = 0.0


@dataclass
class PhaseObjective:
    """A single phase within the strategic plan."""

    name: str = ""
    description: str = ""
    area: Optional[MissionArea] = None
    mission_type: MissionType = MissionType.RECON
    estimated_duration_s: float = 0.0
    drone_count: int = 4
    priority: float = 0.5


@dataclass
class StrategicPlan:
    """Horizon 3: mission-level planning, resource allocation, phasing.

    Runs at 1/120 Hz with 2000 particles.  Determines mission phases,
    resource allocation across sub-swarms, staging areas, and withdrawal
    criteria.
    """

    phases: list[PhaseObjective] = field(default_factory=list)
    total_duration_s: float = 0.0
    resource_allocation: dict[int, str] = field(default_factory=dict)
    regime: RegimeLabel = RegimeLabel.PATROL
    confidence: float = 0.0
    timestamp: float = 0.0


@dataclass
class IntegratedPlan:
    """The cascaded result of all three horizons.

    Contains the active plans from each horizon, plus metadata about
    coherence between levels.
    """

    tactical: dict[int, TacticalPlan] = field(default_factory=dict)
    operational: Optional[OperationalPlan] = None
    strategic: Optional[StrategicPlan] = None
    coherent: bool = True
    incoherence_reason: str = ""
    cascade_timestamp: float = 0.0


# ---------------------------------------------------------------------------
# Fleet state snapshot for planning
# ---------------------------------------------------------------------------


@dataclass
class FleetState:
    """Snapshot of the fleet state used as input to all planning horizons."""

    drones: list[DroneSnapshot] = field(default_factory=list)
    centroid: Vec3 = field(default_factory=Vec3)
    dispersion_m: float = 0.0
    regime: RegimeLabel = RegimeLabel.PATROL
    threats: list[Vec3] = field(default_factory=list)
    obstacles: list[tuple[Vec3, float]] = field(default_factory=list)  # (center, radius)
    timestamp: float = 0.0

    def recompute_metrics(self) -> None:
        alive = [d for d in self.drones if d.alive]
        n = len(alive)
        if n == 0:
            self.centroid = Vec3()
            self.dispersion_m = 0.0
            return
        cx = sum(d.position.x for d in alive) / n
        cy = sum(d.position.y for d in alive) / n
        cz = sum(d.position.z for d in alive) / n
        self.centroid = Vec3(cx, cy, cz)
        self.dispersion_m = (sum(d.position.distance_to(self.centroid) ** 2 for d in alive) / n) ** 0.5


# ---------------------------------------------------------------------------
# MultiHorizonPlanner
# ---------------------------------------------------------------------------


class MultiHorizonPlanner:
    """t-CoT (temporal Chain of Thought) planner.

    Manages three parallel planning horizons and cascades constraints
    between them.

    H1 (tactical):    dt=0.1s,  100 particles -- obstacle avoidance
    H2 (operational): dt=5s,    500 particles -- formation
    H3 (strategic):   dt=60s,  2000 particles -- mission planning

    The cascade logic:

    1. Strategic plan sets mission phases, resource budgets, and the
       operating regime.
    2. Operational plan converts phase objectives into formation
       geometry and coordination targets, respecting strategic budgets.
    3. Tactical plan generates immediate waypoints that avoid obstacles
       while tracking the operational formation target.

    Bottom-up feedback:

    - If a tactical plan is infeasible (terrain, threat), it raises a
      *veto* that forces the operational plan to re-route.
    - If the operational plan cannot satisfy the strategic phase
      objective (insufficient drones, energy), it flags the strategic
      plan for re-evaluation.
    """

    def __init__(self, configs: dict[Horizon, HorizonConfig] | None = None) -> None:
        self._configs = configs or dict(DEFAULT_CONFIGS)
        self._strategic: Optional[StrategicPlan] = None
        self._operational: Optional[OperationalPlan] = None
        self._tactical: dict[int, TacticalPlan] = {}
        self._last_plan_time: dict[Horizon, float] = {h: 0.0 for h in Horizon}
        logger.info("MultiHorizonPlanner initialized (3 horizons)")

    # -- Public planning API ------------------------------------------------

    def plan_tactical(self, state: FleetState, dt: float = 0.1) -> dict[int, TacticalPlan]:
        """Plan at Horizon 1: obstacle avoidance for each drone.

        Runs at 10 Hz.  For each alive drone, generates a short trajectory
        that avoids obstacles and maintains minimum separation.

        Parameters
        ----------
        state : FleetState
            Current fleet state.
        dt : float
            Time step (default 0.1s).

        Returns
        -------
        dict[int, TacticalPlan]
            Per-drone tactical plans keyed by drone_id.
        """
        cfg = self._configs[Horizon.TACTICAL]
        now = time.monotonic()
        plans: dict[int, TacticalPlan] = {}

        alive_drones = [d for d in state.drones if d.alive]

        for drone in alive_drones:
            waypoints = self._generate_tactical_waypoints(drone, state, cfg)
            clearance = self._compute_waypoint_clearance(waypoints, state.obstacles)
            separation = self._compute_rollout_min_separation(drone, waypoints, alive_drones)

            veto_reasons = []
            if clearance <= 2.0:
                veto_reasons.append(f"obstacle clearance {clearance:.1f}m < 2m")
            if separation <= 3.0:
                veto_reasons.append(f"separation {separation:.1f}m < 3m")

            plan = TacticalPlan(
                drone_id=drone.drone_id,
                waypoints=waypoints,
                obstacle_clearance_m=clearance,
                min_separation_m=separation,
                valid=clearance > 2.0 and separation > 3.0 and len(waypoints) > 0,
                veto_reason="; ".join(veto_reasons),
                timestamp=now,
            )
            plans[drone.drone_id] = plan

        self._tactical = plans
        self._last_plan_time[Horizon.TACTICAL] = now
        return plans

    def plan_operational(self, state: FleetState, dt: float = 5.0) -> OperationalPlan:
        """Plan at Horizon 2: formation geometry and coordination.

        Runs at 0.2 Hz.  Computes optimal formation for the current
        mission phase, assigns formation slots, and sets cruise speed.

        Parameters
        ----------
        state : FleetState
            Current fleet state.
        dt : float
            Time step (default 5s).

        Returns
        -------
        OperationalPlan
            Formation and coordination plan.
        """
        cfg = self._configs[Horizon.OPERATIONAL]
        now = time.monotonic()
        alive = [d for d in state.drones if d.alive]
        n = len(alive)

        # Generate formation slots
        slots = self._generate_formation(alive, state)

        # Compute sensor coverage
        coverage = self._estimate_sensor_coverage(alive, state)

        plan = OperationalPlan(
            formation=slots,
            formation_centroid=state.centroid,
            formation_heading_rad=self._estimate_formation_heading(alive, state),
            speed_ms=self._select_cruise_speed(state),
            sensor_coverage_pct=coverage,
            valid=n >= 2,
            timestamp=now,
        )

        self._operational = plan
        self._last_plan_time[Horizon.OPERATIONAL] = now
        return plan

    def plan_strategic(self, state: FleetState, dt: float = 60.0) -> StrategicPlan:
        """Plan at Horizon 3: mission phasing and resource allocation.

        Runs at ~1/120 Hz.  Determines mission phases, allocates drones
        to sub-swarms, sets operating regime, and computes withdrawal
        criteria.

        Parameters
        ----------
        state : FleetState
            Current fleet state.
        dt : float
            Time step (default 60s).

        Returns
        -------
        StrategicPlan
            High-level mission plan with phases and resource allocation.
        """
        cfg = self._configs[Horizon.STRATEGIC]
        now = time.monotonic()
        alive = [d for d in state.drones if d.alive]

        # Generate phases based on current situation
        phases = self._generate_phases(state)

        # Resource allocation: assign drones to phases
        allocation = {}
        drone_idx = 0
        for phase in phases:
            for _ in range(min(phase.drone_count, len(alive) - drone_idx)):
                if drone_idx < len(alive):
                    allocation[alive[drone_idx].drone_id] = phase.name
                    drone_idx += 1

        total_duration = sum(p.estimated_duration_s for p in phases)

        plan = StrategicPlan(
            phases=phases,
            total_duration_s=total_duration,
            resource_allocation=allocation,
            regime=state.regime,
            confidence=0.7 if len(alive) >= 4 else 0.4,
            timestamp=now,
        )

        self._strategic = plan
        self._last_plan_time[Horizon.STRATEGIC] = now
        return plan

    def cascade_plans(self) -> IntegratedPlan:
        """Integrate all three horizons into a coherent plan.

        Performs top-down constraint propagation and bottom-up veto
        checking.  Returns an IntegratedPlan with a coherence flag.
        """
        now = time.monotonic()

        # Check for tactical vetoes
        vetoed = [p for p in self._tactical.values() if not p.valid]
        coherent = len(vetoed) == 0
        reason = ""

        if vetoed:
            ids = [p.drone_id for p in vetoed]
            reasons = [p.veto_reason for p in vetoed]
            reason = f"Tactical vetoes from drones {ids}: {reasons}"
            logger.warning("Cascade incoherence: %s", reason)

        return IntegratedPlan(
            tactical=dict(self._tactical),
            operational=self._operational,
            strategic=self._strategic,
            coherent=coherent,
            incoherence_reason=reason,
            cascade_timestamp=now,
        )

    def needs_replan(self, horizon: Horizon) -> bool:
        """Check if a horizon needs replanning based on its interval."""
        now = time.monotonic()
        interval = self._configs[horizon].replan_interval_s
        return (now - self._last_plan_time[horizon]) >= interval

    # -- Private helpers -----------------------------------------------------

    @staticmethod
    def _generate_tactical_waypoints(
        drone: DroneSnapshot,
        state: FleetState,
        cfg: HorizonConfig,
    ) -> list[Waypoint]:
        """Generate a short trajectory via a small heading/speed lattice search."""
        steps = min(cfg.steps, 20)
        if steps <= 0:
            return []

        preferred_direction = MultiHorizonPlanner._preferred_direction(drone, state)
        nominal_speed = max(drone.velocity.norm(), 6.0)
        heading_offsets = (
            0.0,
            math.radians(15.0),
            math.radians(-15.0),
            math.radians(30.0),
            math.radians(-30.0),
            math.radians(45.0),
            math.radians(-45.0),
            math.radians(60.0),
            math.radians(-60.0),
            math.radians(75.0),
            math.radians(-75.0),
        )
        speed_scales = (0.55, 0.85, 1.0, 1.15)

        best_waypoints: list[Waypoint] = []
        best_score = float("-inf")

        for heading_offset in heading_offsets:
            heading_direction = MultiHorizonPlanner._rotate_xy(preferred_direction, heading_offset)
            for speed_scale in speed_scales:
                speed_ms = max(3.0, nominal_speed * speed_scale)
                # Use APF if obstacles/threats exist, else constant velocity
                obs_list = [(o[0], o[1]) for o in getattr(state, 'obstacles', [])]
                thr_list = [(t[0], t[1]) for t in getattr(state, 'threats', [])]
                if obs_list or thr_list:
                    candidate = MultiHorizonPlanner._rollout_apf(
                        drone=drone,
                        direction=heading_direction,
                        speed_ms=speed_ms,
                        cfg=cfg,
                        steps=steps,
                        obstacles=obs_list,
                        threats=thr_list,
                    )
                else:
                    candidate = MultiHorizonPlanner._rollout_constant_velocity(
                        drone=drone,
                        direction=heading_direction,
                        speed_ms=speed_ms,
                        cfg=cfg,
                        steps=steps,
                    )
                score = MultiHorizonPlanner._score_tactical_rollout(
                    drone=drone,
                    state=state,
                    waypoints=candidate,
                    preferred_direction=preferred_direction,
                    heading_offset_rad=heading_offset,
                    speed_ms=speed_ms,
                    nominal_speed_ms=nominal_speed,
                )
                if score > best_score:
                    best_score = score
                    best_waypoints = candidate

        return best_waypoints

    @staticmethod
    def _rollout_constant_velocity(
        drone: DroneSnapshot,
        direction: Vec3,
        speed_ms: float,
        cfg: HorizonConfig,
        steps: int,
    ) -> list[Waypoint]:
        """Roll out a constant-velocity candidate over the short tactical horizon."""
        heading = math.atan2(direction.y, direction.x)
        velocity = Vec3(direction.x * speed_ms, direction.y * speed_ms, drone.velocity.z)
        waypoints: list[Waypoint] = []

        for step in range(steps):
            t = (step + 1) * cfg.dt
            waypoints.append(
                Waypoint(
                    position=Vec3(
                        drone.position.x + velocity.x * t,
                        drone.position.y + velocity.y * t,
                        drone.position.z + velocity.z * t,
                    ),
                    velocity=Vec3(velocity.x, velocity.y, velocity.z),
                    time_s=t,
                    heading_rad=heading,
                )
            )

        return waypoints

    @staticmethod
    def _rollout_apf(
        drone: DroneSnapshot,
        direction: Vec3,
        speed_ms: float,
        cfg: HorizonConfig,
        steps: int,
        obstacles: list[tuple],
        threats: list[tuple],
    ) -> list[Waypoint]:
        """APF-guided rollout: attractive toward goal + repulsive from obstacles/threats."""
        K_REP_OBS = 200.0
        D0_OBS = 20.0
        K_REP_THREAT = 150.0
        D0_THREAT = 60.0

        px, py, pz = drone.position.x, drone.position.y, drone.position.z
        waypoints: list[Waypoint] = []

        for step in range(steps):
            # Attractive force toward goal direction
            fx, fy, fz = direction.x * speed_ms, direction.y * speed_ms, 0.0

            # Repulsive forces from obstacles (position, radius)
            for obs_pos, obs_r in obstacles:
                dx, dy = px - obs_pos.x, py - obs_pos.y
                raw_dist = (dx * dx + dy * dy) ** 0.5
                dist = max(raw_dist - obs_r, 0.1)
                if dist < D0_OBS:
                    rep_mag = K_REP_OBS * (1.0 / dist - 1.0 / D0_OBS) / (dist * dist)
                    norm = max(raw_dist, 0.1)
                    rx, ry = dx / norm, dy / norm
                    # Break collinearity: add perpendicular if nearly head-on
                    cross = abs(rx * direction.y - ry * direction.x)
                    if cross < 0.15:
                        rx, ry = -ry, rx  # rotate 90 degrees
                    fx += rep_mag * rx
                    fy += rep_mag * ry

            # Repulsive forces from threats (position, radius)
            for thr_pos, thr_r in threats:
                dx, dy = px - thr_pos.x, py - thr_pos.y
                dist = max((dx * dx + dy * dy) ** 0.5 - thr_r, 0.1)
                if dist < D0_THREAT:
                    rep_mag = K_REP_THREAT * (1.0 / dist - 1.0 / D0_THREAT) / (dist * dist)
                    norm = max((dx * dx + dy * dy) ** 0.5, 0.1)
                    fx += rep_mag * dx / norm
                    fy += rep_mag * dy / norm

            # Normalize to speed_ms
            f_norm = max((fx * fx + fy * fy + fz * fz) ** 0.5, 1e-6)
            vx = fx / f_norm * speed_ms
            vy = fy / f_norm * speed_ms
            vz = drone.velocity.z  # keep altitude velocity

            px += vx * cfg.dt
            py += vy * cfg.dt
            pz += vz * cfg.dt

            waypoints.append(Waypoint(
                position=Vec3(px, py, pz),
                velocity=Vec3(vx, vy, vz),
                time_s=(step + 1) * cfg.dt,
                heading_rad=math.atan2(vy, vx),
            ))

        return waypoints

    @staticmethod
    def _score_tactical_rollout(
        drone: DroneSnapshot,
        state: FleetState,
        waypoints: list[Waypoint],
        preferred_direction: Vec3,
        heading_offset_rad: float,
        speed_ms: float,
        nominal_speed_ms: float,
    ) -> float:
        """Score a candidate rollout by progress, safety margin, and exposure."""
        if not waypoints:
            return float("-inf")

        clearance = MultiHorizonPlanner._compute_waypoint_clearance(waypoints, state.obstacles)
        alive_drones = [d for d in state.drones if d.alive]
        separation = MultiHorizonPlanner._compute_rollout_min_separation(drone, waypoints, alive_drones)
        threat_exposure = MultiHorizonPlanner._path_threat_exposure(waypoints, state.threats)
        last = waypoints[-1].position
        progress = (
            (last.x - drone.position.x) * preferred_direction.x
            + (last.y - drone.position.y) * preferred_direction.y
        )

        clearance_bonus = 25.0 if math.isinf(clearance) else max(-8.0, min(clearance, 25.0))
        separation_bonus = 20.0 if math.isinf(separation) else max(-8.0, min(separation - 3.0, 20.0))
        clearance_shortfall = 0.0 if math.isinf(clearance) else max(0.0, 2.0 - clearance)
        separation_shortfall = 0.0 if math.isinf(separation) else max(0.0, 3.0 - separation)
        unsafe_penalty = 60.0 if not math.isinf(clearance) and clearance < 2.0 else 0.0
        deconflict_penalty = 40.0 if not math.isinf(separation) and separation < 3.0 else 0.0
        threat_weight = 45.0 if state.regime == RegimeLabel.EVADE else 25.0
        turn_penalty = abs(heading_offset_rad) * 6.0
        speed_penalty = abs(speed_ms - nominal_speed_ms) * 0.25

        return (
            progress
            + 1.8 * clearance_bonus
            + 0.9 * separation_bonus
            - threat_weight * threat_exposure
            - turn_penalty
            - speed_penalty
            - unsafe_penalty
            - deconflict_penalty
            - 40.0 * clearance_shortfall
            - 25.0 * separation_shortfall
        )

    @staticmethod
    def _compute_waypoint_clearance(waypoints: list[Waypoint], obstacles: list[tuple[Vec3, float]]) -> float:
        """Minimum obstacle clearance across the full candidate trajectory."""
        if not obstacles:
            return float("inf")
        min_clearance = float("inf")
        for wp in waypoints:
            for center, radius in obstacles:
                dist = wp.position.distance_to(center) - radius
                min_clearance = min(min_clearance, dist)
        return min_clearance

    @staticmethod
    def _compute_rollout_min_separation(
        drone: DroneSnapshot,
        waypoints: list[Waypoint],
        all_drones: list[DroneSnapshot],
    ) -> float:
        """Minimum predicted separation from the rest of the current fleet snapshot."""
        others = [other for other in all_drones if other.drone_id != drone.drone_id and other.alive]
        if not others:
            return float("inf")

        min_sep = float("inf")
        for wp in waypoints:
            for other in others:
                dist = wp.position.distance_to(other.position)
                min_sep = min(min_sep, dist)
        return min_sep

    @staticmethod
    def _path_threat_exposure(waypoints: list[Waypoint], threats: list[Vec3]) -> float:
        """Average normalized exposure to nearby threats along a candidate path."""
        if not waypoints or not threats:
            return 0.0

        total_exposure = 0.0
        for wp in waypoints:
            step_exposure = 0.0
            for threat in threats:
                dist = wp.position.distance_to(threat)
                if dist < 120.0:
                    step_exposure = max(step_exposure, (120.0 - dist) / 120.0)
            total_exposure += step_exposure
        return total_exposure / len(waypoints)

    @staticmethod
    def _preferred_direction(drone: DroneSnapshot, state: FleetState) -> Vec3:
        """Preferred horizontal direction before tactical safety corrections."""
        horizontal_velocity = Vec3(drone.velocity.x, drone.velocity.y, 0.0)
        if horizontal_velocity.norm() > 1.0:
            return MultiHorizonPlanner._normalize_xy(horizontal_velocity)

        if state.threats and state.regime == RegimeLabel.EVADE:
            nearest = min(state.threats, key=lambda threat: drone.position.distance_to(threat))
            return MultiHorizonPlanner._normalize_xy(
                Vec3(drone.position.x - nearest.x, drone.position.y - nearest.y, 0.0)
            )

        if state.threats and state.regime == RegimeLabel.ENGAGE:
            nearest = min(state.threats, key=lambda threat: drone.position.distance_to(threat))
            return MultiHorizonPlanner._normalize_xy(
                Vec3(nearest.x - drone.position.x, nearest.y - drone.position.y, 0.0)
            )

        to_centroid = Vec3(state.centroid.x - drone.position.x, state.centroid.y - drone.position.y, 0.0)
        if to_centroid.norm() > 1.0:
            return MultiHorizonPlanner._normalize_xy(to_centroid)

        return Vec3(1.0, 0.0, 0.0)

    @staticmethod
    def _normalize_xy(vec: Vec3) -> Vec3:
        """Normalize only the horizontal component of a vector."""
        magnitude = (vec.x**2 + vec.y**2) ** 0.5
        if magnitude <= 1e-6:
            return Vec3(1.0, 0.0, 0.0)
        return Vec3(vec.x / magnitude, vec.y / magnitude, 0.0)

    @staticmethod
    def _rotate_xy(direction: Vec3, angle_rad: float) -> Vec3:
        """Rotate a horizontal direction vector by the given angle."""
        base = MultiHorizonPlanner._normalize_xy(direction)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        return Vec3(
            base.x * cos_a - base.y * sin_a,
            base.x * sin_a + base.y * cos_a,
            0.0,
        )

    @staticmethod
    def _compute_obstacle_clearance(drone: DroneSnapshot, obstacles: list[tuple[Vec3, float]]) -> float:
        """Minimum distance to any obstacle minus its radius."""
        if not obstacles:
            return float("inf")
        min_clearance = float("inf")
        for center, radius in obstacles:
            dist = drone.position.distance_to(center) - radius
            min_clearance = min(min_clearance, dist)
        return min_clearance

    @staticmethod
    def _compute_min_separation(drone: DroneSnapshot, all_drones: list[DroneSnapshot]) -> float:
        """Minimum distance to any other drone."""
        min_sep = float("inf")
        for other in all_drones:
            if other.drone_id != drone.drone_id:
                dist = drone.position.distance_to(other.position)
                min_sep = min(min_sep, dist)
        return min_sep

    @staticmethod
    def _generate_formation(drones: list[DroneSnapshot], state: FleetState) -> list[FormationSlot]:
        """Generate a V-formation or line-abreast based on drone count."""
        n = len(drones)
        slots = []
        spacing = 20.0  # meters

        for i, drone in enumerate(drones):
            if i == 0:
                role = "leader"
                offset = Vec3(0, 0, 0)
            else:
                role = "wingman"
                side = 1 if i % 2 == 1 else -1
                rank = (i + 1) // 2
                # V-formation: each rank further back and to the side
                offset = Vec3(
                    -rank * spacing * 0.7,  # behind leader
                    side * rank * spacing,   # to the side
                    0,
                )
            slots.append(FormationSlot(drone_id=drone.drone_id, offset=offset, role=role))

        return slots

    @staticmethod
    def _estimate_formation_heading(drones: list[DroneSnapshot], state: FleetState) -> float:
        """Estimate formation heading from the fleet's aggregate motion."""
        if drones:
            mean_vx = sum(drone.velocity.x for drone in drones) / len(drones)
            mean_vy = sum(drone.velocity.y for drone in drones) / len(drones)
            if abs(mean_vx) > 1e-6 or abs(mean_vy) > 1e-6:
                return math.atan2(mean_vy, mean_vx)

        if state.threats:
            nearest = min(state.threats, key=lambda threat: state.centroid.distance_to(threat))
            return math.atan2(nearest.y - state.centroid.y, nearest.x - state.centroid.x)

        return 0.0

    @staticmethod
    def _estimate_sensor_coverage(drones: list[DroneSnapshot], state: FleetState) -> float:
        """Rough estimate of sensor area coverage (0-100%)."""
        if not drones:
            return 0.0
        # Each drone covers a ~100m radius with its sensor
        sensor_radius = 100.0
        individual_area = 3.14159 * sensor_radius**2
        total_sensor_area = len(drones) * individual_area * 0.7  # 30% overlap assumed

        # Compare against the mission area (approximate from dispersion)
        mission_radius = max(state.dispersion_m * 2, 200.0)
        mission_area = 3.14159 * mission_radius**2

        return min(100.0, (total_sensor_area / max(mission_area, 1.0)) * 100.0)

    @staticmethod
    def _select_cruise_speed(state: FleetState) -> float:
        """Select cruise speed based on regime."""
        speeds = {
            RegimeLabel.PATROL: 10.0,
            RegimeLabel.ENGAGE: 20.0,
            RegimeLabel.EVADE: 25.0,
        }
        return speeds.get(state.regime, 15.0)

    @staticmethod
    def _generate_phases(state: FleetState) -> list[PhaseObjective]:
        """Generate mission phases based on current situation."""
        alive = sum(1 for d in state.drones if d.alive)
        has_threats = len(state.threats) > 0

        phases = []

        if has_threats:
            phases.append(
                PhaseObjective(
                    name="threat_assessment",
                    description="Assess and classify detected threats",
                    mission_type=MissionType.RECON,
                    estimated_duration_s=120.0,
                    drone_count=min(2, alive),
                    priority=0.9,
                )
            )
            phases.append(
                PhaseObjective(
                    name="engagement",
                    description="Engage or suppress identified threats",
                    mission_type=MissionType.STRIKE,
                    estimated_duration_s=300.0,
                    drone_count=min(4, alive),
                    priority=0.8,
                )
            )
        else:
            phases.append(
                PhaseObjective(
                    name="area_recon",
                    description="Systematic area reconnaissance",
                    mission_type=MissionType.RECON,
                    estimated_duration_s=600.0,
                    drone_count=min(4, alive),
                    priority=0.5,
                )
            )

        phases.append(
            PhaseObjective(
                name="overwatch",
                description="Maintain overwatch and relay",
                mission_type=MissionType.PATROL,
                estimated_duration_s=300.0,
                drone_count=min(2, alive),
                priority=0.3,
            )
        )

        return phases


def tactical_to_decisions(
    plans: dict[int, "TacticalPlan"],
) -> list:
    """Convert tactical plans to Decision objects for the mission brain.

    Valid plans → GOTO decisions targeting first waypoint.
    Invalid plans → LOITER decisions with veto reason.
    """
    from strix.brain import Decision, DecisionKind

    decisions = []
    for drone_id, plan in plans.items():
        if plan.valid and plan.waypoints:
            wp = plan.waypoints[0]
            vel_norm = (wp.velocity.x ** 2 + wp.velocity.y ** 2 + wp.velocity.z ** 2) ** 0.5
            decisions.append(Decision(
                drone_id=drone_id,
                kind=DecisionKind.GOTO,
                target_position=wp.position,
                speed_ms=max(vel_norm, 1.0),
                reason=f"tactical H1 waypoint (clearance={plan.obstacle_clearance_m:.1f}m)",
                confidence=1.0,
            ))
        else:
            decisions.append(Decision(
                drone_id=drone_id,
                kind=DecisionKind.LOITER,
                target_position=None,
                speed_ms=0.0,
                reason=plan.veto_reason or "tactical plan invalid",
                confidence=0.3,
            ))
    return decisions
