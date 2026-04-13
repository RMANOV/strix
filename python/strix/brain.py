# SPDX-License-Identifier: Apache-2.0

"""Mission Brain -- orchestration loop for the public autonomy core.

Layer 1 of the STRIX architecture.  Orchestrates the particle filter loop,
regime management, combinatorial auctions, and mesh coordination to convert
high-level human intent into executable platform assignments.

The main loop mirrors a quantitative trading engine:

    predict -> update -> regime_check -> auction -> assign

Each tick runs this pipeline at 10 Hz, producing a list of Decisions that the
adapter layer (`strix-adapters`) translates into platform commands.
"""

# Coordinate Convention — NED (North-East-Down)
# All positions and velocities in this module use the NED frame:
#   X = North (positive = North)
#   Y = East  (positive = East)
#   Z = Down  (positive = down; negative Z = altitude above ground)
# Example: a drone at 50 m altitude has position.z = -50.0

from __future__ import annotations

import asyncio
import dataclasses
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from strix.orientation import OrientationEngine, OrientationSnapshot

logger = logging.getLogger("strix.brain")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BrainConfig:
    """Tunable parameters for the MissionBrain."""

    tick_hz: float = 10.0
    """Main loop frequency (Hz)."""

    n_particles: int = 1000
    """Particle count for the friendly-side navigation filter."""

    n_enemy_particles: int = 500
    """Particle count for the contact-side filter."""

    regime_transition_matrix: list[list[float]] = field(
        default_factory=lambda: [
            [0.90, 0.07, 0.03],
            [0.10, 0.80, 0.10],
            [0.15, 0.10, 0.75],
        ]
    )
    """3x3 Markov transition matrix: [PATROL, ENGAGE, EVADE]."""

    auction_interval_ticks: int = 5
    """Re-run the task auction every N ticks."""

    ess_resample_threshold: float = 0.5
    """Effective-sample-size fraction below which resampling is triggered."""

    max_fleet_size: int = 256
    """Hard cap on fleet members (governs pre-allocated arrays)."""

    default_packet_success_rate: float = 0.85
    """Nominal packet success used for planning validation under degraded comms."""

    stale_state_age_s: float = 0.25
    """Nominal neighbor-state age used for planner validation."""

    regime_hysteresis_margin: float = 0.15
    """Minimum evidence gap required to change regimes."""


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class MissionType(Enum):
    """High-level mission categories."""

    RECON = 1
    STRIKE = 2
    INTERDICT = 2
    ESCORT = 3
    DEFEND = 4
    SECURE = 4
    PATROL = 5
    RELAY = 6


class RegimeLabel(Enum):
    """Battlespace operating regimes, mirroring the Rust `Regime` enum."""

    PATROL = 0
    ENGAGE = 1
    TRACK = 1
    EVADE = 2
    AVOID = 2


class DecisionKind(Enum):
    """Atomic decision types emitted by the brain each tick."""

    GOTO = 1
    LOITER = 2
    ENGAGE_TARGET = 3
    ACT_ON_TARGET = 3
    EVADE = 4
    RETURN_TO_BASE = 5
    RELAY_STATION = 6
    LAND = 7


_THREAT_TYPE_ALIASES: dict[str, str] = {
    "area_denial": "sam",
    "air_denial": "sam",
    "local_hazard": "small_arms",
    "interference": "electronic_warfare",
    "ew": "electronic_warfare",
    "jamming": "electronic_warfare",
}


def _normalize_threat_type_label(label: str | None) -> str:
    """Map neutral aliases onto the canonical threat labels used internally."""

    if not label:
        return "unknown"

    head, sep, tail = label.strip().partition(":")
    normalized_head = head.lower().replace("-", "_").replace(" ", "_")
    normalized_head = _THREAT_TYPE_ALIASES.get(normalized_head, normalized_head)
    if not sep:
        return normalized_head
    return f"{normalized_head}:{tail}"


@dataclass
class Vec3:
    """Minimal 3-D vector used across the Python layer."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def distance_to(self, other: Vec3) -> float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2) ** 0.5

    def norm(self) -> float:
        return (self.x**2 + self.y**2 + self.z**2) ** 0.5


@dataclass
class MissionArea:
    """Geographic bounding region for a mission."""

    center: Vec3 = field(default_factory=Vec3)
    radius_m: float = 500.0
    min_alt_m: float = 30.0
    max_alt_m: float = 120.0


@dataclass
class Constraint:
    """A constraint on mission execution."""

    name: str = ""
    description: str = ""
    area: Optional[MissionArea] = None
    avoid: bool = False


@dataclass
class MissionIntent:
    """Structured representation of what the operator wants."""

    mission_type: MissionType = MissionType.RECON
    area: Optional[MissionArea] = None
    constraints: list[Constraint] = field(default_factory=list)
    deadline_s: float = 600.0
    priority: float = 0.7
    drone_count: Optional[int] = None
    description: str = ""


@dataclass
class DroneSnapshot:
    """Lightweight snapshot of one drone's state, used within the brain."""

    drone_id: int = 0
    position: Vec3 = field(default_factory=Vec3)
    velocity: Vec3 = field(default_factory=Vec3)
    regime: RegimeLabel = RegimeLabel.PATROL
    energy: float = 1.0
    alive: bool = True
    capabilities: int = 0
    timestamp_s: float = 0.0


@dataclass
class ThreatObservation:
    """An observation about an external contact."""

    threat_id: int = 0
    position: Vec3 = field(default_factory=Vec3)
    velocity: Vec3 = field(default_factory=Vec3)
    confidence: float = 0.5
    threat_type: str = "unknown"
    timestamp: float = 0.0


@dataclass
class CommsState:
    """Observed communications quality for one drone or the fleet as a whole."""

    packet_success_rate: float = 1.0
    state_age_s: float = 0.0
    latency_ms: float = 0.0


class _FallbackLinkQualityTracker:
    """Small sliding-window link tracker used when Rust comms FFI is unavailable."""

    def __init__(self, window_size: int = 16) -> None:
        self._window_size = max(window_size, 1)
        self._history: dict[int, list[bool]] = {}

    def record_success(self, peer_id: int) -> None:
        self._push(peer_id, True)

    def record_failure(self, peer_id: int) -> None:
        self._push(peer_id, False)

    def quality(self, peer_id: int) -> float:
        window = self._history.get(peer_id)
        if not window:
            return 0.5
        return sum(1.0 for sample in window if sample) / len(window)

    def remove(self, peer_id: int) -> None:
        self._history.pop(peer_id, None)

    def _push(self, peer_id: int, success: bool) -> None:
        history = self._history.setdefault(peer_id, [])
        history.append(success)
        if len(history) > self._window_size:
            del history[0 : len(history) - self._window_size]


@dataclass
class Decision:
    """A single atomic decision produced by a brain tick."""

    drone_id: int = 0
    task_id: Optional[int] = None
    kind: DecisionKind = DecisionKind.LOITER
    target_position: Optional[Vec3] = None
    speed_ms: float = 10.0
    reason: str = ""
    confidence: float = 1.0


@dataclass
class MissionPlan:
    """A complete plan produced from a MissionIntent."""

    intent: MissionIntent = field(default_factory=MissionIntent)
    assignments: list[Decision] = field(default_factory=list)
    estimated_duration_s: float = 0.0
    confidence: float = 0.0
    regime: RegimeLabel = RegimeLabel.PATROL
    explanation: str = ""


# ---------------------------------------------------------------------------
# MissionBrain
# ---------------------------------------------------------------------------


class MissionBrain:
    """Layer 1: Market Brain -- Mission Planner.

    Orchestrates the Rust core functions to run the particle filter loop,
    manage regimes, run auctions, and coordinate the mesh.

    The brain maintains three internal models:

    1. **Fleet State**: particle-filter estimates for every friendly drone.
    2. **Threat State**: adversarial particle-filter estimates for every
       detected enemy entity.
    3. **Regime State**: current Markov regime for the entire battlespace
       (PATROL / ENGAGE / EVADE), updated from sensor evidence.

    Each ``tick()`` call runs:
        predict -> update -> regime_check -> auction -> assign
    and returns the resulting list of :class:`Decision` objects.
    """

    def __init__(self, config: BrainConfig | None = None) -> None:
        self.config = config or BrainConfig()
        self._fleet: dict[int, DroneSnapshot] = {}
        self._threats: dict[int, ThreatObservation] = {}
        self._regime = RegimeLabel.PATROL
        self._tick_count = 0
        self._active_plan: MissionPlan | None = None
        self._pending_intents: deque[MissionIntent] = deque()
        self._running = False
        self._last_auction_tick = 0
        self._next_task_id = 0
        self._network_comms = CommsState(
            packet_success_rate=self.config.default_packet_success_rate,
            state_age_s=self.config.stale_state_age_s,
        )
        self._drone_comms: dict[int, CommsState] = {}
        self._last_arrival_s: dict[int, float] = {}
        self._stale_windows: dict[int, int] = {}
        self._runtime_comms_active = False
        self._LinkQualityTracker = _FallbackLinkQualityTracker

        # Rust FFI: particle filters + auctioneer (graceful fallback)
        self._filters: dict[int, object] = {}
        self._rust_available = False
        self._auctioneer = None
        try:
            from strix._strix_core import Auctioneer, ParticleNavFilter  # noqa: F401

            self._ParticleNavFilter = ParticleNavFilter
            self._auctioneer = Auctioneer()
            self._rust_available = True
            logger.info("MissionBrain: Rust FFI available")
        except ImportError:
            self._ParticleNavFilter = None
            logger.warning("MissionBrain: Rust FFI unavailable, using Python fallbacks")
        else:
            try:
                from strix._strix_core import LinkQualityTracker  # noqa: F401

                self._LinkQualityTracker = LinkQualityTracker
            except ImportError:
                pass
        self._link_tracker = self._LinkQualityTracker(max(int(self.config.tick_hz * 2), 8))
        self._orientation = OrientationEngine()

        logger.info("MissionBrain initialised (particles=%d)", self.config.n_particles)

    # -- Fleet management ----------------------------------------------------

    def register_drone(self, drone: DroneSnapshot) -> None:
        """Register or update a drone in the fleet."""
        previous = self._fleet.get(drone.drone_id)
        if len(self._fleet) >= self.config.max_fleet_size and drone.drone_id not in self._fleet:
            raise ValueError(f"Fleet full ({self.config.max_fleet_size} drones)")
        self._fleet[drone.drone_id] = drone
        self._drone_comms.setdefault(
            drone.drone_id,
            CommsState(
                packet_success_rate=self._network_comms.packet_success_rate,
                state_age_s=self._network_comms.state_age_s,
                latency_ms=self._network_comms.latency_ms,
            ),
        )
        if previous is not None or drone.timestamp_s > 0.0:
            self._runtime_comms_active = True
            self._link_tracker.record_success(drone.drone_id)
            self._stale_windows[drone.drone_id] = 0
        self._last_arrival_s[drone.drone_id] = time.monotonic()
        self._refresh_runtime_comms()

        if not self._rust_available:
            return

        needs_filter_reset = drone.drone_id not in self._filters
        if previous is not None:
            position_jumped = previous.position.distance_to(drone.position) >= 5.0
            if position_jumped or previous.alive != drone.alive:
                needs_filter_reset = True

        if needs_filter_reset:
            self._filters[drone.drone_id] = self._ParticleNavFilter(
                n_particles=self.config.n_particles,
                position=[drone.position.x, drone.position.y, drone.position.z],
                drone_id=drone.drone_id,
            )

    def remove_drone(self, drone_id: int) -> None:
        """Mark a drone as lost and remove it from active fleet."""
        self._fleet.pop(drone_id, None)
        self._filters.pop(drone_id, None)
        self._drone_comms.pop(drone_id, None)
        self._last_arrival_s.pop(drone_id, None)
        self._stale_windows.pop(drone_id, None)
        self._link_tracker.remove(drone_id)

    @property
    def fleet_size(self) -> int:
        return sum(1 for d in self._fleet.values() if d.alive)

    def update_network_state(
        self,
        packet_success_rate: float,
        state_age_s: float | None = None,
        latency_ms: float | None = None,
    ) -> None:
        """Update fleet-wide communications health from an external source."""
        self._network_comms = CommsState(
            packet_success_rate=max(0.0, min(packet_success_rate, 1.0)),
            state_age_s=self.config.stale_state_age_s if state_age_s is None else max(0.0, state_age_s),
            latency_ms=0.0 if latency_ms is None else max(0.0, latency_ms),
        )

    def update_link_state(
        self,
        drone_id: int,
        packet_success_rate: float,
        state_age_s: float | None = None,
        latency_ms: float | None = None,
    ) -> None:
        """Update observed link quality for one drone."""
        self._drone_comms[drone_id] = CommsState(
            packet_success_rate=max(0.0, min(packet_success_rate, 1.0)),
            state_age_s=self._network_comms.state_age_s if state_age_s is None else max(0.0, state_age_s),
            latency_ms=self._network_comms.latency_ms if latency_ms is None else max(0.0, latency_ms),
        )

    # -- Intent pipeline -----------------------------------------------------

    def process_intent_sync(self, intent: MissionIntent) -> MissionPlan:
        """Convert a high-level intent into a concrete mission plan.

        Steps:
            1. Validate intent feasibility (enough drones, within range).
            2. Compute optimal allocation via combinatorial auction.
            3. Generate waypoints and assignments.
            4. Package into a MissionPlan for operator review.
        """
        logger.info("Processing intent: %s over %s", intent.mission_type.name, intent.area)
        self._refresh_runtime_comms()

        # Determine how many drones to allocate
        available = [d for d in self._fleet.values() if d.alive]
        requested = intent.drone_count or self._auto_drone_count(intent, len(available))
        requested = min(requested, len(available))

        if requested == 0:
            return MissionPlan(
                intent=intent,
                confidence=0.0,
                explanation="No drones available for tasking.",
            )

        # Run the auction: score each drone's fitness for this mission
        scored = self._score_drones_for_intent(available, intent)
        selected = scored[:requested]
        assignments, plan_confidence, plan_summary = self._build_assignments_for_intent(selected, intent)
        estimated_duration = self._estimate_duration(selected, intent)
        assignments = [self._with_task_id(assignment) for assignment in assignments]
        plan = MissionPlan(
            intent=intent,
            assignments=assignments,
            estimated_duration_s=estimated_duration,
            confidence=plan_confidence if len(selected) >= requested else max(0.35, plan_confidence * 0.75),
            regime=self._regime,
            explanation=(
                f"Allocated {len(selected)}/{requested} drones. "
                f"Current regime: {self._regime.name}. "
                f"Estimated time: {estimated_duration:.0f}s. "
                f"{plan_summary}"
            ),
        )

        orientation = self._refresh_orientation(planner_confidence=plan.confidence)
        plan = self._apply_orientation_to_plan(plan, orientation)
        self._active_plan = plan
        return plan

    async def process_intent(self, intent: MissionIntent) -> MissionPlan:
        return self.process_intent_sync(intent)

    # -- Main loop -----------------------------------------------------------

    def tick_sync(self, dt: float) -> list[Decision]:
        """Execute one brain cycle: predict -> update -> regime -> auction -> assign.

        Parameters
        ----------
        dt : float
            Wall-clock time step in seconds since last tick.

        Returns
        -------
        list[Decision]
            Zero or more decisions for the adapter layer to execute.
        """
        self._tick_count += 1
        decisions: list[Decision] = []
        self._refresh_runtime_comms()

        # 1. PREDICT -- propagate particle filters forward
        self._predict_step(dt)

        # 2. UPDATE -- incorporate new sensor observations
        self._update_step()
        orientation = self._refresh_orientation(
            planner_confidence=self._active_plan.confidence if self._active_plan else None
        )
        if self._active_plan is not None:
            self._active_plan = self._apply_orientation_to_plan(self._active_plan, orientation)

        # 3. REGIME CHECK -- evaluate Markov transitions
        proposed_regime = self._check_regime()
        regime_name = self._orientation.recommend_regime(self._regime.name, proposed_regime.name)
        new_regime = RegimeLabel[regime_name]
        if new_regime != self._regime:
            logger.info("Regime transition: %s -> %s", self._regime.name, new_regime.name)
            self._regime = new_regime

        # 4. AUCTION -- periodically re-run task allocation
        ticks_since_auction = self._tick_count - self._last_auction_tick
        if self._should_run_auction(ticks_since_auction):
            self._run_auction()
            if self._active_plan is not None:
                self._active_plan = self._apply_orientation_to_plan(self._active_plan, orientation)
            self._last_auction_tick = self._tick_count

        # 5. ASSIGN -- generate per-drone decisions from active plan
        if self._active_plan:
            for assignment in self._active_plan.assignments:
                drone = self._fleet.get(assignment.drone_id)
                if drone and drone.alive:
                    decisions.append(assignment)

        return decisions

    async def tick(self, dt: float) -> list[Decision]:
        return self.tick_sync(dt)

    def handle_loss_sync(self, drone_id: int, cause: str) -> None:
        """React to the loss of a drone -- anti-fragile response.

        The swarm does NOT simply degrade.  It adapts:
        1. Mark the loss location as a kill zone (increased risk in bidding).
        2. Re-auction the lost drone's tasks.
        3. If attrition exceeds drawdown threshold, trigger regime shift to EVADE.
        """
        drone = self._fleet.get(drone_id)
        if drone is None:
            logger.warning("Loss reported for unknown drone %d", drone_id)
            return

        logger.warning("DRONE LOST: id=%d cause=%s pos=(%s)", drone_id, cause, drone.position)

        # Record kill zone (offset 0x8000_0000 avoids collision with real threat IDs)
        self._threats[0x8000_0000 + drone_id] = ThreatObservation(
            threat_id=0x8000_0000 + drone_id,
            position=drone.position,
            confidence=0.9,
            threat_type=f"kill_zone:{cause}",
            timestamp=time.monotonic(),
        )

        self._fleet[drone_id] = dataclasses.replace(drone, alive=False)
        self._orientation.register_broken_assumption(
            "friendly_loss",
            0.75,
            time.monotonic(),
            details=f"drone={drone_id} cause={cause}",
        )

        # Check attrition level
        total = len(self._fleet)
        alive = self.fleet_size
        attrition = 1.0 - (alive / max(total, 1))
        if attrition > 0.3:
            logger.critical("ATTRITION %.0f%% -- forcing EVADE regime", attrition * 100)
            self._regime = RegimeLabel.EVADE

        # Re-auction: remove dead drone's assignments
        if self._active_plan:
            self._active_plan.assignments = [a for a in self._active_plan.assignments if a.drone_id != drone_id]

        self._trigger_reauction()

    async def handle_loss(self, drone_id: int, cause: str) -> None:
        self.handle_loss_sync(drone_id, cause)

    def update_threat_sync(self, observation: ThreatObservation) -> None:
        """Incorporate a new threat observation into the adversarial filter."""
        normalized_type = _normalize_threat_type_label(observation.threat_type)
        if normalized_type != observation.threat_type:
            observation = dataclasses.replace(observation, threat_type=normalized_type)
        self._threats[observation.threat_id] = observation
        logger.info(
            "Threat updated: id=%d type=%s confidence=%.2f",
            observation.threat_id,
            observation.threat_type,
            observation.confidence,
        )
        source_label = observation.threat_type.split(":", 1)[0]
        self._orientation.observe_source(source_label, observation.confidence, time.monotonic())

        # High-confidence threat near the fleet -> consider regime shift
        if observation.confidence > 0.7:
            centroid = self._fleet_centroid()
            dist = centroid.distance_to(observation.position)
            if dist < 300.0:
                link_quality = self._planner_packet_success_rate()
                stale_age = self._network_state_age()
                if link_quality < 0.5 or stale_age > 1.0:
                    logger.warning(
                        "High-confidence threat at %.0fm under degraded comms "
                        "(link=%.2f stale=%.2fs) -- biasing to EVADE",
                        dist,
                        link_quality,
                        stale_age,
                    )
                    self._regime = RegimeLabel.EVADE
                else:
                    logger.warning("High-confidence threat at %.0fm -- shifting to ENGAGE", dist)
                    self._regime = RegimeLabel.ENGAGE
                self._trigger_reauction()

    async def update_threat(self, observation: ThreatObservation) -> None:
        self.update_threat_sync(observation)

    # -- Async runner --------------------------------------------------------

    async def run(self) -> None:
        """Run the brain main loop indefinitely at the configured tick rate."""
        self._running = True
        dt = 1.0 / self.config.tick_hz
        logger.info("Brain main loop started (%.0f Hz)", self.config.tick_hz)

        while self._running:
            t0 = time.monotonic()

            # Process at most one queued intent per tick
            if self._pending_intents:
                intent = self._pending_intents.popleft()
                self.process_intent_sync(intent)

            elapsed = time.monotonic() - t0
            actual_dt = max(elapsed, dt)  # at least nominal, never negative
            if elapsed > dt * 1.5:
                logger.warning("Tick overrun: elapsed=%.3fs nominal=%.3fs", elapsed, dt)

            decisions = self.tick_sync(actual_dt)
            if decisions:
                logger.debug("Tick %d: %d decisions", self._tick_count, len(decisions))

            # Rate-limit to target Hz
            elapsed = time.monotonic() - t0
            sleep_time = max(0.0, dt - elapsed)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

    def stop(self) -> None:
        """Signal the main loop to stop."""
        self._running = False
        logger.info("Brain stop requested")

    # -- Private helpers -----------------------------------------------------

    def _predict_step(self, dt: float) -> None:
        """Propagate particle filters forward via Rust FFI.

        Each filter.step() does predict+update+resample in one call.
        Falls back to naive kinematics if Rust is unavailable.
        """
        for drone in self._fleet.values():
            if not drone.alive:
                continue
            pf = self._filters.get(drone.drone_id) if self._rust_available else None
            if pf is not None:
                # NED: position.z is negative altitude, negate to get positive altitude for barometer
                observations = [("barometer", -drone.position.z)]
                tb = self._nearest_threat_bearing(drone)
                pos, vel, _regime_probs = pf.step(observations, tb, 1.0, dt)
                drone.position = Vec3(*pos)
                drone.velocity = Vec3(*vel)
            else:
                drone.position.x += drone.velocity.x * dt
                drone.position.y += drone.velocity.y * dt
                drone.position.z += drone.velocity.z * dt

    def _update_step(self) -> None:
        """No-op: predict already did the full predict+update cycle.

        The Rust ParticleNavFilter.step() combines predict, update, and
        resample in one call, so _predict_step already incorporates
        sensor observations.
        """

    def _check_regime(self) -> RegimeLabel:
        """Evaluate regime transitions from threat environment.

        Uses weighted threat pressure with hysteresis rather than a pure count
        of nearby contacts.  This reduces oscillation and makes adversarial
        motion toward the fleet matter more than a static contact count.
        """
        if not self._threats:
            return RegimeLabel.PATROL

        centroid = self._fleet_centroid()
        engage_pressure = 0.0
        evade_pressure = 0.0

        for threat in self._threats.values():
            distance = centroid.distance_to(threat.position)
            proximity = max(0.0, 1.0 - distance / 700.0)
            if proximity <= 0.0:
                continue

            approach_score = 0.0
            speed = threat.velocity.norm()
            if speed > 1e-6 and distance > 1e-6:
                to_centroid = Vec3(
                    centroid.x - threat.position.x,
                    centroid.y - threat.position.y,
                    centroid.z - threat.position.z,
                )
                approach_projection = (
                    threat.velocity.x * to_centroid.x
                    + threat.velocity.y * to_centroid.y
                    + threat.velocity.z * to_centroid.z
                ) / distance
                approach_score = max(0.0, min(approach_projection / 25.0, 1.0))

            threat_label = _normalize_threat_type_label(threat.threat_type)
            kill_zone_bias = 0.25 if threat_label.startswith("kill_zone:") else 0.0
            confidence = max(0.0, min(threat.confidence, 1.0))

            engage_pressure += confidence * (0.65 * proximity + 0.35 * approach_score)
            evade_pressure += confidence * (0.45 * proximity + 0.55 * approach_score + kill_zone_bias)

        if self._active_plan and self._active_plan.intent.mission_type == MissionType.STRIKE:
            engage_pressure += 0.1 * self._active_plan.intent.priority

        link_quality = self._planner_packet_success_rate()
        stale_age = self._network_state_age()
        if link_quality < 0.55 and (engage_pressure > 0.2 or evade_pressure > 0.2):
            evade_pressure += (0.55 - link_quality) * 1.6
        if stale_age > 0.75 and (engage_pressure > 0.2 or evade_pressure > 0.15):
            evade_pressure += min((stale_age - 0.75) * 0.35, 0.45)
        if link_quality < 0.45 and stale_age > 1.0 and engage_pressure > 0.25:
            evade_pressure += 0.35

        margin = self.config.regime_hysteresis_margin
        if self._regime == RegimeLabel.EVADE and evade_pressure >= 0.35:
            return RegimeLabel.EVADE
        if self._regime == RegimeLabel.ENGAGE and engage_pressure >= 0.25:
            return RegimeLabel.ENGAGE
        if evade_pressure >= 0.8 or (evade_pressure >= 0.6 and evade_pressure > engage_pressure + margin):
            return RegimeLabel.EVADE
        if engage_pressure >= 0.35 or engage_pressure > evade_pressure + margin:
            return RegimeLabel.ENGAGE
        return RegimeLabel.PATROL

    def _run_auction(self) -> list[Decision]:
        """Run the combinatorial task auction via Rust FFI.

        Builds AuctionDroneState/Task/ThreatState lists from internal state,
        calls Auctioneer.run_auction(), and converts Assignment results to
        Decision objects.
        """
        if not self._rust_available or self._auctioneer is None:
            return []

        from strix._strix_core import AuctionDroneState, Task as AuctionTask, ThreatState as AuctionThreat

        # Build drone states for auction
        auction_drones = []
        for drone in self._fleet.values():
            if not drone.alive:
                continue
            auction_drones.append(
                AuctionDroneState(
                    id=drone.drone_id,
                    position=[drone.position.x, drone.position.y, drone.position.z],
                    velocity=[drone.velocity.x, drone.velocity.y, drone.velocity.z],
                    regime_index=drone.regime.value,
                    energy=drone.energy,
                    alive=True,
                )
            )

        # Build tasks from active plan assignments (if any)
        auction_tasks = []
        task_templates: dict[int, Decision] = {}
        if self._active_plan:
            self._active_plan.assignments = [self._with_task_id(assignment) for assignment in self._active_plan.assignments]
            for assignment in self._active_plan.assignments:
                if assignment.target_position:
                    tp = assignment.target_position
                    if assignment.task_id is None:
                        continue
                    task_templates[assignment.task_id] = assignment
                    auction_tasks.append(
                        AuctionTask(
                            id=assignment.task_id,
                            location=[tp.x, tp.y, tp.z],
                            priority=assignment.confidence,
                        )
                    )

        if not auction_drones or not auction_tasks:
            return []

        # Build threats
        auction_threats = [
            AuctionThreat(
                id=t.threat_id,
                position=[t.position.x, t.position.y, t.position.z],
            )
            for t in self._threats.values()
        ]

        result = self._auctioneer.run_auction(auction_drones, auction_tasks, auction_threats)

        # Convert assignments to decisions
        decisions: list[Decision] = []

        for assignment in result.assignments:
            # Find the corresponding task's location by ID (not index).
            template = task_templates.get(assignment.task_id)
            task_pos = template.target_position if template is not None else None
            kind = template.kind if template is not None else DecisionKind.GOTO
            speed_ms = template.speed_ms if template is not None else 10.0
            confidence = max(0.0, min(assignment.bid_score / 10.0, 1.0))
            reason = f"Auction assignment (score={assignment.bid_score:.2f})"
            if template is not None:
                confidence = max(confidence, template.confidence * 0.75)
                reason = (
                    f"Auction assignment (score={assignment.bid_score:.2f}, "
                    f"seeded_from={template.kind.name})"
                )

            decisions.append(
                Decision(
                    drone_id=assignment.drone_id,
                    task_id=assignment.task_id,
                    kind=kind,
                    target_position=task_pos,
                    speed_ms=speed_ms,
                    reason=reason,
                    confidence=confidence,
                )
            )

        if self._active_plan is not None and decisions:
            self._active_plan.assignments = decisions

        return decisions

    def _should_run_auction(self, ticks_since_auction: int) -> bool:
        if ticks_since_auction >= self.config.auction_interval_ticks:
            return True
        return bool(getattr(self._auctioneer, "needs_reauction", False))

    def _trigger_reauction(self) -> None:
        trigger = getattr(self._auctioneer, "trigger_reauction", None)
        if callable(trigger):
            trigger()

    def _with_task_id(self, decision: Decision) -> Decision:
        if decision.task_id is not None:
            return decision

        task_id = self._next_task_id
        self._next_task_id += 1
        return Decision(
            drone_id=decision.drone_id,
            task_id=task_id,
            kind=decision.kind,
            target_position=decision.target_position,
            speed_ms=decision.speed_ms,
            reason=decision.reason,
            confidence=decision.confidence,
        )

    def _nearest_threat_bearing(self, drone: DroneSnapshot) -> list[float]:
        """Compute unit vector from drone toward nearest threat."""
        if not self._threats:
            return [0.0, 0.0, 0.0]
        nearest = min(self._threats.values(), key=lambda t: drone.position.distance_to(t.position))
        dx = nearest.position.x - drone.position.x
        dy = nearest.position.y - drone.position.y
        dz = nearest.position.z - drone.position.z
        dist = (dx**2 + dy**2 + dz**2) ** 0.5
        if dist < 1e-6:
            return [0.0, 0.0, 0.0]
        return [dx / dist, dy / dist, dz / dist]

    def _score_drones_for_intent(
        self,
        drones: list[DroneSnapshot],
        intent: MissionIntent,
    ) -> list[tuple[DroneSnapshot, float]]:
        """Score and rank drones for a mission intent.

        Scoring mirrors the Rust bidder:
            total = urgency*10 + capability*3 + proximity*5 + energy*2 - risk*4
        """
        target = intent.area.center if intent.area else Vec3()
        scored: list[tuple[DroneSnapshot, float]] = []

        for drone in drones:
            dist = drone.position.distance_to(target)
            proximity = 1.0 / max(dist, 1.0)
            capability = self._capability_match(drone, intent)
            energy = drone.energy
            route_risk = self._line_threat_exposure(drone.position, target)
            local_risk = self._threat_exposure(drone.position)
            target_pressure = self._target_threat_pressure(target)
            risk = 0.35 * local_risk + 0.4 * route_risk + 0.25 * target_pressure
            reserve_bias = self._reserve_bias(drone, intent)

            score = (
                intent.priority * 10.0
                + capability * 3.0
                + proximity * 5.0
                + energy * 2.0
                + reserve_bias
                - risk * 4.0
            )
            scored.append((drone, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _threat_exposure(self, pos: Vec3) -> float:
        """Compute threat exposure at a position [0, 1]."""
        if not self._threats:
            return 0.0
        exposure = 0.0
        for t in self._threats.values():
            dist = pos.distance_to(t.position)
            lethal_radius = 200.0  # default
            if dist < lethal_radius:
                exposure += (1.0 - dist / lethal_radius) * t.confidence
        return min(exposure, 1.0)

    def _line_threat_exposure(self, start: Vec3, target: Vec3) -> float:
        """Approximate route risk by sampling a straight-line path to target."""
        if not self._threats:
            return 0.0
        samples = []
        for alpha in (0.25, 0.5, 0.75, 1.0):
            probe = Vec3(
                start.x + (target.x - start.x) * alpha,
                start.y + (target.y - start.y) * alpha,
                start.z + (target.z - start.z) * alpha,
            )
            samples.append(self._threat_exposure(probe))
        return sum(samples) / len(samples)

    def _target_threat_pressure(self, target: Vec3) -> float:
        """Measure how contested the mission area itself appears to be."""
        if not self._threats:
            return 0.0

        pressure = 0.0
        for threat in self._threats.values():
            dist = target.distance_to(threat.position)
            proximity = max(0.0, 1.0 - dist / 350.0)
            if proximity <= 0.0:
                continue
            pressure += proximity * threat.confidence
        return min(pressure, 1.5)

    def _fleet_centroid(self) -> Vec3:
        """Compute the weighted centroid of all alive drones."""
        alive = [d for d in self._fleet.values() if d.alive]
        if not alive:
            return Vec3()
        n = len(alive)
        return Vec3(
            x=sum(d.position.x for d in alive) / n,
            y=sum(d.position.y for d in alive) / n,
            z=sum(d.position.z for d in alive) / n,
        )

    @staticmethod
    def _auto_drone_count(intent: MissionIntent, available: int) -> int:
        """Heuristic for how many drones a mission type needs."""
        base = {
            MissionType.RECON: 4,
            MissionType.STRIKE: 6,
            MissionType.ESCORT: 3,
            MissionType.DEFEND: 8,
            MissionType.PATROL: 4,
            MissionType.RELAY: 2,
        }
        return min(base.get(intent.mission_type, 4), available)

    @staticmethod
    def _intent_to_decision_kind(mission_type: MissionType) -> DecisionKind:
        """Map mission type to the primary decision kind."""
        mapping = {
            MissionType.RECON: DecisionKind.GOTO,
            MissionType.STRIKE: DecisionKind.ENGAGE_TARGET,
            MissionType.ESCORT: DecisionKind.GOTO,
            MissionType.DEFEND: DecisionKind.LOITER,
            MissionType.PATROL: DecisionKind.LOITER,
            MissionType.RELAY: DecisionKind.RELAY_STATION,
        }
        return mapping.get(mission_type, DecisionKind.GOTO)

    def _build_assignments_for_intent(
        self,
        selected: list[tuple[DroneSnapshot, float]],
        intent: MissionIntent,
    ) -> tuple[list[Decision], float, str]:
        """Translate selected drones into routed mission assignments."""
        selected_drones = [drone for drone, _ in selected]
        if not selected_drones:
            return [], 0.0, "No candidate drones selected."

        if intent.area is None:
            assignments = []
            for drone, score in selected:
                confidence = max(0.2, min(0.95, 0.45 + score / 25.0))
                assignments.append(
                    Decision(
                        drone_id=drone.drone_id,
                        kind=self._intent_to_decision_kind(intent.mission_type),
                        target_position=None,
                        speed_ms=self._mission_speed(intent, drone, 12.0),
                        reason=f"{intent.mission_type.name} fallback assignment without mission area",
                        confidence=confidence,
                    )
                )
            avg_conf = sum(decision.confidence for decision in assignments) / len(assignments)
            return assignments, avg_conf, "No mission area provided; generated fallback tasking."

        from strix.temporal.multi_horizon import (
            FleetState,
            MultiHorizonPlanner,
            planner_validation_metrics,
            tactical_to_decisions,
        )

        planner = MultiHorizonPlanner()
        state = FleetState(
            drones=[
                DroneSnapshot(
                    drone_id=drone.drone_id,
                    position=Vec3(drone.position.x, drone.position.y, drone.position.z),
                    velocity=Vec3(drone.velocity.x, drone.velocity.y, drone.velocity.z),
                    regime=drone.regime,
                    energy=drone.energy,
                    alive=drone.alive,
                    capabilities=drone.capabilities,
                )
                for drone in selected_drones
            ],
            regime=self._regime,
            threats=[threat.position for threat in self._threats.values() if threat.confidence > 0.25],
            obstacles=self._intent_obstacles(intent),
            neighbor_state_ages_s=self._neighbor_state_ages(selected_drones),
            packet_success_rate=self._planner_packet_success_rate(selected_drones),
            timestamp=time.monotonic(),
        )
        state.recompute_metrics()

        tactical_plans = planner.plan_tactical(state)
        operational_plan = planner.plan_operational(state)
        planner.plan_strategic(state)
        integrated = planner.cascade_plans()
        tactical_decisions = tactical_to_decisions(tactical_plans)
        validation = planner_validation_metrics(tactical_plans)
        score_by_drone = {drone.drone_id: score for drone, score in selected}

        assignments: list[Decision] = []
        for decision in tactical_decisions:
            drone = self._fleet.get(decision.drone_id)
            if drone is None:
                continue

            if decision.kind == DecisionKind.LOITER:
                assignments.append(
                    Decision(
                        drone_id=decision.drone_id,
                        kind=DecisionKind.LOITER,
                        target_position=None,
                        speed_ms=0.0,
                        reason=f"{intent.mission_type.name} vetoed by planner: {decision.reason}",
                        confidence=max(0.15, decision.confidence),
                    )
                )
                continue

            target_position = self._clamp_to_area(decision.target_position or intent.area.center, intent.area)
            route_risk = self._line_threat_exposure(drone.position, target_position)
            bid_score = score_by_drone.get(decision.drone_id, 0.0)
            mission_speed = self._mission_speed(intent, drone, decision.speed_ms)
            confidence = 0.45 * decision.confidence + 0.3 * operational_plan.coordination_confidence
            confidence += 0.15 * max(0.0, min(bid_score / 20.0, 1.0))
            confidence += 0.10 * max(0.0, 1.0 - route_risk)
            confidence = max(0.15, min(confidence, 0.98))

            assignments.append(
                Decision(
                    drone_id=decision.drone_id,
                    kind=self._intent_to_decision_kind(intent.mission_type),
                    target_position=target_position,
                    speed_ms=mission_speed,
                    reason=(
                        f"{intent.mission_type.name} planner route; "
                        f"validation={decision.confidence:.2f}; route_risk={route_risk:.2f}"
                    ),
                    confidence=confidence,
                )
            )

        if not assignments:
            return [], 0.0, "Planner produced no executable assignments."

        overall_confidence = sum(assignment.confidence for assignment in assignments) / len(assignments)
        if not integrated.coherent:
            overall_confidence *= 0.7
        summary = (
            f"Planner valid_fraction={validation['valid_fraction']:.2f}; "
            f"mean_validation={validation['mean_validation_confidence']:.2f}; "
            f"link={validation['packet_success_rate']:.2f}; "
            f"stale_max={validation['max_neighbor_age_s']:.2f}s; "
            f"coordination={operational_plan.coordination_confidence:.2f}"
        )
        return assignments, overall_confidence, summary

    def _refresh_orientation(self, planner_confidence: float | None = None) -> OrientationSnapshot:
        source_confidences: dict[str, float] = {}
        doctrine_scores: dict[str, float] = {}
        threat_confidences: list[float] = []

        for threat in self._threats.values():
            confidence = max(0.0, min(threat.confidence, 1.0))
            threat_confidences.append(confidence)
            label = _normalize_threat_type_label(threat.threat_type).split(":", 1)[0]
            source_confidences[label] = max(source_confidences.get(label, 0.0), confidence)
            doctrine_scores[label] = doctrine_scores.get(label, 0.0) + confidence

        alive_fraction = self.fleet_size / max(len(self._fleet), 1)
        return self._orientation.update(
            now_s=time.monotonic(),
            regime=self._regime.name,
            planner_confidence=planner_confidence,
            comms_quality=self._planner_packet_success_rate(),
            stale_age_s=self._network_state_age(),
            fleet_size=max(len(self._fleet), 1),
            alive_fraction=alive_fraction,
            threat_confidences=threat_confidences,
            source_confidences=source_confidences,
            doctrine_scores=doctrine_scores,
        )

    def _apply_orientation_to_plan(
        self,
        plan: MissionPlan,
        orientation: OrientationSnapshot,
    ) -> MissionPlan:
        assignments = [
            Decision(
                drone_id=assignment.drone_id,
                task_id=assignment.task_id,
                kind=assignment.kind,
                target_position=assignment.target_position,
                speed_ms=assignment.speed_ms,
                reason=(
                    f"{assignment.reason.split('; orientation integrity=', 1)[0]}; "
                    f"{orientation.reason_fragment()}"
                ),
                confidence=self._orientation.reweight_confidence(assignment.confidence),
            )
            for assignment in plan.assignments
        ]
        base_explanation = (plan.explanation or "Orientation-adjusted plan.").split(
            " orientation integrity=",
            1,
        )[0].rstrip()
        broken_fragment = ""
        if orientation.broken_assumptions:
            broken_fragment = (
                f" Broken assumptions: {', '.join(orientation.broken_assumptions)}."
            )
        explanation = f"{base_explanation}{broken_fragment} {orientation.reason_fragment()}."
        return MissionPlan(
            intent=plan.intent,
            assignments=assignments,
            estimated_duration_s=plan.estimated_duration_s,
            confidence=self._orientation.reweight_confidence(plan.confidence),
            regime=plan.regime,
            explanation=explanation,
        )

    def _planner_packet_success_rate(self, drones: list[DroneSnapshot] | None = None) -> float:
        self._refresh_runtime_comms()
        observed = [
            self._effective_comms_state(drone.drone_id).packet_success_rate
            for drone in (drones or [])
            if drone.alive
        ]
        if observed:
            return max(0.0, min(sum(observed) / len(observed), 1.0))
        return self._network_comms.packet_success_rate

    def _network_state_age(self) -> float:
        self._refresh_runtime_comms()
        observed = [state.state_age_s for state in self._drone_comms.values()]
        if observed:
            return max(max(observed), self._network_comms.state_age_s)
        return self._network_comms.state_age_s

    def _neighbor_state_ages(self, drones: list[DroneSnapshot]) -> dict[int, float]:
        return {
            drone.drone_id: self._effective_comms_state(drone.drone_id).state_age_s
            for drone in drones
        }

    def _effective_comms_state(self, drone_id: int) -> CommsState:
        return self._drone_comms.get(drone_id, self._network_comms)

    def _refresh_runtime_comms(self, now: float | None = None) -> None:
        """Derive comms health from telemetry freshness when live updates are flowing."""
        if not self._runtime_comms_active:
            return

        now = time.monotonic() if now is None else now
        expected_interval = max(1.0 / max(self.config.tick_hz, 1e-6), 0.05)
        overdue_window = max(expected_interval * 1.5, self.config.stale_state_age_s)
        observed_qualities: list[float] = []
        observed_ages: list[float] = []

        for drone_id, drone in self._fleet.items():
            if not drone.alive:
                continue

            arrival = self._last_arrival_s.get(drone_id)
            if arrival is None:
                continue

            age = max(0.0, now - arrival)
            missed_windows = int(age / overdue_window) if overdue_window > 1e-6 else 0
            already_recorded = self._stale_windows.get(drone_id, 0)
            while already_recorded < missed_windows:
                self._link_tracker.record_failure(drone_id)
                already_recorded += 1
            self._stale_windows[drone_id] = missed_windows

            base_state = self._drone_comms.get(drone_id, self._network_comms)
            quality = max(0.0, min(self._link_tracker.quality(drone_id), 1.0))
            self._drone_comms[drone_id] = CommsState(
                packet_success_rate=quality,
                state_age_s=age,
                latency_ms=base_state.latency_ms,
            )
            observed_qualities.append(quality)
            observed_ages.append(age)

        if observed_qualities:
            self._network_comms = CommsState(
                packet_success_rate=sum(observed_qualities) / len(observed_qualities),
                state_age_s=max(observed_ages),
                latency_ms=self._network_comms.latency_ms,
            )

    @staticmethod
    def _intent_obstacles(intent: MissionIntent) -> list[tuple[Vec3, float]]:
        obstacles: list[tuple[Vec3, float]] = []
        for constraint in intent.constraints:
            if constraint.avoid and constraint.area is not None:
                obstacles.append((constraint.area.center, max(constraint.area.radius_m, 10.0)))
        return obstacles

    @staticmethod
    def _clamp_to_area(target: Vec3, area: MissionArea) -> Vec3:
        dx = target.x - area.center.x
        dy = target.y - area.center.y
        horizontal = (dx**2 + dy**2) ** 0.5
        if horizontal > area.radius_m and horizontal > 1e-6:
            scale = area.radius_m / horizontal
            dx *= scale
            dy *= scale
        return Vec3(area.center.x + dx, area.center.y + dy, target.z)

    @staticmethod
    def _mission_speed(intent: MissionIntent, drone: DroneSnapshot, planner_speed: float) -> float:
        base = {
            MissionType.RECON: 12.0,
            MissionType.STRIKE: 18.0,
            MissionType.ESCORT: 14.0,
            MissionType.DEFEND: 9.0,
            MissionType.PATROL: 8.0,
            MissionType.RELAY: 10.0,
        }.get(intent.mission_type, 12.0)
        if drone.energy < 0.25:
            base *= 0.8
        return max(4.0, min(max(base, planner_speed), 25.0))

    @staticmethod
    def _capability_match(drone: DroneSnapshot, intent: MissionIntent) -> float:
        has_sensor = bool(drone.capabilities & 0b0001)
        has_weapon = bool(drone.capabilities & 0b0010)
        has_relay = bool(drone.capabilities & 0b1000)

        if intent.mission_type == MissionType.RECON:
            return 1.2 if has_sensor else 0.7
        if intent.mission_type == MissionType.STRIKE:
            return 1.3 if has_weapon else 0.5
        if intent.mission_type == MissionType.RELAY:
            return 1.2 if has_relay else 0.6
        return 1.0

    @staticmethod
    def _reserve_bias(drone: DroneSnapshot, intent: MissionIntent) -> float:
        if intent.mission_type in {MissionType.DEFEND, MissionType.RELAY}:
            return 0.75 * drone.energy
        if intent.mission_type == MissionType.STRIKE and drone.energy < 0.2:
            return -1.0
        return 0.0

    @staticmethod
    def _estimate_duration(
        selected: list[tuple[DroneSnapshot, float]],
        intent: MissionIntent,
    ) -> float:
        """Rough time estimate based on distance and mission type."""
        if not selected or not intent.area:
            return intent.deadline_s

        max_dist = max(d.position.distance_to(intent.area.center) for d, _ in selected)
        travel_time = max_dist / 15.0  # assume 15 m/s cruise
        mission_time = {
            MissionType.RECON: 300.0,
            MissionType.STRIKE: 120.0,
            MissionType.ESCORT: intent.deadline_s,
            MissionType.DEFEND: intent.deadline_s,
            MissionType.PATROL: intent.deadline_s,
            MissionType.RELAY: intent.deadline_s,
        }.get(intent.mission_type, 300.0)

        return travel_time + mission_time
