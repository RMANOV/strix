"""Mission Brain -- the central market-inspired mission planner.

Layer 1 of the STRIX architecture.  Orchestrates the particle filter loop,
regime management, combinatorial auctions, and mesh coordination to convert
high-level human intent into executable drone assignments.

The main loop mirrors a quantitative trading engine:

    predict -> update -> regime_check -> auction -> assign

Each tick runs this pipeline at 10 Hz, producing a list of Decisions that the
Puppet Master layer (strix-adapters) translates into platform commands.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

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
    """Particle count for the adversarial (enemy) filter."""

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


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class MissionType(Enum):
    """High-level mission categories."""

    RECON = auto()
    STRIKE = auto()
    ESCORT = auto()
    DEFEND = auto()
    PATROL = auto()
    RELAY = auto()


class RegimeLabel(Enum):
    """Battlespace operating regimes, mirroring the Rust `Regime` enum."""

    PATROL = 0
    ENGAGE = 1
    EVADE = 2


class DecisionKind(Enum):
    """Atomic decision types emitted by the brain each tick."""

    GOTO = auto()
    LOITER = auto()
    ENGAGE_TARGET = auto()
    EVADE = auto()
    RETURN_TO_BASE = auto()
    RELAY_STATION = auto()
    LAND = auto()


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
    """Structured representation of what the commander wants."""

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


@dataclass
class ThreatObservation:
    """An observation about an enemy entity."""

    threat_id: int = 0
    position: Vec3 = field(default_factory=Vec3)
    velocity: Vec3 = field(default_factory=Vec3)
    confidence: float = 0.5
    threat_type: str = "unknown"
    timestamp: float = 0.0


@dataclass
class Decision:
    """A single atomic decision produced by a brain tick."""

    drone_id: int = 0
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
        self._pending_intents: list[MissionIntent] = []
        self._running = False
        self._last_auction_tick = 0

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

        logger.info("MissionBrain initialised (particles=%d)", self.config.n_particles)

    # -- Fleet management ----------------------------------------------------

    def register_drone(self, drone: DroneSnapshot) -> None:
        """Register or update a drone in the fleet."""
        previous = self._fleet.get(drone.drone_id)
        if len(self._fleet) >= self.config.max_fleet_size and drone.drone_id not in self._fleet:
            raise ValueError(f"Fleet full ({self.config.max_fleet_size} drones)")
        self._fleet[drone.drone_id] = drone
        if not self._rust_available:
            return

        needs_filter_reset = drone.drone_id not in self._filters
        if previous is not None:
            position_changed = previous.position.distance_to(drone.position) > 1e-6
            velocity_changed = previous.velocity.distance_to(drone.velocity) > 1e-6
            if position_changed or velocity_changed or previous.alive != drone.alive:
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

    @property
    def fleet_size(self) -> int:
        return sum(1 for d in self._fleet.values() if d.alive)

    # -- Intent pipeline -----------------------------------------------------

    def process_intent_sync(self, intent: MissionIntent) -> MissionPlan:
        """Convert a high-level intent into a concrete mission plan.

        Steps:
            1. Validate intent feasibility (enough drones, within range).
            2. Compute optimal allocation via combinatorial auction.
            3. Generate waypoints and assignments.
            4. Package into a MissionPlan for commander review.
        """
        logger.info("Processing intent: %s over %s", intent.mission_type.name, intent.area)

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

        # Generate assignments
        assignments = []
        for drone_snap, _score in selected:
            target = intent.area.center if intent.area else Vec3()
            decision = Decision(
                drone_id=drone_snap.drone_id,
                kind=self._intent_to_decision_kind(intent.mission_type),
                target_position=target,
                speed_ms=15.0,
                reason=f"Assigned to {intent.mission_type.name} mission",
                confidence=0.8,
            )
            assignments.append(decision)

        estimated_duration_s = self._estimate_duration(selected, intent)
        plan = MissionPlan(
            intent=intent,
            assignments=assignments,
            estimated_duration_s=estimated_duration_s,
            confidence=0.8 if len(selected) >= requested else 0.5,
            regime=self._regime,
            explanation=(
                f"Allocated {len(selected)}/{requested} drones. "
                f"Current regime: {self._regime.name}. "
                f"Estimated time: {estimated_duration_s:.0f}s."
            ),
        )

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
            Zero or more decisions for the Puppet Master to execute.
        """
        self._tick_count += 1
        decisions: list[Decision] = []

        # 1. PREDICT -- propagate particle filters forward
        self._predict_step(dt)

        # 2. UPDATE -- incorporate new sensor observations
        self._update_step()

        # 3. REGIME CHECK -- evaluate Markov transitions
        new_regime = self._check_regime()
        if new_regime != self._regime:
            logger.info("Regime transition: %s -> %s", self._regime.name, new_regime.name)
            self._regime = new_regime

        # 4. AUCTION -- periodically re-run task allocation
        ticks_since_auction = self._tick_count - self._last_auction_tick
        if ticks_since_auction >= self.config.auction_interval_ticks:
            self._run_auction()
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

        # Record kill zone
        self._threats[10000 + drone_id] = ThreatObservation(
            threat_id=10000 + drone_id,
            position=drone.position,
            confidence=0.9,
            threat_type=f"kill_zone:{cause}",
            timestamp=time.monotonic(),
        )

        drone.alive = False

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

    async def handle_loss(self, drone_id: int, cause: str) -> None:
        self.handle_loss_sync(drone_id, cause)

    def update_threat_sync(self, observation: ThreatObservation) -> None:
        """Incorporate a new threat observation into the adversarial filter."""
        self._threats[observation.threat_id] = observation
        logger.info(
            "Threat updated: id=%d type=%s confidence=%.2f",
            observation.threat_id,
            observation.threat_type,
            observation.confidence,
        )

        # High-confidence threat near the fleet -> consider regime shift
        if observation.confidence > 0.7:
            centroid = self._fleet_centroid()
            dist = centroid.distance_to(observation.position)
            if dist < 300.0:
                logger.warning("High-confidence threat at %.0fm -- shifting to ENGAGE", dist)
                self._regime = RegimeLabel.ENGAGE

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

            # Process any queued intents
            while self._pending_intents:
                intent = self._pending_intents.pop(0)
                self.process_intent_sync(intent)

            decisions = self.tick_sync(dt)
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

        Uses the Markov transition matrix weighted by current evidence:
        - High threat density near fleet -> ENGAGE
        - Active fire / losses -> EVADE
        - No threats detected -> PATROL
        """
        if not self._threats:
            return RegimeLabel.PATROL

        centroid = self._fleet_centroid()
        close_threats = sum(
            1
            for t in self._threats.values()
            if centroid.distance_to(t.position) < 500.0 and t.confidence > 0.5
        )

        if close_threats >= 3:
            return RegimeLabel.EVADE
        elif close_threats >= 1:
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
            for i, assignment in enumerate(self._active_plan.assignments):
                if assignment.target_position:
                    tp = assignment.target_position
                    task_templates[i] = assignment
                    auction_tasks.append(
                        AuctionTask(
                            id=i,
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
            template = task_templates.get(assignment.task_id)
            task_pos = template.target_position if template is not None else None

            decisions.append(
                Decision(
                    drone_id=assignment.drone_id,
                    kind=template.kind if template is not None else DecisionKind.GOTO,
                    target_position=task_pos,
                    speed_ms=template.speed_ms if template is not None else 10.0,
                    reason=(
                        f"Auction assignment (score={assignment.bid_score:.2f})"
                        if template is None
                        else f"{template.reason} via auction (score={assignment.bid_score:.2f})"
                    ),
                    confidence=max(0.0, min(assignment.bid_score / 10.0, 1.0)),
                )
            )

        if self._active_plan is not None and decisions:
            self._active_plan.assignments = decisions

        return decisions

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
            capability = 1.0  # placeholder: full match assumed
            energy = drone.energy
            risk = self._threat_exposure(drone.position)

            score = intent.priority * 10.0 + capability * 3.0 + proximity * 5.0 + energy * 2.0 - risk * 4.0
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
