"""Digital Twin -- 3D state representation and world model.

Maintains a synchronized mirror of the physical battlespace: every drone,
threat, pheromone deposit, mission area, and terrain feature is tracked as
a first-class entity with full state history.

The twin serves three purposes:

1. **Visualization**: feeds the rerun.io / WebGL 3D display.
2. **What-if analysis**: the Rehearsal module clones the twin and runs
   hypothetical scenarios without affecting the real state.
3. **Explainability**: every state transition is logged, enabling
   after-action replay and decision auditing.

The twin is the single source of truth for the Python orchestration layer.
All other modules read from the twin rather than maintaining their own
copies of world state.
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from strix.brain import RegimeLabel, Vec3

logger = logging.getLogger("strix.digital_twin.twin")

# ---------------------------------------------------------------------------
# State types
# ---------------------------------------------------------------------------


@dataclass
class DroneState:
    """Full state of a single drone in the digital twin."""

    drone_id: int = 0
    position: Vec3 = field(default_factory=Vec3)
    velocity: Vec3 = field(default_factory=Vec3)
    attitude: tuple[float, float, float] = (0.0, 0.0, 0.0)  # roll, pitch, yaw (rad)
    regime: RegimeLabel = RegimeLabel.PATROL
    energy: float = 1.0
    alive: bool = True
    capabilities: int = 0
    last_update: float = 0.0

    # Particle filter uncertainty
    position_uncertainty_m: float = 0.0
    velocity_uncertainty_ms: float = 0.0


@dataclass
class ThreatState:
    """State of a tracked threat entity."""

    threat_id: int = 0
    position: Vec3 = field(default_factory=Vec3)
    velocity: Vec3 = field(default_factory=Vec3)
    threat_type: str = "unknown"
    lethal_radius_m: float = 200.0
    confidence: float = 0.5
    behavior: str = "unknown"
    last_update: float = 0.0


@dataclass
class PheromoneDeposit:
    """A digital pheromone marking in the environment."""

    position: Vec3 = field(default_factory=Vec3)
    ptype: str = "explored"  # explored | danger | interest | relay
    intensity: float = 1.0
    depositor_id: int = 0
    timestamp: float = 0.0


@dataclass
class MissionAreaState:
    """State of a mission area being tracked."""

    name: str = ""
    center: Vec3 = field(default_factory=Vec3)
    radius_m: float = 500.0
    coverage_pct: float = 0.0
    active: bool = True


@dataclass
class WorldSnapshot:
    """Immutable snapshot of the entire world state at a point in time.

    Used for visualization, logging, and what-if cloning.
    """

    timestamp: float = 0.0
    drones: dict[int, DroneState] = field(default_factory=dict)
    threats: dict[int, ThreatState] = field(default_factory=dict)
    pheromones: list[PheromoneDeposit] = field(default_factory=list)
    mission_areas: list[MissionAreaState] = field(default_factory=list)
    regime: RegimeLabel = RegimeLabel.PATROL
    fleet_centroid: Vec3 = field(default_factory=Vec3)
    fleet_dispersion_m: float = 0.0
    active_drone_count: int = 0


# ---------------------------------------------------------------------------
# DigitalTwin
# ---------------------------------------------------------------------------


class DigitalTwin:
    """Maintains a 3D mirror of the real-world state.

    Every drone, threat, pheromone field, and mission area is represented
    in the twin for visualization and what-if analysis.

    Thread safety: the twin is designed to be updated from the brain's
    asyncio event loop (single-threaded).  The ``get_snapshot()`` method
    produces an immutable deep copy safe for cross-thread consumption
    (e.g., by the visualization layer).

    Usage::

        twin = DigitalTwin()
        twin.update_drone(1, DroneState(drone_id=1, position=Vec3(10, 20, -50)))
        twin.update_threat(100, ThreatState(threat_id=100, position=Vec3(300, 0, -80)))

        snapshot = twin.get_snapshot()
        # Pass snapshot to visualization, rehearsal, or logging
    """

    def __init__(self) -> None:
        self._drones: dict[int, DroneState] = {}
        self._threats: dict[int, ThreatState] = {}
        self._pheromones: list[PheromoneDeposit] = []
        self._mission_areas: list[MissionAreaState] = []
        self._regime = RegimeLabel.PATROL
        self._history: list[WorldSnapshot] = []
        self._max_history = 1000
        self._pheromone_decay_rate = 0.05  # per second
        logger.info("DigitalTwin initialized")

    # -- Drone management ---------------------------------------------------

    def update_drone(self, drone_id: int, state: DroneState) -> None:
        """Create or update a drone in the twin."""
        state.drone_id = drone_id
        state.last_update = time.monotonic()
        self._drones[drone_id] = state

    def remove_drone(self, drone_id: int) -> None:
        """Mark a drone as dead (kept in state for history, but marked inactive)."""
        if drone_id in self._drones:
            self._drones[drone_id].alive = False
            logger.info("Drone %d marked as lost in twin", drone_id)

    def get_drone(self, drone_id: int) -> Optional[DroneState]:
        """Get the current state of a specific drone."""
        return self._drones.get(drone_id)

    @property
    def alive_drones(self) -> list[DroneState]:
        """All alive drones."""
        return [d for d in self._drones.values() if d.alive]

    # -- Threat management --------------------------------------------------

    def update_threat(self, threat_id: int, state: ThreatState) -> None:
        """Create or update a threat entity in the twin."""
        state.threat_id = threat_id
        state.last_update = time.monotonic()
        self._threats[threat_id] = state

    def remove_threat(self, threat_id: int) -> None:
        """Remove a threat (confirmed destroyed or false alarm)."""
        self._threats.pop(threat_id, None)

    def get_threat(self, threat_id: int) -> Optional[ThreatState]:
        return self._threats.get(threat_id)

    @property
    def active_threats(self) -> list[ThreatState]:
        return list(self._threats.values())

    # -- Pheromone field ----------------------------------------------------

    def deposit_pheromone(self, deposit: PheromoneDeposit) -> None:
        """Add a pheromone deposit to the environment."""
        deposit.timestamp = time.monotonic()
        self._pheromones.append(deposit)

    def decay_pheromones(self, dt: float) -> None:
        """Apply exponential decay to all pheromone deposits.

        Removes deposits whose intensity falls below 0.01.
        """
        import math
        decay_factor = math.exp(-self._pheromone_decay_rate * dt)
        surviving = []
        for p in self._pheromones:
            p.intensity *= decay_factor
            if p.intensity > 0.01:
                surviving.append(p)
        self._pheromones = surviving

    def pheromone_intensity_at(self, position: Vec3, ptype: str = "explored") -> float:
        """Query the accumulated pheromone intensity at a position.

        Intensity falls off with distance (Gaussian kernel, sigma=20m).
        """
        import math

        sigma = 20.0
        total = 0.0
        for p in self._pheromones:
            if p.ptype == ptype:
                dist_sq = (
                    (p.position.x - position.x) ** 2
                    + (p.position.y - position.y) ** 2
                    + (p.position.z - position.z) ** 2
                )
                total += p.intensity * math.exp(-0.5 * dist_sq / (sigma * sigma))
        return total

    # -- Mission areas ------------------------------------------------------

    def add_mission_area(self, area: MissionAreaState) -> None:
        self._mission_areas.append(area)

    def update_coverage(self, area_name: str, coverage_pct: float) -> None:
        for area in self._mission_areas:
            if area.name == area_name:
                area.coverage_pct = coverage_pct
                break

    # -- Regime -------------------------------------------------------------

    def set_regime(self, regime: RegimeLabel) -> None:
        if regime != self._regime:
            logger.info("Twin regime: %s -> %s", self._regime.name, regime.name)
            self._regime = regime

    # -- Snapshot and history -----------------------------------------------

    def get_snapshot(self) -> WorldSnapshot:
        """Get an immutable deep-copy snapshot of the current world state.

        Safe to pass to visualization or other threads.
        """
        alive = [d for d in self._drones.values() if d.alive]
        n = len(alive)

        if n > 0:
            cx = sum(d.position.x for d in alive) / n
            cy = sum(d.position.y for d in alive) / n
            cz = sum(d.position.z for d in alive) / n
            centroid = Vec3(cx, cy, cz)
            dispersion = (sum(d.position.distance_to(centroid) ** 2 for d in alive) / n) ** 0.5
        else:
            centroid = Vec3()
            dispersion = 0.0

        snapshot = WorldSnapshot(
            timestamp=time.monotonic(),
            drones=copy.deepcopy(self._drones),
            threats=copy.deepcopy(self._threats),
            pheromones=copy.deepcopy(self._pheromones),
            mission_areas=copy.deepcopy(self._mission_areas),
            regime=self._regime,
            fleet_centroid=centroid,
            fleet_dispersion_m=dispersion,
            active_drone_count=n,
        )

        # Archive to history (bounded)
        self._history.append(snapshot)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        return snapshot

    def get_history(self, last_n: int = 100) -> list[WorldSnapshot]:
        """Get the last N snapshots from history."""
        return self._history[-last_n:]

    def clone(self) -> DigitalTwin:
        """Deep-clone the entire twin for what-if simulation."""
        return copy.deepcopy(self)
