"""Rehearsal Engine -- test commands before sending them to real drones.

"What if I send 3 drones north while keeping 2 on overwatch?"

The rehearsal module clones the digital twin and runs a fast-forward
simulation of proposed plans.  This gives the commander:

- Predicted outcomes and risk estimates
- Side-by-side comparison of alternative plans
- Confidence intervals on mission success
- Early warning of plan failure modes

This is the military equivalent of paper-trading a strategy before
going live -- a core risk management practice from quantitative finance.
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from strix.brain import DecisionKind, MissionPlan, RegimeLabel, Vec3
from strix.digital_twin.twin import DigitalTwin, DroneState, WorldSnapshot

logger = logging.getLogger("strix.digital_twin.rehearsal")

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class SimMetrics:
    """Quantitative metrics from a simulated plan execution."""

    area_coverage_pct: float = 0.0
    time_to_objective_s: float = 0.0
    drones_lost: int = 0
    energy_consumed_pct: float = 0.0
    max_threat_exposure: float = 0.0
    min_separation_m: float = float("inf")
    position_error_m: float = 0.0


@dataclass
class SimResult:
    """Complete result of a rehearsal simulation."""

    plan: MissionPlan = field(default_factory=MissionPlan)
    metrics: SimMetrics = field(default_factory=SimMetrics)
    snapshots: list[WorldSnapshot] = field(default_factory=list)
    success: bool = False
    failure_reason: str = ""
    wall_time_ms: float = 0.0
    sim_time_s: float = 0.0


@dataclass
class Comparison:
    """Side-by-side comparison of multiple simulated plans."""

    results: list[SimResult] = field(default_factory=list)
    recommended_index: int = 0
    recommendation_reason: str = ""


# ---------------------------------------------------------------------------
# Rehearsal
# ---------------------------------------------------------------------------


class Rehearsal:
    """Test commands before sending them to real drones.

    Usage::

        twin = DigitalTwin()
        # ... populate twin with current state ...

        rehearsal = Rehearsal(twin)

        plan_a = MissionPlan(...)
        plan_b = MissionPlan(...)

        result_a = rehearsal.simulate(plan_a, steps=100)
        comparison = rehearsal.compare_plans([plan_a, plan_b])

        if comparison.recommended_index == 0:
            execute(plan_a)
    """

    def __init__(self, twin: DigitalTwin) -> None:
        self._twin = twin
        self._sim_dt = 0.5  # simulation time step (seconds)
        logger.info("Rehearsal engine initialized")

    def simulate(self, plan: MissionPlan, steps: int = 100) -> SimResult:
        """Simulate a mission plan in a cloned twin.

        Parameters
        ----------
        plan : MissionPlan
            The mission plan to simulate.
        steps : int
            Number of simulation steps to run.

        Returns
        -------
        SimResult
            Simulation outcome with metrics and state history.
        """
        t0_wall = time.monotonic()
        sim_twin = self._twin.clone()
        snapshots: list[WorldSnapshot] = []
        sim_time = 0.0

        # Initial snapshot
        snapshots.append(sim_twin.get_snapshot())

        for step in range(steps):
            sim_time = step * self._sim_dt

            # Apply plan decisions to the twin
            for decision in plan.assignments:
                drone_state = sim_twin.get_drone(decision.drone_id)
                if drone_state is None or not drone_state.alive:
                    continue

                self._apply_decision(drone_state, decision, self._sim_dt)
                sim_twin.update_drone(decision.drone_id, drone_state)

            # Decay pheromones
            sim_twin.decay_pheromones(self._sim_dt)

            # Record snapshot periodically
            if step % 10 == 0:
                snapshots.append(sim_twin.get_snapshot())

        # Final snapshot
        final_snap = sim_twin.get_snapshot()
        snapshots.append(final_snap)

        # Compute metrics
        metrics = self._compute_metrics(snapshots, plan)
        wall_time_ms = (time.monotonic() - t0_wall) * 1000

        # Determine success
        success = metrics.drones_lost == 0 and metrics.area_coverage_pct > 50.0
        failure_reason = ""
        if metrics.drones_lost > 0:
            failure_reason = f"Lost {metrics.drones_lost} drone(s) during simulation."
        elif metrics.area_coverage_pct < 50.0:
            failure_reason = f"Insufficient coverage: {metrics.area_coverage_pct:.0f}%."

        result = SimResult(
            plan=plan,
            metrics=metrics,
            snapshots=snapshots,
            success=success,
            failure_reason=failure_reason,
            wall_time_ms=wall_time_ms,
            sim_time_s=sim_time,
        )

        logger.info(
            "Simulation complete: %d steps, %.1fs sim time, %.1fms wall, success=%s",
            steps,
            sim_time,
            wall_time_ms,
            success,
        )
        return result

    def compare_plans(self, plans: list[MissionPlan]) -> Comparison:
        """Simulate multiple plans and recommend the best one.

        Ranking criteria (weighted):
        1. Mission success (binary gate)
        2. Area coverage (40%)
        3. Drone survivability (30%)
        4. Energy efficiency (15%)
        5. Speed to objective (15%)

        Parameters
        ----------
        plans : list[MissionPlan]
            Plans to compare.

        Returns
        -------
        Comparison
            Results for each plan and a recommendation.
        """
        results = [self.simulate(plan) for plan in plans]

        # Score each plan
        scores: list[float] = []
        for r in results:
            m = r.metrics
            if not r.success:
                scores.append(-1.0)
                continue

            score = (
                m.area_coverage_pct / 100.0 * 0.40
                + (1.0 - m.drones_lost / max(len(r.plan.assignments), 1)) * 0.30
                + (1.0 - m.energy_consumed_pct / 100.0) * 0.15
                + (1.0 / max(m.time_to_objective_s, 1.0)) * 100.0 * 0.15
            )
            scores.append(score)

        best_idx = max(range(len(scores)), key=lambda i: scores[i])

        reason = f"Plan {best_idx} scored highest ({scores[best_idx]:.3f})"
        if scores[best_idx] < 0:
            reason = "No plan succeeded in simulation. Recommend re-planning."

        comparison = Comparison(
            results=results,
            recommended_index=best_idx,
            recommendation_reason=reason,
        )

        logger.info("Compared %d plans. Recommended: plan %d (%s)", len(plans), best_idx, reason)
        return comparison

    # -- Private simulation helpers -----------------------------------------

    @staticmethod
    def _apply_decision(drone: DroneState, decision, dt: float) -> None:
        """Apply a decision to a drone state for one simulation step."""
        if decision.target_position is None:
            return

        target = decision.target_position
        dx = target.x - drone.position.x
        dy = target.y - drone.position.y
        dz = target.z - drone.position.z
        dist = (dx**2 + dy**2 + dz**2) ** 0.5

        if dist < 1.0:
            # Arrived
            drone.velocity = Vec3(0, 0, 0)
            return

        # Move toward target at commanded speed
        speed = min(decision.speed_ms, dist / dt)
        drone.velocity = Vec3(
            speed * dx / dist,
            speed * dy / dist,
            speed * dz / dist,
        )
        drone.position = Vec3(
            drone.position.x + drone.velocity.x * dt,
            drone.position.y + drone.velocity.y * dt,
            drone.position.z + drone.velocity.z * dt,
        )

        # Energy consumption (rough model: 0.1% per second at cruise)
        drone.energy = max(0.0, drone.energy - 0.001 * dt)

    @staticmethod
    def _compute_metrics(snapshots: list[WorldSnapshot], plan: MissionPlan) -> SimMetrics:
        """Compute aggregate metrics from simulation snapshots."""
        if not snapshots:
            return SimMetrics()

        initial = snapshots[0]
        final = snapshots[-1]

        initial_alive = sum(1 for d in initial.drones.values() if d.alive)
        final_alive = sum(1 for d in final.drones.values() if d.alive)
        drones_lost = initial_alive - final_alive

        # Energy consumed
        initial_energy = sum(d.energy for d in initial.drones.values() if d.alive)
        final_energy = sum(d.energy for d in final.drones.values() if d.alive)
        energy_pct = ((initial_energy - final_energy) / max(initial_energy, 0.01)) * 100.0

        # Coverage estimate (from dispersion as proxy)
        coverage = min(100.0, final.fleet_dispersion_m * 0.5)

        # Time to objective: when centroid first reached target area
        time_to_obj = 0.0
        if plan.intent.area:
            target = plan.intent.area.center
            for snap in snapshots:
                if snap.fleet_centroid.distance_to(target) < plan.intent.area.radius_m:
                    time_to_obj = snap.timestamp - snapshots[0].timestamp
                    break
            else:
                time_to_obj = snapshots[-1].timestamp - snapshots[0].timestamp

        # Min separation across all snapshots
        min_sep = float("inf")
        for snap in snapshots:
            alive = [d for d in snap.drones.values() if d.alive]
            for i, d1 in enumerate(alive):
                for d2 in alive[i + 1 :]:
                    dist = d1.position.distance_to(d2.position)
                    min_sep = min(min_sep, dist)

        return SimMetrics(
            area_coverage_pct=coverage,
            time_to_objective_s=time_to_obj,
            drones_lost=drones_lost,
            energy_consumed_pct=energy_pct,
            max_threat_exposure=0.0,
            min_separation_m=min_sep if min_sep != float("inf") else 0.0,
            position_error_m=0.0,
        )
