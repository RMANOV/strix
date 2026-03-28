"""Adversarial Prediction Engine -- dual particle filter for enemy behaviour.

The KEY INNOVATION of STRIX: predict enemy behaviour *before* they complete
their manoeuvre, by maintaining a separate particle filter where each particle
is a hypothesis about the enemy's intent.

This is the direct battlefield analogue of the counterparty prediction model
from quantitative trading.  In finance, you model the likely actions of other
market participants to front-run large orders.  Here, you model the likely
actions of an adversary to gain tactical advantage.

Enemy particles encode:
    [x, y, z, vx, vy, vz]  -- 6D kinematic state
    regime                  -- intent hypothesis (DEFENDING, ATTACKING, RETREATING)

The regime-switching Markov model detects transitions:
    - Stationary clustering -> DEFENDING
    - Velocity vector toward friendly centroid -> ATTACKING
    - Velocity vector away from engagement area -> RETREATING
"""

from __future__ import annotations

import bisect
import logging
import math
import random
from dataclasses import dataclass, field, replace
from enum import Enum, auto
from typing import Optional

from strix.brain import Vec3

logger = logging.getLogger("strix.adversarial")

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class EnemyBehavior(Enum):
    """Classified enemy intent -- each particle votes on one of these."""

    DEFENDING = auto()
    ATTACKING = auto()
    RETREATING = auto()
    UNKNOWN = auto()


class EnemyDoctrine(Enum):
    """Doctrinal hypothesis library used for online model selection."""

    FIXED_DEFENSE = auto()
    DIRECT_ASSAULT = auto()
    FLANKING = auto()
    FIGHTING_WITHDRAWAL = auto()
    DECOY = auto()
    EW_JAMMING = auto()
    PROBING = auto()
    FEINT = auto()
    COORDINATED_ATTACK = auto()


@dataclass
class SensorReading:
    """A single sensor observation of an enemy entity."""

    threat_id: int = 0
    position: Optional[Vec3] = None
    velocity: Optional[Vec3] = None
    bearing_rad: Optional[float] = None
    signal_strength_dbm: Optional[float] = None
    radar_cross_section: Optional[float] = None
    confidence: float = 0.5
    timestamp: float = 0.0
    sensor_type: str = "unknown"


@dataclass
class EnemyEstimate:
    """Aggregated estimate of one enemy entity from the particle filter."""

    threat_id: int = 0
    position: Vec3 = field(default_factory=Vec3)
    velocity: Vec3 = field(default_factory=Vec3)
    position_uncertainty_m: float = 0.0
    behavior: EnemyBehavior = EnemyBehavior.UNKNOWN
    behavior_probabilities: dict[EnemyBehavior, float] = field(default_factory=dict)
    dominant_doctrine: EnemyDoctrine = EnemyDoctrine.FIXED_DEFENSE
    doctrine_probabilities: dict[EnemyDoctrine, float] = field(default_factory=dict)
    deception_score: float = 0.0
    time_to_contact_s: float = float("inf")
    confidence: float = 0.0


@dataclass
class _Particle:
    """Internal representation of a single enemy hypothesis particle."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    regime: int = 0  # 0=DEFENDING, 1=ATTACKING, 2=RETREATING
    weight: float = 1.0


# ---------------------------------------------------------------------------
# Confidence Calibrator
# ---------------------------------------------------------------------------


class ConfidenceCalibrator:
    """Isotonic regression calibrator for confidence scores.

    Before fitting: calibrate() is identity (passthrough).
    After fitting with (raw_confidences, outcomes): calibrate() maps
    raw scores through a monotonically non-decreasing step function.
    """

    def __init__(self) -> None:
        self._fitted = False
        self._xs: list[float] = []
        self._ys: list[float] = []

    def fit(self, raw_confidences: list[float], outcomes: list[int]) -> None:
        """Fit isotonic regression from raw confidence → actual outcome rate.

        Computes empirical outcome rates per unique x-value, then applies the
        Pool Adjacent Violators (PAV) algorithm to enforce isotonicity on the
        per-bin means.  The resulting piecewise-linear function is stored as
        (self._xs, self._ys) breakpoint pairs.
        """
        # Pre-average duplicate x-values into empirical bins (weighted by count)
        bin_sums: dict[float, float] = {}
        bin_counts: dict[float, int] = {}
        for x, y in zip(raw_confidences, outcomes):
            bin_sums[x] = bin_sums.get(x, 0.0) + float(y)
            bin_counts[x] = bin_counts.get(x, 0) + 1
        sorted_xs = sorted(bin_sums)
        bin_means = [bin_sums[x] / bin_counts[x] for x in sorted_xs]

        # Pool Adjacent Violators on pre-averaged bins
        # Each "block" is a list of original bin indices
        blocks: list[list[int]] = [[i] for i in range(len(sorted_xs))]
        means: list[float] = list(bin_means)
        changed = True
        while changed:
            changed = False
            i = 0
            new_blocks: list[list[int]] = []
            new_means: list[float] = []
            while i < len(blocks):
                if i + 1 < len(blocks) and means[i] > means[i + 1]:
                    merged = blocks[i] + blocks[i + 1]
                    merged_count = sum(bin_counts[sorted_xs[j]] for j in merged)
                    merged_sum = sum(bin_sums[sorted_xs[j]] for j in merged)
                    new_blocks.append(merged)
                    new_means.append(merged_sum / merged_count)
                    changed = True
                    i += 2
                else:
                    new_blocks.append(blocks[i])
                    new_means.append(means[i])
                    i += 1
            blocks = new_blocks
            means = new_means

        # Build breakpoint arrays for piecewise-linear interpolation.
        # Between consecutive blocks, use their respective means as endpoints
        # so that linear interpolation produces a smooth, non-decreasing curve.
        self._xs = []
        self._ys = []
        for idx, (block, mean) in enumerate(zip(blocks, means)):
            x_lo = sorted_xs[block[0]]
            x_hi = sorted_xs[block[-1]]
            # Avoid duplicate x entries for single-point blocks
            if not self._xs or self._xs[-1] != x_lo:
                self._xs.append(x_lo)
                self._ys.append(mean)
            if x_hi != x_lo:
                self._xs.append(x_hi)
                self._ys.append(mean)
        self._fitted = True

    def calibrate(self, raw: float) -> float:
        """Map raw confidence through the calibration function."""
        if not self._fitted:
            return raw
        if raw <= self._xs[0]:
            return self._ys[0]
        if raw >= self._xs[-1]:
            return self._ys[-1]
        i = bisect.bisect_left(self._xs, raw)
        if i == 0:
            return self._ys[0]
        t = (raw - self._xs[i - 1]) / max(self._xs[i] - self._xs[i - 1], 1e-12)
        return self._ys[i - 1] + t * (self._ys[i] - self._ys[i - 1])


# ---------------------------------------------------------------------------
# Adversarial Engine
# ---------------------------------------------------------------------------

# Regime indices
_DEFEND = 0
_ATTACK = 1
_RETREAT = 2

# Default Markov transition matrix for threat regimes (mirrors Rust)
_THREAT_TRANSITION = [
    [0.85, 0.10, 0.05],  # DEFEND  -> [DEF, ATK, RET]
    [0.08, 0.82, 0.10],  # ATTACK  -> [DEF, ATK, RET]
    [0.12, 0.08, 0.80],  # RETREAT -> [DEF, ATK, RET]
]


class AdversarialEngine:
    """Dual particle filter: friendly + enemy.

    The KEY INNOVATION: predict enemy behavior before they complete their
    maneuver.  Enemy particles represent hypotheses about enemy intent.

    Each tracked threat gets its own cloud of particles.  The engine
    maintains regime probabilities per threat and computes:

    - Estimated position and velocity
    - Classified behavior (DEFENDING / ATTACKING / RETREATING)
    - Time-to-contact with the friendly fleet centroid
    - Position uncertainty (particle spread)

    Parameters
    ----------
    n_enemy_particles : int
        Number of particles per tracked threat.  More particles give
        better estimates at higher computational cost.
    """

    def __init__(
        self,
        n_enemy_particles: int = 500,
        seed: int | None = None,
        calibrator: ConfidenceCalibrator | None = None,
    ) -> None:
        self.n_particles = n_enemy_particles
        self._rng = random.Random(seed)
        self._calibrator = calibrator
        self._tracks: dict[int, list[_Particle]] = {}
        self._friendly_centroid = Vec3()
        self._transition = _THREAT_TRANSITION
        self._logical_time_s = 0.0
        self._doctrine_posteriors: dict[int, dict[EnemyDoctrine, float]] = {}
        self._velocity_history: dict[int, list[float]] = {}  # closing speed history per track
        self._behavior_history: dict[int, list[EnemyBehavior]] = {}  # for temporal deception
        self._last_observations: dict[int, SensorReading] = {}
        self._doctrine_prior = self._normalize_distribution(
            {
                EnemyDoctrine.FIXED_DEFENSE: 0.25,
                EnemyDoctrine.DIRECT_ASSAULT: 0.17,
                EnemyDoctrine.FLANKING: 0.13,
                EnemyDoctrine.FIGHTING_WITHDRAWAL: 0.12,
                EnemyDoctrine.DECOY: 0.08,
                EnemyDoctrine.EW_JAMMING: 0.08,
                EnemyDoctrine.PROBING: 0.08,
                EnemyDoctrine.FEINT: 0.05,
                EnemyDoctrine.COORDINATED_ATTACK: 0.04,
            }
        )

        # Process noise parameters per regime
        self._noise = {
            _DEFEND: {"pos": 0.1, "vel": 0.2},
            _ATTACK: {"pos": 0.3, "vel": 0.5},
            _RETREAT: {"pos": 0.25, "vel": 0.4},
        }

    # -- Public API ----------------------------------------------------------

    def set_friendly_centroid(self, centroid: Vec3) -> None:
        """Update the friendly fleet centroid for attack-vector computation."""
        self._friendly_centroid = centroid

    def init_track(self, threat_id: int, initial_pos: Vec3) -> None:
        """Initialize a new threat track with particles around the observed position."""
        particles = []
        for _ in range(self.n_particles):
            p = _Particle(
                x=initial_pos.x + self._rng.gauss(0, 5.0),
                y=initial_pos.y + self._rng.gauss(0, 5.0),
                z=initial_pos.z + self._rng.gauss(0, 2.0),
                vx=self._rng.gauss(0, 1.0),
                vy=self._rng.gauss(0, 1.0),
                vz=self._rng.gauss(0, 0.5),
                regime=self._rng.choices([0, 1, 2], weights=[0.6, 0.2, 0.2])[0],
                weight=1.0 / self.n_particles,
            )
            particles.append(p)
        self._tracks[threat_id] = particles
        self._doctrine_posteriors[threat_id] = dict(self._doctrine_prior)
        logger.info("Initialized threat track %d with %d particles", threat_id, self.n_particles)

    def predict_enemy(self, dt: float) -> dict[int, EnemyEstimate]:
        """Propagate all enemy tracks forward by dt seconds.

        Returns estimated state for every tracked threat.

        Dynamics per regime:
        - DEFENDING: mean-reverting velocity (alpha=0.5), low noise
        - ATTACKING: velocity tracks vector toward friendly centroid
        - RETREATING: velocity tracks vector away from engagement
        """
        self._logical_time_s += max(dt, 0.0)
        estimates = {}
        for threat_id, particles in self._tracks.items():
            self._predict_particles(particles, dt)
            doctrine_probs = self._update_doctrine_posterior(
                threat_id,
                particles=particles,
            )
            estimates[threat_id] = self._estimate(
                threat_id,
                particles,
                doctrine_probs=doctrine_probs,
                update_history=True,
            )
        return estimates

    def update_from_sensor(self, observation: SensorReading) -> None:
        """Incorporate a sensor observation to update particle weights.

        Supports position observations, bearing-only, and velocity hints.
        """
        if observation.timestamp <= 0.0:
            observation = replace(observation, timestamp=self._logical_time_s)

        threat_id = observation.threat_id
        if threat_id not in self._tracks:
            # Auto-initialize if we have position
            if observation.position:
                self.init_track(threat_id, observation.position)
            else:
                logger.warning("Observation for unknown threat %d with no position", threat_id)
                return

        particles = self._tracks[threat_id]

        if observation.position:
            self._update_position(particles, observation.position, observation.confidence)

        if observation.velocity:
            self._update_velocity(particles, observation.velocity, observation.confidence)

        if observation.bearing_rad is not None:
            self._update_bearing(particles, observation.bearing_rad, observation.confidence)

        # Normalize weights
        self._normalize_weights(particles)

        # Resample if ESS too low
        ess = self._effective_sample_size(particles)
        if ess < 0.5 * len(particles):
            particles = self._systematic_resample(particles)
            self._tracks[threat_id] = particles

        self._update_doctrine_posterior(threat_id, observation=observation, particles=particles)
        self._last_observations[threat_id] = observation

        # Track closing speed history for PROBING detection
        if observation.velocity:
            mean_pos, _ = self._weighted_mean_state(particles)
            dx = self._friendly_centroid.x - mean_pos.x
            dy = self._friendly_centroid.y - mean_pos.y
            dz = self._friendly_centroid.z - mean_pos.z
            dist = (dx * dx + dy * dy + dz * dz) ** 0.5
            if dist > 1e-3:
                closing = (
                    observation.velocity.x * dx
                    + observation.velocity.y * dy
                    + observation.velocity.z * dz
                ) / dist
                hist = self._velocity_history.setdefault(threat_id, [])
                hist.append(closing)
                if len(hist) > 10:
                    hist[:] = hist[-10:]

    def _is_closing(self, particles: list) -> bool:
        """Check if a track's weighted mean velocity is toward friendly centroid."""
        mean_pos, mean_vel = self._weighted_mean_state(particles)
        dx = self._friendly_centroid.x - mean_pos.x
        dy = self._friendly_centroid.y - mean_pos.y
        dz = self._friendly_centroid.z - mean_pos.z
        dist = (dx * dx + dy * dy + dz * dz) ** 0.5
        if dist < 1e-3:
            return False
        closing = (mean_vel.x * dx + mean_vel.y * dy + mean_vel.z * dz) / dist
        return closing > 2.0

    def classify_enemy_behavior(self, threat_id: int | None = None) -> dict[int, EnemyBehavior]:
        """Classify the most likely behavior for each (or a specific) threat.

        Classification is the regime with the highest weighted vote
        across all particles for that track.
        """
        targets = {threat_id: self._tracks[threat_id]} if threat_id is not None else self._tracks
        result = {}
        for tid, particles in targets.items():
            probs = self._regime_probabilities(particles)
            max_regime = max(probs, key=probs.get)  # type: ignore[arg-type]
            result[tid] = max_regime
        return result

    def classify_enemy_doctrine(self, threat_id: int | None = None) -> dict[int, EnemyDoctrine]:
        """Classify the most likely doctrinal hypothesis for each threat."""
        targets = {threat_id: self._tracks[threat_id]} if threat_id is not None else self._tracks
        result = {}
        for tid, particles in targets.items():
            posterior = self._compute_doctrine_posterior(
                tid,
                particles=particles,
            )
            result[tid] = max(posterior, key=posterior.get)
        return result

    def time_to_contact(self, threat_id: int | None = None) -> dict[int, float]:
        """Estimate time-to-contact with the friendly centroid for each threat.

        Uses the weighted mean velocity toward the centroid.
        Returns inf if the threat is moving away or stationary.
        """
        targets = {threat_id: self._tracks[threat_id]} if threat_id is not None else self._tracks
        result = {}
        for tid, particles in targets.items():
            est = self._estimate(
                tid,
                particles,
                doctrine_probs=self._compute_doctrine_posterior(tid, particles=particles),
                update_history=False,
            )
            result[tid] = est.time_to_contact_s
        return result

    def drop_track(self, threat_id: int) -> None:
        """Stop tracking a threat (e.g., confirmed destroyed)."""
        self._tracks.pop(threat_id, None)
        self._doctrine_posteriors.pop(threat_id, None)
        self._velocity_history.pop(threat_id, None)
        self._behavior_history.pop(threat_id, None)
        self._last_observations.pop(threat_id, None)

    @property
    def tracked_threats(self) -> list[int]:
        """List of currently tracked threat IDs."""
        return list(self._tracks.keys())

    # -- Private: Prediction -------------------------------------------------

    def _predict_particles(self, particles: list[_Particle], dt: float) -> None:
        """Propagate particles forward with regime-specific dynamics."""
        dt_sqrt = max(dt, 1e-8) ** 0.5
        fc = self._friendly_centroid

        for p in particles:
            # Regime transition (stochastic)
            p.regime = self._sample_regime_transition(p.regime)

            noise = self._noise[p.regime]

            if p.regime == _DEFEND:
                # Mean-reverting velocity
                p.vx = 0.5 * p.vx + self._rng.gauss(0, noise["vel"]) * dt_sqrt
                p.vy = 0.5 * p.vy + self._rng.gauss(0, noise["vel"]) * dt_sqrt
                p.vz = 0.5 * p.vz + self._rng.gauss(0, noise["vel"] * 0.5) * dt_sqrt

            elif p.regime == _ATTACK:
                # Velocity tracks vector toward friendly centroid
                dx = fc.x - p.x
                dy = fc.y - p.y
                dz = fc.z - p.z
                dist = max((dx**2 + dy**2 + dz**2) ** 0.5, 1.0)
                target_speed = 20.0  # assumed attack speed m/s
                tvx = target_speed * dx / dist
                tvy = target_speed * dy / dist
                tvz = target_speed * dz / dist
                beta = 0.3
                p.vx += beta * (tvx - p.vx) * dt + self._rng.gauss(0, noise["vel"]) * dt_sqrt
                p.vy += beta * (tvy - p.vy) * dt + self._rng.gauss(0, noise["vel"]) * dt_sqrt
                p.vz += beta * (tvz - p.vz) * dt + self._rng.gauss(0, noise["vel"] * 0.5) * dt_sqrt

            else:  # RETREAT
                # Velocity tracks vector away from engagement
                dx = p.x - fc.x
                dy = p.y - fc.y
                dz = 0.0  # retreat is mostly lateral
                dist = max((dx**2 + dy**2) ** 0.5, 1.0)
                target_speed = 15.0
                tvx = target_speed * dx / dist
                tvy = target_speed * dy / dist
                p.vx += 0.2 * (tvx - p.vx) * dt + self._rng.gauss(0, noise["vel"]) * dt_sqrt
                p.vy += 0.2 * (tvy - p.vy) * dt + self._rng.gauss(0, noise["vel"]) * dt_sqrt
                p.vz += self._rng.gauss(0, noise["vel"] * 0.3) * dt_sqrt

            # Position update
            p.x += p.vx * dt + self._rng.gauss(0, noise["pos"]) * dt_sqrt
            p.y += p.vy * dt + self._rng.gauss(0, noise["pos"]) * dt_sqrt
            p.z += p.vz * dt + self._rng.gauss(0, noise["pos"] * 0.5) * dt_sqrt

    def _sample_regime_transition(self, current: int) -> int:
        """Sample next regime from the Markov transition matrix."""
        row = self._transition[current]
        r = self._rng.random()
        cumsum = 0.0
        for regime_idx, prob in enumerate(row):
            cumsum += prob
            if r <= cumsum:
                return regime_idx
        return current

    # -- Private: Update -----------------------------------------------------

    def _update_position(self, particles: list[_Particle], obs_pos: Vec3, confidence: float) -> None:
        """Update particle weights from a position observation."""
        sigma2 = (10.0 / max(confidence, 0.1)) ** 2  # tighter sigma for higher confidence
        for p in particles:
            dx = p.x - obs_pos.x
            dy = p.y - obs_pos.y
            dz = p.z - obs_pos.z
            dist_sq = dx * dx + dy * dy + dz * dz
            likelihood = math.exp(-0.5 * dist_sq / sigma2)
            p.weight *= max(likelihood, 1e-300)

    def _update_velocity(self, particles: list[_Particle], obs_vel: Vec3, confidence: float) -> None:
        """Update particle weights from a velocity observation."""
        sigma2 = (5.0 / max(confidence, 0.1)) ** 2
        for p in particles:
            dvx = p.vx - obs_vel.x
            dvy = p.vy - obs_vel.y
            dvz = p.vz - obs_vel.z
            dist_sq = dvx * dvx + dvy * dvy + dvz * dvz
            likelihood = math.exp(-0.5 * dist_sq / sigma2)
            p.weight *= max(likelihood, 1e-300)

    def _update_bearing(self, particles: list[_Particle], bearing_rad: float, confidence: float) -> None:
        """Update particle weights from a bearing-only observation."""
        sigma2 = (0.2 / max(confidence, 0.1)) ** 2  # radians
        for p in particles:
            particle_bearing = math.atan2(p.y, p.x)
            diff = particle_bearing - bearing_rad
            # Wrap to [-pi, pi]
            diff = (diff + math.pi) % (2 * math.pi) - math.pi
            likelihood = math.exp(-0.5 * diff * diff / sigma2)
            p.weight *= max(likelihood, 1e-300)

    @staticmethod
    def _normalize_weights(particles: list[_Particle]) -> None:
        """Normalize particle weights with underflow protection."""
        for p in particles:
            p.weight += 1e-300
        total = sum(p.weight for p in particles)
        for p in particles:
            p.weight /= total

    @staticmethod
    def _effective_sample_size(particles: list[_Particle]) -> float:
        """ESS = 1 / sum(w^2)."""
        sum_sq = sum(p.weight**2 for p in particles)
        return 1.0 / (sum_sq + 1e-12)

    def _systematic_resample(self, particles: list[_Particle]) -> list[_Particle]:
        """O(N) systematic resampling, same algorithm as the Rust core."""
        n = len(particles)
        weights = [p.weight for p in particles]

        # Cumulative sum
        cumsum = [0.0] * n
        cumsum[0] = weights[0]
        for i in range(1, n):
            cumsum[i] = cumsum[i - 1] + weights[i]
        cumsum[-1] = 1.0  # pin

        step = 1.0 / n
        start = self._rng.uniform(0.0, step)
        uniform_weight = 1.0 / n

        new_particles = []
        j = 0
        for i in range(n):
            pos = start + step * i
            while j < n - 1 and cumsum[j] < pos:
                j += 1
            src = particles[j]
            new_particles.append(
                _Particle(
                    x=src.x,
                    y=src.y,
                    z=src.z,
                    vx=src.vx,
                    vy=src.vy,
                    vz=src.vz,
                    regime=src.regime,
                    weight=uniform_weight,
                )
            )

        return new_particles

    # -- Private: Estimation -------------------------------------------------

    @staticmethod
    def _normalize_distribution(distribution: dict, fallback: Optional[dict] = None) -> dict:
        """Normalize a positive-valued distribution with safe fallback."""
        cleaned = {key: max(float(value), 0.0) for key, value in distribution.items()}
        total = sum(cleaned.values())
        if total <= 1e-12:
            if fallback is not None:
                return dict(fallback)
            if not cleaned:
                return {}
            uniform = 1.0 / len(cleaned)
            return {key: uniform for key in cleaned}
        return {key: value / total for key, value in cleaned.items()}

    @staticmethod
    def _weighted_mean_state(particles: list[_Particle]) -> tuple[Vec3, Vec3]:
        """Compute weighted mean position and velocity for a particle cloud."""
        mx = my = mz = 0.0
        mvx = mvy = mvz = 0.0
        for p in particles:
            mx += p.x * p.weight
            my += p.y * p.weight
            mz += p.z * p.weight
            mvx += p.vx * p.weight
            mvy += p.vy * p.weight
            mvz += p.vz * p.weight
        return Vec3(mx, my, mz), Vec3(mvx, mvy, mvz)

    def _doctrine_memory_prior(self, threat_id: int) -> dict[EnemyDoctrine, float]:
        posterior = self._doctrine_posteriors.get(threat_id)
        if posterior is None:
            return dict(self._doctrine_prior)

        observation = self._last_observations.get(threat_id)
        if observation is None:
            return dict(posterior)

        age_s = max(0.0, self._logical_time_s - observation.timestamp)
        memory = math.exp(-age_s / 15.0)
        blended = {
            doctrine: posterior.get(doctrine, 0.0) * memory + self._doctrine_prior[doctrine] * (1.0 - memory)
            for doctrine in self._doctrine_prior
        }
        return self._normalize_distribution(blended, self._doctrine_prior)

    def _compute_doctrine_posterior(
        self,
        threat_id: int,
        observation: SensorReading | None = None,
        particles: Optional[list[_Particle]] = None,
    ) -> dict[EnemyDoctrine, float]:
        """Compute doctrinal hypothesis posterior from kinematics and sensor cues."""
        particles = particles or self._tracks.get(threat_id)
        if not particles:
            return dict(self._doctrine_prior)

        prior = self._doctrine_memory_prior(threat_id)
        obs_likelihood = self._doctrine_likelihoods_from_observation(observation, particles)
        behavior_likelihood = self._behavior_to_doctrine_likelihoods(particles)
        behavior_weight = 1.0 if observation is not None else 0.0

        posterior = {}
        for doctrine, prior_prob in prior.items():
            posterior[doctrine] = (
                prior_prob
                * obs_likelihood.get(doctrine, 1.0)
                * behavior_likelihood.get(doctrine, 1.0) ** behavior_weight
            )

        return self._normalize_distribution(
            posterior,
            self._doctrine_prior,
        )

    def _update_doctrine_posterior(
        self,
        threat_id: int,
        observation: SensorReading | None = None,
        particles: Optional[list[_Particle]] = None,
    ) -> dict[EnemyDoctrine, float]:
        posterior = self._compute_doctrine_posterior(threat_id, observation=observation, particles=particles)
        self._doctrine_posteriors[threat_id] = posterior
        return posterior

    def _doctrine_likelihoods_from_observation(
        self,
        observation: SensorReading | None,
        particles: list[_Particle],
    ) -> dict[EnemyDoctrine, float]:
        """Score doctrinal hypotheses from current observation geometry."""
        mean_pos, mean_vel = self._weighted_mean_state(particles)
        pos = observation.position if observation and observation.position is not None else mean_pos
        vel = observation.velocity if observation and observation.velocity is not None else mean_vel
        confidence = observation.confidence if observation is not None else 0.4

        dx = self._friendly_centroid.x - pos.x
        dy = self._friendly_centroid.y - pos.y
        dz = self._friendly_centroid.z - pos.z
        dist = max((dx**2 + dy**2 + dz**2) ** 0.5, 1.0)

        speed = vel.norm()
        closing_speed = (vel.x * dx + vel.y * dy + vel.z * dz) / dist
        lateral_speed = max(speed * speed - closing_speed * closing_speed, 0.0) ** 0.5
        lateral_ratio = lateral_speed / max(speed, 1.0)
        signal_strength = observation.signal_strength_dbm if observation is not None else None
        radar_cross_section = observation.radar_cross_section if observation is not None else None
        sensor_type = (observation.sensor_type if observation is not None else "unknown").lower()

        likelihoods = {
            EnemyDoctrine.FIXED_DEFENSE: 0.65,
            EnemyDoctrine.DIRECT_ASSAULT: 0.65,
            EnemyDoctrine.FLANKING: 0.65,
            EnemyDoctrine.FIGHTING_WITHDRAWAL: 0.65,
            EnemyDoctrine.DECOY: 0.65,
            EnemyDoctrine.EW_JAMMING: 0.65,
        }

        if speed < 4.0 and dist < 800.0:
            likelihoods[EnemyDoctrine.FIXED_DEFENSE] += 0.7 * confidence
        if closing_speed > 4.0 and lateral_ratio < 0.4:
            likelihoods[EnemyDoctrine.DIRECT_ASSAULT] += 1.0 * confidence
        if closing_speed > 0.5 and lateral_ratio >= 0.45:
            likelihoods[EnemyDoctrine.FLANKING] += 1.0 * confidence
        if closing_speed < -2.0:
            likelihoods[EnemyDoctrine.FIGHTING_WITHDRAWAL] += 1.1 * confidence

        if signal_strength is not None:
            if signal_strength > -55.0:
                likelihoods[EnemyDoctrine.EW_JAMMING] += 1.8 * confidence
            if signal_strength > -45.0:
                likelihoods[EnemyDoctrine.EW_JAMMING] += 1.0 * confidence
            if signal_strength > -50.0 and (radar_cross_section is None or radar_cross_section < 0.2):
                likelihoods[EnemyDoctrine.DECOY] += 0.9 * confidence

        if sensor_type in {"ew", "rf", "sigint"}:
            likelihoods[EnemyDoctrine.EW_JAMMING] += 1.4 * confidence
            if signal_strength is not None and signal_strength > -40.0:
                likelihoods[EnemyDoctrine.EW_JAMMING] += 0.8 * confidence
        if observation is not None and observation.bearing_rad is not None and observation.position is None:
            likelihoods[EnemyDoctrine.DECOY] += 0.4 * confidence
        if radar_cross_section is not None and radar_cross_section < 0.15:
            likelihoods[EnemyDoctrine.DECOY] += 0.5 * confidence

        # PROBING: oscillating approach (uses velocity history)
        history = self._velocity_history.get(
            observation.threat_id if observation else None, []
        )
        if len(history) >= 3:
            sign_changes = sum(
                1 for a, b in zip(history[-3:], history[-2:]) if (a > 0) != (b > 0)
            )
            if sign_changes >= 1:
                likelihoods[EnemyDoctrine.PROBING] = 0.65 + 0.9 * confidence
        # FEINT: attack signals + deception indicators
        if closing_speed > 3.0 and (
            radar_cross_section is not None and radar_cross_section < 0.2
        ):
            likelihoods[EnemyDoctrine.FEINT] = 0.65 + 0.8 * confidence
        # COORDINATED_ATTACK: multiple tracks closing simultaneously
        n_closing = sum(
            1 for tid, pts in self._tracks.items()
            if tid != (observation.threat_id if observation else None)
            and len(pts) > 0
            and self._is_closing(pts)
        )
        if n_closing >= 1 and closing_speed > 2.0:
            likelihoods[EnemyDoctrine.COORDINATED_ATTACK] = 0.65 + 0.7 * confidence * min(n_closing, 3) / 3.0

        return likelihoods

    def _behavior_to_doctrine_likelihoods(
        self,
        particles: list[_Particle],
    ) -> dict[EnemyDoctrine, float]:
        """Map coarse particle regimes to richer doctrinal hypotheses."""
        behavior_probs = self._regime_probabilities(particles)
        defend = behavior_probs.get(EnemyBehavior.DEFENDING, 0.0)
        attack = behavior_probs.get(EnemyBehavior.ATTACKING, 0.0)
        retreat = behavior_probs.get(EnemyBehavior.RETREATING, 0.0)

        return {
            EnemyDoctrine.FIXED_DEFENSE: 0.7 + defend * 1.2,
            EnemyDoctrine.DIRECT_ASSAULT: 0.7 + attack * 1.3,
            EnemyDoctrine.FLANKING: 0.7 + attack * 0.9,
            EnemyDoctrine.FIGHTING_WITHDRAWAL: 0.7 + retreat * 1.4,
            EnemyDoctrine.DECOY: 0.7 + defend * 0.3 + retreat * 0.2,
            EnemyDoctrine.EW_JAMMING: 0.7 + defend * 0.2 + attack * 0.2,
            EnemyDoctrine.PROBING: 0.7 + attack * 0.4 + defend * 0.3,
            EnemyDoctrine.FEINT: 0.7 + attack * 0.5 + retreat * 0.2,
            EnemyDoctrine.COORDINATED_ATTACK: 0.7 + attack * 1.0,
        }

    def _estimate(
        self,
        threat_id: int,
        particles: list[_Particle],
        doctrine_probs: dict[EnemyDoctrine, float] | None = None,
        update_history: bool = True,
    ) -> EnemyEstimate:
        """Compute weighted estimate from a particle cloud."""
        mean_pos, mean_vel = self._weighted_mean_state(particles)

        # Position uncertainty: weighted RMS distance from mean
        var = 0.0
        for p in particles:
            var += p.weight * ((p.x - mean_pos.x) ** 2 + (p.y - mean_pos.y) ** 2 + (p.z - mean_pos.z) ** 2)
        uncertainty = var**0.5

        # Regime probabilities
        probs = self._regime_probabilities(particles)
        max_behavior = max(probs, key=probs.get)  # type: ignore[arg-type]

        doctrine_probs = dict(doctrine_probs or self._doctrine_posteriors.get(threat_id, self._doctrine_prior))
        dominant_doctrine = max(doctrine_probs, key=doctrine_probs.get)
        base_deception = min(
            1.0,
            doctrine_probs.get(EnemyDoctrine.DECOY, 0.0)
            + doctrine_probs.get(EnemyDoctrine.EW_JAMMING, 0.0),
        )
        dominant_behavior = max(probs, key=probs.get)
        hist = list(self._behavior_history.get(threat_id, []))
        if update_history:
            stored_hist = self._behavior_history.setdefault(threat_id, [])
            stored_hist.append(dominant_behavior)
            if len(stored_hist) > 10:
                stored_hist[:] = stored_hist[-10:]
            hist = stored_hist

        transition_rate = 0.0
        if len(hist) >= 2:
            transitions = sum(1 for a, b in zip(hist, hist[1:]) if a != b)
            transition_rate = transitions / (len(hist) - 1)
        deception_score = min(1.0, base_deception + 0.3 * transition_rate)

        # Time-to-contact
        ttc = self._compute_ttc(mean_pos, mean_vel)

        # Overall confidence blends kinematic concentration with doctrine certainty.
        kinematic_confidence = max(0.0, min(1.0, 1.0 - uncertainty / 100.0))
        doctrine_certainty = max(doctrine_probs.values()) if doctrine_probs else 0.0
        confidence = max(
            0.0,
            min(
                1.0,
                kinematic_confidence * 0.75 + doctrine_certainty * 0.25 - deception_score * 0.15,
            ),
        )
        if self._calibrator is not None:
            confidence = self._calibrator.calibrate(confidence)

        return EnemyEstimate(
            threat_id=threat_id,
            position=mean_pos,
            velocity=mean_vel,
            position_uncertainty_m=uncertainty,
            behavior=max_behavior,
            behavior_probabilities=probs,
            dominant_doctrine=dominant_doctrine,
            doctrine_probabilities=doctrine_probs,
            deception_score=deception_score,
            time_to_contact_s=ttc,
            confidence=confidence,
        )

    def _regime_probabilities(self, particles: list[_Particle]) -> dict[EnemyBehavior, float]:
        """Compute weighted regime probabilities from particles."""
        probs = {
            EnemyBehavior.DEFENDING: 0.0,
            EnemyBehavior.ATTACKING: 0.0,
            EnemyBehavior.RETREATING: 0.0,
        }
        regime_map = {
            _DEFEND: EnemyBehavior.DEFENDING,
            _ATTACK: EnemyBehavior.ATTACKING,
            _RETREAT: EnemyBehavior.RETREATING,
        }

        for p in particles:
            behavior = regime_map.get(p.regime, EnemyBehavior.DEFENDING)
            probs[behavior] += p.weight

        return probs

    def _compute_ttc(self, pos: Vec3, vel: Vec3) -> float:
        """Compute time-to-contact with the friendly centroid.

        Projects the velocity onto the threat-to-centroid vector.
        Returns inf if closing speed is zero or negative (moving away).
        """
        dx = self._friendly_centroid.x - pos.x
        dy = self._friendly_centroid.y - pos.y
        dz = self._friendly_centroid.z - pos.z
        dist = (dx**2 + dy**2 + dz**2) ** 0.5

        if dist < 1.0:
            return 0.0

        # Project velocity onto the threat-to-centroid direction
        closing_speed = (vel.x * dx + vel.y * dy + vel.z * dz) / dist

        if closing_speed <= 0.0:
            return float("inf")

        return dist / closing_speed


def adversarial_to_risk_context(estimates: dict[int, EnemyEstimate]) -> dict:
    """Convert adversarial estimates to risk context for the auction system.

    Returns a dict with threat_density, confidence, doom_value, max_deception
    suitable for feeding into ScenarioContext on the Auctioneer.
    """
    if not estimates:
        return {"threat_density": 0.0, "confidence": 0.5, "doom_value": 0.0, "max_deception": 0.0}

    n = len(estimates)
    avg_confidence = sum(e.confidence for e in estimates.values()) / n
    max_deception = max(e.deception_score for e in estimates.values())
    attackers = sum(1 for e in estimates.values() if e.behavior == EnemyBehavior.ATTACKING)
    threat_density = attackers / max(n, 1)

    min_ttc = min((e.time_to_contact_s for e in estimates.values()), default=float("inf"))
    doom_value = -1.0 * attackers * (1.0 / max(min_ttc, 1.0))

    return {
        "threat_density": threat_density,
        "confidence": avg_confidence * (1.0 - max_deception * 0.3),
        "doom_value": doom_value,
        "max_deception": max_deception,
    }
