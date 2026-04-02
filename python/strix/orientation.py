"""Explicit orientation and integrity layer for STRIX mission planning.

This module makes the OODA-style `orientation` phase explicit instead of
leaving it implicit inside regime transitions and planner heuristics.

The layer tracks five things:
- trust by information source
- doctrine posterior drift
- novelty / surprise in the operating picture
- self-model mismatch between expected and observed state
- memory of broken assumptions
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    if not math.isfinite(value):
        return low
    return max(low, min(high, value))


def _mean(values: Iterable[float], default: float = 0.0) -> float:
    items = [value for value in values if math.isfinite(value)]
    if not items:
        return default
    return sum(items) / len(items)


def _normalize(weights: dict[str, float]) -> dict[str, float]:
    cleaned = {key: max(float(value), 0.0) for key, value in weights.items()}
    total = sum(cleaned.values())
    if total <= 1e-12:
        return {}
    return {key: value / total for key, value in cleaned.items()}


def _total_variation_distance(left: dict[str, float], right: dict[str, float]) -> float:
    keys = set(left) | set(right)
    if not keys:
        return 0.0
    return 0.5 * sum(abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in keys)


@dataclass(frozen=True)
class BrokenAssumption:
    """A planner assumption that failed against the current environment."""

    name: str
    severity: float
    observed_at_s: float
    details: str = ""


@dataclass(frozen=True)
class OrientationConfig:
    """Tunable parameters for the orientation layer."""

    trust_learning_rate: float = 0.12
    trust_decay_halflife_s: float = 45.0
    novelty_window: int = 32
    broken_assumption_capacity: int = 32
    mismatch_alert_threshold: float = 0.55
    survive_attrition_floor: float = 0.85
    surprise_floor: float = 0.02


@dataclass(frozen=True)
class OrientationSnapshot:
    """Read-only summary of the current orientation state."""

    regime: str = "PATROL"
    trust_by_source: dict[str, float] = field(default_factory=dict)
    doctrine_posterior: dict[str, float] = field(default_factory=dict)
    novelty_score: float = 0.0
    self_model_mismatch: float = 0.0
    integrity_score: float = 1.0
    evade_bias: float = 0.0
    engage_bias: float = 0.0
    confidence_multiplier: float = 1.0
    broken_assumptions: tuple[str, ...] = ()

    def reason_fragment(self) -> str:
        return (
            f"orientation integrity={self.integrity_score:.2f}; "
            f"surprise={self.novelty_score:.2f}; "
            f"mismatch={self.self_model_mismatch:.2f}"
        )


class OrientationEngine:
    """Tracks trust, novelty, mismatch and broken assumptions over time."""

    def __init__(self, config: OrientationConfig | None = None) -> None:
        self.config = config or OrientationConfig()
        self._trust_by_source: dict[str, float] = {}
        self._last_source_update_s: dict[str, float] = {}
        self._doctrine_posterior: dict[str, float] = {}
        self._pressure_history: deque[float] = deque(maxlen=self.config.novelty_window)
        self._broken_assumptions: deque[BrokenAssumption] = deque(
            maxlen=self.config.broken_assumption_capacity
        )
        self._snapshot = OrientationSnapshot()

    def snapshot(self) -> OrientationSnapshot:
        return self._snapshot

    def observe_source(self, source: str, confidence: float, now_s: float) -> None:
        """Update trust for a named source from its latest confidence sample."""
        name = (source or "unknown").strip().lower() or "unknown"
        confidence = _clamp(confidence)
        trust = self._trust_by_source.get(name, 0.5)
        age_s = max(0.0, now_s - self._last_source_update_s.get(name, now_s))
        if self.config.trust_decay_halflife_s > 1e-6 and age_s > 0.0:
            memory = math.exp(-math.log(2.0) * age_s / self.config.trust_decay_halflife_s)
            trust = 0.5 + (trust - 0.5) * memory
        lr = _clamp(self.config.trust_learning_rate)
        trust = (1.0 - lr) * trust + lr * confidence
        self._trust_by_source[name] = _clamp(trust, 0.05, 0.99)
        self._last_source_update_s[name] = max(0.0, now_s)

    def observe_doctrine_distribution(self, doctrine_scores: dict[str, float]) -> None:
        """Blend a new doctrine distribution into the running posterior."""
        incoming = _normalize({(key or "unknown").lower(): value for key, value in doctrine_scores.items()})
        if not incoming:
            return
        if not self._doctrine_posterior:
            self._doctrine_posterior = incoming
            return

        blended: dict[str, float] = {}
        keys = set(self._doctrine_posterior) | set(incoming)
        for key in keys:
            blended[key] = 0.8 * self._doctrine_posterior.get(key, 0.0) + 0.2 * incoming.get(key, 0.0)
        self._doctrine_posterior = _normalize(blended)

    def register_broken_assumption(
        self,
        name: str,
        severity: float,
        now_s: float,
        details: str = "",
    ) -> None:
        self._broken_assumptions.append(
            BrokenAssumption(
                name=name,
                severity=_clamp(severity),
                observed_at_s=max(0.0, now_s),
                details=details,
            )
        )

    def update(
        self,
        *,
        now_s: float,
        regime: str,
        planner_confidence: float | None,
        comms_quality: float,
        stale_age_s: float,
        fleet_size: int,
        alive_fraction: float,
        threat_confidences: Iterable[float],
        source_confidences: dict[str, float] | None = None,
        doctrine_scores: dict[str, float] | None = None,
    ) -> OrientationSnapshot:
        """Recompute the current orientation snapshot from the latest evidence."""
        if source_confidences:
            for source, confidence in source_confidences.items():
                self.observe_source(source, confidence, now_s)
        previous_doctrine = dict(self._doctrine_posterior)
        if doctrine_scores:
            self.observe_doctrine_distribution(doctrine_scores)

        mean_trust = _mean(self._trust_by_source.values(), default=0.5)
        average_threat = _mean(threat_confidences, default=0.0)
        alive_fraction = _clamp(alive_fraction)
        comms_quality = _clamp(comms_quality)
        stale_penalty = _clamp(stale_age_s / 2.0)
        doctrine_shift = _total_variation_distance(previous_doctrine, self._doctrine_posterior)
        attrition_gap = max(0.0, self.config.survive_attrition_floor - alive_fraction)
        attrition_penalty = _clamp(attrition_gap / max(self.config.survive_attrition_floor, 1e-6))

        pressure = _clamp(
            0.40 * average_threat
            + 0.25 * (1.0 - comms_quality)
            + 0.15 * stale_penalty
            + 0.20 * attrition_penalty
        )
        baseline_pressure = _mean(self._pressure_history, default=pressure)
        novelty = _clamp(abs(pressure - baseline_pressure) * 0.65 + doctrine_shift * 0.35)
        if self._pressure_history:
            novelty = max(novelty, self.config.surprise_floor)
        self._pressure_history.append(pressure)

        expected_state = _clamp(planner_confidence) if planner_confidence is not None else _clamp(1.0 - average_threat)
        observed_state = _clamp(1.0 - pressure)
        self_model_mismatch = _clamp(abs(expected_state - observed_state) + 0.35 * doctrine_shift)

        if self_model_mismatch >= self.config.mismatch_alert_threshold:
            self.register_broken_assumption(
                "planner_world_model_gap",
                self_model_mismatch,
                now_s,
                details=(
                    f"regime={regime.lower()} fleet={fleet_size} "
                    f"expected={expected_state:.2f} observed={observed_state:.2f}"
                ),
            )
        if attrition_penalty > 0.0:
            self.register_broken_assumption(
                "attrition_floor_breach",
                attrition_penalty,
                now_s,
                details=f"alive_fraction={alive_fraction:.2f}",
            )

        integrity = _clamp(
            1.0 - 0.50 * self_model_mismatch - 0.25 * novelty - 0.25 * (1.0 - mean_trust),
            0.05,
            1.0,
        )

        doctrine_aggression = max(
            (
                value
                for key, value in self._doctrine_posterior.items()
                if any(token in key for token in ("attack", "assault", "strike", "engage", "coordinated"))
            ),
            default=0.0,
        )
        engage_bias = _clamp(doctrine_aggression * mean_trust * (1.0 - 0.5 * self_model_mismatch))
        evade_bias = _clamp(0.55 * self_model_mismatch + 0.30 * novelty + 0.15 * (1.0 - mean_trust))
        confidence_multiplier = _clamp(0.55 + 0.45 * integrity - 0.10 * novelty, 0.35, 1.0)

        recent_breaks = tuple(item.name for item in list(self._broken_assumptions)[-4:])
        self._snapshot = OrientationSnapshot(
            regime=regime,
            trust_by_source=dict(sorted(self._trust_by_source.items())),
            doctrine_posterior=dict(sorted(self._doctrine_posterior.items())),
            novelty_score=novelty,
            self_model_mismatch=self_model_mismatch,
            integrity_score=integrity,
            evade_bias=evade_bias,
            engage_bias=engage_bias,
            confidence_multiplier=confidence_multiplier,
            broken_assumptions=recent_breaks,
        )
        return self._snapshot

    def recommend_regime(self, current_regime: str, proposed_regime: str) -> str:
        """Bias regime selection using the latest integrity snapshot."""
        snapshot = self._snapshot
        current = (current_regime or "PATROL").upper()
        proposed = (proposed_regime or current).upper()
        if snapshot.integrity_score < 0.35 or snapshot.evade_bias > 0.70:
            return "EVADE"
        if proposed != "EVADE" and snapshot.engage_bias > 0.55 and snapshot.integrity_score > 0.45:
            return "ENGAGE"
        return proposed

    def reweight_confidence(self, confidence: float) -> float:
        """Apply the current integrity gate to a planner or assignment confidence."""
        return _clamp(_clamp(confidence) * self._snapshot.confidence_multiplier, 0.05, 1.0)
