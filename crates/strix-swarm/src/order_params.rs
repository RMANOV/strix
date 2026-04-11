//! Global order parameters for swarm health monitoring.
//!
//! Five macroscopic metrics computed from per-drone state each tick.
//! These close the regulation loop at the system level — without them,
//! criticality scheduler works only from local signals.

use nalgebra::Vector3;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use strix_core::state::Regime;

/// Five global order parameters for the swarm.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct OrderParameters {
    /// Vicsek-style alignment: mean normalised velocity dot product ∈ [0, 1].
    /// 1.0 = all drones flying same direction, 0.0 = random headings.
    pub alignment_order: f64,
    /// Fragmentation index ∈ [0, 1]. 0.0 = tight cluster, 1.0 = maximally dispersed.
    pub fragmentation_index: f64,
    /// Shannon entropy of trust values ∈ [0, 1]. High = uniform trust, low = polarised.
    pub trust_entropy: f64,
    /// Coverage dispersion: normalised standard deviation of pairwise distances ∈ [0, 1].
    pub coverage_dispersion: f64,
    /// Mission progress surrogate ∈ [0, 1]. Fraction of drones in non-Patrol regime.
    pub mission_progress: f64,
}

impl OrderParameters {
    /// Compute all five metrics from per-drone data.
    ///
    /// - `positions`: (drone_id, position) pairs
    /// - `velocities`: drone_id → velocity
    /// - `regimes`: drone_id → regime
    /// - `per_drone_trust`: drone_id → aggregate trust score ∈ [0, 1].
    ///   Pass actual trust values from `TrustTracker::aggregate_trust()`.
    ///   Falls back to `per_drone_fear` semantics if trust is unavailable
    ///   (same computation, different interpretation).
    pub fn compute(
        positions: &[(u32, Vector3<f64>)],
        velocities: &HashMap<u32, Vector3<f64>>,
        regimes: &HashMap<u32, Regime>,
        per_drone_trust: &HashMap<u32, f64>,
    ) -> Self {
        let alignment_order = compute_alignment(positions, velocities);
        let fragmentation_index = compute_fragmentation(positions);
        let trust_entropy = compute_trust_entropy(per_drone_trust);
        let coverage_dispersion = compute_coverage_dispersion(positions);
        let mission_progress = compute_mission_progress(regimes);

        Self {
            alignment_order,
            fragmentation_index,
            trust_entropy,
            coverage_dispersion,
            mission_progress,
        }
    }

    /// Generate epistemic gate signals from computed order parameters.
    ///
    /// - High entropy + low alignment → macro XOR conflict (swarm is incoherent)
    /// - Low entropy + high alignment → macro XNOR corroboration (swarm is coherent)
    pub fn gate_signals(&self, now: f64) -> Vec<strix_mesh::bool_gates::GateSignal> {
        use strix_mesh::bool_gates::{GateSignal, SignalSource};
        let mut signals = Vec::new();

        if self.trust_entropy > 0.8 && self.alignment_order < 0.3 {
            let severity =
                ((self.trust_entropy - 0.8) + (0.3 - self.alignment_order)).clamp(0.0, 1.0);
            signals.push(GateSignal::Conflict {
                source_a: SignalSource::OrderParams,
                source_b: SignalSource::OrderParams,
                severity,
                timestamp: now,
            });
        }

        if self.trust_entropy < 0.2 && self.alignment_order > 0.8 {
            signals.push(GateSignal::Corroboration {
                sources: vec![SignalSource::OrderParams, SignalSource::OrderParams],
                independence: 0.7,
                confidence_boost: 0.01,
                timestamp: now,
            });
        }

        signals
    }
}

/// Vicsek alignment: |mean(v̂_i)| where v̂ = v/|v| for moving drones.
fn compute_alignment(
    positions: &[(u32, Vector3<f64>)],
    velocities: &HashMap<u32, Vector3<f64>>,
) -> f64 {
    let mut sum = Vector3::zeros();
    let mut count = 0usize;

    for (id, _) in positions {
        if let Some(v) = velocities.get(id) {
            let speed = v.norm();
            if speed > 1e-3 {
                sum += v / speed;
                count += 1;
            }
        }
    }

    if count == 0 {
        return 1.0; // all stationary = perfectly aligned (trivially)
    }

    (sum / count as f64).norm().clamp(0.0, 1.0)
}

/// Fragmentation: normalised mean distance from centroid.
fn compute_fragmentation(positions: &[(u32, Vector3<f64>)]) -> f64 {
    let n = positions.len();
    if n <= 1 {
        return 0.0;
    }

    let centroid: Vector3<f64> = positions.iter().map(|(_, p)| p).sum::<Vector3<f64>>() / n as f64;

    let mean_dist: f64 = positions
        .iter()
        .map(|(_, p)| (p - centroid).norm())
        .sum::<f64>()
        / n as f64;

    // Normalise by characteristic scale (100m comm radius).
    (mean_dist / 100.0).clamp(0.0, 1.0)
}

/// Shannon entropy of per-peer trust scores, normalised to [0, 1].
///
/// High entropy = uniform trust (healthy disagreement or uniform trust).
/// Low entropy = polarised trust (some peers highly trusted, others not).
fn compute_trust_entropy(per_drone_trust: &HashMap<u32, f64>) -> f64 {
    let n = per_drone_trust.len();
    if n <= 1 {
        return 0.0;
    }

    // Bucket into 10 bins
    let mut bins = [0u32; 10];
    for &f in per_drone_trust.values() {
        let f = f.clamp(0.0, 1.0);
        let idx = ((f * 9.999) as usize).min(9);
        bins[idx] += 1;
    }

    let n_f = n as f64;
    let mut entropy = 0.0;
    for &count in &bins {
        if count > 0 {
            let p = count as f64 / n_f;
            entropy -= p * p.ln();
        }
    }

    // Normalise by max entropy (ln(10))
    let max_entropy = (10.0_f64).ln();
    if max_entropy > 1e-12 {
        (entropy / max_entropy).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Coverage dispersion: coefficient of variation of pairwise distances.
fn compute_coverage_dispersion(positions: &[(u32, Vector3<f64>)]) -> f64 {
    let n = positions.len();
    if n <= 1 {
        return 0.0;
    }

    // Sample pairwise distances (cap at 200 pairs to avoid O(n²) for large swarms)
    let mut distances = Vec::new();
    let max_pairs = 200;
    'outer: for i in 0..n {
        for j in (i + 1)..n {
            distances.push((positions[i].1 - positions[j].1).norm());
            if distances.len() >= max_pairs {
                break 'outer;
            }
        }
    }

    if distances.is_empty() {
        return 0.0;
    }

    let mean = distances.iter().sum::<f64>() / distances.len() as f64;
    if mean < 1e-6 {
        return 0.0;
    }

    let variance =
        distances.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / distances.len() as f64;
    let cv = variance.sqrt() / mean;

    // Normalise: CV of 1.0 = dispersion of 1.0
    cv.clamp(0.0, 1.0)
}

/// Mission progress: fraction of drones NOT in Patrol.
fn compute_mission_progress(regimes: &HashMap<u32, Regime>) -> f64 {
    if regimes.is_empty() {
        return 0.0;
    }
    let active = regimes
        .values()
        .filter(|r| !matches!(r, Regime::Patrol))
        .count();
    active as f64 / regimes.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_swarm_returns_defaults() {
        let op = OrderParameters::compute(&[], &HashMap::new(), &HashMap::new(), &HashMap::new());
        // alignment=1.0 (trivially aligned when no drones), frag=0, progress=0
        assert_eq!(op.fragmentation_index, 0.0);
        assert_eq!(op.mission_progress, 0.0);
        assert_eq!(op.coverage_dispersion, 0.0);
    }

    #[test]
    fn perfect_alignment() {
        let positions = vec![
            (0, Vector3::new(0.0, 0.0, 0.0)),
            (1, Vector3::new(10.0, 0.0, 0.0)),
        ];
        let mut velocities = HashMap::new();
        velocities.insert(0, Vector3::new(5.0, 0.0, 0.0));
        velocities.insert(1, Vector3::new(5.0, 0.0, 0.0));

        let op =
            OrderParameters::compute(&positions, &velocities, &HashMap::new(), &HashMap::new());
        assert!(
            (op.alignment_order - 1.0).abs() < 1e-6,
            "same direction = 1.0"
        );
    }

    #[test]
    fn opposite_alignment() {
        let positions = vec![
            (0, Vector3::new(0.0, 0.0, 0.0)),
            (1, Vector3::new(10.0, 0.0, 0.0)),
        ];
        let mut velocities = HashMap::new();
        velocities.insert(0, Vector3::new(5.0, 0.0, 0.0));
        velocities.insert(1, Vector3::new(-5.0, 0.0, 0.0));

        let op =
            OrderParameters::compute(&positions, &velocities, &HashMap::new(), &HashMap::new());
        assert!(op.alignment_order < 0.1, "opposite directions ≈ 0.0");
    }

    #[test]
    fn mission_progress_counts_non_patrol() {
        let mut regimes = HashMap::new();
        regimes.insert(0, Regime::Patrol);
        regimes.insert(1, Regime::Engage);
        regimes.insert(2, Regime::Evade);
        regimes.insert(3, Regime::Patrol);

        let op = OrderParameters::compute(&[], &HashMap::new(), &regimes, &HashMap::new());
        assert!((op.mission_progress - 0.5).abs() < 1e-6, "2/4 active");
    }

    #[test]
    fn trust_entropy_uniform_is_high() {
        let mut fear = HashMap::new();
        for i in 0..100 {
            fear.insert(i, (i as f64) / 100.0); // spread across 0-1
        }
        let op = OrderParameters::compute(&[], &HashMap::new(), &HashMap::new(), &fear);
        assert!(
            op.trust_entropy > 0.7,
            "uniform spread = high entropy, got {}",
            op.trust_entropy
        );
    }

    #[test]
    fn trust_entropy_concentrated_is_low() {
        let mut fear = HashMap::new();
        for i in 0..100 {
            fear.insert(i, 0.5); // all identical
        }
        let op = OrderParameters::compute(&[], &HashMap::new(), &HashMap::new(), &fear);
        assert!(
            op.trust_entropy < 0.2,
            "concentrated = low entropy, got {}",
            op.trust_entropy
        );
    }
}
