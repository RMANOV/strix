//! NeuralBarrier — learned safety filter using GCBF+ GNN.
//!
//! This is the primary public interface for GCBF+ evaluation.
//! It wraps the GNN encoder and provides per-agent safety filtering
//! compatible with the classical CBF result interface.

use super::config::GcbfConfig;
use super::graph::GraphTopology;
use super::nn::GnnEncoder;
use super::weights::{default_weights, GcbfWeights, WeightError};
use crate::cbf::{CbfResult, NoFlyZone};
use nalgebra::Vector3;
use std::collections::HashMap;

/// Learned neural control barrier function.
pub struct NeuralBarrier {
    encoder: GnnEncoder,
    config: GcbfConfig,
}

impl NeuralBarrier {
    /// Create from pre-loaded weights.
    pub fn from_weights(weights: GcbfWeights, config: GcbfConfig) -> Result<Self, WeightError> {
        let encoder = weights.into_encoder()?;
        Ok(Self { encoder, config })
    }

    /// Create with default (untrained) weights for testing.
    pub fn with_default_weights(config: GcbfConfig, hidden_dim: usize) -> Self {
        let weights = default_weights(hidden_dim);
        let encoder = weights.into_encoder().expect("default weights are valid");
        Self { encoder, config }
    }

    /// Load from a JSON weight file.
    pub fn load(config: GcbfConfig, path: &str) -> Result<Self, WeightError> {
        let weights = GcbfWeights::load_json(path)?;
        Self::from_weights(weights, config)
    }

    /// Filter velocities for all agents in a single batched pass.
    ///
    /// This is the main entry point, replacing the O(n²) classical CBF loop.
    ///
    /// Returns a map of drone_id → CbfResult with safe velocities.
    pub fn filter_all(
        &self,
        positions: &[(u32, Vector3<f64>)],
        velocities: &HashMap<u32, Vector3<f64>>,
        desired_velocities: &HashMap<u32, Vector3<f64>>,
        nfz: &[NoFlyZone],
        fear: f64,
    ) -> (HashMap<u32, CbfResult>, u32) {
        let config = self.config.with_fear(fear);
        let n = positions.len();
        if n == 0 {
            return (HashMap::new(), 0);
        }

        // Extract flat arrays for graph construction.
        let pos_vec: Vec<Vector3<f64>> = positions.iter().map(|(_, p)| *p).collect();
        let vel_vec: Vec<Vector3<f64>> = positions
            .iter()
            .map(|(id, _)| velocities.get(id).copied().unwrap_or_else(Vector3::zeros))
            .collect();
        // Goals: use desired velocity direction as proxy goal (position + desired_vel * 10s).
        let goal_vec: Vec<Vector3<f64>> = positions
            .iter()
            .map(|(id, pos)| {
                let dv = desired_velocities
                    .get(id)
                    .copied()
                    .unwrap_or_else(Vector3::zeros);
                pos + dv * 10.0
            })
            .collect();

        // Build graph and run GNN.
        let graph = GraphTopology::build(&pos_vec, &vel_vec, &goal_vec, &config);
        let (barrier_values, action_corrections) =
            self.encoder.forward(&graph, config.max_correction);

        // Convert GNN outputs to CbfResults + apply classical altitude/NFZ constraints.
        let mut results = HashMap::with_capacity(n);
        let mut total_active = 0u32;

        for (local_idx, (drone_id, drone_pos)) in positions.iter().enumerate() {
            let base_vel = velocities
                .get(drone_id)
                .copied()
                .unwrap_or_else(Vector3::zeros);
            let desired = desired_velocities
                .get(drone_id)
                .copied()
                .unwrap_or(base_vel);

            // Neural barrier correction (agent-agent safety).
            let h_value = barrier_values[local_idx];
            let neural_correction = Vector3::new(
                action_corrections[local_idx][0],
                action_corrections[local_idx][1],
                action_corrections[local_idx][2],
            );

            // Apply neural correction only when barrier is near or below zero.
            let mut correction = if h_value < config.safety_margin {
                // Scale correction by urgency: stronger when barrier is more violated.
                let urgency =
                    ((config.safety_margin - h_value) / config.safety_margin).clamp(0.0, 1.0);
                neural_correction * urgency
            } else {
                Vector3::zeros()
            };

            // Classical altitude constraint (always applied, exact).
            let altitude_correction =
                classical_altitude_correction(drone_pos, &desired, &correction, &config);
            correction += altitude_correction;

            // Classical NFZ constraint (always applied, exact).
            let nfz_correction =
                classical_nfz_correction(drone_pos, &desired, &correction, nfz, &config);
            correction += nfz_correction;

            // Clamp total.
            let mag = correction.norm();
            if mag > config.max_correction {
                correction *= config.max_correction / mag;
            }

            let any_active = correction.norm() > 1e-6;
            let active_count = if any_active { 1 } else { 0 }
                + if altitude_correction.norm() > 1e-6 {
                    1
                } else {
                    0
                }
                + if nfz_correction.norm() > 1e-6 { 1 } else { 0 };

            if any_active {
                total_active += active_count;
            }

            results.insert(
                *drone_id,
                CbfResult {
                    safe_velocity: desired + correction,
                    correction,
                    any_active,
                    active_count,
                },
            );
        }

        (results, total_active)
    }
}

/// Classical altitude constraint — copied from cbf.rs for hybrid operation.
fn classical_altitude_correction(
    pos: &Vector3<f64>,
    desired: &Vector3<f64>,
    existing_correction: &Vector3<f64>,
    config: &GcbfConfig,
) -> Vector3<f64> {
    let mut correction = Vector3::zeros();
    let effective_vz = desired.z + existing_correction.z;

    // Floor (max altitude in NED).
    let climb_margin = 1.0 + (-effective_vz).max(0.0) * 0.5;
    let floor_limit = config.altitude_floor_ned + climb_margin;
    let h_floor = pos.z - floor_limit;
    if h_floor < 0.0 {
        let penetration = (-h_floor).max(0.5);
        correction.z +=
            (config.gamma * penetration + (-effective_vz).max(0.0)).min(config.max_correction);
    }

    // Ceiling (min altitude in NED).
    let effective_vz2: f64 = desired.z + existing_correction.z + correction.z;
    let descent_margin = 1.0 + effective_vz2.max(0.0) * 0.5;
    let ceiling_limit = config.altitude_ceiling_ned - descent_margin;
    let h_ceil = ceiling_limit - pos.z;
    if h_ceil < 0.0 {
        let penetration = (-h_ceil).max(0.5);
        correction.z -=
            (config.gamma * penetration + effective_vz2.max(0.0)).min(config.max_correction);
    }

    correction
}

/// Classical NFZ constraint — spherical exclusion zones.
fn classical_nfz_correction(
    pos: &Vector3<f64>,
    desired: &Vector3<f64>,
    existing_correction: &Vector3<f64>,
    nfz: &[NoFlyZone],
    config: &GcbfConfig,
) -> Vector3<f64> {
    let mut correction = Vector3::zeros();
    for zone in nfz {
        let effective_vel = desired + existing_correction + correction;
        let diff = pos - zone.center;
        let dist = diff.norm();
        let direction = if dist > 1e-6 {
            diff / dist
        } else {
            Vector3::new(1.0, 0.0, 0.0)
        };
        let inward_speed = (-effective_vel.dot(&direction)).max(0.0);
        let effective_radius = zone.radius + 0.5 * inward_speed;
        let h = dist * dist - effective_radius * effective_radius;

        if h < 0.0 {
            let penetration = (effective_radius - dist).max(0.5);
            let push = (config.gamma * penetration + inward_speed).min(config.max_correction);
            correction += direction * push;
        }
    }
    correction
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> GcbfConfig {
        GcbfConfig {
            k_neighbors: 3,
            comm_radius: 50.0,
            grid_cell_size: 50.0,
            ..GcbfConfig::default()
        }
    }

    #[test]
    fn filter_all_empty() {
        let barrier = NeuralBarrier::with_default_weights(test_config(), 8);
        let (results, active) = barrier.filter_all(&[], &HashMap::new(), &HashMap::new(), &[], 0.0);
        assert!(results.is_empty());
        assert_eq!(active, 0);
    }

    #[test]
    fn filter_all_single_drone() {
        let barrier = NeuralBarrier::with_default_weights(test_config(), 8);
        let positions = vec![(1, Vector3::new(0.0, 0.0, -50.0))];
        let vel: HashMap<u32, Vector3<f64>> = [(1, Vector3::zeros())].into();
        let desired: HashMap<u32, Vector3<f64>> = [(1, Vector3::new(5.0, 0.0, 0.0))].into();
        let (results, _) = barrier.filter_all(&positions, &vel, &desired, &[], 0.0);
        assert_eq!(results.len(), 1);
        let r = &results[&1];
        assert!(r.safe_velocity.norm().is_finite());
    }

    #[test]
    fn filter_all_produces_finite_outputs() {
        let barrier = NeuralBarrier::with_default_weights(test_config(), 16);
        let positions: Vec<(u32, Vector3<f64>)> = (0..10)
            .map(|i| (i, Vector3::new(i as f64 * 10.0, 0.0, -50.0)))
            .collect();
        let vel: HashMap<u32, Vector3<f64>> = positions
            .iter()
            .map(|(id, _)| (*id, Vector3::zeros()))
            .collect();
        let desired = vel.clone();
        let (results, _) = barrier.filter_all(&positions, &vel, &desired, &[], 0.0);
        for (_, r) in &results {
            assert!(r.safe_velocity.x.is_finite());
            assert!(r.safe_velocity.y.is_finite());
            assert!(r.safe_velocity.z.is_finite());
            assert!(r.correction.norm() <= test_config().max_correction + 1e-6);
        }
    }

    #[test]
    fn altitude_constraint_enforced() {
        let barrier = NeuralBarrier::with_default_weights(test_config(), 8);
        // Drone at 3m altitude (below 5m minimum).
        let positions = vec![(1, Vector3::new(0.0, 0.0, -3.0))];
        let vel: HashMap<u32, Vector3<f64>> = [(1, Vector3::zeros())].into();
        let desired: HashMap<u32, Vector3<f64>> = [(1, Vector3::new(0.0, 0.0, 1.0))].into(); // descending
        let (results, _) = barrier.filter_all(&positions, &vel, &desired, &[], 0.0);
        let r = &results[&1];
        // Correction should push upward (decrease z in NED).
        assert!(
            r.correction.z < 0.0,
            "should push up: correction.z = {}",
            r.correction.z
        );
    }

    #[test]
    fn nfz_constraint_enforced() {
        let barrier = NeuralBarrier::with_default_weights(test_config(), 8);
        let nfz = vec![NoFlyZone {
            center: Vector3::new(5.0, 0.0, -50.0),
            radius: 10.0,
        }];
        let positions = vec![(1, Vector3::new(2.0, 0.0, -50.0))]; // inside NFZ
        let vel: HashMap<u32, Vector3<f64>> = [(1, Vector3::zeros())].into();
        let desired: HashMap<u32, Vector3<f64>> = [(1, Vector3::new(5.0, 0.0, 0.0))].into();
        let (results, _) = barrier.filter_all(&positions, &vel, &desired, &nfz, 0.0);
        let r = &results[&1];
        assert!(r.any_active, "should activate inside NFZ");
    }

    #[test]
    fn load_python_exported_weights() {
        // Cross-validation: load weights exported by Python model.py.
        let path = "/tmp/test_gcbf_weights.json";
        if !std::path::Path::new(path).exists() {
            // Skip if weights not generated (CI).
            return;
        }
        let barrier =
            NeuralBarrier::load(test_config(), path).expect("should load Python-exported weights");
        let positions: Vec<(u32, Vector3<f64>)> = (0..5)
            .map(|i| (i, Vector3::new(i as f64 * 20.0, 0.0, -50.0)))
            .collect();
        let vel: HashMap<u32, Vector3<f64>> = positions
            .iter()
            .map(|(id, _)| (*id, Vector3::zeros()))
            .collect();
        let desired = vel.clone();
        let (results, _) = barrier.filter_all(&positions, &vel, &desired, &[], 0.0);
        assert_eq!(results.len(), 5);
        for (_, r) in &results {
            assert!(r.safe_velocity.x.is_finite());
            assert!(r.safe_velocity.y.is_finite());
            assert!(r.safe_velocity.z.is_finite());
        }
    }
}
