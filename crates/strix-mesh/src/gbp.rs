//! Gaussian Belief Propagation for distributed position fusion.
//!
//! Each drone maintains a Gaussian belief N(μ, Σ) about its 3D position.
//! Beliefs are exchanged via gossip and fused additively in **information form**:
//!
//! ```text
//! Λ = Σ⁻¹           (precision matrix)
//! η = Λ μ            (information vector)
//! Fusion: Λ_fused = Λ_self + Σ Λ_i,  η_fused = η_self + Σ η_i
//! Recovery: μ_fused = Λ_fused⁻¹ η_fused
//! ```
//!
//! Uses diagonal covariance (3 floats per drone) for bandwidth efficiency.

use nalgebra::{Matrix3, Vector3};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::NodeId;

// ---------------------------------------------------------------------------
// Gaussian belief
// ---------------------------------------------------------------------------

/// A 3D Gaussian belief in information (canonical) form.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaussianBelief {
    /// Mean position μ.
    pub mean: Vector3<f64>,
    /// Precision matrix Λ = Σ⁻¹ (stored as full 3×3 for generality).
    pub precision: Matrix3<f64>,
}

impl GaussianBelief {
    /// Construct from diagonal covariance [var_x, var_y, var_z].
    /// Precision = diag(1/var_x, 1/var_y, 1/var_z).
    pub fn from_diagonal_covariance(mean: Vector3<f64>, var: [f64; 3]) -> Self {
        let px = 1.0 / var[0].max(1e-10);
        let py = 1.0 / var[1].max(1e-10);
        let pz = 1.0 / var[2].max(1e-10);
        Self {
            mean,
            precision: Matrix3::from_diagonal(&Vector3::new(px, py, pz)),
        }
    }

    /// Information vector η = Λ μ.
    pub fn information_vector(&self) -> Vector3<f64> {
        self.precision * self.mean
    }

    /// Recover covariance Σ = Λ⁻¹. Returns None if precision is singular.
    pub fn covariance(&self) -> Option<Matrix3<f64>> {
        self.precision.try_inverse()
    }

    /// Diagonal of the covariance [var_x, var_y, var_z].
    /// Falls back to [1e10, 1e10, 1e10] if precision is singular.
    pub fn covariance_diagonal(&self) -> [f64; 3] {
        match self.covariance() {
            Some(cov) => [cov[(0, 0)], cov[(1, 1)], cov[(2, 2)]],
            None => [1e10, 1e10, 1e10],
        }
    }

    /// Check if all elements are finite and diagonal precision is positive.
    pub fn is_valid(&self) -> bool {
        self.mean.iter().all(|v| v.is_finite())
            && self.precision.iter().all(|v| v.is_finite())
            && self.precision[(0, 0)] > 0.0
            && self.precision[(1, 1)] > 0.0
            && self.precision[(2, 2)] > 0.0
    }

    /// Uninformative prior: zero precision, zero mean. Identity element for fusion.
    pub fn uninformative() -> Self {
        Self {
            mean: Vector3::zeros(),
            precision: Matrix3::zeros(),
        }
    }
}

// ---------------------------------------------------------------------------
// GBP configuration
// ---------------------------------------------------------------------------

/// Configuration for Gaussian Belief Propagation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GbpConfig {
    /// Enable GBP. Default: false (backward-compatible).
    pub enabled: bool,
    /// Number of message-passing iterations per tick. Default: 1.
    pub max_iterations: u32,
    /// Minimum per-axis precision to accept from neighbor. Default: 0.01.
    pub min_precision_diag: f64,
    /// Damping factor ∈ [0, 1]. Scales neighbor contributions. Default: 0.5.
    pub damping: f64,
}

impl Default for GbpConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_iterations: 1,
            min_precision_diag: 0.01,
            damping: 0.5,
        }
    }
}

// ---------------------------------------------------------------------------
// GBP node (per-drone)
// ---------------------------------------------------------------------------

/// Per-drone GBP state: self belief + neighbor beliefs → fused belief.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GbpNode {
    self_id: NodeId,
    self_belief: GaussianBelief,
    neighbor_beliefs: HashMap<NodeId, GaussianBelief>,
    fused_belief: GaussianBelief,
    config: GbpConfig,
}

impl GbpNode {
    pub fn new(self_id: NodeId, config: GbpConfig) -> Self {
        Self {
            self_id,
            self_belief: GaussianBelief::uninformative(),
            neighbor_beliefs: HashMap::new(),
            fused_belief: GaussianBelief::uninformative(),
            config,
        }
    }

    /// Set self belief from particle filter covariance (called each tick).
    pub fn set_self_belief(&mut self, belief: GaussianBelief) {
        if belief.is_valid() {
            self.self_belief = belief;
        }
    }

    /// Set a neighbor's belief (from gossip). Validates before accepting.
    pub fn set_neighbor_belief(&mut self, id: NodeId, belief: GaussianBelief) {
        if id != self.self_id && belief.is_valid() {
            self.neighbor_beliefs.insert(id, belief);
        }
    }

    /// Remove a neighbor (e.g. pruned from gossip).
    pub fn remove_neighbor(&mut self, id: NodeId) {
        self.neighbor_beliefs.remove(&id);
    }

    /// Run n iterations of GBP message passing.
    ///
    /// For star topology (self + direct neighbors), 1 iteration is sufficient:
    /// additive fusion in information form.
    pub fn iterate(&mut self, n: u32) {
        for _ in 0..n {
            // Start with self belief in information form.
            let mut lambda_fused = self.self_belief.precision;
            let mut eta_fused = self.self_belief.information_vector();

            // Accumulate neighbor contributions (damped).
            let damping = self.config.damping.clamp(0.0, 1.0);
            for belief in self.neighbor_beliefs.values() {
                // Skip neighbors with too-low precision (very uncertain).
                let diag_ok = belief.precision[(0, 0)] >= self.config.min_precision_diag
                    && belief.precision[(1, 1)] >= self.config.min_precision_diag
                    && belief.precision[(2, 2)] >= self.config.min_precision_diag;
                if !diag_ok {
                    continue;
                }

                lambda_fused += belief.precision * damping;
                eta_fused += belief.information_vector() * damping;
            }

            // Recover fused mean: μ = Λ⁻¹ η
            if let Some(cov_fused) = lambda_fused.try_inverse() {
                let mean_fused = cov_fused * eta_fused;
                if mean_fused.iter().all(|v| v.is_finite())
                    && lambda_fused.iter().all(|v| v.is_finite())
                {
                    self.fused_belief = GaussianBelief {
                        mean: mean_fused,
                        precision: lambda_fused,
                    };
                }
                // else: keep previous fused_belief (NaN safety)
            }
            // else: singular precision → keep previous fused_belief
        }
    }

    /// Get the fused belief.
    pub fn fused_belief(&self) -> &GaussianBelief {
        &self.fused_belief
    }

    /// Convenience: fused position mean.
    pub fn fused_position(&self) -> Vector3<f64> {
        self.fused_belief.mean
    }

    /// Convenience: fused diagonal covariance.
    pub fn fused_covariance_diagonal(&self) -> [f64; 3] {
        self.fused_belief.covariance_diagonal()
    }

    /// Number of neighbor beliefs currently held.
    pub fn neighbor_count(&self) -> usize {
        self.neighbor_beliefs.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_diagonal_covariance() {
        let b =
            GaussianBelief::from_diagonal_covariance(Vector3::new(1.0, 2.0, 3.0), [4.0, 4.0, 4.0]);
        assert!((b.precision[(0, 0)] - 0.25).abs() < 1e-6, "1/4 = 0.25");
        assert!((b.precision[(1, 1)] - 0.25).abs() < 1e-6);
        assert!((b.precision[(0, 1)]).abs() < 1e-10, "off-diagonal = 0");
    }

    #[test]
    fn information_vector_correct() {
        let b = GaussianBelief::from_diagonal_covariance(
            Vector3::new(10.0, 20.0, 30.0),
            [1.0, 1.0, 1.0],
        );
        let eta = b.information_vector();
        // Λ = I, η = I * μ = μ
        assert!((eta.x - 10.0).abs() < 1e-6);
        assert!((eta.y - 20.0).abs() < 1e-6);
    }

    #[test]
    fn single_node_fusion() {
        let cfg = GbpConfig {
            enabled: true,
            max_iterations: 1,
            ..GbpConfig::default()
        };
        let mut node = GbpNode::new(NodeId(0), cfg);
        let self_b = GaussianBelief::from_diagonal_covariance(
            Vector3::new(5.0, 10.0, 15.0),
            [1.0, 1.0, 1.0],
        );
        node.set_self_belief(self_b);
        node.iterate(1);

        let fused = node.fused_position();
        assert!((fused.x - 5.0).abs() < 1e-6, "self only → fused = self");
        assert!((fused.y - 10.0).abs() < 1e-6);
    }

    #[test]
    fn two_node_fusion_weighted_mean() {
        let cfg = GbpConfig {
            enabled: true,
            max_iterations: 1,
            damping: 1.0, // no damping for this test
            ..GbpConfig::default()
        };
        let mut node = GbpNode::new(NodeId(0), cfg);

        // Self at (0,0,0) with var=[1,1,1] → precision diag=[1,1,1]
        node.set_self_belief(GaussianBelief::from_diagonal_covariance(
            Vector3::new(0.0, 0.0, 0.0),
            [1.0, 1.0, 1.0],
        ));

        // Neighbor at (10,0,0) with same variance → equal precision
        node.set_neighbor_belief(
            NodeId(1),
            GaussianBelief::from_diagonal_covariance(Vector3::new(10.0, 0.0, 0.0), [1.0, 1.0, 1.0]),
        );

        node.iterate(1);

        let fused = node.fused_position();
        // Equal precision → fused at midpoint (5, 0, 0)
        assert!(
            (fused.x - 5.0).abs() < 1e-6,
            "equal precision → midpoint, got {}",
            fused.x
        );

        // Fused variance should be 0.5 (precision = 2)
        let cov = node.fused_covariance_diagonal();
        assert!(
            (cov[0] - 0.5).abs() < 1e-6,
            "fused var = 0.5, got {}",
            cov[0]
        );
    }

    #[test]
    fn asymmetric_precision_favors_confident() {
        let cfg = GbpConfig {
            enabled: true,
            max_iterations: 1,
            damping: 1.0,
            ..GbpConfig::default()
        };
        let mut node = GbpNode::new(NodeId(0), cfg);

        // Self at (0,0,0), very precise (var=0.1 → precision=10)
        node.set_self_belief(GaussianBelief::from_diagonal_covariance(
            Vector3::new(0.0, 0.0, 0.0),
            [0.1, 0.1, 0.1],
        ));

        // Neighbor at (100,0,0), very uncertain (var=100 → precision=0.01)
        node.set_neighbor_belief(
            NodeId(1),
            GaussianBelief::from_diagonal_covariance(
                Vector3::new(100.0, 0.0, 0.0),
                [100.0, 100.0, 100.0],
            ),
        );

        node.iterate(1);

        let fused = node.fused_position();
        // Fused should be very close to self (much higher precision)
        assert!(
            fused.x < 2.0,
            "confident self should dominate, got {}",
            fused.x
        );
    }

    #[test]
    fn zero_precision_ignored() {
        let cfg = GbpConfig {
            enabled: true,
            max_iterations: 1,
            damping: 1.0,
            min_precision_diag: 0.01,
            ..GbpConfig::default()
        };
        let mut node = GbpNode::new(NodeId(0), cfg);

        node.set_self_belief(GaussianBelief::from_diagonal_covariance(
            Vector3::new(5.0, 5.0, 5.0),
            [1.0, 1.0, 1.0],
        ));

        // Neighbor with near-zero precision (below threshold)
        let mut uninf = GaussianBelief::uninformative();
        uninf.mean = Vector3::new(999.0, 999.0, 999.0);
        node.set_neighbor_belief(NodeId(1), uninf);

        node.iterate(1);

        let fused = node.fused_position();
        // Uninformative neighbor should not pull the fused position
        assert!(
            (fused.x - 5.0).abs() < 1e-6,
            "zero precision neighbor ignored, got {}",
            fused.x
        );
    }

    #[test]
    fn nan_rejection() {
        let cfg = GbpConfig {
            enabled: true,
            max_iterations: 1,
            ..GbpConfig::default()
        };
        let mut node = GbpNode::new(NodeId(0), cfg);

        node.set_self_belief(GaussianBelief::from_diagonal_covariance(
            Vector3::new(5.0, 5.0, 5.0),
            [1.0, 1.0, 1.0],
        ));

        // NaN belief — should be rejected by set_neighbor_belief
        let bad = GaussianBelief {
            mean: Vector3::new(f64::NAN, 0.0, 0.0),
            precision: Matrix3::identity(),
        };
        node.set_neighbor_belief(NodeId(1), bad);

        assert_eq!(node.neighbor_count(), 0, "NaN belief should be rejected");
    }

    #[test]
    fn damping_reduces_neighbor_influence() {
        let cfg_full = GbpConfig {
            enabled: true,
            max_iterations: 1,
            damping: 1.0,
            ..GbpConfig::default()
        };
        let cfg_half = GbpConfig {
            enabled: true,
            max_iterations: 1,
            damping: 0.5,
            ..GbpConfig::default()
        };

        let self_b =
            GaussianBelief::from_diagonal_covariance(Vector3::new(0.0, 0.0, 0.0), [1.0, 1.0, 1.0]);
        let neighbor_b =
            GaussianBelief::from_diagonal_covariance(Vector3::new(10.0, 0.0, 0.0), [1.0, 1.0, 1.0]);

        let mut node_full = GbpNode::new(NodeId(0), cfg_full);
        node_full.set_self_belief(self_b.clone());
        node_full.set_neighbor_belief(NodeId(1), neighbor_b.clone());
        node_full.iterate(1);

        let mut node_half = GbpNode::new(NodeId(0), cfg_half);
        node_half.set_self_belief(self_b);
        node_half.set_neighbor_belief(NodeId(1), neighbor_b);
        node_half.iterate(1);

        let full_x = node_full.fused_position().x;
        let half_x = node_half.fused_position().x;

        // With damping=0.5, neighbor pulls less → fused closer to self (0)
        assert!(
            half_x < full_x,
            "damping should reduce neighbor pull: half={half_x}, full={full_x}"
        );
    }
}
