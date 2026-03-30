//! GCBF+ configuration — tunable parameters for graph-based neural safety.

use serde::{Deserialize, Serialize};

/// Configuration for the Graph Control Barrier Function filter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GcbfConfig {
    /// Number of nearest neighbors per agent (k in k-NN).
    pub k_neighbors: usize,
    /// Communication/sensing radius — only consider neighbors within this range (meters).
    pub comm_radius: f64,
    /// Safety margin added to neural barrier output.
    pub safety_margin: f64,
    /// Decay rate for neural barrier constraint (analogous to alpha in classical CBF).
    pub gamma: f64,
    /// Maximum correction magnitude (m/s).
    pub max_correction: f64,
    /// Grid cell size for spatial index (meters). Should be >= comm_radius.
    pub grid_cell_size: f64,
    /// Whether to fall back to classical CBF if neural barrier is unavailable.
    pub fallback_to_classical: bool,
    /// Altitude floor in NED (negative = above ground). Classical constraint, always applied.
    pub altitude_floor_ned: f64,
    /// Altitude ceiling in NED (negative = above ground). Classical constraint, always applied.
    pub altitude_ceiling_ned: f64,
}

impl Default for GcbfConfig {
    fn default() -> Self {
        Self {
            k_neighbors: 8,
            comm_radius: 100.0,
            safety_margin: 0.5,
            gamma: 1.0,
            max_correction: 10.0,
            grid_cell_size: 100.0,
            fallback_to_classical: true,
            altitude_floor_ned: -500.0,
            altitude_ceiling_ned: -5.0,
        }
    }
}

impl GcbfConfig {
    /// Modulate GCBF+ parameters by fear level f ∈ [0,1].
    ///
    /// Higher fear → more neighbors, wider radius, larger margins.
    pub fn with_fear(&self, f: f64) -> Self {
        let f = if f.is_finite() {
            f.clamp(0.0, 1.0)
        } else {
            0.0
        };
        Self {
            k_neighbors: self.k_neighbors + (f * 4.0) as usize, // 8→12
            comm_radius: self.comm_radius + f * 50.0,           // 100→150m
            safety_margin: self.safety_margin + f * 1.0,        // 0.5→1.5
            gamma: self.gamma + f * 1.0,                        // 1.0→2.0
            max_correction: self.max_correction + f * 15.0,     // 10→25 m/s
            ..self.clone()
        }
    }
}
