//! Weight serialization and loading for GCBF+ neural network.
//!
//! Weights are stored as a flat binary format with a version header.
//! Dimensions are encoded inline so the loader can reconstruct the
//! `GnnEncoder` without knowing the architecture at compile time.

use super::nn::{GcnLayer, GnnEncoder, MlpHead};
use serde::{Deserialize, Serialize};

const WEIGHT_MAGIC: u32 = 0x47434246; // "GCBF"
const WEIGHT_VERSION: u32 = 1;

/// Serializable weight container.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GcbfWeights {
    pub magic: u32,
    pub version: u32,
    pub hidden_dim: usize,
    pub edge_dim_l1: usize,
    pub node_dim_l1: usize,
    pub layer1_w_edge: Vec<f64>,
    pub layer1_b_edge: Vec<f64>,
    pub layer1_w_node: Vec<f64>,
    pub layer1_b_node: Vec<f64>,
    pub layer2_w_edge: Vec<f64>,
    pub layer2_b_edge: Vec<f64>,
    pub layer2_w_node: Vec<f64>,
    pub layer2_b_node: Vec<f64>,
    pub barrier_w: Vec<f64>,
    pub barrier_b: Vec<f64>,
    pub action_w: Vec<f64>,
    pub action_b: Vec<f64>,
}

/// Errors during weight loading.
#[derive(Debug)]
pub enum WeightError {
    InvalidMagic(u32),
    UnsupportedVersion(u32),
    DimensionMismatch(String),
    Io(std::io::Error),
    Deserialize(String),
}

impl std::fmt::Display for WeightError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidMagic(m) => {
                write!(f, "invalid magic: 0x{m:08X}, expected 0x{WEIGHT_MAGIC:08X}")
            }
            Self::UnsupportedVersion(v) => {
                write!(f, "unsupported version {v}, expected {WEIGHT_VERSION}")
            }
            Self::DimensionMismatch(s) => write!(f, "dimension mismatch: {s}"),
            Self::Io(e) => write!(f, "IO error: {e}"),
            Self::Deserialize(s) => write!(f, "deserialization error: {s}"),
        }
    }
}

impl std::error::Error for WeightError {}

impl From<std::io::Error> for WeightError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl GcbfWeights {
    /// Validate header.
    fn validate(&self) -> Result<(), WeightError> {
        if self.magic != WEIGHT_MAGIC {
            return Err(WeightError::InvalidMagic(self.magic));
        }
        if self.version != WEIGHT_VERSION {
            return Err(WeightError::UnsupportedVersion(self.version));
        }
        Ok(())
    }

    /// Reconstruct a `GnnEncoder` from these weights.
    pub fn into_encoder(self) -> Result<GnnEncoder, WeightError> {
        self.validate()?;
        let h = self.hidden_dim;

        let layer1 = build_gcn_layer(
            &self.layer1_w_edge,
            &self.layer1_b_edge,
            &self.layer1_w_node,
            &self.layer1_b_node,
            h,
            self.edge_dim_l1,
            self.node_dim_l1 + h,
        )?;

        let layer2 = build_gcn_layer(
            &self.layer2_w_edge,
            &self.layer2_b_edge,
            &self.layer2_w_node,
            &self.layer2_b_node,
            h,
            7, // edge features stay 7D in layer 2
            h + h,
        )?;

        let barrier_head = build_mlp_head(&self.barrier_w, &self.barrier_b, 1, h)?;
        let action_head = build_mlp_head(&self.action_w, &self.action_b, 3, h)?;

        Ok(GnnEncoder {
            layer1,
            layer2,
            barrier_head,
            action_head,
        })
    }

    /// Load weights from a JSON file.
    pub fn load_json(path: &str) -> Result<Self, WeightError> {
        let data = std::fs::read_to_string(path)?;
        serde_json::from_str(&data).map_err(|e| WeightError::Deserialize(e.to_string()))
    }
}

/// Create default weights (small random initialization) for a given architecture.
///
/// Used for testing and as a starting point before training.
pub fn default_weights(hidden_dim: usize) -> GcbfWeights {
    let edge_dim = 7;
    let node_dim = 9;

    // Xavier-like initialization: scale by sqrt(2 / fan_in).
    let scale_edge = (2.0 / edge_dim as f64).sqrt();
    let scale_node = (2.0 / (node_dim + hidden_dim) as f64).sqrt();
    let scale_h = (2.0 / (hidden_dim * 2) as f64).sqrt();

    let mut seed = 12345u64;
    let mut next = || -> f64 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((seed >> 33) as f64 / u32::MAX as f64 - 0.5) * 2.0
    };

    let mut gen = |n: usize, scale: f64| -> Vec<f64> { (0..n).map(|_| next() * scale).collect() };

    GcbfWeights {
        magic: WEIGHT_MAGIC,
        version: WEIGHT_VERSION,
        hidden_dim,
        edge_dim_l1: edge_dim,
        node_dim_l1: node_dim,
        layer1_w_edge: gen(hidden_dim * edge_dim, scale_edge),
        layer1_b_edge: vec![0.0; hidden_dim],
        layer1_w_node: gen(hidden_dim * (node_dim + hidden_dim), scale_node),
        layer1_b_node: vec![0.0; hidden_dim],
        layer2_w_edge: gen(hidden_dim * 7, scale_h),
        layer2_b_edge: vec![0.0; hidden_dim],
        layer2_w_node: gen(hidden_dim * (hidden_dim + hidden_dim), scale_h),
        layer2_b_node: vec![0.0; hidden_dim],
        barrier_w: gen(hidden_dim, (2.0 / hidden_dim as f64).sqrt()),
        barrier_b: vec![0.5], // positive bias → default safe
        action_w: gen(3 * hidden_dim, (2.0 / hidden_dim as f64).sqrt()),
        action_b: vec![0.0; 3],
    }
}

fn build_gcn_layer(
    w_edge_flat: &[f64],
    b_edge: &[f64],
    w_node_flat: &[f64],
    b_node: &[f64],
    hidden: usize,
    edge_in: usize,
    node_in: usize,
) -> Result<GcnLayer, WeightError> {
    if w_edge_flat.len() != hidden * edge_in {
        return Err(WeightError::DimensionMismatch(format!(
            "w_edge: expected {}x{}={}, got {}",
            hidden,
            edge_in,
            hidden * edge_in,
            w_edge_flat.len()
        )));
    }
    if w_node_flat.len() != hidden * node_in {
        return Err(WeightError::DimensionMismatch(format!(
            "w_node: expected {}x{}={}, got {}",
            hidden,
            node_in,
            hidden * node_in,
            w_node_flat.len()
        )));
    }

    Ok(GcnLayer {
        w_edge: w_edge_flat.to_vec(),
        b_edge: b_edge.to_vec(),
        w_node: w_node_flat.to_vec(),
        b_node: b_node.to_vec(),
        hidden_dim: hidden,
        edge_in,
        node_in,
    })
}

fn build_mlp_head(
    w_flat: &[f64],
    b: &[f64],
    out_dim: usize,
    in_dim: usize,
) -> Result<MlpHead, WeightError> {
    if w_flat.len() != out_dim * in_dim {
        return Err(WeightError::DimensionMismatch(format!(
            "mlp_head: expected {}x{}={}, got {}",
            out_dim,
            in_dim,
            out_dim * in_dim,
            w_flat.len()
        )));
    }
    Ok(MlpHead {
        w: w_flat.to_vec(),
        b: b.to_vec(),
        output_dim: out_dim,
        input_dim: in_dim,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_weights_roundtrip() {
        let w = default_weights(16);
        assert_eq!(w.magic, WEIGHT_MAGIC);
        assert_eq!(w.version, WEIGHT_VERSION);
        let encoder = w.into_encoder().expect("should build encoder");
        assert_eq!(encoder.layer1.hidden_dim, 16);
        assert_eq!(encoder.layer2.hidden_dim, 16);
        assert_eq!(encoder.barrier_head.output_dim, 1);
        assert_eq!(encoder.action_head.output_dim, 3);
    }

    #[test]
    fn invalid_magic_rejected() {
        let mut w = default_weights(8);
        w.magic = 0xDEADBEEF;
        assert!(w.into_encoder().is_err());
    }

    #[test]
    fn json_serialization_roundtrip() {
        let w = default_weights(8);
        let json = serde_json::to_string(&w).expect("serialize");
        let w2: GcbfWeights = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(w2.hidden_dim, 8);
        assert_eq!(w2.layer1_w_edge.len(), w.layer1_w_edge.len());
    }
}
