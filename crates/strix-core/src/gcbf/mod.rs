//! **GCBF+** — Graph Control Barrier Functions for scalable swarm safety.
//!
//! Replaces the O(n²) pairwise CBF with a learned neural barrier function
//! evaluated over a k-nearest-neighbor graph, achieving O(n·k) complexity.
//!
//! Based on: "GCBF+: A Neural Graph Control Barrier Function Framework
//! for Distributed Safe Multi-Agent Control" (arXiv:2401.14554).
//!
//! # Architecture
//!
//! 1. **Spatial Grid** (`knn`): O(n) construction, O(1) amortized k-NN queries.
//! 2. **Graph Topology** (`graph`): Sparse directed graph with per-node/edge features.
//! 3. **GNN Encoder** (`nn`): 2-layer GCN with mean aggregation.
//! 4. **Neural Barrier** (`barrier`): h_neural evaluation + safe velocity correction.
//! 5. **Hybrid**: Agent-agent via GNN, altitude + NFZ via classical constraints.
//!
//! # Usage
//!
//! ```rust,ignore
//! use strix_core::gcbf::{NeuralBarrier, GcbfConfig};
//!
//! let config = GcbfConfig::default();
//! let barrier = NeuralBarrier::with_default_weights(config, 16);
//! let (results, active) = barrier.filter_all(&positions, &velocities, &desired, &nfz, fear);
//! ```

pub mod barrier;
pub mod config;
pub mod graph;
pub mod knn;
pub mod nn;
pub mod weights;

pub use barrier::NeuralBarrier;
pub use config::GcbfConfig;
pub use graph::GraphTopology;
pub use weights::{default_weights, GcbfWeights};
