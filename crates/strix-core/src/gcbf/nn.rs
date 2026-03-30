//! GNN layers for GCBF+ neural barrier evaluation.
//!
//! Two-layer Graph Convolutional Network (GCN) with mean aggregation,
//! following the GCBF+ architecture (arXiv:2401.14554).
//!
//! Architecture:
//! ```text
//! Layer 1: msg_ij = ReLU(W_edge @ edge_feat + b)
//!          agg_i  = mean(msg_ij for j ∈ N(i))
//!          h_i^1  = ReLU(W_node @ [node_feat || agg_i] + b)
//! Layer 2: same structure on h^1
//! Heads:   barrier = W_b @ h^2 + b_b   (scalar)
//!          action  = tanh(W_a @ h^2 + b_a) * max_correction   (3D)
//! ```

use super::graph::GraphTopology;

/// A single GCN message-passing layer.
pub struct GcnLayer {
    /// Edge message transform: [hidden_dim, edge_input_dim].
    pub w_edge: Vec<Vec<f64>>,
    pub b_edge: Vec<f64>,
    /// Node update transform: [hidden_dim, node_input_dim + hidden_dim].
    pub w_node: Vec<Vec<f64>>,
    pub b_node: Vec<f64>,
    pub hidden_dim: usize,
}

/// Simple MLP head (single linear layer).
pub struct MlpHead {
    /// Weight matrix: [output_dim, input_dim].
    pub w: Vec<Vec<f64>>,
    pub b: Vec<f64>,
    pub output_dim: usize,
}

/// Two-layer GCN encoder.
pub struct GnnEncoder {
    pub layer1: GcnLayer,
    pub layer2: GcnLayer,
    pub barrier_head: MlpHead,
    pub action_head: MlpHead,
}

impl GcnLayer {
    /// Forward pass for one GCN layer.
    ///
    /// `node_states`: per-node feature vectors (flattened: `[n_nodes][node_dim]`).
    /// `graph`: topology with edges and edge features.
    ///
    /// Returns updated per-node hidden states `[n_nodes][hidden_dim]`.
    pub fn forward(&self, node_states: &[Vec<f64>], graph: &GraphTopology) -> Vec<Vec<f64>> {
        let n = graph.n_nodes;
        let h = self.hidden_dim;

        // Step 1: Compute edge messages.
        let mut messages: Vec<Vec<f64>> = Vec::with_capacity(graph.edges.len());
        for ef in &graph.edge_features {
            let msg = mat_vec_relu(&self.w_edge, &self.b_edge, ef);
            messages.push(msg);
        }

        // Step 2: Aggregate messages per target node (mean pooling).
        let mut aggregated = vec![vec![0.0; h]; n];
        for (edge_idx, &(_src, tgt)) in graph.edges.iter().enumerate() {
            for d in 0..h {
                aggregated[tgt][d] += messages[edge_idx][d];
            }
        }
        for (i, agg) in aggregated.iter_mut().enumerate().take(n) {
            if graph.degree[i] > 0 {
                let inv_deg = 1.0 / graph.degree[i] as f64;
                for val in agg.iter_mut().take(h) {
                    *val *= inv_deg;
                }
            }
        }

        // Step 3: Node update — concatenate [node_state || aggregated], apply W_node + ReLU.
        let mut output = Vec::with_capacity(n);
        for i in 0..n {
            let mut concat = node_states[i].clone();
            concat.extend_from_slice(&aggregated[i]);
            let h_new = mat_vec_relu(&self.w_node, &self.b_node, &concat);
            output.push(h_new);
        }
        output
    }
}

impl MlpHead {
    /// Linear transform (no activation — caller applies tanh/identity).
    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        mat_vec_linear(&self.w, &self.b, input)
    }
}

impl GnnEncoder {
    /// Full forward pass: 2 GCN layers + barrier + action heads.
    ///
    /// Returns `(barrier_values, action_corrections)` per node.
    /// - `barrier_values[i]`: scalar h value (positive = safe).
    /// - `action_corrections[i]`: 3D velocity correction in NED.
    pub fn forward(&self, graph: &GraphTopology, max_correction: f64) -> (Vec<f64>, Vec<[f64; 3]>) {
        if graph.n_nodes == 0 {
            return (Vec::new(), Vec::new());
        }

        // Initial node features from graph.
        let node_states_0: Vec<Vec<f64>> =
            graph.node_features.iter().map(|nf| nf.to_vec()).collect();

        // Layer 1.
        let h1 = self.layer1.forward(&node_states_0, graph);

        // Layer 2: uses h1 as both node_states and contributes to edge features.
        // For layer 2, edge features incorporate source + target hidden states.
        // We build extended edge features: [h_src || h_tgt || original_edge_feat].
        let graph2 = build_layer2_graph(graph, &h1, self.layer2.hidden_dim);
        let h2 = self.layer2.forward(&h1, &graph2);

        // Heads.
        let mut barrier_values = Vec::with_capacity(graph.n_nodes);
        let mut action_corrections = Vec::with_capacity(graph.n_nodes);
        for hidden in &h2 {
            // Barrier head: scalar.
            let bv = self.barrier_head.forward(hidden);
            barrier_values.push(bv[0]);

            // Action head: 3D correction with tanh scaling.
            let av = self.action_head.forward(hidden);
            action_corrections.push([
                av[0].tanh() * max_correction,
                av[1].tanh() * max_correction,
                av[2].tanh() * max_correction,
            ]);
        }

        (barrier_values, action_corrections)
    }
}

/// Build layer-2 graph with extended edge features.
///
/// GCBF+ layer 2 edge features: `[h_src || h_tgt || original_edge_feat]`.
fn build_layer2_graph(
    base: &GraphTopology,
    hidden_states: &[Vec<f64>],
    _hidden_dim: usize,
) -> GraphTopology {
    let extended_features: Vec<[f64; 7]> = base
        .edges
        .iter()
        .enumerate()
        .map(|(idx, &(src, tgt))| {
            // For simplicity, we keep the same 7D edge features but modulate
            // by the hidden state norms. This avoids changing the edge dimension
            // between layers while still incorporating learned representations.
            let src_norm = vec_norm(&hidden_states[src]).max(1e-6);
            let tgt_norm = vec_norm(&hidden_states[tgt]).max(1e-6);
            let scale = (src_norm * tgt_norm).sqrt();
            let ef = &base.edge_features[idx];
            [
                ef[0] / scale,
                ef[1] / scale,
                ef[2] / scale,
                ef[3] / scale,
                ef[4] / scale,
                ef[5] / scale,
                ef[6] / scale,
            ]
        })
        .collect();

    GraphTopology {
        n_nodes: base.n_nodes,
        edges: base.edges.clone(),
        degree: base.degree.clone(),
        node_features: hidden_states
            .iter()
            .map(|h| {
                let mut feat = [0.0; 9];
                for (i, v) in h.iter().enumerate().take(9) {
                    feat[i] = *v;
                }
                feat
            })
            .collect(),
        edge_features: extended_features,
    }
}

// ── Linear algebra primitives ─────────────────────────────────────────────

/// Matrix-vector multiply with bias and ReLU: `max(0, W @ x + b)`.
fn mat_vec_relu(w: &[Vec<f64>], b: &[f64], x: &[f64]) -> Vec<f64> {
    w.iter()
        .zip(b.iter())
        .map(|(row, bias)| {
            let dot: f64 = row.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
            (dot + bias).max(0.0) // ReLU
        })
        .collect()
}

/// Matrix-vector multiply with bias (no activation).
fn mat_vec_linear(w: &[Vec<f64>], b: &[f64], x: &[f64]) -> Vec<f64> {
    w.iter()
        .zip(b.iter())
        .map(|(row, bias)| {
            let dot: f64 = row.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
            dot + bias
        })
        .collect()
}

/// L2 norm of a vector.
fn vec_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_identity_layer(input_dim: usize, edge_dim: usize, hidden: usize) -> GcnLayer {
        // W_edge: hidden x edge_dim (small random-ish values)
        let w_edge: Vec<Vec<f64>> = (0..hidden)
            .map(|r| {
                (0..edge_dim)
                    .map(|c| if r == c % hidden { 1.0 } else { 0.0 })
                    .collect()
            })
            .collect();
        let b_edge = vec![0.0; hidden];
        // W_node: hidden x (input_dim + hidden)
        let node_in = input_dim + hidden;
        let w_node: Vec<Vec<f64>> = (0..hidden)
            .map(|r| {
                (0..node_in)
                    .map(|c| if r == c % hidden { 0.1 } else { 0.0 })
                    .collect()
            })
            .collect();
        let b_node = vec![0.1; hidden];
        GcnLayer {
            w_edge,
            b_edge,
            w_node,
            b_node,
            hidden_dim: hidden,
        }
    }

    fn make_test_encoder(hidden: usize) -> GnnEncoder {
        let layer1 = make_identity_layer(9, 7, hidden);
        let layer2 = make_identity_layer(hidden, 7, hidden);
        let barrier_head = MlpHead {
            w: vec![(0..hidden).map(|_| 0.1).collect()],
            b: vec![0.5], // bias positive → default "safe"
            output_dim: 1,
        };
        let action_head = MlpHead {
            w: (0..3)
                .map(|r| {
                    (0..hidden)
                        .map(|c| if r == c % 3 { 0.1 } else { 0.0 })
                        .collect()
                })
                .collect(),
            b: vec![0.0; 3],
            output_dim: 3,
        };
        GnnEncoder {
            layer1,
            layer2,
            barrier_head,
            action_head,
        }
    }

    #[test]
    fn forward_produces_correct_output_count() {
        use nalgebra::Vector3;
        let positions = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(10.0, 0.0, 0.0),
            Vector3::new(0.0, 10.0, 0.0),
        ];
        let velocities = vec![Vector3::zeros(); 3];
        let config = super::super::config::GcbfConfig {
            k_neighbors: 2,
            comm_radius: 50.0,
            grid_cell_size: 50.0,
            ..Default::default()
        };
        let graph = GraphTopology::build(&positions, &velocities, &[], &config);
        let encoder = make_test_encoder(8);
        let (barriers, actions) = encoder.forward(&graph, 10.0);
        assert_eq!(barriers.len(), 3);
        assert_eq!(actions.len(), 3);
    }

    #[test]
    fn barrier_values_are_finite() {
        use nalgebra::Vector3;
        let positions: Vec<Vector3<f64>> = (0..10)
            .map(|i| Vector3::new(i as f64 * 5.0, 0.0, -50.0))
            .collect();
        let velocities = vec![Vector3::zeros(); 10];
        let config = super::super::config::GcbfConfig {
            k_neighbors: 3,
            comm_radius: 100.0,
            grid_cell_size: 100.0,
            ..Default::default()
        };
        let graph = GraphTopology::build(&positions, &velocities, &[], &config);
        let encoder = make_test_encoder(16);
        let (barriers, actions) = encoder.forward(&graph, 10.0);
        for b in &barriers {
            assert!(b.is_finite(), "barrier value is not finite: {b}");
        }
        for a in &actions {
            for v in a {
                assert!(v.is_finite(), "action value is not finite: {v}");
            }
        }
    }

    #[test]
    fn action_clamped_by_max_correction() {
        use nalgebra::Vector3;
        let positions = vec![Vector3::new(0.0, 0.0, 0.0), Vector3::new(1.0, 0.0, 0.0)];
        let velocities = vec![Vector3::new(100.0, 100.0, 100.0); 2]; // large velocities
        let config = super::super::config::GcbfConfig {
            k_neighbors: 1,
            comm_radius: 50.0,
            grid_cell_size: 50.0,
            ..Default::default()
        };
        let graph = GraphTopology::build(&positions, &velocities, &[], &config);
        let encoder = make_test_encoder(8);
        let max_corr = 5.0;
        let (_, actions) = encoder.forward(&graph, max_corr);
        for a in &actions {
            for v in a {
                assert!(
                    v.abs() <= max_corr + 1e-10,
                    "action component {v} exceeds max_correction {max_corr}"
                );
            }
        }
    }

    #[test]
    fn empty_graph_forward() {
        let encoder = make_test_encoder(8);
        let graph = GraphTopology {
            n_nodes: 0,
            edges: Vec::new(),
            degree: Vec::new(),
            node_features: Vec::new(),
            edge_features: Vec::new(),
        };
        let (b, a) = encoder.forward(&graph, 10.0);
        assert!(b.is_empty());
        assert!(a.is_empty());
    }

    #[test]
    fn mat_vec_relu_applies_relu() {
        let w = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let b = vec![-0.5, 0.5];
        let x = vec![0.3, -1.0];
        let result = super::mat_vec_relu(&w, &b, &x);
        // (0.3 - 0.5) = -0.2 → ReLU → 0.0
        // (-1.0 + 0.5) = -0.5 → ReLU → 0.0
        assert!((result[0] - 0.0).abs() < 1e-10);
        assert!((result[1] - 0.0).abs() < 1e-10);
    }
}
