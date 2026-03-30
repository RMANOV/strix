//! GNN layers for GCBF+ neural barrier evaluation.
//!
//! Two-layer Graph Convolutional Network (GCN) with mean aggregation,
//! following the GCBF+ architecture (arXiv:2401.14554).
//!
//! Weight matrices are stored as flat row-major `Vec<f64>` for cache locality.

use super::graph::GraphTopology;

/// A single GCN message-passing layer.
pub struct GcnLayer {
    /// Edge message transform, row-major: [hidden_dim × edge_in].
    pub w_edge: Vec<f64>,
    pub b_edge: Vec<f64>,
    /// Node update transform, row-major: [hidden_dim × (node_in + hidden_dim)].
    pub w_node: Vec<f64>,
    pub b_node: Vec<f64>,
    pub hidden_dim: usize,
    pub edge_in: usize,
    pub node_in: usize,
}

/// Single linear layer (barrier or action head).
pub struct MlpHead {
    /// Row-major: [output_dim × input_dim].
    pub w: Vec<f64>,
    pub b: Vec<f64>,
    pub output_dim: usize,
    pub input_dim: usize,
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
    /// Topology (`edges`, `degree`) and `edge_features` are separate so layer 2
    /// can reuse the same topology with different edge features (no clone).
    pub fn forward(
        &self,
        node_states: &[Vec<f64>],
        edges: &[(usize, usize)],
        degree: &[usize],
        edge_features: &[[f64; 7]],
        n_nodes: usize,
    ) -> Vec<Vec<f64>> {
        let h = self.hidden_dim;

        // Step 1: edge messages — one mat-vec per edge.
        let mut msg_buf = vec![0.0; h];
        let mut aggregated = vec![vec![0.0; h]; n_nodes];

        for (edge_idx, ef) in edge_features.iter().enumerate() {
            flat_mat_vec_relu(
                &self.w_edge,
                &self.b_edge,
                ef,
                h,
                self.edge_in,
                &mut msg_buf,
            );
            let (_src, tgt) = edges[edge_idx];
            let agg = &mut aggregated[tgt];
            for d in 0..h {
                agg[d] += msg_buf[d];
            }
        }

        // Step 2: normalize by degree (mean pooling).
        for (i, agg) in aggregated.iter_mut().enumerate() {
            if degree[i] > 0 {
                let inv_deg = 1.0 / degree[i] as f64;
                for val in agg.iter_mut() {
                    *val *= inv_deg;
                }
            }
        }

        // Step 3: node update — concat [node_state || aggregated], apply W_node + ReLU.
        let concat_dim = self.node_in;
        let mut concat_buf = vec![0.0; concat_dim];
        let mut output = Vec::with_capacity(n_nodes);
        for i in 0..n_nodes {
            let ns = &node_states[i];
            let ns_len = ns.len();
            concat_buf[..ns_len].copy_from_slice(ns);
            concat_buf[ns_len..ns_len + h].copy_from_slice(&aggregated[i]);
            let mut h_new = vec![0.0; h];
            flat_mat_vec_relu(
                &self.w_node,
                &self.b_node,
                &concat_buf[..concat_dim],
                h,
                concat_dim,
                &mut h_new,
            );
            output.push(h_new);
        }
        output
    }
}

impl MlpHead {
    /// Linear transform into pre-sized output buffer (no activation).
    pub fn forward_into(&self, input: &[f64], out: &mut [f64]) {
        for (r, out_val) in out.iter_mut().enumerate().take(self.output_dim) {
            let row_start = r * self.input_dim;
            let dot: f64 = self.w[row_start..row_start + self.input_dim]
                .iter()
                .zip(input.iter())
                .map(|(a, b)| a * b)
                .sum();
            *out_val = dot + self.b[r];
        }
    }
}

impl GnnEncoder {
    /// Full forward pass: 2 GCN layers + barrier + action heads.
    pub fn forward(&self, graph: &GraphTopology, max_correction: f64) -> (Vec<f64>, Vec<[f64; 3]>) {
        if graph.n_nodes == 0 {
            return (Vec::new(), Vec::new());
        }

        let node_states_0: Vec<Vec<f64>> =
            graph.node_features.iter().map(|nf| nf.to_vec()).collect();

        // Layer 1: use graph's edge features directly.
        let h1 = self.layer1.forward(
            &node_states_0,
            &graph.edges,
            &graph.degree,
            &graph.edge_features,
            graph.n_nodes,
        );

        // Layer 2: reuse topology, modulate edge features by hidden state norms.
        let ef2 = modulate_edge_features(&graph.edges, &graph.edge_features, &h1);
        let h2 = self.layer2.forward(
            &h1,
            &graph.edges,  // borrowed, not cloned
            &graph.degree, // borrowed, not cloned
            &ef2,
            graph.n_nodes,
        );

        // Heads — reuse small output buffers.
        let mut barrier_values = Vec::with_capacity(graph.n_nodes);
        let mut action_corrections = Vec::with_capacity(graph.n_nodes);
        let mut bv_buf = [0.0_f64; 1];
        let mut av_buf = [0.0_f64; 3];
        for hidden in &h2 {
            self.barrier_head.forward_into(hidden, &mut bv_buf);
            barrier_values.push(bv_buf[0]);
            self.action_head.forward_into(hidden, &mut av_buf);
            action_corrections.push([
                av_buf[0].tanh() * max_correction,
                av_buf[1].tanh() * max_correction,
                av_buf[2].tanh() * max_correction,
            ]);
        }

        (barrier_values, action_corrections)
    }
}

/// Modulate edge features by hidden state norms for layer-2 context.
fn modulate_edge_features(
    edges: &[(usize, usize)],
    base_features: &[[f64; 7]],
    hidden_states: &[Vec<f64>],
) -> Vec<[f64; 7]> {
    edges
        .iter()
        .enumerate()
        .map(|(idx, &(src, tgt))| {
            let src_norm = slice_norm(&hidden_states[src]).max(1e-6);
            let tgt_norm = slice_norm(&hidden_states[tgt]).max(1e-6);
            let inv_scale = 1.0 / (src_norm * tgt_norm).sqrt();
            let ef = &base_features[idx];
            [
                ef[0] * inv_scale,
                ef[1] * inv_scale,
                ef[2] * inv_scale,
                ef[3] * inv_scale,
                ef[4] * inv_scale,
                ef[5] * inv_scale,
                ef[6] * inv_scale,
            ]
        })
        .collect()
}

// ── Linear algebra primitives ─────────────────────────────────────────────

/// Row-major matrix-vector multiply with bias and ReLU, writing into `out`.
fn flat_mat_vec_relu(w: &[f64], b: &[f64], x: &[f64], rows: usize, cols: usize, out: &mut [f64]) {
    for r in 0..rows {
        let row_start = r * cols;
        let dot: f64 = w[row_start..row_start + cols]
            .iter()
            .zip(x.iter())
            .map(|(a, b)| a * b)
            .sum();
        out[r] = (dot + b[r]).max(0.0);
    }
}

/// L2 norm of a slice.
#[inline]
fn slice_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_identity_layer(input_dim: usize, edge_dim: usize, hidden: usize) -> GcnLayer {
        let mut w_edge = vec![0.0; hidden * edge_dim];
        for r in 0..hidden {
            let c = r % edge_dim;
            w_edge[r * edge_dim + c] = 1.0;
        }
        let node_in = input_dim + hidden;
        let mut w_node = vec![0.0; hidden * node_in];
        for r in 0..hidden {
            let c = r % node_in;
            w_node[r * node_in + c] = 0.1;
        }
        GcnLayer {
            w_edge,
            b_edge: vec![0.0; hidden],
            w_node,
            b_node: vec![0.1; hidden],
            hidden_dim: hidden,
            edge_in: edge_dim,
            node_in,
        }
    }

    fn make_test_encoder(hidden: usize) -> GnnEncoder {
        let layer1 = make_identity_layer(9, 7, hidden);
        let layer2 = make_identity_layer(hidden, 7, hidden);
        let barrier_head = MlpHead {
            w: vec![0.1; hidden],
            b: vec![0.5],
            output_dim: 1,
            input_dim: hidden,
        };
        let mut action_w = vec![0.0; 3 * hidden];
        for r in 0..3 {
            for c in 0..hidden {
                if r == c % 3 {
                    action_w[r * hidden + c] = 0.1;
                }
            }
        }
        let action_head = MlpHead {
            w: action_w,
            b: vec![0.0; 3],
            output_dim: 3,
            input_dim: hidden,
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
        let velocities = vec![Vector3::new(100.0, 100.0, 100.0); 2];
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
    fn flat_mat_vec_relu_applies_relu() {
        let w = vec![1.0, 0.0, 0.0, 1.0]; // 2×2 identity
        let b = vec![-0.5, 0.5];
        let x = vec![0.3, -1.0];
        let mut out = vec![0.0; 2];
        super::flat_mat_vec_relu(&w, &b, &x, 2, 2, &mut out);
        assert!((out[0] - 0.0).abs() < 1e-10); // 0.3 - 0.5 = -0.2 → ReLU → 0
        assert!((out[1] - 0.0).abs() < 1e-10); // -1.0 + 0.5 = -0.5 → ReLU → 0
    }
}
