//! Sparse graph topology for GNN message passing.
//!
//! Builds a k-NN directed graph from drone positions using [`SpatialGrid`].
//! Each node carries 9D features (relative position, velocity, goal in NED)
//! and each edge carries 7D features (relative position, velocity, distance).

use super::config::GcbfConfig;
use super::knn::SpatialGrid;
use nalgebra::Vector3;

/// Sparse directed graph for GNN message passing.
pub struct GraphTopology {
    pub n_nodes: usize,
    /// Directed edges: `(source, target)` — target receives message from source.
    pub edges: Vec<(usize, usize)>,
    /// Per-node neighbor count.
    pub degree: Vec<usize>,
    /// Node features: [rel_goal_x, rel_goal_y, rel_goal_z, vel_x, vel_y, vel_z, 0, 0, 0].
    /// Positions are relative to self (always zero for self-node), so we store
    /// goal-relative position + velocity. The remaining 3 dims are reserved.
    pub node_features: Vec<[f64; 9]>,
    /// Edge features: [rel_pos_x, rel_pos_y, rel_pos_z, rel_vel_x, rel_vel_y, rel_vel_z, distance].
    pub edge_features: Vec<[f64; 7]>,
}

impl GraphTopology {
    /// Build a k-NN graph from positions, velocities, and goals.
    ///
    /// `goals` may be empty — in which case goal-relative features are zero.
    pub fn build(
        positions: &[Vector3<f64>],
        velocities: &[Vector3<f64>],
        goals: &[Vector3<f64>],
        config: &GcbfConfig,
    ) -> Self {
        let n = positions.len();
        if n == 0 {
            return Self {
                n_nodes: 0,
                edges: Vec::new(),
                degree: Vec::new(),
                node_features: Vec::new(),
                edge_features: Vec::new(),
            };
        }

        let grid = SpatialGrid::build(positions, config.grid_cell_size);

        let mut edges = Vec::with_capacity(n * config.k_neighbors);
        let mut edge_features = Vec::with_capacity(n * config.k_neighbors);
        let mut degree = vec![0usize; n];

        for i in 0..n {
            let neighbors = grid.k_nearest(i, positions, config.k_neighbors, config.comm_radius);
            for (j, dist_sq) in &neighbors {
                edges.push((*j, i)); // message flows from j → i (source, target)
                let rel_pos = positions[*j] - positions[i];
                let rel_vel = velocities.get(*j).copied().unwrap_or_else(Vector3::zeros)
                    - velocities.get(i).copied().unwrap_or_else(Vector3::zeros);
                let dist = dist_sq.sqrt();
                edge_features.push([
                    rel_pos.x, rel_pos.y, rel_pos.z, rel_vel.x, rel_vel.y, rel_vel.z, dist,
                ]);
            }
            degree[i] = neighbors.len();
        }

        // Node features: goal-relative position + velocity.
        let node_features: Vec<[f64; 9]> = (0..n)
            .map(|i| {
                let vel = velocities.get(i).copied().unwrap_or_else(Vector3::zeros);
                let goal_rel = if i < goals.len() {
                    goals[i] - positions[i]
                } else {
                    Vector3::zeros()
                };
                [
                    goal_rel.x, goal_rel.y, goal_rel.z, vel.x, vel.y, vel.z, 0.0, 0.0, 0.0,
                ]
            })
            .collect();

        Self {
            n_nodes: n,
            edges,
            degree,
            node_features,
            edge_features,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> GcbfConfig {
        GcbfConfig {
            k_neighbors: 3,
            comm_radius: 50.0,
            grid_cell_size: 50.0,
            ..GcbfConfig::default()
        }
    }

    #[test]
    fn empty_graph() {
        let g = GraphTopology::build(&[], &[], &[], &default_config());
        assert_eq!(g.n_nodes, 0);
        assert!(g.edges.is_empty());
    }

    #[test]
    fn single_node_no_edges() {
        let pos = vec![Vector3::new(0.0, 0.0, -50.0)];
        let vel = vec![Vector3::zeros()];
        let g = GraphTopology::build(&pos, &vel, &[], &default_config());
        assert_eq!(g.n_nodes, 1);
        assert!(g.edges.is_empty());
        assert_eq!(g.degree[0], 0);
    }

    #[test]
    fn four_nodes_correct_edge_count() {
        let pos = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(10.0, 0.0, 0.0),
            Vector3::new(0.0, 10.0, 0.0),
            Vector3::new(10.0, 10.0, 0.0),
        ];
        let vel = vec![Vector3::zeros(); 4];
        let cfg = default_config(); // k=3
        let g = GraphTopology::build(&pos, &vel, &[], &cfg);
        assert_eq!(g.n_nodes, 4);
        // Each node has 3 neighbors (all others), so 4*3=12 edges.
        assert_eq!(g.edges.len(), 12);
        assert_eq!(g.edge_features.len(), 12);
        for &d in &g.degree {
            assert_eq!(d, 3);
        }
    }

    #[test]
    fn edge_features_have_correct_distance() {
        let pos = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(3.0, 4.0, 0.0), // distance = 5.0
        ];
        let vel = vec![Vector3::zeros(); 2];
        let cfg = GcbfConfig {
            k_neighbors: 1,
            comm_radius: 50.0,
            grid_cell_size: 50.0,
            ..GcbfConfig::default()
        };
        let g = GraphTopology::build(&pos, &vel, &[], &cfg);
        assert_eq!(g.edges.len(), 2); // bidirectional
                                      // Check distance field (index 6).
        for ef in &g.edge_features {
            assert!(
                (ef[6] - 5.0).abs() < 1e-10,
                "distance should be 5.0, got {}",
                ef[6]
            );
        }
    }

    #[test]
    fn node_features_include_goal() {
        let pos = vec![Vector3::new(0.0, 0.0, 0.0)];
        let vel = vec![Vector3::new(1.0, 2.0, 3.0)];
        let goals = vec![Vector3::new(100.0, 200.0, -50.0)];
        let g = GraphTopology::build(&pos, &vel, &goals, &default_config());
        let nf = &g.node_features[0];
        assert!((nf[0] - 100.0).abs() < 1e-10); // goal_rel_x
        assert!((nf[1] - 200.0).abs() < 1e-10); // goal_rel_y
        assert!((nf[2] - -50.0).abs() < 1e-10); // goal_rel_z
        assert!((nf[3] - 1.0).abs() < 1e-10); // vel_x
        assert!((nf[4] - 2.0).abs() < 1e-10); // vel_y
        assert!((nf[5] - 3.0).abs() < 1e-10); // vel_z
    }
}
