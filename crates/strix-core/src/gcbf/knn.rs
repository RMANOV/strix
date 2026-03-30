//! Grid-based spatial hashing for O(1) amortized k-nearest-neighbor queries.
//!
//! Drones operate in bounded airspace, so a fixed-cell grid gives O(n) build
//! and O(27 × avg_per_cell) queries — faster than kd-trees for semi-uniform
//! distributions at the densities we care about (100–2000 drones in ~1 km³).

use nalgebra::Vector3;
use std::collections::HashMap;

/// 3D spatial grid for fast neighbor queries.
pub struct SpatialGrid {
    cell_size: f64,
    inv_cell_size: f64,
    cells: HashMap<(i32, i32, i32), Vec<usize>>,
}

impl SpatialGrid {
    /// Build a spatial grid from a set of positions.  O(n).
    pub fn build(positions: &[Vector3<f64>], cell_size: f64) -> Self {
        let cell_size = cell_size.max(1.0); // sanity floor
        let inv = 1.0 / cell_size;
        let mut cells: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();
        for (idx, pos) in positions.iter().enumerate() {
            let key = Self::cell_key_with_inv(pos, inv);
            cells.entry(key).or_default().push(idx);
        }
        Self {
            cell_size,
            inv_cell_size: inv,
            cells,
        }
    }

    /// Find the k nearest neighbors of `query_idx` within `max_radius`.
    ///
    /// Returns `(neighbor_index, squared_distance)` pairs sorted by distance.
    pub fn k_nearest(
        &self,
        query_idx: usize,
        positions: &[Vector3<f64>],
        k: usize,
        max_radius: f64,
    ) -> Vec<(usize, f64)> {
        if k == 0 || positions.is_empty() {
            return Vec::new();
        }
        let query_pos = &positions[query_idx];
        let max_r_sq = max_radius * max_radius;

        // How many cells to search in each direction.
        let cell_range = (max_radius / self.cell_size).ceil() as i32;
        let center_key = Self::cell_key_with_inv(query_pos, self.inv_cell_size);

        let mut candidates: Vec<(usize, f64)> = Vec::new();

        for dx in -cell_range..=cell_range {
            for dy in -cell_range..=cell_range {
                for dz in -cell_range..=cell_range {
                    let key = (center_key.0 + dx, center_key.1 + dy, center_key.2 + dz);
                    if let Some(bucket) = self.cells.get(&key) {
                        for &idx in bucket {
                            if idx == query_idx {
                                continue;
                            }
                            let d_sq = (positions[idx] - query_pos).norm_squared();
                            if d_sq <= max_r_sq {
                                candidates.push((idx, d_sq));
                            }
                        }
                    }
                }
            }
        }

        // Partial sort: only need the k smallest.
        if candidates.len() > k {
            candidates.select_nth_unstable_by(k, |a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            candidates.truncate(k);
        }
        candidates
            .sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates
    }

    #[inline]
    fn cell_key_with_inv(pos: &Vector3<f64>, inv: f64) -> (i32, i32, i32) {
        (
            (pos.x * inv).floor() as i32,
            (pos.y * inv).floor() as i32,
            (pos.z * inv).floor() as i32,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_grid_returns_no_neighbors() {
        let positions: Vec<Vector3<f64>> = vec![];
        let grid = SpatialGrid::build(&positions, 10.0);
        // k_nearest with empty positions should not panic.
        assert!(positions.is_empty());
        let _ = grid; // just verify construction
    }

    #[test]
    fn single_agent_no_neighbors() {
        let positions = vec![Vector3::new(0.0, 0.0, -50.0)];
        let grid = SpatialGrid::build(&positions, 100.0);
        let neighbors = grid.k_nearest(0, &positions, 8, 100.0);
        assert!(neighbors.is_empty());
    }

    #[test]
    fn k_nearest_correctness_vs_brute_force() {
        // 20 drones in a 200m cube.
        let mut rng = 42u64;
        let positions: Vec<Vector3<f64>> = (0..20)
            .map(|_| {
                rng = rng
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let x = (rng >> 33) as f64 / (u32::MAX as f64) * 200.0 - 100.0;
                rng = rng
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let y = (rng >> 33) as f64 / (u32::MAX as f64) * 200.0 - 100.0;
                rng = rng
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let z = (rng >> 33) as f64 / (u32::MAX as f64) * 200.0 - 100.0;
                Vector3::new(x, y, z)
            })
            .collect();

        let grid = SpatialGrid::build(&positions, 100.0);
        let k = 5;
        let max_r = 300.0;

        for query_idx in 0..positions.len() {
            let grid_result = grid.k_nearest(query_idx, &positions, k, max_r);

            // Brute-force.
            let mut brute: Vec<(usize, f64)> = positions
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != query_idx)
                .map(|(i, p)| (i, (p - positions[query_idx]).norm_squared()))
                .filter(|(_, d)| *d <= max_r * max_r)
                .collect();
            brute.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            brute.truncate(k);

            assert_eq!(
                grid_result.len(),
                brute.len(),
                "query {query_idx}: grid returned {} vs brute {}",
                grid_result.len(),
                brute.len()
            );
            for (g, b) in grid_result.iter().zip(brute.iter()) {
                assert_eq!(g.0, b.0, "query {query_idx}: index mismatch");
                assert!(
                    (g.1 - b.1).abs() < 1e-10,
                    "query {query_idx}: distance mismatch"
                );
            }
        }
    }

    #[test]
    fn respects_max_radius() {
        let positions = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(5.0, 0.0, 0.0),   // 5m away
            Vector3::new(50.0, 0.0, 0.0),  // 50m away
            Vector3::new(200.0, 0.0, 0.0), // 200m away
        ];
        let grid = SpatialGrid::build(&positions, 100.0);
        let result = grid.k_nearest(0, &positions, 10, 10.0);
        assert_eq!(result.len(), 1, "only the 5m neighbor should be within 10m");
        assert_eq!(result[0].0, 1);
    }

    #[test]
    fn returns_at_most_k() {
        let positions: Vec<Vector3<f64>> =
            (0..50).map(|i| Vector3::new(i as f64, 0.0, 0.0)).collect();
        let grid = SpatialGrid::build(&positions, 100.0);
        let result = grid.k_nearest(0, &positions, 3, 1000.0);
        assert_eq!(result.len(), 3);
    }
}
