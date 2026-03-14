//! Fleet-as-Portfolio Optimization.
//!
//! Borrows concepts from quantitative portfolio management:
//!
//! - **Diversification**: don't over-concentrate drones on a single task.
//! - **Correlation / coverage overlap**: manage redundant sensor footprints.
//! - **Risk budgeting**: allocate drones proportional to task criticality.
//! - **Position sizing**: critical tasks get more drones, patrol tasks get fewer.
//! - **Rebalancing**: adjust allocation as new intelligence arrives.

use serde::{Deserialize, Serialize};

use crate::{Assignment, DroneState, Position, Task};

// ────────────────────────────────────────────────────────────────────────────────
// Types
// ────────────────────────────────────────────────────────────────────────────────

/// NxN coverage overlap matrix between drone sensor footprints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageMatrix {
    /// Number of drones.
    pub n: usize,
    /// Flattened row-major matrix: `data[i * n + j]` = overlap between drone i and j.
    pub data: Vec<f64>,
}

impl CoverageMatrix {
    /// Build a coverage matrix from drone positions and a sensor range.
    pub fn from_drones(drones: &[DroneState], sensor_range: f64) -> Self {
        let n = drones.len();
        let mut data = vec![0.0; n * n];

        for i in 0..n {
            for j in 0..n {
                if i == j {
                    data[i * n + j] = 1.0;
                } else {
                    let dist = drones[i].position.distance_to(&drones[j].position);
                    // Overlap is 1 when positions coincide, 0 when >= 2*sensor_range apart.
                    let max_overlap_dist = 2.0 * sensor_range;
                    data[i * n + j] = if dist >= max_overlap_dist {
                        0.0
                    } else {
                        1.0 - dist / max_overlap_dist
                    };
                }
            }
        }

        Self { n, data }
    }

    /// Get the overlap value between drone i and drone j.
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.n + j]
    }

    /// Average pairwise overlap (excluding self-overlap). Higher means more redundancy.
    pub fn average_overlap(&self) -> f64 {
        if self.n < 2 {
            return 0.0;
        }
        let sum: f64 = self
            .data
            .iter()
            .enumerate()
            .filter(|(idx, _)| {
                let i = idx / self.n;
                let j = idx % self.n;
                i != j
            })
            .map(|(_, &v)| v)
            .sum();
        sum / (self.n * (self.n - 1)) as f64
    }
}

/// Per-task allocation recommendation from the portfolio optimizer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskAllocation {
    pub task_id: u32,
    /// Recommended number of drones for this task.
    pub recommended_drones: usize,
    /// Risk-budget weight [0, 1].
    pub risk_weight: f64,
}

/// Fleet portfolio optimizer.
#[derive(Debug, Clone)]
pub struct PortfolioOptimizer {
    /// Maximum fraction of fleet that may be assigned to a single task.
    pub max_concentration: f64,
    /// Sensor footprint range (metres) used for coverage overlap calculations.
    pub sensor_range: f64,
    /// Minimum drones to keep in reserve (not assigned to any task).
    pub reserve_count: usize,
}

// ────────────────────────────────────────────────────────────────────────────────
// Implementation
// ────────────────────────────────────────────────────────────────────────────────

impl Default for PortfolioOptimizer {
    fn default() -> Self {
        Self {
            max_concentration: 0.5,
            sensor_range: 500.0,
            reserve_count: 1,
        }
    }
}

impl PortfolioOptimizer {
    /// Create a new optimizer with default parameters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder: set maximum concentration per task.
    pub fn with_max_concentration(mut self, max: f64) -> Self {
        self.max_concentration = max.clamp(0.0, 1.0);
        self
    }

    /// Builder: set sensor range for coverage overlap.
    pub fn with_sensor_range(mut self, range: f64) -> Self {
        self.sensor_range = range;
        self
    }

    /// Builder: set reserve count.
    pub fn with_reserve(mut self, count: usize) -> Self {
        self.reserve_count = count;
        self
    }

    /// Compute the coverage (overlap) matrix for the fleet.
    pub fn coverage_matrix(&self, drones: &[DroneState]) -> CoverageMatrix {
        CoverageMatrix::from_drones(drones, self.sensor_range)
    }

    /// Optimise fleet allocation: decide how many drones each task should get.
    ///
    /// Uses a risk-budget approach: critical tasks receive a proportionally
    /// larger share of the fleet, while patrol tasks receive fewer drones.
    /// Diversification caps prevent over-concentration.
    pub fn optimize_fleet_allocation(
        &self,
        drones: &[DroneState],
        tasks: &[Task],
    ) -> Vec<TaskAllocation> {
        let alive_count = drones.iter().filter(|d| d.alive).count();
        if alive_count == 0 || tasks.is_empty() {
            return tasks
                .iter()
                .map(|t| TaskAllocation {
                    task_id: t.id,
                    recommended_drones: 0,
                    risk_weight: 0.0,
                })
                .collect();
        }

        let available = alive_count.saturating_sub(self.reserve_count);

        // Compute raw risk weights from task priority * urgency.
        let raw_weights: Vec<f64> = tasks
            .iter()
            .map(|t| t.priority * t.urgency + 0.01)
            .collect();
        let total_weight: f64 = raw_weights.iter().sum();

        let max_per_task = ((available as f64) * self.max_concentration).ceil() as usize;

        let mut allocations: Vec<TaskAllocation> = Vec::with_capacity(tasks.len());
        let mut total_allocated = 0usize;

        for (i, task) in tasks.iter().enumerate() {
            let risk_weight = raw_weights[i] / total_weight;
            let raw_count = (available as f64 * risk_weight).round() as usize;
            let capped = raw_count.min(max_per_task).max(1);
            let assigned = capped.min(available.saturating_sub(total_allocated));

            allocations.push(TaskAllocation {
                task_id: task.id,
                recommended_drones: assigned,
                risk_weight,
            });
            total_allocated += assigned;
        }

        allocations
    }

    /// Rebalance existing assignments after new intel arrives.
    ///
    /// Returns a list of reassignment recommendations (new allocations) that
    /// differ from the current state. The caller is responsible for triggering
    /// re-auctions for affected tasks.
    pub fn rebalance(
        &self,
        drones: &[DroneState],
        tasks: &[Task],
        current_assignments: &[Assignment],
    ) -> Vec<TaskAllocation> {
        let optimal = self.optimize_fleet_allocation(drones, tasks);

        // Count current drones per task.
        let mut current_counts: std::collections::HashMap<u32, usize> =
            std::collections::HashMap::new();
        for a in current_assignments {
            *current_counts.entry(a.task_id).or_insert(0) += 1;
        }

        // Only return tasks where the recommendation differs from current state.
        optimal
            .into_iter()
            .filter(|alloc| {
                let current = current_counts.get(&alloc.task_id).copied().unwrap_or(0);
                current != alloc.recommended_drones
            })
            .collect()
    }

    /// Evaluate portfolio diversification quality.
    ///
    /// Returns a score in [0, 1] where 1 means perfectly diversified (no overlap,
    /// even distribution) and 0 means fully concentrated.
    pub fn diversification_score(
        &self,
        drones: &[DroneState],
        assignments: &[Assignment],
        tasks: &[Task],
    ) -> f64 {
        if assignments.is_empty() || tasks.is_empty() {
            return 0.0;
        }

        // Factor 1: concentration — how evenly are drones spread across tasks?
        // Two sub-factors:
        //   (a) Evenness: are assigned tasks getting equal drone counts?
        //   (b) Coverage: what fraction of available tasks have at least one drone?
        let mut counts: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
        for a in assignments {
            *counts.entry(a.task_id).or_insert(0) += 1;
        }
        let max_count = *counts.values().max().unwrap_or(&0) as f64;
        let ideal_count = assignments.len() as f64 / tasks.len().max(1) as f64;
        let evenness = if max_count > 0.0 {
            (ideal_count / max_count).min(1.0)
        } else {
            0.0
        };
        let coverage = counts.len() as f64 / tasks.len().max(1) as f64;
        let concentration_score = evenness * 0.5 + coverage * 0.5;

        // Factor 2: coverage overlap — lower is better for diversification.
        let cov = self.coverage_matrix(drones);
        let overlap_score = 1.0 - cov.average_overlap();

        // Weighted combination.
        concentration_score * 0.6 + overlap_score * 0.4
    }
}

/// Compute distance from a position to the centroid of a group of drones.
pub fn centroid_distance(drones: &[DroneState], target: &Position) -> f64 {
    if drones.is_empty() {
        return f64::INFINITY;
    }
    let cx: f64 = drones.iter().map(|d| d.position.x).sum::<f64>() / drones.len() as f64;
    let cy: f64 = drones.iter().map(|d| d.position.y).sum::<f64>() / drones.len() as f64;
    let cz: f64 = drones.iter().map(|d| d.position.z).sum::<f64>() / drones.len() as f64;
    let centroid = Position::new(cx, cy, cz);
    centroid.distance_to(target)
}

// ────────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Capabilities, Regime};

    fn drone_at(id: u32, x: f64, y: f64) -> DroneState {
        DroneState {
            id,
            position: Position::new(x, y, 100.0),
            velocity: [0.0; 3],
            regime: Regime::Patrol,
            capabilities: Capabilities::default(),
            energy: 1.0,
            alive: true,
        }
    }

    fn task_at(id: u32, x: f64, y: f64, priority: f64, urgency: f64) -> Task {
        Task {
            id,
            location: Position::new(x, y, 50.0),
            required_capabilities: Capabilities::default(),
            priority,
            urgency,
            bundle_id: None,
            dark_pool: None,
        }
    }

    #[test]
    fn test_coverage_matrix_self_overlap() {
        let drones = vec![drone_at(1, 0.0, 0.0), drone_at(2, 1000.0, 1000.0)];
        let cov = CoverageMatrix::from_drones(&drones, 500.0);
        assert!((cov.get(0, 0) - 1.0).abs() < 1e-12);
        assert!((cov.get(1, 1) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_coverage_matrix_no_overlap() {
        let drones = vec![drone_at(1, 0.0, 0.0), drone_at(2, 5000.0, 5000.0)];
        let cov = CoverageMatrix::from_drones(&drones, 500.0);
        assert!((cov.get(0, 1)).abs() < 1e-12);
    }

    #[test]
    fn test_coverage_matrix_partial_overlap() {
        let drones = vec![drone_at(1, 0.0, 0.0), drone_at(2, 500.0, 0.0)];
        let cov = CoverageMatrix::from_drones(&drones, 500.0);
        let overlap = cov.get(0, 1);
        assert!(
            overlap > 0.0 && overlap < 1.0,
            "partial overlap = {overlap}"
        );
    }

    #[test]
    fn test_optimize_fleet_allocation_proportional() {
        let drones: Vec<DroneState> = (0..10)
            .map(|i| drone_at(i, i as f64 * 100.0, 0.0))
            .collect();
        let tasks = vec![
            task_at(1, 0.0, 0.0, 0.9, 0.9),   // high priority
            task_at(2, 500.0, 0.0, 0.1, 0.1), // low priority
        ];

        let opt = PortfolioOptimizer::new().with_reserve(0);
        let allocs = opt.optimize_fleet_allocation(&drones, &tasks);

        assert_eq!(allocs.len(), 2);
        let high = allocs.iter().find(|a| a.task_id == 1).unwrap();
        let low = allocs.iter().find(|a| a.task_id == 2).unwrap();
        assert!(
            high.recommended_drones >= low.recommended_drones,
            "high-priority task should get >= drones: {} vs {}",
            high.recommended_drones,
            low.recommended_drones,
        );
    }

    #[test]
    fn test_optimize_fleet_allocation_concentration_cap() {
        let drones: Vec<DroneState> = (0..10).map(|i| drone_at(i, 0.0, 0.0)).collect();
        let tasks = vec![task_at(1, 0.0, 0.0, 1.0, 1.0)]; // single very high priority task

        let opt = PortfolioOptimizer::new()
            .with_max_concentration(0.3)
            .with_reserve(0);
        let allocs = opt.optimize_fleet_allocation(&drones, &tasks);

        assert_eq!(allocs.len(), 1);
        assert!(
            allocs[0].recommended_drones <= 3,
            "concentration cap 30% of 10 = 3 max, got {}",
            allocs[0].recommended_drones,
        );
    }

    #[test]
    fn test_optimize_fleet_allocation_no_drones() {
        let opt = PortfolioOptimizer::new();
        let allocs = opt.optimize_fleet_allocation(&[], &[task_at(1, 0.0, 0.0, 0.5, 0.5)]);
        assert_eq!(allocs[0].recommended_drones, 0);
    }

    #[test]
    fn test_rebalance_detects_change() {
        let drones: Vec<DroneState> = (0..6).map(|i| drone_at(i, i as f64 * 100.0, 0.0)).collect();
        let tasks = vec![
            task_at(1, 0.0, 0.0, 0.9, 0.9),
            task_at(2, 500.0, 0.0, 0.1, 0.1),
        ];

        // Current: everything on task 2 (obviously sub-optimal).
        let current = vec![
            Assignment {
                drone_id: 0,
                task_id: 2,
                bid_score: 1.0,
            },
            Assignment {
                drone_id: 1,
                task_id: 2,
                bid_score: 1.0,
            },
            Assignment {
                drone_id: 2,
                task_id: 2,
                bid_score: 1.0,
            },
            Assignment {
                drone_id: 3,
                task_id: 2,
                bid_score: 1.0,
            },
        ];

        let opt = PortfolioOptimizer::new().with_reserve(0);
        let rebalance = opt.rebalance(&drones, &tasks, &current);

        // Should recommend changes since current state is lopsided.
        assert!(!rebalance.is_empty(), "rebalance should detect imbalance");
    }

    #[test]
    fn test_diversification_score() {
        let drones: Vec<DroneState> = (0..4)
            .map(|i| drone_at(i, i as f64 * 1000.0, 0.0))
            .collect();
        let tasks = vec![
            task_at(1, 0.0, 0.0, 0.5, 0.5),
            task_at(2, 3000.0, 0.0, 0.5, 0.5),
        ];

        // Even split.
        let even = vec![
            Assignment {
                drone_id: 0,
                task_id: 1,
                bid_score: 1.0,
            },
            Assignment {
                drone_id: 1,
                task_id: 1,
                bid_score: 1.0,
            },
            Assignment {
                drone_id: 2,
                task_id: 2,
                bid_score: 1.0,
            },
            Assignment {
                drone_id: 3,
                task_id: 2,
                bid_score: 1.0,
            },
        ];

        // Concentrated.
        let concentrated = vec![
            Assignment {
                drone_id: 0,
                task_id: 1,
                bid_score: 1.0,
            },
            Assignment {
                drone_id: 1,
                task_id: 1,
                bid_score: 1.0,
            },
            Assignment {
                drone_id: 2,
                task_id: 1,
                bid_score: 1.0,
            },
            Assignment {
                drone_id: 3,
                task_id: 1,
                bid_score: 1.0,
            },
        ];

        let opt = PortfolioOptimizer::new();
        let score_even = opt.diversification_score(&drones, &even, &tasks);
        let score_conc = opt.diversification_score(&drones, &concentrated, &tasks);

        assert!(
            score_even > score_conc,
            "even split ({score_even}) should score higher than concentrated ({score_conc})"
        );
    }

    #[test]
    fn test_centroid_distance() {
        let drones = vec![drone_at(1, 0.0, 0.0), drone_at(2, 10.0, 0.0)];
        let target = Position::new(5.0, 0.0, 100.0);
        let dist = centroid_distance(&drones, &target);
        assert!(
            dist < 1e-9,
            "centroid at (5,0,100) should be at target, got {dist}"
        );
    }
}
