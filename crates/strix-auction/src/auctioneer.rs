//! Auction Engine — combinatorial task allocation via modified Hungarian algorithm.
//!
//! The [`Auctioneer`] collects sealed bids from all drones, handles task bundling
//! and dark-pool filtering, then solves the assignment problem to produce an
//! optimal drone-to-task mapping.
//!
//! ## Features
//!
//! - **Hungarian algorithm** (O(n^3)) for optimal assignment.
//! - **Task bundling**: tasks sharing a `bundle_id` are awarded together.
//! - **Dark pool**: compartmentalized tasks visible only to designated sub-swarms.
//! - **Re-auction triggers**: when conditions change (drone loss, new intel).

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::bidder::{Bid, Bidder, ScenarioContext};
use crate::{Assignment, DroneState, Task, ThreatState};

// ────────────────────────────────────────────────────────────────────────────────
// Types
// ────────────────────────────────────────────────────────────────────────────────

/// Outcome of a single auction round.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuctionResult {
    /// Finalized drone-to-task assignments.
    pub assignments: Vec<Assignment>,
    /// Tasks that received no valid bid (remain unassigned).
    pub unassigned_tasks: Vec<u32>,
    /// Total market welfare (sum of winning bid scores).
    pub total_welfare: f64,
}

/// The auctioneer that orchestrates the market.
#[derive(Debug, Clone)]
pub struct Auctioneer {
    /// Threshold below which a bid is considered too weak to accept.
    pub min_bid_threshold: f64,
    /// Whether a re-auction is needed (set by external events).
    pub needs_reauction: bool,
    /// Fear level F ∈ [0,1] — passed to bidders for risk-adjusted scoring.
    pub fear: f64,
    /// Per-drone scenario contexts from phi-sim (doom/upside/confidence).
    /// Set before `run_auction` to inject scenario-enriched bid scoring.
    pub scenario_contexts: HashMap<u32, ScenarioContext>,
}

// ────────────────────────────────────────────────────────────────────────────────
// Implementation
// ────────────────────────────────────────────────────────────────────────────────

impl Default for Auctioneer {
    fn default() -> Self {
        Self {
            min_bid_threshold: 0.0,
            needs_reauction: false,
            fear: 0.0,
            scenario_contexts: HashMap::new(),
        }
    }
}

impl Auctioneer {
    /// Create a new auctioneer with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the minimum bid threshold below which bids are discarded.
    pub fn with_min_bid(mut self, threshold: f64) -> Self {
        self.min_bid_threshold = threshold;
        self
    }

    /// Run a full auction round: collect bids, solve assignment, return results.
    ///
    /// # Arguments
    /// - `drones`: fleet state (each drone bids independently).
    /// - `tasks`: tasks available for auction this round.
    /// - `threats`: current threat picture (affects bid risk scoring).
    /// - `sub_swarm_map`: maps `drone_id → sub_swarm_id` for dark-pool eligibility.
    /// - `kill_zone_penalties`: global kill-zone list from antifragile module.
    pub fn run_auction(
        &mut self,
        drones: &[DroneState],
        tasks: &[Task],
        threats: &[ThreatState],
        sub_swarm_map: &HashMap<u32, u32>,
        kill_zone_penalties: &[(crate::Position, f64, f64)],
    ) -> AuctionResult {
        // 1. Collect bids from all drones.
        let all_bids =
            self.collect_bids(drones, tasks, threats, sub_swarm_map, kill_zone_penalties);

        // 2. Group tasks by bundle.
        let bundles = group_bundles(tasks);

        // 3. Build cost matrix & solve assignment.
        let result = self.solve_assignment(tasks, &all_bids, &bundles);

        self.needs_reauction = false;
        result
    }

    /// Set scenario context for a specific drone (from phi-sim decisions).
    pub fn set_scenario_context(&mut self, drone_id: u32, ctx: ScenarioContext) {
        self.scenario_contexts.insert(drone_id, ctx);
    }

    /// Clear all scenario contexts (e.g. between auction rounds).
    pub fn clear_scenario_contexts(&mut self) {
        self.scenario_contexts.clear();
    }

    /// Signal that conditions have changed and a re-auction is warranted.
    pub fn trigger_reauction(&mut self) {
        self.needs_reauction = true;
    }

    /// Finalize assignments and return the result. In a real system this would
    /// send commands to the winning drones.
    pub fn clear_market(&self, result: &AuctionResult) -> Vec<Assignment> {
        result.assignments.clone()
    }

    // ── Private ─────────────────────────────────────────────────────────────

    fn collect_bids(
        &self,
        drones: &[DroneState],
        tasks: &[Task],
        threats: &[ThreatState],
        sub_swarm_map: &HashMap<u32, u32>,
        kill_zone_penalties: &[(crate::Position, f64, f64)],
    ) -> Vec<Bid> {
        let mut all_bids = Vec::new();
        for drone in drones {
            let mut bidder = Bidder::new(drone.clone()).with_fear(self.fear);
            if let Some(&ssid) = sub_swarm_map.get(&drone.id) {
                bidder = bidder.with_sub_swarm(ssid);
            }
            if !kill_zone_penalties.is_empty() {
                bidder = bidder.with_kill_zone_penalties(kill_zone_penalties.to_vec());
            }
            if let Some(ctx) = self.scenario_contexts.get(&drone.id) {
                bidder = bidder.with_scenario_context(ctx.clone());
            }
            let bids = bidder.bid_on_tasks(tasks, threats);
            all_bids.extend(bids);
        }
        // Filter bids below threshold.
        all_bids.retain(|b| b.score >= self.min_bid_threshold);
        all_bids
    }

    fn solve_assignment(
        &self,
        tasks: &[Task],
        bids: &[Bid],
        bundles: &HashMap<u32, Vec<u32>>,
    ) -> AuctionResult {
        // Collect unique drone IDs that have bids.
        let drone_ids: Vec<u32> = {
            let mut s: Vec<u32> = bids.iter().map(|b| b.drone_id).collect();
            s.sort_unstable();
            s.dedup();
            s
        };
        let all_task_ids: Vec<u32> = {
            let mut s: Vec<u32> = tasks.iter().map(|t| t.id).collect();
            s.sort_unstable();
            s.dedup();
            s
        };

        if drone_ids.is_empty() || all_task_ids.is_empty() {
            return AuctionResult {
                assignments: vec![],
                unassigned_tasks: all_task_ids,
                total_welfare: 0.0,
            };
        }

        // Build cost matrix. Filter NaN bid scores to prevent corruption.
        let max_score = bids
            .iter()
            .map(|b| b.score)
            .filter(|s| s.is_finite())
            .fold(0.0_f64, f64::max);
        let big_penalty = (max_score.abs() + 1.0) * 10.0;

        // Build a flat bid lookup: bid_scores[drone_idx * n_tasks + task_idx] → score.
        // Sentinel value -big_penalty means no bid. drone_ids and all_task_ids are
        // already sorted+deduped, so binary_search gives O(log N) dense indices.
        let n_drones = drone_ids.len();
        let n_tasks = all_task_ids.len();
        let mut bid_scores = vec![-big_penalty; n_drones * n_tasks];
        for b in bids {
            if let (Ok(di), Ok(ti)) = (
                drone_ids.binary_search(&b.drone_id),
                all_task_ids.binary_search(&b.task_id),
            ) {
                let s = if b.score.is_finite() {
                    b.score
                } else {
                    -big_penalty
                };
                bid_scores[di * n_tasks + ti] = s;
            }
        }

        // Helper: look up score for (drone_idx, task_id); returns None if sentinel.
        let lookup = |di: usize, tid: u32| -> Option<f64> {
            let ti = all_task_ids.binary_search(&tid).ok()?;
            let s = bid_scores[di * n_tasks + ti];
            if s <= -big_penalty {
                None
            } else {
                Some(s)
            }
        };

        // ── Bundle handling ─────────────────────────────────────────────────
        // Collapse bundled tasks into "super-tasks" for the assignment matrix.
        // A super-task's score = sum of scores for all member tasks.
        // `slot_task_ids[slot]` = vec of real task IDs in that slot.
        let mut slot_task_ids: Vec<Vec<u32>> = Vec::new();
        let mut already_bundled: HashSet<u32> = HashSet::new();

        // First pass: add bundles as single slots.
        for member_ids in bundles.values() {
            if member_ids.len() > 1 {
                slot_task_ids.push(member_ids.clone());
                for &tid in member_ids {
                    already_bundled.insert(tid);
                }
            }
        }

        // Second pass: add un-bundled tasks as individual slots.
        for &tid in &all_task_ids {
            if !already_bundled.contains(&tid) {
                slot_task_ids.push(vec![tid]);
            }
        }

        let n = drone_ids.len().max(slot_task_ids.len());

        let mut cost_matrix = vec![vec![0.0f64; n]; n];
        for (di, _did) in drone_ids.iter().enumerate() {
            for (si, slot) in slot_task_ids.iter().enumerate() {
                // Super-task score = sum of individual bid scores.
                let total_score: f64 = slot
                    .iter()
                    .map(|&tid| lookup(di, tid).unwrap_or(-big_penalty))
                    .sum();
                cost_matrix[di][si] = -total_score;
            }
            cost_matrix[di][slot_task_ids.len()..n].fill(0.0);
        }
        for row in cost_matrix.iter_mut().skip(drone_ids.len()) {
            row.fill(0.0);
        }

        // Run Hungarian algorithm.
        let assignment_vec = hungarian(&cost_matrix);

        // Interpret results — expand super-tasks back to individual assignments.
        let mut assignments = Vec::new();
        let mut assigned_tasks: HashSet<u32> = HashSet::new();
        let mut total_welfare = 0.0;

        for (di, &si) in assignment_vec.iter().enumerate() {
            if di >= drone_ids.len() || si >= slot_task_ids.len() {
                continue;
            }
            let slot = &slot_task_ids[si];

            // Check that the drone has valid bids for at least one task in the slot.
            let has_any_bid = slot
                .iter()
                .any(|&tid| lookup(di, tid).is_some_and(|s| s >= self.min_bid_threshold));

            if has_any_bid {
                let did = drone_ids[di];
                for &tid in slot {
                    let score = lookup(di, tid).unwrap_or(0.0);
                    assignments.push(Assignment {
                        drone_id: did,
                        task_id: tid,
                        bid_score: score,
                    });
                    assigned_tasks.insert(tid);
                    total_welfare += score;
                }
            }
        }

        let unassigned_tasks: Vec<u32> = all_task_ids
            .iter()
            .filter(|tid| !assigned_tasks.contains(tid))
            .copied()
            .collect();

        AuctionResult {
            assignments,
            unassigned_tasks,
            total_welfare,
        }
    }
}

/// Group tasks by their `bundle_id`.
fn group_bundles(tasks: &[Task]) -> HashMap<u32, Vec<u32>> {
    let mut map: HashMap<u32, Vec<u32>> = HashMap::new();
    for task in tasks {
        if let Some(bid) = task.bundle_id {
            map.entry(bid).or_default().push(task.id);
        }
    }
    map
}

// ────────────────────────────────────────────────────────────────────────────────
// Hungarian Algorithm  (O(n^3))
// ────────────────────────────────────────────────────────────────────────────────

/// Classic O(n^3) Hungarian algorithm for square cost matrix.
///
/// Returns a vector `assignment` where `assignment[row] = col`.
///
/// Based on the Kuhn-Munkres method with potential (dual) variables.
pub fn hungarian(cost: &[Vec<f64>]) -> Vec<usize> {
    let n = cost.len();
    if n == 0 {
        return vec![];
    }

    // u[i] = potential for row i,  v[j] = potential for column j.
    let mut u = vec![0.0f64; n + 1];
    let mut v = vec![0.0f64; n + 1];
    // p[j] = row assigned to column j (1-indexed, 0 = unassigned).
    let mut p = vec![0usize; n + 1];
    // way[j] = column in the shortest-path tree that leads to j.
    let mut way = vec![0usize; n + 1];

    let mut min_v = vec![f64::INFINITY; n + 1];
    let mut used = vec![false; n + 1];

    for i in 1..=n {
        p[0] = i;
        let mut j0: usize = 0;
        min_v.fill(f64::INFINITY);
        used.fill(false);

        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = f64::INFINITY;
            let mut j1: usize = 0;

            for j in 1..=n {
                if used[j] {
                    continue;
                }
                let cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
                if cur < min_v[j] {
                    min_v[j] = cur;
                    way[j] = j0;
                }
                if min_v[j] < delta {
                    delta = min_v[j];
                    j1 = j;
                }
            }

            for j in 0..=n {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    min_v[j] -= delta;
                }
            }

            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }

        // Trace back the augmenting path.
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }

    // Build result: assignment[row] = col  (0-indexed).
    let mut result = vec![0usize; n];
    for j in 1..=n {
        if p[j] > 0 {
            result[p[j] - 1] = j - 1;
        }
    }
    result
}

// ────────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Capabilities, DroneState, Position, Regime, Task};

    fn make_drone(id: u32, x: f64, y: f64) -> DroneState {
        DroneState {
            id,
            position: Position::new(x, y, 100.0),
            velocity: [0.0; 3],
            regime: Regime::Patrol,
            capabilities: Capabilities {
                has_sensor: true,
                has_weapon: true,
                has_ew: false,
                has_relay: false,
            },
            energy: 0.9,
            alive: true,
        }
    }

    fn make_task(id: u32, x: f64, y: f64) -> Task {
        Task {
            id,
            location: Position::new(x, y, 50.0),
            required_capabilities: Capabilities {
                has_sensor: true,
                has_weapon: false,
                has_ew: false,
                has_relay: false,
            },
            priority: 0.5,
            urgency: 0.5,
            bundle_id: None,
            dark_pool: None,
        }
    }

    // ── Hungarian algorithm ─────────────────────────────────────────────────

    #[test]
    fn test_hungarian_identity() {
        // Optimal assignment for an identity-like matrix is the diagonal.
        let cost = vec![
            vec![1.0, 100.0, 100.0],
            vec![100.0, 2.0, 100.0],
            vec![100.0, 100.0, 3.0],
        ];
        let result = hungarian(&cost);
        assert_eq!(result, vec![0, 1, 2]);
    }

    #[test]
    fn test_hungarian_swap() {
        // Swapped assignment is cheaper.
        let cost = vec![vec![10.0, 1.0], vec![1.0, 10.0]];
        let result = hungarian(&cost);
        assert_eq!(result, vec![1, 0]);
    }

    #[test]
    fn test_hungarian_3x3() {
        let cost = vec![
            vec![82.0, 83.0, 69.0],
            vec![77.0, 37.0, 49.0],
            vec![11.0, 69.0, 5.0],
        ];
        let result = hungarian(&cost);
        // Optimal: row0→col2 (69), row1→col1 (37), row2→col0 (11) = 117
        let total: f64 = result.iter().enumerate().map(|(i, &j)| cost[i][j]).sum();
        assert!((total - 117.0).abs() < 1e-9, "total = {total}");
    }

    #[test]
    fn test_hungarian_empty() {
        let cost: Vec<Vec<f64>> = vec![];
        let result = hungarian(&cost);
        assert!(result.is_empty());
    }

    #[test]
    fn test_hungarian_1x1() {
        let cost = vec![vec![42.0]];
        let result = hungarian(&cost);
        assert_eq!(result, vec![0]);
    }

    // ── Full auction flow ───────────────────────────────────────────────────

    #[test]
    fn test_run_auction_basic() {
        let drones = vec![make_drone(1, 0.0, 0.0), make_drone(2, 100.0, 100.0)];
        let tasks = vec![make_task(10, 5.0, 5.0), make_task(20, 95.0, 95.0)];

        let mut auctioneer = Auctioneer::new();
        let result = auctioneer.run_auction(&drones, &tasks, &[], &HashMap::new(), &[]);

        assert_eq!(result.assignments.len(), 2, "both tasks should be assigned");
        assert!(result.unassigned_tasks.is_empty());
        assert!(result.total_welfare > 0.0);

        // Verify each task is assigned to the closer drone.
        let a1 = result.assignments.iter().find(|a| a.task_id == 10).unwrap();
        let a2 = result.assignments.iter().find(|a| a.task_id == 20).unwrap();
        assert_eq!(
            a1.drone_id, 1,
            "task near (5,5) should go to drone at (0,0)"
        );
        assert_eq!(
            a2.drone_id, 2,
            "task near (95,95) should go to drone at (100,100)"
        );
    }

    #[test]
    fn test_run_auction_no_drones() {
        let mut auctioneer = Auctioneer::new();
        let result =
            auctioneer.run_auction(&[], &[make_task(1, 0.0, 0.0)], &[], &HashMap::new(), &[]);
        assert!(result.assignments.is_empty());
        assert_eq!(result.unassigned_tasks.len(), 1);
    }

    #[test]
    fn test_run_auction_more_tasks_than_drones() {
        let drones = vec![make_drone(1, 0.0, 0.0)];
        let tasks = vec![
            make_task(10, 5.0, 5.0),
            make_task(20, 50.0, 50.0),
            make_task(30, 90.0, 90.0),
        ];

        let mut auctioneer = Auctioneer::new();
        let result = auctioneer.run_auction(&drones, &tasks, &[], &HashMap::new(), &[]);

        // Only 1 drone, so at most 1 task assigned (plus possible bundle mates).
        assert_eq!(result.assignments.len(), 1);
        assert_eq!(result.unassigned_tasks.len(), 2);
    }

    #[test]
    fn test_reauction_trigger() {
        let mut auctioneer = Auctioneer::new();
        assert!(!auctioneer.needs_reauction);
        auctioneer.trigger_reauction();
        assert!(auctioneer.needs_reauction);
        // Running an auction clears the flag.
        let _ = auctioneer.run_auction(&[], &[], &[], &HashMap::new(), &[]);
        assert!(!auctioneer.needs_reauction);
    }

    #[test]
    fn test_clear_market() {
        let drones = vec![make_drone(1, 0.0, 0.0)];
        let tasks = vec![make_task(10, 5.0, 5.0)];
        let mut auctioneer = Auctioneer::new();
        let result = auctioneer.run_auction(&drones, &tasks, &[], &HashMap::new(), &[]);
        let cleared = auctioneer.clear_market(&result);
        assert_eq!(cleared.len(), result.assignments.len());
    }

    #[test]
    fn test_bundle_assignment() {
        let drones = vec![make_drone(1, 0.0, 0.0), make_drone(2, 100.0, 100.0)];
        let tasks = vec![
            Task {
                id: 10,
                location: Position::new(5.0, 5.0, 50.0),
                required_capabilities: Capabilities::default(),
                priority: 0.5,
                urgency: 0.5,
                bundle_id: Some(1),
                dark_pool: None,
            },
            Task {
                id: 11,
                location: Position::new(8.0, 8.0, 50.0),
                required_capabilities: Capabilities::default(),
                priority: 0.5,
                urgency: 0.5,
                bundle_id: Some(1),
                dark_pool: None,
            },
        ];

        let mut auctioneer = Auctioneer::new();
        let result = auctioneer.run_auction(&drones, &tasks, &[], &HashMap::new(), &[]);

        // Both bundled tasks should go to the same drone.
        let assigned_drones: HashSet<u32> = result.assignments.iter().map(|a| a.drone_id).collect();
        assert_eq!(
            assigned_drones.len(),
            1,
            "bundled tasks should go to one drone, got {:?}",
            assigned_drones
        );
    }
}
