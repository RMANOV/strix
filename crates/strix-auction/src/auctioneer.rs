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

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use strix_mesh::hypergraph::{GroupEffect, GroupVote, HyperEdge, HypergraphCoordinator};
use strix_mesh::NodeId;

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
    /// Maximum number of bids retained per drone after local compression.
    pub max_bids_per_drone: Option<usize>,
    /// Bid volume above which the auction falls back to a greedy anytime clear.
    pub greedy_bid_volume_threshold: usize,
    /// Quorum required before bundled tasks are treated as a true group effect.
    pub bundle_quorum_ratio: f64,
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
            max_bids_per_drone: Some(8),
            greedy_bid_volume_threshold: 256,
            bundle_quorum_ratio: 0.5,
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

    /// Limit how many bids each drone exports into the global market clear.
    pub fn with_bid_cap(mut self, limit: usize) -> Self {
        self.max_bids_per_drone = Some(limit.max(1));
        self
    }

    /// Disable per-drone bid compression.
    pub fn without_bid_cap(mut self) -> Self {
        self.max_bids_per_drone = None;
        self
    }

    /// Set the bid-volume threshold for the greedy anytime clear.
    pub fn with_greedy_bid_volume_threshold(mut self, threshold: usize) -> Self {
        self.greedy_bid_volume_threshold = threshold;
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

        // 2. Group tasks by bundle and only keep higher-order groups that
        // actually show quorum effects in the bidding graph.
        let bundles = group_bundles(tasks);
        let bundles = self.validate_bundles(tasks, &all_bids, &bundles);

        // 3. Build cost matrix & solve assignment.
        let result = if self.needs_reauction || all_bids.len() > self.greedy_bid_volume_threshold {
            self.solve_assignment_greedy(tasks, &all_bids, &bundles)
        } else {
            self.solve_assignment(tasks, &all_bids, &bundles)
        };

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
        // Each drone's bidding is independent — parallelize across drones.
        let mut all_bids: Vec<Bid> = drones
            .par_iter()
            .flat_map(|drone| {
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
                self.compress_bids(tasks, bidder.bid_on_tasks(tasks, threats))
            })
            .collect();
        // Filter bids below threshold.
        all_bids.retain(|b| b.score >= self.min_bid_threshold);
        all_bids
    }

    fn compress_bids(&self, tasks: &[Task], bids: Vec<Bid>) -> Vec<Bid> {
        let Some(limit) = self.max_bids_per_drone else {
            return bids;
        };
        let limit = limit.max(1);
        if bids.len() <= limit {
            return bids;
        }

        let bundle_map: HashMap<u32, u32> = tasks
            .iter()
            .filter_map(|task| task.bundle_id.map(|bundle_id| (task.id, bundle_id)))
            .collect();

        let mut selected = Vec::new();
        let mut selected_tasks: HashSet<u32> = HashSet::new();
        let mut selected_bundles: HashSet<u32> = HashSet::new();

        for bid in bids.iter().take(limit) {
            if selected_tasks.insert(bid.task_id) {
                selected.push(bid.clone());
                if let Some(bundle_id) = bundle_map.get(&bid.task_id) {
                    selected_bundles.insert(*bundle_id);
                }
            }
        }

        if selected_bundles.is_empty() {
            return selected;
        }

        for bid in bids.into_iter().skip(limit) {
            if let Some(bundle_id) = bundle_map.get(&bid.task_id) {
                if selected_bundles.contains(bundle_id) && selected_tasks.insert(bid.task_id) {
                    selected.push(bid);
                }
            }
        }

        selected
    }

    fn validate_bundles(
        &self,
        tasks: &[Task],
        bids: &[Bid],
        bundles: &HashMap<u32, Vec<u32>>,
    ) -> HashMap<u32, Vec<u32>> {
        let mut validated = HashMap::new();
        let mut coordinator = HypergraphCoordinator::default();
        let task_lookup: HashMap<u32, &Task> = tasks.iter().map(|task| (task.id, task)).collect();
        let max_score = bids
            .iter()
            .map(|bid| bid.score)
            .filter(|score| score.is_finite())
            .fold(0.0_f64, f64::max)
            .max(1e-6);

        for (bundle_id, members) in bundles {
            if members.len() <= 1 {
                validated.insert(*bundle_id, members.clone());
                continue;
            }

            let mut supporters: HashMap<u32, Vec<f64>> = HashMap::new();
            for bid in bids.iter().filter(|bid| {
                members.binary_search(&bid.task_id).is_ok()
                    && bid.score.is_finite()
                    && bid.score >= self.min_bid_threshold
            }) {
                supporters.entry(bid.drone_id).or_default().push(bid.score);
            }

            let mut participant_ids: Vec<NodeId> = supporters.keys().copied().map(NodeId).collect();
            participant_ids.sort_unstable();
            participant_ids.dedup();
            if participant_ids.len() < 2 {
                continue;
            }

            let mean_priority = members
                .iter()
                .filter_map(|task_id| task_lookup.get(task_id).copied())
                .map(|task| ((task.priority + task.urgency) * 0.5).clamp(0.0, 1.0))
                .sum::<f64>()
                / members.len().max(1) as f64;
            let effect = if mean_priority >= 0.75 {
                GroupEffect::AntiDeception
            } else {
                GroupEffect::BundleBid
            };

            coordinator.add_edge(HyperEdge {
                edge_id: *bundle_id as u64,
                members: participant_ids,
                effect,
                quorum_ratio: self.bundle_quorum_ratio,
            });

            for (drone_id, scores) in supporters {
                let coverage = scores.len() as f64 / members.len().max(1) as f64;
                let mean_score = scores.iter().copied().sum::<f64>() / scores.len().max(1) as f64;
                let confidence = (0.55 * coverage
                    + 0.45 * (mean_score / max_score).clamp(0.0, 1.0))
                .clamp(0.0, 1.0);
                coordinator.record_vote(GroupVote {
                    edge_id: *bundle_id as u64,
                    voter: NodeId(drone_id),
                    confidence,
                    timestamp: 0.0,
                });
            }

            if coordinator
                .resolve(*bundle_id as u64)
                .map(|resolution| resolution.confirmed)
                .unwrap_or(false)
            {
                validated.insert(*bundle_id, members.clone());
            }
        }

        validated
    }
    fn solve_assignment_greedy(
        &self,
        tasks: &[Task],
        bids: &[Bid],
        bundles: &HashMap<u32, Vec<u32>>,
    ) -> AuctionResult {
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

        let max_score = bids
            .iter()
            .map(|b| b.score)
            .filter(|s| s.is_finite())
            .fold(0.0_f64, f64::max);
        let big_penalty = (max_score.abs() + 1.0) * 10.0;

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

        let lookup = |di: usize, tid: u32| -> Option<f64> {
            let ti = all_task_ids.binary_search(&tid).ok()?;
            let s = bid_scores[di * n_tasks + ti];
            if s <= -big_penalty {
                None
            } else {
                Some(s)
            }
        };

        let slot_task_ids = build_slot_task_ids(&all_task_ids, bundles);

        let mut candidates: Vec<(f64, usize, usize)> = Vec::new();
        for di in 0..drone_ids.len() {
            for (si, slot) in slot_task_ids.iter().enumerate() {
                if let Some(total_score) = score_slot(slot, &lookup, di, self.min_bid_threshold) {
                    candidates.push((total_score, di, si));
                }
            }
        }

        candidates.sort_by(|a, b| {
            b.0.total_cmp(&a.0)
                .then_with(|| drone_ids[a.1].cmp(&drone_ids[b.1]))
                .then_with(|| a.2.cmp(&b.2))
        });

        let mut assignments = Vec::new();
        let mut assigned_tasks: HashSet<u32> = HashSet::new();
        let mut used_drones: HashSet<u32> = HashSet::new();

        for (_slot_score, di, si) in candidates {
            let did = drone_ids[di];
            if used_drones.contains(&did) {
                continue;
            }

            let slot = &slot_task_ids[si];
            if slot.iter().any(|tid| assigned_tasks.contains(tid)) {
                continue;
            }

            used_drones.insert(did);
            for &tid in slot {
                let score = lookup(di, tid).unwrap_or(0.0);
                assignments.push(Assignment {
                    drone_id: did,
                    task_id: tid,
                    bid_score: score,
                });
                assigned_tasks.insert(tid);
            }
        }

        finalize_result(&all_task_ids, assignments)
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
        let slot_task_ids = build_slot_task_ids(&all_task_ids, bundles);

        let n = drone_ids.len().max(slot_task_ids.len());

        let mut cost_matrix = vec![vec![0.0f64; n]; n];
        for (di, _did) in drone_ids.iter().enumerate() {
            for (si, slot) in slot_task_ids.iter().enumerate() {
                cost_matrix[di][si] = score_slot(slot, &lookup, di, self.min_bid_threshold)
                    .map_or(big_penalty, |total_score| -total_score);
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

        for (di, &si) in assignment_vec.iter().enumerate() {
            if di >= drone_ids.len() || si >= slot_task_ids.len() {
                continue;
            }
            let slot = &slot_task_ids[si];

            if score_slot(slot, &lookup, di, self.min_bid_threshold).is_some() {
                let did = drone_ids[di];
                for &tid in slot {
                    let score = lookup(di, tid).unwrap_or(0.0);
                    assignments.push(Assignment {
                        drone_id: did,
                        task_id: tid,
                        bid_score: score,
                    });
                }
            }
        }

        finalize_result(&all_task_ids, assignments)
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
    for task_ids in map.values_mut() {
        task_ids.sort_unstable();
    }
    map
}

fn build_slot_task_ids(all_task_ids: &[u32], bundles: &HashMap<u32, Vec<u32>>) -> Vec<Vec<u32>> {
    let mut slot_task_ids: Vec<Vec<u32>> = Vec::new();
    let mut already_bundled: HashSet<u32> = HashSet::new();

    let mut bundle_ids: Vec<u32> = bundles.keys().copied().collect();
    bundle_ids.sort_unstable();

    for bundle_id in bundle_ids {
        let member_ids = &bundles[&bundle_id];
        if member_ids.len() > 1 {
            slot_task_ids.push(member_ids.clone());
            for &tid in member_ids {
                already_bundled.insert(tid);
            }
        }
    }

    for &tid in all_task_ids {
        if !already_bundled.contains(&tid) {
            slot_task_ids.push(vec![tid]);
        }
    }

    slot_task_ids
}

fn score_slot<F>(
    slot: &[u32],
    lookup: &F,
    drone_index: usize,
    min_bid_threshold: f64,
) -> Option<f64>
where
    F: Fn(usize, u32) -> Option<f64>,
{
    let mut total_score = 0.0;
    for &task_id in slot {
        match lookup(drone_index, task_id) {
            Some(score) if score.is_finite() && score >= min_bid_threshold => {
                total_score += score;
            }
            _ => return None,
        }
    }
    total_score.is_finite().then_some(total_score)
}

fn finalize_result(all_task_ids: &[u32], mut assignments: Vec<Assignment>) -> AuctionResult {
    assignments.sort_by(|a, b| {
        a.task_id
            .cmp(&b.task_id)
            .then_with(|| a.drone_id.cmp(&b.drone_id))
    });

    let assigned_tasks: HashSet<u32> = assignments
        .iter()
        .map(|assignment| assignment.task_id)
        .collect();
    let unassigned_tasks = all_task_ids
        .iter()
        .filter(|task_id| !assigned_tasks.contains(*task_id))
        .copied()
        .collect();
    let total_welfare = assignments
        .iter()
        .map(|assignment| assignment.bid_score)
        .sum();

    AuctionResult {
        assignments,
        unassigned_tasks,
        total_welfare,
    }
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
    use crate::bidder::{Bid, BidComponents};
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

    fn make_bid(drone_id: u32, task_id: u32, score: f64) -> Bid {
        Bid {
            drone_id,
            task_id,
            score,
            components: BidComponents {
                proximity: 0.0,
                capability: 1.0,
                energy: 1.0,
                risk_exposure: 0.0,
                urgency_bonus: 0.0,
            },
        }
    }

    fn assignment_pairs(result: &AuctionResult) -> Vec<(u32, u32)> {
        result
            .assignments
            .iter()
            .map(|assignment| (assignment.drone_id, assignment.task_id))
            .collect()
    }

    #[test]
    fn test_validate_bundles_requires_multi_bidder_support() {
        let tasks = vec![
            Task {
                id: 10,
                location: Position::new(5.0, 5.0, 50.0),
                required_capabilities: Capabilities::default(),
                priority: 0.8,
                urgency: 0.8,
                bundle_id: Some(1),
                dark_pool: None,
            },
            Task {
                id: 11,
                location: Position::new(8.0, 8.0, 50.0),
                required_capabilities: Capabilities::default(),
                priority: 0.8,
                urgency: 0.8,
                bundle_id: Some(1),
                dark_pool: None,
            },
        ];
        let bids = vec![make_bid(1, 10, 5.0), make_bid(1, 11, 4.5)];
        let auctioneer = Auctioneer::new();
        let bundles = group_bundles(&tasks);

        assert!(auctioneer
            .validate_bundles(&tasks, &bids, &bundles)
            .is_empty());
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
    fn test_collect_bids_respects_bid_cap() {
        let drone = make_drone(1, 0.0, 0.0);
        let tasks = vec![
            make_task(10, 5.0, 5.0),
            make_task(20, 10.0, 10.0),
            make_task(30, 20.0, 20.0),
            make_task(40, 30.0, 30.0),
        ];

        let auctioneer = Auctioneer::new().with_bid_cap(2);
        let bids = auctioneer.collect_bids(&[drone], &tasks, &[], &HashMap::new(), &[]);
        assert_eq!(
            bids.len(),
            2,
            "bid cap should retain only the strongest bids"
        );
    }

    #[test]
    fn test_collect_bids_keeps_bundle_mates_when_capped() {
        let drone = make_drone(1, 0.0, 0.0);
        let tasks = vec![
            Task {
                id: 10,
                location: Position::new(5.0, 5.0, 50.0),
                required_capabilities: Capabilities::default(),
                priority: 0.5,
                urgency: 0.5,
                bundle_id: Some(7),
                dark_pool: None,
            },
            Task {
                id: 11,
                location: Position::new(6.0, 6.0, 50.0),
                required_capabilities: Capabilities::default(),
                priority: 0.5,
                urgency: 0.5,
                bundle_id: Some(7),
                dark_pool: None,
            },
            make_task(20, 100.0, 100.0),
        ];

        let auctioneer = Auctioneer::new().with_bid_cap(1);
        let bids = auctioneer.collect_bids(&[drone], &tasks, &[], &HashMap::new(), &[]);
        let task_ids: HashSet<u32> = bids.iter().map(|b| b.task_id).collect();

        assert_eq!(bids.len(), 2, "bundle mates should survive bid compression");
        assert!(task_ids.contains(&10));
        assert!(task_ids.contains(&11));
    }

    #[test]
    fn test_run_auction_breaks_equal_score_ties_by_task_id_under_bid_cap() {
        let drones = vec![make_drone(1, 0.0, 0.0)];
        let task_10 = make_task(10, 10.0, 0.0);
        let task_20 = make_task(20, 0.0, 10.0);

        let mut auctioneer_a = Auctioneer::new().with_bid_cap(1);
        let result_a = auctioneer_a.run_auction(
            &drones,
            &[task_20.clone(), task_10.clone()],
            &[],
            &HashMap::new(),
            &[],
        );

        let mut auctioneer_b = Auctioneer::new().with_bid_cap(1);
        let result_b =
            auctioneer_b.run_auction(&drones, &[task_10, task_20], &[], &HashMap::new(), &[]);

        assert_eq!(assignment_pairs(&result_a), vec![(1, 10)]);
        assert_eq!(assignment_pairs(&result_b), vec![(1, 10)]);
        assert!((result_a.total_welfare - result_b.total_welfare).abs() < 1e-12);
    }

    #[test]
    fn test_build_slot_task_ids_sorts_bundle_members_deterministically() {
        let tasks = vec![
            Task {
                id: 11,
                location: Position::new(6.0, 6.0, 50.0),
                required_capabilities: Capabilities::default(),
                priority: 0.5,
                urgency: 0.5,
                bundle_id: Some(7),
                dark_pool: None,
            },
            Task {
                id: 10,
                location: Position::new(5.0, 5.0, 50.0),
                required_capabilities: Capabilities::default(),
                priority: 0.5,
                urgency: 0.5,
                bundle_id: Some(7),
                dark_pool: None,
            },
            make_task(30, 100.0, 100.0),
        ];

        let bundles = group_bundles(&tasks);
        let slot_task_ids = build_slot_task_ids(&[10, 11, 30], &bundles);

        assert_eq!(slot_task_ids, vec![vec![10, 11], vec![30]]);
    }

    #[test]
    fn test_optimal_solver_requires_full_bundle_to_clear_threshold() {
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
        let bids = vec![make_bid(1, 10, 5.0), make_bid(1, 11, 0.5)];

        let auctioneer = Auctioneer::new().with_min_bid(1.0);
        let result = auctioneer.solve_assignment(&tasks, &bids, &group_bundles(&tasks));

        assert!(result.assignments.is_empty());
        assert_eq!(result.unassigned_tasks, vec![10, 11]);
        assert_eq!(result.total_welfare, 0.0);
    }

    #[test]
    fn test_run_auction_is_stable_under_adversarial_task_permutations() {
        let drones = vec![make_drone(1, 0.0, 0.0)];
        let permutations = vec![
            vec![
                make_task(30, 40.0, 0.0),
                make_task(20, 0.0, 10.0),
                make_task(10, 10.0, 0.0),
            ],
            vec![
                make_task(10, 10.0, 0.0),
                make_task(30, 40.0, 0.0),
                make_task(20, 0.0, 10.0),
            ],
            vec![
                make_task(20, 0.0, 10.0),
                make_task(10, 10.0, 0.0),
                make_task(30, 40.0, 0.0),
            ],
            vec![
                make_task(30, 40.0, 0.0),
                make_task(10, 10.0, 0.0),
                make_task(20, 0.0, 10.0),
            ],
        ];

        let mut baseline_auctioneer = Auctioneer::new().with_bid_cap(1);
        let baseline =
            baseline_auctioneer.run_auction(&drones, &permutations[0], &[], &HashMap::new(), &[]);
        let baseline_pairs = assignment_pairs(&baseline);
        let baseline_unassigned = baseline.unassigned_tasks.clone();
        let baseline_total_welfare = baseline.total_welfare;

        for tasks in permutations.into_iter().skip(1) {
            let mut auctioneer = Auctioneer::new().with_bid_cap(1);
            let result = auctioneer.run_auction(&drones, &tasks, &[], &HashMap::new(), &[]);

            assert_eq!(assignment_pairs(&result), baseline_pairs);
            assert_eq!(result.unassigned_tasks, baseline_unassigned);
            assert!((result.total_welfare - baseline_total_welfare).abs() < 1e-12);
        }
    }

    #[test]
    fn test_greedy_solver_assigns_best_slots() {
        let drones = vec![make_drone(1, 0.0, 0.0), make_drone(2, 100.0, 100.0)];
        let tasks = vec![make_task(10, 5.0, 5.0), make_task(20, 95.0, 95.0)];

        let auctioneer = Auctioneer::new();
        let bids = auctioneer.collect_bids(&drones, &tasks, &[], &HashMap::new(), &[]);
        let bundles = group_bundles(&tasks);
        let result = auctioneer.solve_assignment_greedy(&tasks, &bids, &bundles);

        let a1 = result.assignments.iter().find(|a| a.task_id == 10).unwrap();
        let a2 = result.assignments.iter().find(|a| a.task_id == 20).unwrap();
        assert_eq!(a1.drone_id, 1);
        assert_eq!(a2.drone_id, 2);
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
