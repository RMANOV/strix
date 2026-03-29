//! # Fractal Hierarchy
//!
//! Self-similar command structure at every scale:
//!
//! ```text
//! Pair(2) → Squad(5-8) → Platoon(20-30) → Company(100+)
//! ```
//!
//! The same particle filter + auction algorithms run at **each** level,
//! operating on different spatial/temporal granularity. Leaders carry the
//! particle filter state for their level; on leader loss the state is
//! distributed among remaining members so a new leader can reconstruct it.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::NodeId;

fn effective_rank(rank: f64) -> f64 {
    if rank.is_finite() {
        rank
    } else {
        f64::NEG_INFINITY
    }
}

// ---------------------------------------------------------------------------
// Hierarchy level
// ---------------------------------------------------------------------------

/// Scale tier within the fractal hierarchy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub enum HierarchyLevel {
    /// Two drones — the atomic unit.
    Pair,
    /// 5–8 drones (2–4 pairs).
    Squad,
    /// 20–30 drones (4–6 squads).
    Platoon,
    /// 100+ drones (4+ platoons).
    Company,
}

impl HierarchyLevel {
    /// Nominal group sizes (min, max) for each level.
    pub fn size_range(self) -> (usize, usize) {
        match self {
            HierarchyLevel::Pair => (2, 2),
            HierarchyLevel::Squad => (5, 8),
            HierarchyLevel::Platoon => (20, 30),
            HierarchyLevel::Company => (100, usize::MAX),
        }
    }

    /// The level one tier up, or `None` at Company.
    pub fn parent_level(self) -> Option<HierarchyLevel> {
        match self {
            HierarchyLevel::Pair => Some(HierarchyLevel::Squad),
            HierarchyLevel::Squad => Some(HierarchyLevel::Platoon),
            HierarchyLevel::Platoon => Some(HierarchyLevel::Company),
            HierarchyLevel::Company => None,
        }
    }

    /// The level one tier down, or `None` at Pair.
    pub fn child_level(self) -> Option<HierarchyLevel> {
        match self {
            HierarchyLevel::Company => Some(HierarchyLevel::Platoon),
            HierarchyLevel::Platoon => Some(HierarchyLevel::Squad),
            HierarchyLevel::Squad => Some(HierarchyLevel::Pair),
            HierarchyLevel::Pair => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Distributed particle filter state (stub)
// ---------------------------------------------------------------------------

/// Compact representation of particle filter state carried by a leader.
/// In the full system this holds weighted particles; here we store an
/// opaque serializable blob so the fractal module can distribute / merge it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParticleFilterState {
    /// Serialized particles (opaque to this crate).
    pub data: Vec<u8>,
    /// Number of particles.
    pub particle_count: usize,
    /// Timestamp of last filter update.
    pub last_update: f64,
}

impl ParticleFilterState {
    /// Create an empty (initial) state.
    pub fn empty() -> Self {
        Self {
            data: Vec::new(),
            particle_count: 0,
            last_update: 0.0,
        }
    }

    /// Split the state into `n` roughly-equal shards for distribution.
    pub fn distribute(&self, n: usize) -> Vec<ParticleFilterShard> {
        if n == 0 || self.data.is_empty() {
            return vec![ParticleFilterShard::empty(); n.max(1)];
        }
        let chunk_size = self.data.len().div_ceil(n);
        self.data
            .chunks(chunk_size)
            .enumerate()
            .map(|(idx, chunk)| ParticleFilterShard {
                shard_index: idx,
                total_shards: n,
                data: chunk.to_vec(),
                particle_count_hint: self.particle_count / n,
            })
            .collect()
    }

    /// Reconstruct state from collected shards.
    pub fn reconstruct(shards: &mut [ParticleFilterShard], timestamp: f64) -> Self {
        shards.sort_by_key(|s| s.shard_index);
        let data: Vec<u8> = shards.iter().flat_map(|s| s.data.iter().copied()).collect();
        let particle_count: usize = shards.iter().map(|s| s.particle_count_hint).sum();
        Self {
            data,
            particle_count,
            last_update: timestamp,
        }
    }
}

/// One shard of a distributed particle filter state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParticleFilterShard {
    /// Index of this shard.
    pub shard_index: usize,
    /// Total shards the state was split into.
    pub total_shards: usize,
    /// Raw particle data for this shard.
    pub data: Vec<u8>,
    /// Approximate number of particles in this shard.
    pub particle_count_hint: usize,
}

impl ParticleFilterShard {
    /// Create an empty shard.
    pub fn empty() -> Self {
        Self {
            shard_index: 0,
            total_shards: 1,
            data: Vec::new(),
            particle_count_hint: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Fractal node
// ---------------------------------------------------------------------------

/// A node within the fractal hierarchy tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FractalNode {
    /// Identity of this drone.
    pub node_id: NodeId,
    /// Current hierarchy level of the *group* this node belongs to.
    pub level: HierarchyLevel,
    /// Parent node id (leader of the parent group), if any.
    pub parent: Option<NodeId>,
    /// Direct children managed by this node (only populated for leaders).
    pub children: Vec<NodeId>,
    /// Whether this node is the leader of its group.
    pub is_leader: bool,
    /// Rank used for leader election (higher = more capable).
    pub rank: f64,
    /// Particle filter state — only meaningful for leaders.
    pub pf_state: Option<ParticleFilterState>,
    /// Distributed shard held when *not* a leader (for recovery).
    pub pf_shard: Option<ParticleFilterShard>,
}

impl FractalNode {
    /// Create a leaf node with default rank.
    pub fn new(node_id: NodeId) -> Self {
        Self {
            node_id,
            level: HierarchyLevel::Pair,
            parent: None,
            children: Vec::new(),
            is_leader: false,
            rank: 0.0,
            pf_state: None,
            pf_shard: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Hierarchy
// ---------------------------------------------------------------------------

/// The full fractal hierarchy for a swarm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FractalHierarchy {
    /// All nodes indexed by their id.
    pub nodes: HashMap<NodeId, FractalNode>,
    /// Groups at each level: level → list of (leader, members).
    pub groups: HashMap<HierarchyLevel, Vec<Group>>,
}

/// A group at one level of the hierarchy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Group {
    /// Leader of this group.
    pub leader: NodeId,
    /// All members *including* the leader.
    pub members: Vec<NodeId>,
    /// The hierarchy level of this group.
    pub level: HierarchyLevel,
}

/// Build a fractal hierarchy for `n` drones.
///
/// Algorithm:
/// 1. Create pairs from the flat list.
/// 2. Group pairs into squads (3–4 pairs each).
/// 3. Group squads into platoons (4–6 squads each).
/// 4. If enough platoons exist, group into a company.
///
/// The first member of each group is designated leader.
/// Ranks are assigned sequentially (higher id ⇒ higher initial rank for
/// determinism; real ranks come from capability assessment).
pub fn build_hierarchy(drone_ids: &[NodeId]) -> FractalHierarchy {
    let n = drone_ids.len();
    let mut nodes: HashMap<NodeId, FractalNode> = HashMap::new();

    // Initialise all nodes.
    for (i, &id) in drone_ids.iter().enumerate() {
        let mut node = FractalNode::new(id);
        // Simple deterministic rank: index-based.
        node.rank = i as f64;
        nodes.insert(id, node);
    }

    let mut groups: HashMap<HierarchyLevel, Vec<Group>> = HashMap::new();

    if n < 2 {
        // Single drone — no hierarchy, it is its own leader.
        if let Some((&id, node)) = nodes.iter_mut().next() {
            node.is_leader = true;
            node.level = HierarchyLevel::Pair;
            let group = Group {
                leader: id,
                members: vec![id],
                level: HierarchyLevel::Pair,
            };
            groups.entry(HierarchyLevel::Pair).or_default().push(group);
        }
        return FractalHierarchy { nodes, groups };
    }

    // --- Level 1: Pairs ---
    let pair_chunks = build_groups_from_ids(drone_ids, 2, 2);
    let mut pair_leaders: Vec<NodeId> = Vec::new();

    for chunk in &pair_chunks {
        let leader_id = *chunk
            .first()
            .expect("build_groups_from_ids guarantees non-empty chunks");
        pair_leaders.push(leader_id);

        for &id in chunk {
            if let Some(node) = nodes.get_mut(&id) {
                node.level = HierarchyLevel::Pair;
                node.parent = if id == leader_id {
                    None
                } else {
                    Some(leader_id)
                };
                node.is_leader = id == leader_id;
            }
        }

        // Leader knows its children.
        if let Some(leader_node) = nodes.get_mut(&leader_id) {
            leader_node.children = chunk.iter().copied().filter(|&x| x != leader_id).collect();
            leader_node.pf_state = Some(ParticleFilterState::empty());
        }

        groups.entry(HierarchyLevel::Pair).or_default().push(Group {
            leader: leader_id,
            members: chunk.clone(),
            level: HierarchyLevel::Pair,
        });
    }

    // --- Level 2: Squads (3–4 pair leaders per squad) ---
    if pair_leaders.len() >= 2 {
        let squad_chunks = build_groups_from_ids(&pair_leaders, 3, 4);
        let mut squad_leaders: Vec<NodeId> = Vec::new();

        for chunk in &squad_chunks {
            let leader_id = *chunk
                .first()
                .expect("build_groups_from_ids guarantees non-empty chunks");
            squad_leaders.push(leader_id);

            for &id in chunk {
                if let Some(node) = nodes.get_mut(&id) {
                    node.parent = Some(leader_id);
                }
            }
            if let Some(leader_node) = nodes.get_mut(&leader_id) {
                leader_node.level = HierarchyLevel::Squad;
                leader_node.children = chunk.iter().copied().filter(|&x| x != leader_id).collect();
                leader_node.is_leader = true;
                leader_node.pf_state = Some(ParticleFilterState::empty());
            }

            // Collect all drones in the squad (expand pair leaders → their pairs).
            let all_members: Vec<NodeId> = chunk
                .iter()
                .flat_map(|&pl| {
                    let mut members = vec![pl];
                    if let Some(n) = nodes.get(&pl) {
                        // Only extend with Pair-level children.
                        for &child in &n.children {
                            if nodes
                                .get(&child)
                                .is_some_and(|c| c.level == HierarchyLevel::Pair)
                            {
                                members.push(child);
                            }
                        }
                    }
                    members
                })
                .collect();

            groups
                .entry(HierarchyLevel::Squad)
                .or_default()
                .push(Group {
                    leader: leader_id,
                    members: all_members,
                    level: HierarchyLevel::Squad,
                });
        }

        // --- Level 3: Platoons (4–6 squad leaders per platoon) ---
        if squad_leaders.len() >= 2 {
            let platoon_chunks = build_groups_from_ids(&squad_leaders, 4, 6);
            let mut platoon_leaders: Vec<NodeId> = Vec::new();

            for chunk in &platoon_chunks {
                let leader_id = *chunk
                    .first()
                    .expect("build_groups_from_ids guarantees non-empty chunks");
                platoon_leaders.push(leader_id);

                for &id in chunk {
                    if let Some(node) = nodes.get_mut(&id) {
                        node.parent = Some(leader_id);
                    }
                }
                if let Some(leader_node) = nodes.get_mut(&leader_id) {
                    leader_node.level = HierarchyLevel::Platoon;
                    leader_node.children =
                        chunk.iter().copied().filter(|&x| x != leader_id).collect();
                    leader_node.is_leader = true;
                    leader_node.pf_state = Some(ParticleFilterState::empty());
                }

                groups
                    .entry(HierarchyLevel::Platoon)
                    .or_default()
                    .push(Group {
                        leader: leader_id,
                        members: chunk.clone(),
                        level: HierarchyLevel::Platoon,
                    });
            }

            // --- Level 4: Company (if ≥ 2 platoons) ---
            if platoon_leaders.len() >= 2 {
                let leader_id = *platoon_leaders
                    .first()
                    .expect("platoon_leaders non-empty: len >= 2 guard above");
                for &id in &platoon_leaders {
                    if let Some(node) = nodes.get_mut(&id) {
                        node.parent = Some(leader_id);
                    }
                }
                if let Some(leader_node) = nodes.get_mut(&leader_id) {
                    leader_node.level = HierarchyLevel::Company;
                    leader_node.children = platoon_leaders
                        .iter()
                        .copied()
                        .filter(|&x| x != leader_id)
                        .collect();
                    leader_node.is_leader = true;
                    leader_node.pf_state = Some(ParticleFilterState::empty());
                }

                groups
                    .entry(HierarchyLevel::Company)
                    .or_default()
                    .push(Group {
                        leader: leader_id,
                        members: platoon_leaders,
                        level: HierarchyLevel::Company,
                    });
            }
        }
    }

    FractalHierarchy { nodes, groups }
}

/// Split `ids` into groups of `[min_size, max_size]`, distributing remainder.
fn build_groups_from_ids(ids: &[NodeId], min_size: usize, max_size: usize) -> Vec<Vec<NodeId>> {
    if ids.is_empty() {
        return Vec::new();
    }
    let target = min_size.max(1);
    let n = ids.len();
    if n <= max_size {
        return vec![ids.to_vec()];
    }
    let num_groups = n.div_ceil(max_size);
    let base_size = n / num_groups;
    let remainder = n % num_groups;

    let mut result = Vec::new();
    let mut offset = 0;
    for i in 0..num_groups {
        let size = base_size + if i < remainder { 1 } else { 0 };
        let size = size.max(target);
        let end = (offset + size).min(n);
        result.push(ids[offset..end].to_vec());
        offset = end;
    }
    // Remaining items go into last group.
    if offset < n {
        if let Some(last) = result.last_mut() {
            last.extend_from_slice(&ids[offset..n]);
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Leader promotion
// ---------------------------------------------------------------------------

/// Promote a new leader when the current one is lost.
///
/// 1. Distribute the old leader's particle filter state to surviving members.
/// 2. Select the highest-ranked surviving member as new leader.
/// 3. New leader reconstructs PF state from distributed shards.
///
/// Returns the new leader's `NodeId`, or `None` if the group is empty.
pub fn promote_leader(
    hierarchy: &mut FractalHierarchy,
    group_level: HierarchyLevel,
    dead_leader: NodeId,
) -> Option<NodeId> {
    // Find the group.
    let groups = hierarchy.groups.get_mut(&group_level)?;
    let group_idx = groups.iter().position(|g| g.leader == dead_leader)?;
    let group = &mut groups[group_idx];

    // Remove dead leader from members.
    group.members.retain(|&id| id != dead_leader);
    if group.members.is_empty() {
        return None;
    }

    // Distribute PF state among survivors.
    let pf_state = hierarchy
        .nodes
        .get(&dead_leader)
        .and_then(|n| n.pf_state.clone())
        .unwrap_or_else(ParticleFilterState::empty);
    let shards = pf_state.distribute(group.members.len());

    for (member_id, shard) in group.members.iter().zip(shards.into_iter()) {
        if let Some(node) = hierarchy.nodes.get_mut(member_id) {
            node.pf_shard = Some(shard);
        }
    }

    // Elect highest-ranked member.
    let new_leader = *group.members.iter().max_by(|a, b| {
        let ra = hierarchy
            .nodes
            .get(a)
            .map_or(f64::NEG_INFINITY, |n| effective_rank(n.rank));
        let rb = hierarchy
            .nodes
            .get(b)
            .map_or(f64::NEG_INFINITY, |n| effective_rank(n.rank));
        ra.total_cmp(&rb).then_with(|| a.0.cmp(&b.0))
    })?;

    // Collect shards from all members to reconstruct PF state.
    let mut collected_shards: Vec<ParticleFilterShard> = group
        .members
        .iter()
        .filter_map(|id| hierarchy.nodes.get(id).and_then(|n| n.pf_shard.clone()))
        .collect();
    let reconstructed =
        ParticleFilterState::reconstruct(&mut collected_shards, pf_state.last_update);

    // Update new leader.
    if let Some(leader_node) = hierarchy.nodes.get_mut(&new_leader) {
        leader_node.is_leader = true;
        leader_node.level = group_level;
        leader_node.pf_state = Some(reconstructed);
        leader_node.pf_shard = None;
        leader_node.children = group
            .members
            .iter()
            .copied()
            .filter(|&id| id != new_leader)
            .collect();
    }

    // Update group.
    group.leader = new_leader;

    // Update children to point to new parent.
    let members_snapshot: Vec<NodeId> = group.members.clone();
    for &mid in &members_snapshot {
        if mid != new_leader {
            if let Some(node) = hierarchy.nodes.get_mut(&mid) {
                node.parent = Some(new_leader);
                node.is_leader = false;
            }
        }
    }

    // Remove dead node.
    hierarchy.nodes.remove(&dead_leader);

    Some(new_leader)
}

// ---------------------------------------------------------------------------
// Split / merge
// ---------------------------------------------------------------------------

/// Dynamically resize groups based on mission needs.
///
/// - If a group exceeds `max_size` for its level → split into two groups.
/// - If two groups at the same level are both below `min_size` → merge them.
///
/// Returns the list of affected group leaders after the operation.
pub fn split_merge(hierarchy: &mut FractalHierarchy, level: HierarchyLevel) -> Vec<NodeId> {
    let (min_size, max_size) = level.size_range();
    let mut affected_leaders = Vec::new();

    let groups = match hierarchy.groups.get_mut(&level) {
        Some(g) => g,
        None => return affected_leaders,
    };

    // --- Split oversized groups (iterate until all fit) ---
    if max_size < usize::MAX {
        let mut changed = true;
        while changed {
            changed = false;
            let mut new_groups: Vec<Group> = Vec::new();
            let mut to_remove: Vec<usize> = Vec::new();

            for (i, group) in groups.iter().enumerate() {
                if group.members.len() > max_size {
                    let mid = group.members.len() / 2;
                    let left = group.members[..mid].to_vec();
                    let right = group.members[mid..].to_vec();

                    // mid >= 1 since group.members.len() > max_size >= 1
                    let left_leader = *left.first().expect("left half non-empty: mid >= 1");
                    let right_leader = *right.first().expect("right half non-empty: len > mid");

                    new_groups.push(Group {
                        leader: left_leader,
                        members: left,
                        level,
                    });
                    new_groups.push(Group {
                        leader: right_leader,
                        members: right,
                        level,
                    });

                    affected_leaders.push(left_leader);
                    affected_leaders.push(right_leader);
                    to_remove.push(i);
                    changed = true;
                }
            }

            // Remove split groups (in reverse to preserve indices).
            for &i in to_remove.iter().rev() {
                groups.remove(i);
            }
            groups.extend(new_groups);
        }
    }

    // --- Merge undersized groups ---
    // Pair up consecutive undersized groups.
    let mut merged_indices: Vec<usize> = Vec::new();
    let mut merge_pairs: Vec<(usize, usize)> = Vec::new();

    let group_len = groups.len();
    let mut i = 0;
    while i + 1 < group_len {
        if groups[i].members.len() < min_size && groups[i + 1].members.len() < min_size {
            merge_pairs.push((i, i + 1));
            merged_indices.push(i);
            merged_indices.push(i + 1);
            i += 2;
        } else {
            i += 1;
        }
    }

    let mut merged_groups: Vec<Group> = Vec::new();
    for (a, b) in &merge_pairs {
        let mut members = groups[*a].members.clone();
        members.extend_from_slice(&groups[*b].members);
        // members = groups[a].members + groups[b].members, both non-empty.
        let leader = *members
            .first()
            .expect("merged members non-empty: both source groups non-empty");
        merged_groups.push(Group {
            leader,
            members,
            level,
        });
        affected_leaders.push(leader);
    }

    // Remove merged groups in reverse order.
    merged_indices.sort_unstable();
    for &idx in merged_indices.iter().rev() {
        if idx < groups.len() {
            groups.remove(idx);
        }
    }
    groups.extend(merged_groups);

    // Update node metadata for affected leaders.
    for group in groups.iter() {
        let leader_id = group.leader;
        if let Some(leader_node) = hierarchy.nodes.get_mut(&leader_id) {
            leader_node.is_leader = true;
            leader_node.level = level;
            leader_node.children = group
                .members
                .iter()
                .copied()
                .filter(|&id| id != leader_id)
                .collect();
        }
        for &mid in &group.members {
            if mid != leader_id {
                if let Some(node) = hierarchy.nodes.get_mut(&mid) {
                    node.parent = Some(leader_id);
                }
            }
        }
    }

    affected_leaders
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ids(n: u32) -> Vec<NodeId> {
        (0..n).map(NodeId).collect()
    }

    #[test]
    fn build_hierarchy_small() {
        let ids = make_ids(4);
        let h = build_hierarchy(&ids);
        assert_eq!(h.nodes.len(), 4);
        // Should have pairs.
        assert!(h.groups.contains_key(&HierarchyLevel::Pair));
    }

    #[test]
    fn build_hierarchy_squad_size() {
        let ids = make_ids(10);
        let h = build_hierarchy(&ids);
        // 10 drones → 5 pairs → should form squads.
        assert!(h.groups.contains_key(&HierarchyLevel::Pair));
        assert!(
            h.groups.contains_key(&HierarchyLevel::Squad),
            "10 drones should produce squads"
        );
    }

    #[test]
    fn build_hierarchy_single_drone() {
        let ids = make_ids(1);
        let h = build_hierarchy(&ids);
        assert_eq!(h.nodes.len(), 1);
        let node = h.nodes.get(&NodeId(0)).unwrap();
        assert!(node.is_leader);
    }

    #[test]
    fn promote_leader_works() {
        let ids = make_ids(6);
        let mut h = build_hierarchy(&ids);

        // Find a pair group.
        let pair_groups = h.groups.get(&HierarchyLevel::Pair).unwrap();
        let first_group = pair_groups.first().unwrap();
        let old_leader = first_group.leader;

        let new = promote_leader(&mut h, HierarchyLevel::Pair, old_leader);
        assert!(new.is_some());
        let new_leader = new.unwrap();
        assert_ne!(new_leader, old_leader);
        assert!(!h.nodes.contains_key(&old_leader));
        assert!(h.nodes.get(&new_leader).unwrap().is_leader);
    }

    #[test]
    fn promote_leader_ignores_non_finite_ranks() {
        let ids = make_ids(3);
        let mut h = build_hierarchy(&ids);
        h.groups.insert(
            HierarchyLevel::Pair,
            vec![Group {
                leader: NodeId(0),
                members: ids.clone(),
                level: HierarchyLevel::Pair,
            }],
        );
        h.nodes.get_mut(&NodeId(1)).unwrap().rank = f64::NAN;
        h.nodes.get_mut(&NodeId(2)).unwrap().rank = 1.0;

        let new_leader = promote_leader(&mut h, HierarchyLevel::Pair, NodeId(0)).unwrap();
        assert_eq!(new_leader, NodeId(2));
    }

    #[test]
    fn pf_state_distribute_reconstruct_roundtrip() {
        let state = ParticleFilterState {
            data: vec![1, 2, 3, 4, 5, 6, 7, 8],
            particle_count: 100,
            last_update: 42.0,
        };
        let shards = state.distribute(3);
        assert_eq!(shards.len(), 3);

        let mut collected = shards;
        let reconstructed = ParticleFilterState::reconstruct(&mut collected, 42.0);
        assert_eq!(reconstructed.data, vec![1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn split_merge_oversized_group() {
        let ids = make_ids(6);
        let mut h = build_hierarchy(&ids);
        // Manually create an oversized pair group (pairs are max 2).
        h.groups.insert(
            HierarchyLevel::Pair,
            vec![Group {
                leader: NodeId(0),
                members: ids.clone(),
                level: HierarchyLevel::Pair,
            }],
        );

        let affected = split_merge(&mut h, HierarchyLevel::Pair);
        assert!(!affected.is_empty(), "split should produce new leaders");
        // All resulting groups should be ≤ 2.
        for g in h.groups.get(&HierarchyLevel::Pair).unwrap() {
            assert!(g.members.len() <= 2);
        }
    }

    #[test]
    fn hierarchy_level_parent_child() {
        assert_eq!(
            HierarchyLevel::Pair.parent_level(),
            Some(HierarchyLevel::Squad)
        );
        assert_eq!(HierarchyLevel::Company.parent_level(), None);
        assert_eq!(
            HierarchyLevel::Company.child_level(),
            Some(HierarchyLevel::Platoon)
        );
        assert_eq!(HierarchyLevel::Pair.child_level(), None);
    }
}
