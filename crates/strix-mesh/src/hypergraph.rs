use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::NodeId;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GroupEffect {
    ThreatConfirm,
    BundleBid,
    Rally,
    AntiDeception,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperEdge {
    pub edge_id: u64,
    pub members: Vec<NodeId>,
    pub effect: GroupEffect,
    pub quorum_ratio: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupVote {
    pub edge_id: u64,
    pub voter: NodeId,
    pub confidence: f64,
    pub timestamp: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupResolution {
    pub edge_id: u64,
    pub confirmed: bool,
    pub support: usize,
    pub required: usize,
    pub mean_confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HypergraphCoordinator {
    pub anti_deception_delta: f64,
    edges: HashMap<u64, HyperEdge>,
    votes: HashMap<u64, Vec<GroupVote>>,
}

impl Default for HypergraphCoordinator {
    fn default() -> Self {
        Self {
            anti_deception_delta: 0.35,
            edges: HashMap::new(),
            votes: HashMap::new(),
        }
    }
}

impl HypergraphCoordinator {
    pub fn add_edge(&mut self, edge: HyperEdge) {
        self.edges.insert(edge.edge_id, edge);
    }

    pub fn record_vote(&mut self, vote: GroupVote) {
        self.votes.entry(vote.edge_id).or_default().push(vote);
    }

    pub fn resolve(&self, edge_id: u64) -> Option<GroupResolution> {
        let edge = self.edges.get(&edge_id)?;
        let votes = self.votes.get(&edge_id)?;
        let member_set: HashSet<NodeId> = edge.members.iter().copied().collect();
        let mut unique = HashMap::<NodeId, f64>::new();
        for vote in votes {
            if member_set.contains(&vote.voter) {
                unique.insert(vote.voter, vote.confidence.clamp(0.0, 1.0));
            }
        }
        let support = unique.len();
        let required = required_support(edge.members.len(), edge.quorum_ratio);
        let mean_confidence = if support == 0 {
            0.0
        } else {
            unique.values().sum::<f64>() / support as f64
        };
        let confidence_spread = if support <= 1 {
            0.0
        } else {
            let min = unique.values().fold(1.0_f64, |acc, value| acc.min(*value));
            let max = unique.values().fold(0.0_f64, |acc, value| acc.max(*value));
            max - min
        };

        let confirmed = support >= required
            && mean_confidence >= 0.55
            && (edge.effect != GroupEffect::AntiDeception
                || confidence_spread <= self.anti_deception_delta);

        Some(GroupResolution {
            edge_id,
            confirmed,
            support,
            required,
            mean_confidence,
        })
    }
}

fn required_support(member_count: usize, quorum_ratio: f64) -> usize {
    let ratio = quorum_ratio.clamp(0.34, 1.0);
    let required = (member_count as f64 * ratio).ceil() as usize;
    required.max(1).min(member_count.max(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn threat_confirmation_requires_quorum() {
        let mut coordinator = HypergraphCoordinator::default();
        coordinator.add_edge(HyperEdge {
            edge_id: 7,
            members: vec![NodeId(1), NodeId(2), NodeId(3)],
            effect: GroupEffect::ThreatConfirm,
            quorum_ratio: 0.66,
        });
        coordinator.record_vote(GroupVote {
            edge_id: 7,
            voter: NodeId(1),
            confidence: 0.8,
            timestamp: 1.0,
        });
        assert!(!coordinator.resolve(7).unwrap().confirmed);
        coordinator.record_vote(GroupVote {
            edge_id: 7,
            voter: NodeId(2),
            confidence: 0.75,
            timestamp: 1.2,
        });
        assert!(coordinator.resolve(7).unwrap().confirmed);
    }

    #[test]
    fn anti_deception_rejects_divergent_votes() {
        let mut coordinator = HypergraphCoordinator::default();
        coordinator.add_edge(HyperEdge {
            edge_id: 9,
            members: vec![NodeId(1), NodeId(2), NodeId(3)],
            effect: GroupEffect::AntiDeception,
            quorum_ratio: 0.66,
        });
        coordinator.record_vote(GroupVote {
            edge_id: 9,
            voter: NodeId(1),
            confidence: 0.95,
            timestamp: 1.0,
        });
        coordinator.record_vote(GroupVote {
            edge_id: 9,
            voter: NodeId(2),
            confidence: 0.20,
            timestamp: 1.1,
        });
        assert!(!coordinator.resolve(9).unwrap().confirmed);
    }
}
