use std::collections::HashMap;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    pub id: u32,
    pub role: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdge {
    pub src: u32,
    pub dst: u32,
    pub weight: f64,
    pub latency: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatCoupling {
    pub src: u32,
    pub dst: u32,
    pub pressure: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GraphSnapshot {
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<GraphEdge>,
    pub threat_couplings: Vec<ThreatCoupling>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct GlobalEncoding {
    pub mean_degree: f64,
    pub master_signal: f64,
    pub bottleneck_ratio: f64,
    pub oversquashing_risk: f64,
    pub coupling_pressure: f64,
    pub mean_latency: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphSurrogate {
    pub global_gain: f64,
    pub bottleneck_penalty: f64,
    pub coupling_penalty: f64,
}

impl Default for GraphSurrogate {
    fn default() -> Self {
        Self {
            global_gain: 0.25,
            bottleneck_penalty: 0.35,
            coupling_penalty: 0.30,
        }
    }
}

impl GraphSnapshot {
    pub fn global_encoding(&self) -> GlobalEncoding {
        if self.nodes.is_empty() {
            return GlobalEncoding {
                mean_degree: 0.0,
                master_signal: 0.0,
                bottleneck_ratio: 0.0,
                oversquashing_risk: 0.0,
                coupling_pressure: 0.0,
                mean_latency: 0.0,
            };
        }

        let mut degree: HashMap<u32, usize> = self.nodes.iter().map(|node| (node.id, 0)).collect();
        let mut mean_latency = 0.0;
        for edge in &self.edges {
            *degree.entry(edge.src).or_insert(0) += 1;
            *degree.entry(edge.dst).or_insert(0) += 1;
            mean_latency += sanitize(edge.latency);
        }
        mean_latency /= self.edges.len().max(1) as f64;

        let mean_degree = degree.values().copied().sum::<usize>() as f64 / self.nodes.len() as f64;
        let max_degree = degree.values().copied().max().unwrap_or(0) as f64;
        let master_signal = if self.nodes.len() <= 1 {
            0.0
        } else {
            (max_degree / (2.0 * (self.nodes.len() - 1) as f64)).clamp(0.0, 1.0)
        };
        let bottleneck_edges = self
            .edges
            .iter()
            .filter(|edge| sanitize(edge.latency) > mean_latency.max(1e-6) * 1.5)
            .count();
        let bottleneck_ratio = bottleneck_edges as f64 / self.edges.len().max(1) as f64;
        let oversquashing_risk = if mean_degree <= 1e-6 {
            0.0
        } else {
            ((max_degree / mean_degree) - 1.0).clamp(0.0, 2.0) / 2.0
        };
        let coupling_pressure = self
            .threat_couplings
            .iter()
            .map(|coupling| sanitize(coupling.pressure))
            .sum::<f64>()
            / self.threat_couplings.len().max(1) as f64;

        GlobalEncoding {
            mean_degree,
            master_signal,
            bottleneck_ratio,
            oversquashing_risk,
            coupling_pressure,
            mean_latency,
        }
    }
}

impl GraphSurrogate {
    pub fn score(&self, snapshot: &GraphSnapshot) -> [f64; 3] {
        let encoding = snapshot.global_encoding();
        let survival = clamp01(
            1.0 - self.bottleneck_penalty * encoding.bottleneck_ratio
                - self.coupling_penalty * encoding.coupling_pressure * 0.6
                + self.global_gain * encoding.master_signal,
        );
        let continuity = clamp01(
            1.0 - self.bottleneck_penalty * encoding.oversquashing_risk
                - self.coupling_penalty * encoding.coupling_pressure * 0.4
                + self.global_gain * encoding.mean_degree / 6.0,
        );
        let efficiency = clamp01(
            1.0 - sanitize(encoding.mean_latency) * 0.25
                - self.bottleneck_penalty * encoding.bottleneck_ratio * 0.5
                + self.global_gain * encoding.master_signal * 0.5,
        );
        [survival, continuity, efficiency]
    }
}

fn clamp01(value: f64) -> f64 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn sanitize(value: f64) -> f64 {
    if value.is_finite() {
        value.max(0.0)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_snapshot() -> GraphSnapshot {
        GraphSnapshot {
            nodes: vec![
                GraphNode {
                    id: 1,
                    role: "scout".into(),
                },
                GraphNode {
                    id: 2,
                    role: "relay".into(),
                },
                GraphNode {
                    id: 3,
                    role: "strike".into(),
                },
            ],
            edges: vec![
                GraphEdge {
                    src: 1,
                    dst: 2,
                    weight: 1.0,
                    latency: 0.1,
                },
                GraphEdge {
                    src: 2,
                    dst: 3,
                    weight: 1.0,
                    latency: 0.1,
                },
            ],
            threat_couplings: vec![ThreatCoupling {
                src: 1,
                dst: 3,
                pressure: 0.2,
            }],
        }
    }

    #[test]
    fn global_encoding_is_finite() {
        let encoding = sample_snapshot().global_encoding();
        assert!(encoding.mean_degree.is_finite());
        assert!(encoding.master_signal.is_finite());
        assert!(encoding.bottleneck_ratio.is_finite());
    }

    #[test]
    fn surrogate_penalizes_bottlenecks() {
        let surrogate = GraphSurrogate::default();
        let mut smooth = sample_snapshot();
        let mut bottlenecked = sample_snapshot();
        bottlenecked.edges[1].latency = 2.0;

        let smooth_score = surrogate.score(&smooth);
        let bottleneck_score = surrogate.score(&bottlenecked);
        assert!(smooth_score[2] > bottleneck_score[2]);

        smooth.threat_couplings.clear();
        assert!(surrogate.score(&smooth)[0] >= bottleneck_score[0]);
    }
}
