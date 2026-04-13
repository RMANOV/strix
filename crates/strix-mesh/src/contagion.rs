use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::{MeshMessage, NodeId};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContagionMode {
    Simple,
    Complex,
    Damped,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContagionPolicy {
    pub simple_ttl_s: f64,
    pub complex_min_sources: usize,
    pub complex_reinforcement_threshold: f64,
    pub damped_half_life_s: f64,
    pub damped_floor: f64,
}

impl Default for ContagionPolicy {
    fn default() -> Self {
        Self {
            simple_ttl_s: 2.0,
            complex_min_sources: 2,
            complex_reinforcement_threshold: 1.1,
            damped_half_life_s: 4.0,
            damped_floor: 0.45,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct SignalAccumulator {
    sources: HashSet<NodeId>,
    last_timestamp: f64,
    energy: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContagionEngine {
    pub policy: ContagionPolicy,
    signals: HashMap<String, SignalAccumulator>,
    forwarded: u64,
    blocked: u64,
}

impl ContagionEngine {
    pub fn new(policy: ContagionPolicy) -> Self {
        Self {
            policy,
            signals: HashMap::new(),
            forwarded: 0,
            blocked: 0,
        }
    }

    /// Number of messages forwarded since last reset.
    pub fn forwarded_count(&self) -> u64 {
        self.forwarded
    }

    /// Number of messages blocked since last reset.
    pub fn blocked_count(&self) -> u64 {
        self.blocked
    }

    /// Reset forwarding metrics (call at tick boundary).
    pub fn reset_counters(&mut self) {
        self.forwarded = 0;
        self.blocked = 0;
    }

    pub fn mode_for(message: &MeshMessage) -> ContagionMode {
        match message {
            MeshMessage::ThreatAlert { .. }
            | MeshMessage::StateUpdate { .. }
            | MeshMessage::Heartbeat { .. }
            | MeshMessage::PheromoneDeposit { .. } => ContagionMode::Simple,
            MeshMessage::TaskAssignment { .. } | MeshMessage::CoordinationDirective { .. } => {
                ContagionMode::Complex
            }
            MeshMessage::AffectSignal { .. } => ContagionMode::Damped,
        }
    }

    pub fn should_forward(&mut self, message: &MeshMessage, now: f64) -> bool {
        let signature = signature(message);
        let mode = Self::mode_for(message);
        let sender = message.sender();
        let timestamp = sanitize_time(message.timestamp());
        let intensity = message_intensity(message);
        let entry = self.signals.entry(signature).or_default();

        let result = match mode {
            ContagionMode::Simple => {
                let stale = now - entry.last_timestamp > self.policy.simple_ttl_s;
                let first_source = entry.sources.insert(sender);
                if first_source || stale || timestamp > entry.last_timestamp {
                    entry.last_timestamp = timestamp.max(now);
                    entry.energy = intensity;
                    true
                } else {
                    false
                }
            }
            ContagionMode::Complex => {
                entry.sources.insert(sender);
                entry.energy += intensity;
                entry.last_timestamp = timestamp.max(entry.last_timestamp);
                entry.sources.len() >= self.policy.complex_min_sources
                    || entry.energy >= self.policy.complex_reinforcement_threshold
            }
            ContagionMode::Damped => {
                let previous_timestamp = entry.last_timestamp;
                let dt = (now - previous_timestamp).max(0.0);
                let decay = if self.policy.damped_half_life_s <= 1e-6 {
                    0.0
                } else {
                    f64::exp(-std::f64::consts::LN_2 * dt / self.policy.damped_half_life_s)
                };
                let novel_signal = entry.sources.insert(sender) || timestamp > previous_timestamp;
                entry.energy *= decay;
                if novel_signal {
                    entry.energy += intensity;
                }
                entry.last_timestamp = now.max(timestamp);
                entry.energy >= self.policy.damped_floor
            }
        };
        if result {
            self.forwarded += 1;
        } else {
            self.blocked += 1;
        }
        result
    }
}

fn signature(message: &MeshMessage) -> String {
    match message {
        MeshMessage::Heartbeat { sender, .. } => format!("hb:{}", sender.0),
        MeshMessage::StateUpdate { sender, regime, .. } => {
            format!("state:{}:{}", sender.0, regime)
        }
        MeshMessage::TaskAssignment {
            task_id, assignee, ..
        } => format!("task:{}:{}", task_id, assignee.0),
        MeshMessage::ThreatAlert {
            description,
            position,
            ..
        } => format!(
            "threat:{}:{:.0}:{:.0}:{:.0}",
            description, position.0[0], position.0[1], position.0[2]
        ),
        MeshMessage::PheromoneDeposit {
            depositor,
            ptype,
            position,
            ..
        } => format!(
            "pheromone:{}:{:?}:{:.0}:{:.0}:{:.0}",
            depositor.0, ptype, position.0[0], position.0[1], position.0[2]
        ),
        MeshMessage::CoordinationDirective {
            directive, focus, ..
        } => match focus {
            Some(position) => format!(
                "directive:{:?}:{:.0}:{:.0}:{:.0}",
                directive, position.0[0], position.0[1], position.0[2]
            ),
            None => format!("directive:{directive:?}:none"),
        },
        MeshMessage::AffectSignal { label, .. } => format!("affect:{}", label),
    }
}

fn message_intensity(message: &MeshMessage) -> f64 {
    match message {
        MeshMessage::ThreatAlert { threat_level, .. } => threat_level.clamp(0.0, 1.0),
        MeshMessage::PheromoneDeposit { intensity, .. }
        | MeshMessage::CoordinationDirective { intensity, .. }
        | MeshMessage::AffectSignal { intensity, .. } => intensity.clamp(0.0, 1.0),
        MeshMessage::TaskAssignment { .. } => 0.6,
        MeshMessage::StateUpdate { .. } => 0.4,
        MeshMessage::Heartbeat { .. } => 0.2,
    }
}

fn sanitize_time(value: f64) -> f64 {
    if value.is_finite() {
        value.max(0.0)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stigmergy::PheromoneType;
    use crate::{CoordinationDirectiveKind, Position3D};

    #[test]
    fn simple_contagion_forwards_fresh_threats() {
        let mut engine = ContagionEngine::new(ContagionPolicy::default());
        let threat = MeshMessage::ThreatAlert {
            reporter: NodeId(1),
            position: Position3D([10.0, 0.0, 0.0]),
            threat_level: 0.9,
            description: "sam".into(),
            timestamp: 1.0,
        };
        assert!(engine.should_forward(&threat, 1.0));
        assert!(!engine.should_forward(&threat, 1.1));
    }

    #[test]
    fn complex_contagion_requires_reinforcement() {
        let mut engine = ContagionEngine::new(ContagionPolicy::default());
        let first = MeshMessage::CoordinationDirective {
            sender: NodeId(1),
            directive: CoordinationDirectiveKind::StrikeCommit,
            focus: Some(Position3D([5.0, 5.0, 0.0])),
            intensity: 0.4,
            timestamp: 1.0,
        };
        let second = MeshMessage::CoordinationDirective {
            sender: NodeId(2),
            directive: CoordinationDirectiveKind::StrikeCommit,
            focus: Some(Position3D([5.0, 5.0, 0.0])),
            intensity: 0.4,
            timestamp: 1.1,
        };
        assert!(!engine.should_forward(&first, 1.0));
        assert!(engine.should_forward(&second, 1.1));
    }

    #[test]
    fn damped_contagion_cools_down() {
        let mut engine = ContagionEngine::new(ContagionPolicy::default());
        let panic = MeshMessage::AffectSignal {
            sender: NodeId(1),
            label: "panic".into(),
            intensity: 0.5,
            timestamp: 1.0,
        };
        assert!(engine.should_forward(&panic, 1.0));
        assert!(!engine.should_forward(&panic, 9.0));
    }

    #[test]
    fn pheromone_deposits_are_simple_contagion() {
        let msg = MeshMessage::PheromoneDeposit {
            depositor: NodeId(3),
            position: Position3D([0.0, 0.0, 0.0]),
            ptype: PheromoneType::Threat,
            intensity: 0.7,
            timestamp: 2.0,
        };
        assert_eq!(ContagionEngine::mode_for(&msg), ContagionMode::Simple);
    }

    // -- Edge-case / boundary tests --

    #[test]
    fn simple_ttl_expiry_re_forwards() {
        let policy = ContagionPolicy {
            simple_ttl_s: 2.0,
            ..ContagionPolicy::default()
        };
        let mut engine = ContagionEngine::new(policy);
        let msg = MeshMessage::Heartbeat {
            sender: NodeId(10),
            timestamp: 1.0,
        };
        assert!(engine.should_forward(&msg, 1.0), "fresh: forward");
        assert!(!engine.should_forward(&msg, 1.5), "within TTL: block");
        // now=4.0, last=1.0, dt=3.0 > TTL=2.0 → stale → re-forward
        assert!(
            engine.should_forward(&msg, 4.0),
            "after TTL expiry: must re-forward"
        );
    }

    #[test]
    fn damped_energy_below_floor_blocks() {
        let policy = ContagionPolicy {
            damped_half_life_s: 1.0,
            damped_floor: 0.45,
            ..ContagionPolicy::default()
        };
        let mut engine = ContagionEngine::new(policy);
        let msg = MeshMessage::AffectSignal {
            sender: NodeId(5),
            label: "fear".into(),
            intensity: 0.5,
            timestamp: 0.0,
        };
        assert!(engine.should_forward(&msg, 0.0), "initial above floor");
        // After several half-lives, energy decays below floor
        assert!(
            !engine.should_forward(&msg, 10.0),
            "energy decayed below floor"
        );
    }

    #[test]
    fn counter_values_and_reset() {
        let mut engine = ContagionEngine::new(ContagionPolicy::default());
        let a = MeshMessage::Heartbeat {
            sender: NodeId(1),
            timestamp: 0.0,
        };
        let b = MeshMessage::Heartbeat {
            sender: NodeId(2),
            timestamp: 0.0,
        };
        engine.should_forward(&a, 0.0); // forward (new)
        engine.should_forward(&a, 0.5); // block (dup within TTL)
        engine.should_forward(&b, 0.0); // forward (different sender → different sig)
        assert_eq!(engine.forwarded_count(), 2);
        assert_eq!(engine.blocked_count(), 1);
        engine.reset_counters();
        assert_eq!(engine.forwarded_count(), 0);
        assert_eq!(engine.blocked_count(), 0);
    }

    #[test]
    fn complex_min_sources_boundary() {
        let policy = ContagionPolicy {
            complex_min_sources: 2,
            complex_reinforcement_threshold: 100.0, // high → only sources path
            ..ContagionPolicy::default()
        };
        let mut engine = ContagionEngine::new(policy);
        let d1 = MeshMessage::CoordinationDirective {
            sender: NodeId(1),
            directive: CoordinationDirectiveKind::Retreat,
            focus: Some(Position3D([0.0, 0.0, 0.0])),
            intensity: 0.1,
            timestamp: 1.0,
        };
        let d2 = MeshMessage::CoordinationDirective {
            sender: NodeId(2),
            directive: CoordinationDirectiveKind::Retreat,
            focus: Some(Position3D([0.0, 0.0, 0.0])),
            intensity: 0.1,
            timestamp: 1.1,
        };
        assert!(!engine.should_forward(&d1, 1.0), "1 source < min_sources=2");
        assert!(
            engine.should_forward(&d2, 1.1),
            "exactly min_sources=2 must forward"
        );
    }

    #[test]
    fn mode_classification_all_variants() {
        let hb = MeshMessage::Heartbeat {
            sender: NodeId(1),
            timestamp: 0.0,
        };
        assert_eq!(ContagionEngine::mode_for(&hb), ContagionMode::Simple);

        let ta = MeshMessage::TaskAssignment {
            assigner: NodeId(1),
            assignee: NodeId(2),
            task_id: 1,
            description: String::new(),
            timestamp: 0.0,
        };
        assert_eq!(ContagionEngine::mode_for(&ta), ContagionMode::Complex);

        let af = MeshMessage::AffectSignal {
            sender: NodeId(1),
            label: "x".into(),
            intensity: 0.5,
            timestamp: 0.0,
        };
        assert_eq!(ContagionEngine::mode_for(&af), ContagionMode::Damped);
    }
}
