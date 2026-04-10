//! Epistemic evidence graph with boolean-gated inference.
//!
//! Fixed-topology graph where nodes are existing STRIX subsystems
//! (Byzantine, Trust, Quarantine, Gossip, GBP, Pheromone, OrderParams)
//! and edges carry boolean gate signals between them.
//!
//! The graph collects signals during a tick, then batch-processes them
//! to produce `FeedbackAction`s that close the loop:
//! - XOR conflicts → lower trust, widen uncertainty
//! - XNOR corroboration → boost trust (independence-scaled)
//! - NOR vacuums → flag information gaps, increase gossip
//! - NAND violations → escalate
//! - High sustained XOR → reset GBP priors (forgetting)

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

use crate::bool_gates::{FeedbackAction, GateSignal, SignalSource};
use crate::trust::TrustDimension;
use crate::NodeId;

/// Configuration for the epistemic evidence graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceGraphConfig {
    /// Master switch — disabled by default for backward compatibility.
    pub enabled: bool,
    /// Trust penalty per XOR conflict.
    pub xor_trust_delta: f64,
    /// Trust boost per XNOR corroboration.
    pub xnor_trust_delta: f64,
    /// Seconds of silence before NOR vacuum triggers.
    pub vacuum_threshold_s: f64,
    /// XOR rate above which to escalate.
    pub escalation_xor_rate: f64,
    /// XOR rate above which to trigger prior reset.
    pub prior_reset_threshold: f64,
    /// Forgetting factor α when resetting priors (0 = no reset, 1 = full).
    pub prior_reset_alpha: f64,
    /// Structural memory EMA decay per tick (0.95 = slow forget).
    pub memory_forgetting_factor: f64,
    /// Rolling history window size for XOR/XNOR tracking.
    pub history_window: usize,
}

impl Default for EvidenceGraphConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            xor_trust_delta: -0.05,
            xnor_trust_delta: 0.02,
            vacuum_threshold_s: 15.0,
            escalation_xor_rate: 0.7,
            prior_reset_threshold: 0.8,
            prior_reset_alpha: 0.3,
            memory_forgetting_factor: 0.95,
            history_window: 20,
        }
    }
}

/// Epistemic evidence graph — collects signals, processes them in batch,
/// produces feedback actions for the orchestrator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceGraph {
    config: EvidenceGraphConfig,
    /// Signals collected during the current tick.
    pending_signals: Vec<GateSignal>,
    /// Per-peer rolling XOR conflict history (true = conflict occurred).
    xor_history: HashMap<NodeId, VecDeque<bool>>,
    /// Per-peer rolling XNOR corroboration history (true = corroboration).
    xnor_history: HashMap<NodeId, VecDeque<bool>>,
    /// Structural memory: edge reliability ∈ [0, 1].
    /// Key is ordered (min, max) to avoid duplicates.
    edge_reliability: HashMap<(SignalSource, SignalSource), f64>,
    /// Last-seen timestamp per source (for NOR vacuum detection).
    last_seen: HashMap<SignalSource, f64>,
    /// Counters for SwarmDecision reporting.
    tick_conflicts: u32,
    tick_corroborations: u32,
    tick_vacuums: u32,
    tick_escalations: u32,
}

impl EvidenceGraph {
    /// Create a new evidence graph.
    pub fn new(config: EvidenceGraphConfig) -> Self {
        Self {
            config,
            pending_signals: Vec::new(),
            xor_history: HashMap::new(),
            xnor_history: HashMap::new(),
            edge_reliability: HashMap::new(),
            last_seen: HashMap::new(),
            tick_conflicts: 0,
            tick_corroborations: 0,
            tick_vacuums: 0,
            tick_escalations: 0,
        }
    }

    // ── Signal collection (called by subsystems during tick) ───────────

    /// Ingest a pre-built gate signal into the graph.
    /// Dispatches to the appropriate emit_* method.
    pub fn ingest_signal(&mut self, signal: GateSignal) {
        match signal {
            GateSignal::Conflict {
                source_a,
                source_b,
                severity,
                timestamp,
            } => self.emit_conflict(source_a, source_b, severity, timestamp),
            GateSignal::Corroboration {
                sources,
                independence,
                timestamp,
                ..
            } => self.emit_corroboration(sources, independence, timestamp),
            GateSignal::Violation {
                condition_a,
                condition_b,
                severity,
            } => self.emit_violation(&condition_a, &condition_b, severity),
            GateSignal::ImplicationBreak {
                antecedent,
                consequent,
                context,
            } => self.emit_implication_break(&antecedent, &consequent, &context),
            GateSignal::Vacuum { .. } => {
                // Vacuums are detected via record_activity + detect_vacuums.
            }
        }
    }

    /// Record an XOR conflict between two signal sources.
    pub fn emit_conflict(&mut self, a: SignalSource, b: SignalSource, severity: f64, now: f64) {
        if !self.config.enabled {
            return;
        }
        let severity = severity.clamp(0.0, 1.0);
        if !severity.is_finite() {
            return;
        }
        self.pending_signals.push(GateSignal::Conflict {
            source_a: a,
            source_b: b,
            severity,
            timestamp: now,
        });
    }

    /// Record an XNOR corroboration between sources.
    pub fn emit_corroboration(&mut self, sources: Vec<SignalSource>, independence: f64, now: f64) {
        if !self.config.enabled || sources.len() < 2 {
            return;
        }
        let independence = independence.clamp(0.0, 1.0);
        if !independence.is_finite() {
            return;
        }
        let confidence_boost = self.config.xnor_trust_delta * independence;
        self.pending_signals.push(GateSignal::Corroboration {
            sources,
            independence,
            confidence_boost,
            timestamp: now,
        });
    }

    /// Record a NAND violation (forbidden state combination).
    pub fn emit_violation(&mut self, cond_a: &str, cond_b: &str, severity: f64) {
        if !self.config.enabled {
            return;
        }
        self.pending_signals.push(GateSignal::Violation {
            condition_a: cond_a.to_string(),
            condition_b: cond_b.to_string(),
            severity: severity.clamp(0.0, 1.0),
        });
    }

    /// Record a broken implication (A → B violated).
    pub fn emit_implication_break(&mut self, ante: &str, cons: &str, ctx: &str) {
        if !self.config.enabled {
            return;
        }
        self.pending_signals.push(GateSignal::ImplicationBreak {
            antecedent: ante.to_string(),
            consequent: cons.to_string(),
            context: ctx.to_string(),
        });
    }

    /// Record that a signal source was active at `now`.
    pub fn record_activity(&mut self, source: SignalSource, now: f64) {
        if self.config.enabled {
            self.last_seen.insert(source, now);
        }
    }

    // ── Processing (called once per tick) ──────────────────────────────

    /// Process all pending signals and produce feedback actions.
    ///
    /// This is the heart of the evidence graph — called once per tick
    /// after all subsystems have emitted their signals.
    pub fn process(&mut self, now: f64) -> Vec<FeedbackAction> {
        if !self.config.enabled {
            return Vec::new();
        }

        self.tick_conflicts = 0;
        self.tick_corroborations = 0;
        self.tick_vacuums = 0;
        self.tick_escalations = 0;

        let mut actions = Vec::new();
        let mut conflicted_peers: std::collections::HashSet<NodeId> =
            std::collections::HashSet::new();

        // Drain pending signals (take ownership).
        let signals = std::mem::take(&mut self.pending_signals);

        for signal in &signals {
            match signal {
                GateSignal::Conflict {
                    source_a,
                    source_b,
                    severity,
                    ..
                } => {
                    self.tick_conflicts += 1;
                    if let Some(p) = Self::extract_peer(*source_a) {
                        conflicted_peers.insert(p);
                    }
                    if let Some(p) = Self::extract_peer(*source_b) {
                        conflicted_peers.insert(p);
                    }
                    self.handle_conflict(*source_a, *source_b, *severity, &mut actions);
                }
                GateSignal::Corroboration {
                    sources,
                    independence,
                    ..
                } => {
                    self.tick_corroborations += 1;
                    self.handle_corroboration(sources, *independence, &mut actions);
                }
                GateSignal::Violation {
                    condition_a,
                    condition_b,
                    severity,
                } => {
                    self.tick_escalations += 1;
                    actions.push(FeedbackAction::Escalate {
                        reason: format!(
                            "NAND violation: '{}' AND '{}' (severity {:.2})",
                            condition_a, condition_b, severity
                        ),
                        severity: *severity,
                    });
                }
                GateSignal::ImplicationBreak {
                    antecedent,
                    consequent,
                    context,
                } => {
                    self.tick_escalations += 1;
                    actions.push(FeedbackAction::Escalate {
                        reason: format!(
                            "Implication break: {} → {} in {}",
                            antecedent, consequent, context
                        ),
                        severity: 0.5,
                    });
                }
                GateSignal::Vacuum { .. } => {
                    // Vacuums are detected in detect_vacuums(), not from pending.
                }
            }
        }

        // Record "no conflict" for active peers that had corroboration
        // but no conflict this tick — dilutes XOR rate over time.
        for peer in self.last_seen.keys() {
            if let Some(p) = Self::extract_peer(*peer) {
                if !conflicted_peers.contains(&p) {
                    let history = self.xor_history.entry(p).or_default();
                    history.push_back(false);
                    if history.len() > self.config.history_window {
                        history.pop_front();
                    }
                }
            }
        }

        // NOR vacuum detection.
        self.detect_vacuums(now, &mut actions);

        // Escalation from sustained high XOR rate.
        self.check_escalation(&mut actions);

        // XOR-triggered prior reset.
        self.xor_triggered_reset(&mut actions);

        // Decay structural memory toward 0.5 (uninformative).
        self.decay_structural_memory();

        actions
    }

    // ── Conflict handling ──────────────────────────────────────────────

    fn handle_conflict(
        &mut self,
        source_a: SignalSource,
        source_b: SignalSource,
        severity: f64,
        actions: &mut Vec<FeedbackAction>,
    ) {
        // Update XOR history for involved peers.
        if let Some(peer) = Self::extract_peer(source_a).or(Self::extract_peer(source_b)) {
            let history = self.xor_history.entry(peer).or_default();
            history.push_back(true);
            if history.len() > self.config.history_window {
                history.pop_front();
            }

            // Determine which trust dimension to penalize.
            let dimension = Self::conflict_to_trust_dimension(source_a);
            let delta = (self.config.xor_trust_delta * severity).clamp(-0.1, 0.0);
            if delta.is_finite() {
                actions.push(FeedbackAction::UpdateTrust {
                    peer,
                    dimension,
                    delta,
                });
            }
        }

        // Update structural memory: this edge is unreliable.
        self.update_structural_memory(source_a, source_b, false);
    }

    // ── Corroboration handling ─────────────────────────────────────────

    fn handle_corroboration(
        &mut self,
        sources: &[SignalSource],
        independence: f64,
        actions: &mut Vec<FeedbackAction>,
    ) {
        // Update XNOR history for all involved peers.
        for source in sources {
            if let Some(peer) = Self::extract_peer(*source) {
                let history = self.xnor_history.entry(peer).or_default();
                history.push_back(true);
                if history.len() > self.config.history_window {
                    history.pop_front();
                }

                // Boost trust on integrity dimension.
                let delta = (self.config.xnor_trust_delta * independence).clamp(0.0, 0.05);
                if delta.is_finite() && delta > 1e-6 {
                    actions.push(FeedbackAction::UpdateTrust {
                        peer,
                        dimension: TrustDimension::Integrity,
                        delta,
                    });
                }
            }
        }

        // Update structural memory for all pairs: these edges are reliable.
        for i in 0..sources.len() {
            for j in (i + 1)..sources.len() {
                self.update_structural_memory(sources[i], sources[j], true);
            }
        }
    }

    // ── NOR vacuum detection ──────────────────────────────────────────

    fn detect_vacuums(&mut self, now: f64, actions: &mut Vec<FeedbackAction>) {
        let threshold = self.config.vacuum_threshold_s;
        // Collect vacuums from last_seen entries that are stale.
        let vacuums: Vec<(SignalSource, f64)> = self
            .last_seen
            .iter()
            .filter_map(|(source, &last)| {
                let age = now - last;
                if age > threshold && age.is_finite() {
                    Some((*source, age))
                } else {
                    None
                }
            })
            .collect();

        for (source, duration) in vacuums {
            self.tick_vacuums += 1;
            let severity = (duration / (threshold * 3.0)).clamp(0.0, 1.0);
            actions.push(FeedbackAction::MarkVacuum { source, severity });

            // If the vacuum is for a gossip source, also reset GBP prior.
            if let SignalSource::Gossip(peer) = source {
                actions.push(FeedbackAction::ResetPrior {
                    peer,
                    alpha: (severity * 0.5).clamp(0.0, 1.0),
                });
            }
        }
    }

    // ── Escalation from sustained conflict ────────────────────────────

    fn check_escalation(&mut self, actions: &mut Vec<FeedbackAction>) {
        let threshold = self.config.escalation_xor_rate;
        let min_history = self.config.history_window / 2;

        for (peer, history) in &self.xor_history {
            if history.len() < min_history {
                continue;
            }
            let rate = history.iter().filter(|&&x| x).count() as f64 / history.len() as f64;
            if rate > threshold {
                self.tick_escalations += 1;
                actions.push(FeedbackAction::Escalate {
                    reason: format!(
                        "Sustained XOR conflict rate {:.2} for peer {} (threshold {:.2})",
                        rate, peer, threshold
                    ),
                    severity: rate,
                });
            }
        }
    }

    // ── XOR-triggered prior reset ─────────────────────────────────────

    fn xor_triggered_reset(&self, actions: &mut Vec<FeedbackAction>) {
        let threshold = self.config.prior_reset_threshold;
        let alpha = self.config.prior_reset_alpha;
        let min_history = self.config.history_window / 2;

        for (peer, history) in &self.xor_history {
            if history.len() < min_history {
                continue;
            }
            let rate = history.iter().filter(|&&x| x).count() as f64 / history.len() as f64;
            if rate > threshold {
                actions.push(FeedbackAction::ResetPrior { peer: *peer, alpha });
            }
        }
    }

    // ── Structural memory ─────────────────────────────────────────────

    fn update_structural_memory(
        &mut self,
        source_a: SignalSource,
        source_b: SignalSource,
        xnor: bool,
    ) {
        // Canonical key ordering for consistency.
        let key = if source_a <= source_b {
            (source_a, source_b)
        } else {
            (source_b, source_a)
        };

        let reliability = self.edge_reliability.entry(key).or_insert(0.5);
        let lr = 0.1; // structural memory learning rate
        let observation = if xnor { 1.0 } else { 0.0 };
        *reliability = (1.0 - lr) * *reliability + lr * observation;
        *reliability = reliability.clamp(0.0, 1.0);
    }

    fn decay_structural_memory(&mut self) {
        let decay = self.config.memory_forgetting_factor;
        for reliability in self.edge_reliability.values_mut() {
            // Decay toward 0.5 (uninformative).
            *reliability = 0.5 + (*reliability - 0.5) * decay;
        }
        // Prune entries that have fully decayed to uninformative.
        self.edge_reliability.retain(|_, v| (*v - 0.5).abs() > 0.01);
        // Prune empty history deques (peer no longer active).
        self.xor_history.retain(|_, h| !h.is_empty());
        self.xnor_history.retain(|_, h| !h.is_empty());
    }

    // ── Queries ───────────────────────────────────────────────────────

    /// Rolling XOR conflict rate for a peer.
    pub fn peer_xor_rate(&self, peer: NodeId) -> f64 {
        self.xor_history
            .get(&peer)
            .map(|h| {
                if h.is_empty() {
                    0.0
                } else {
                    h.iter().filter(|&&x| x).count() as f64 / h.len() as f64
                }
            })
            .unwrap_or(0.0)
    }

    /// Rolling XNOR corroboration rate for a peer.
    pub fn peer_xnor_rate(&self, peer: NodeId) -> f64 {
        self.xnor_history
            .get(&peer)
            .map(|h| {
                if h.is_empty() {
                    0.0
                } else {
                    h.iter().filter(|&&x| x).count() as f64 / h.len() as f64
                }
            })
            .unwrap_or(0.0)
    }

    /// Reliability score for an edge between two sources.
    pub fn edge_reliability(&self, a: SignalSource, b: SignalSource) -> f64 {
        let key_ab = (a, b);
        let key_ba = (b, a);
        self.edge_reliability
            .get(&key_ab)
            .or_else(|| self.edge_reliability.get(&key_ba))
            .copied()
            .unwrap_or(0.5)
    }

    /// Count of conflicts in the current tick.
    pub fn tick_conflicts(&self) -> u32 {
        self.tick_conflicts
    }

    /// Count of corroborations in the current tick.
    pub fn tick_corroborations(&self) -> u32 {
        self.tick_corroborations
    }

    /// Count of vacuums in the current tick.
    pub fn tick_vacuums(&self) -> u32 {
        self.tick_vacuums
    }

    /// Count of escalations in the current tick.
    pub fn tick_escalations(&self) -> u32 {
        self.tick_escalations
    }

    /// Number of pending signals not yet processed.
    pub fn pending_signal_count(&self) -> usize {
        self.pending_signals.len()
    }

    // ── Helpers ───────────────────────────────────────────────────────

    fn extract_peer(source: SignalSource) -> Option<NodeId> {
        match source {
            SignalSource::Byzantine(id)
            | SignalSource::Trust(id)
            | SignalSource::Quarantine(id)
            | SignalSource::Gossip(id)
            | SignalSource::Gbp(id) => Some(id),
            SignalSource::Pheromone(_) | SignalSource::OrderParams => None,
        }
    }

    fn conflict_to_trust_dimension(source: SignalSource) -> TrustDimension {
        match source {
            SignalSource::Byzantine(_) => TrustDimension::Kinematic,
            SignalSource::Trust(_) => TrustDimension::Integrity,
            SignalSource::Quarantine(_) => TrustDimension::Consensus,
            SignalSource::Gossip(_) => TrustDimension::Timeliness,
            SignalSource::Gbp(_) => TrustDimension::Kinematic,
            SignalSource::Pheromone(_) | SignalSource::OrderParams => TrustDimension::Integrity,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn enabled_config() -> EvidenceGraphConfig {
        EvidenceGraphConfig {
            enabled: true,
            history_window: 10,
            vacuum_threshold_s: 5.0,
            escalation_xor_rate: 0.6,
            prior_reset_threshold: 0.8,
            ..EvidenceGraphConfig::default()
        }
    }

    #[test]
    fn disabled_graph_returns_empty() {
        let mut graph = EvidenceGraph::new(EvidenceGraphConfig::default());
        graph.emit_conflict(
            SignalSource::Byzantine(NodeId(1)),
            SignalSource::Gossip(NodeId(1)),
            0.9,
            10.0,
        );
        let actions = graph.process(10.0);
        assert!(actions.is_empty());
        assert_eq!(graph.tick_conflicts(), 0);
    }

    #[test]
    fn single_conflict_produces_trust_delta() {
        let mut graph = EvidenceGraph::new(enabled_config());
        graph.emit_conflict(
            SignalSource::Byzantine(NodeId(3)),
            SignalSource::Gossip(NodeId(3)),
            0.8,
            1.0,
        );
        let actions = graph.process(1.0);
        assert_eq!(graph.tick_conflicts(), 1);

        let trust_updates: Vec<_> = actions
            .iter()
            .filter(|a| matches!(a, FeedbackAction::UpdateTrust { .. }))
            .collect();
        assert!(!trust_updates.is_empty());

        if let FeedbackAction::UpdateTrust { peer, delta, .. } = &trust_updates[0] {
            assert_eq!(*peer, NodeId(3));
            assert!(*delta < 0.0, "conflict should produce negative delta");
        }
    }

    #[test]
    fn corroboration_boosts_trust() {
        let mut graph = EvidenceGraph::new(enabled_config());
        graph.emit_corroboration(
            vec![
                SignalSource::Byzantine(NodeId(5)),
                SignalSource::Gossip(NodeId(5)),
            ],
            0.9, // high independence
            2.0,
        );
        let actions = graph.process(2.0);
        assert_eq!(graph.tick_corroborations(), 1);

        let trust_updates: Vec<_> = actions
            .iter()
            .filter(|a| matches!(a, FeedbackAction::UpdateTrust { delta, .. } if *delta > 0.0))
            .collect();
        assert!(
            !trust_updates.is_empty(),
            "corroboration should boost trust"
        );
    }

    #[test]
    fn low_independence_reduces_boost() {
        let mut graph = EvidenceGraph::new(enabled_config());
        // High independence
        graph.emit_corroboration(
            vec![
                SignalSource::Byzantine(NodeId(1)),
                SignalSource::Gossip(NodeId(1)),
            ],
            1.0,
            1.0,
        );
        let actions_hi = graph.process(1.0);

        let mut graph2 = EvidenceGraph::new(enabled_config());
        // Low independence
        graph2.emit_corroboration(
            vec![
                SignalSource::Byzantine(NodeId(1)),
                SignalSource::Gossip(NodeId(1)),
            ],
            0.1,
            1.0,
        );
        let actions_lo = graph2.process(1.0);

        let delta_hi = actions_hi.iter().find_map(|a| match a {
            FeedbackAction::UpdateTrust { delta, .. } => Some(*delta),
            _ => None,
        });
        let delta_lo = actions_lo.iter().find_map(|a| match a {
            FeedbackAction::UpdateTrust { delta, .. } => Some(*delta),
            _ => None,
        });

        assert!(
            delta_hi.unwrap_or(0.0) > delta_lo.unwrap_or(0.0),
            "high independence should produce larger trust boost"
        );
    }

    #[test]
    fn vacuum_detection_timing() {
        let mut graph = EvidenceGraph::new(enabled_config());
        // Record activity at t=0.
        graph.record_activity(SignalSource::Gossip(NodeId(7)), 0.0);
        // Process at t=3 — not yet stale (threshold = 5s).
        let actions = graph.process(3.0);
        assert_eq!(graph.tick_vacuums(), 0);

        // Process at t=10 — now stale.
        let actions = graph.process(10.0);
        assert!(graph.tick_vacuums() > 0);
        let has_vacuum = actions
            .iter()
            .any(|a| matches!(a, FeedbackAction::MarkVacuum { .. }));
        assert!(has_vacuum, "should detect vacuum after threshold");
    }

    #[test]
    fn nand_violation_escalates() {
        let mut graph = EvidenceGraph::new(enabled_config());
        graph.emit_violation("high_risk_decision", "low_quorum", 0.9);
        let actions = graph.process(1.0);
        assert_eq!(graph.tick_escalations(), 1);
        let has_escalate = actions
            .iter()
            .any(|a| matches!(a, FeedbackAction::Escalate { .. }));
        assert!(has_escalate);
    }

    #[test]
    fn xor_rate_rolling_window() {
        let mut graph = EvidenceGraph::new(enabled_config());
        // Register peer as active so non-conflict ticks push false.
        graph.record_activity(SignalSource::Gossip(NodeId(1)), 0.0);
        // 8 conflicts out of 10 ticks → rate = 0.8.
        for i in 0..10 {
            if i < 8 {
                graph.emit_conflict(
                    SignalSource::Byzantine(NodeId(1)),
                    SignalSource::Gossip(NodeId(1)),
                    0.5,
                    i as f64,
                );
            }
            graph.process(i as f64);
        }
        let rate = graph.peer_xor_rate(NodeId(1));
        assert!(
            (rate - 0.8).abs() < 0.15,
            "XOR rate should be ~0.8, got {rate}"
        );
    }

    #[test]
    fn structural_memory_update() {
        let mut graph = EvidenceGraph::new(enabled_config());
        let a = SignalSource::Byzantine(NodeId(1));
        let b = SignalSource::Gossip(NodeId(1));

        // Initially 0.5 (uninformative).
        assert!((graph.edge_reliability(a, b) - 0.5).abs() < 1e-6);

        // Several corroborations should push reliability above 0.5.
        for _ in 0..5 {
            graph.emit_corroboration(vec![a, b], 1.0, 1.0);
            graph.process(1.0);
        }
        assert!(
            graph.edge_reliability(a, b) > 0.6,
            "reliability should increase with XNOR"
        );
    }

    #[test]
    fn xor_triggered_reset_fires() {
        let cfg = EvidenceGraphConfig {
            enabled: true,
            history_window: 5,
            prior_reset_threshold: 0.6,
            prior_reset_alpha: 0.3,
            escalation_xor_rate: 0.9, // high so it doesn't interfere
            vacuum_threshold_s: 1000.0,
            ..EvidenceGraphConfig::default()
        };
        let mut graph = EvidenceGraph::new(cfg);

        // 4 conflicts out of 5 → rate = 0.8 > threshold 0.6.
        for i in 0..5 {
            if i < 4 {
                graph.emit_conflict(
                    SignalSource::Byzantine(NodeId(2)),
                    SignalSource::Gossip(NodeId(2)),
                    0.5,
                    i as f64,
                );
            }
            graph.process(i as f64);
        }

        // Process once more to trigger reset check.
        let actions = graph.process(5.0);
        let has_reset = actions
            .iter()
            .any(|a| matches!(a, FeedbackAction::ResetPrior { .. }));
        assert!(has_reset, "should fire prior reset on high XOR rate");
    }

    #[test]
    fn escalation_requires_sustained_conflict() {
        let cfg = EvidenceGraphConfig {
            enabled: true,
            history_window: 10,
            escalation_xor_rate: 0.6,
            vacuum_threshold_s: 1000.0,
            prior_reset_threshold: 0.99,
            ..EvidenceGraphConfig::default()
        };
        let mut graph = EvidenceGraph::new(cfg);

        // Only 2 conflicts — not enough history (min = window/2 = 5).
        for i in 0..2 {
            graph.emit_conflict(
                SignalSource::Byzantine(NodeId(1)),
                SignalSource::Gossip(NodeId(1)),
                0.5,
                i as f64,
            );
            graph.process(i as f64);
        }

        let actions = graph.process(3.0);
        let escalations: Vec<_> = actions
            .iter()
            .filter(|a| {
                matches!(a, FeedbackAction::Escalate { reason, .. } if reason.contains("Sustained"))
            })
            .collect();
        assert!(
            escalations.is_empty(),
            "should not escalate without sustained history"
        );
    }
}
