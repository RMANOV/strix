//! # Radio Abstraction Layer
//!
//! Abstracts communication regardless of physical layer. Provides:
//!
//! - **`CommChannel`** trait: send/broadcast/receive with link quality metrics.
//! - **`SimulatedChannel`**: in-memory channel for testing (perfect, lossy, partitioned).
//! - **`BandwidthManager`**: priority-based message throttling.
//! - **`LinkQualityTracker`**: per-peer link quality tracking over time.

use serde::{Deserialize, Serialize};
use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use crate::{MeshMessage, NodeId};

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Communication errors.
#[derive(Debug, Clone, thiserror::Error, Serialize, Deserialize)]
pub enum CommError {
    /// Target node is unreachable.
    #[error("unreachable: {0}")]
    Unreachable(String),
    /// Bandwidth exhausted — message was dropped.
    #[error("bandwidth exceeded: available {available} B/s, required {required} B/s")]
    BandwidthExceeded { available: u32, required: u32 },
    /// Channel is partitioned from the target.
    #[error("network partition: cannot reach {0}")]
    Partitioned(String),
    /// Serialization failure.
    #[error("serialization error: {0}")]
    SerializationError(String),
    /// Generic I/O error.
    #[error("io error: {0}")]
    IoError(String),
}

// ---------------------------------------------------------------------------
// CommChannel trait
// ---------------------------------------------------------------------------

/// Abstract communication channel.
///
/// Implementations may wrap real radios (via strix-adapters) or simulated
/// channels for testing.
pub trait CommChannel: Send + Sync {
    /// Send a message to a specific node.
    fn send(&self, target: NodeId, msg: &MeshMessage) -> Result<(), CommError>;

    /// Broadcast a message to all reachable nodes.
    fn broadcast(&self, msg: &MeshMessage) -> Result<(), CommError>;

    /// Receive the next pending message, if any.
    fn receive(&self) -> Result<Option<MeshMessage>, CommError>;

    /// Link quality to a specific peer (0.0 = dead, 1.0 = perfect).
    fn link_quality(&self, target: NodeId) -> f64;

    /// Currently available bandwidth in bytes/sec.
    fn bandwidth(&self) -> u32;
}

// ---------------------------------------------------------------------------
// Simulated channel
// ---------------------------------------------------------------------------

/// Network condition presets for testing.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SimulationMode {
    /// All messages delivered perfectly.
    Perfect,
    /// Messages dropped with given probability (0.0–1.0).
    Lossy { drop_rate: f64 },
    /// Specific nodes cannot communicate.
    Partitioned,
}

/// Shared message bus for simulated channels.
///
/// Each `SimulatedChannel` instance represents one node's view of the bus.
#[derive(Debug, Clone)]
pub struct SimulatedBus {
    inner: Arc<Mutex<SimulatedBusInner>>,
}

#[derive(Debug)]
struct SimulatedBusInner {
    /// Per-node inbox.
    inboxes: HashMap<NodeId, VecDeque<MeshMessage>>,
    /// Set of (a, b) pairs that are partitioned.
    partitions: Vec<(NodeId, NodeId)>,
}

impl SimulatedBus {
    /// Create a new shared bus.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(SimulatedBusInner {
                inboxes: HashMap::new(),
                partitions: Vec::new(),
            })),
        }
    }

    /// Register a node on the bus.
    pub fn register(&self, node: NodeId) {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        inner.inboxes.entry(node).or_default();
    }

    /// Create a partition between two nodes.
    pub fn partition(&self, a: NodeId, b: NodeId) {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        inner.partitions.push((a, b));
    }

    /// Remove a partition.
    pub fn heal_partition(&self, a: NodeId, b: NodeId) {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        inner
            .partitions
            .retain(|&(x, y)| !((x == a && y == b) || (x == b && y == a)));
    }

    /// Check if two nodes are partitioned.
    fn is_partitioned(&self, a: NodeId, b: NodeId) -> bool {
        let inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        inner
            .partitions
            .iter()
            .any(|&(x, y)| (x == a && y == b) || (x == b && y == a))
    }

    /// Deliver a message to a node's inbox.
    fn deliver(&self, target: NodeId, msg: MeshMessage) -> Result<(), CommError> {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        inner.inboxes.entry(target).or_default().push_back(msg);
        Ok(())
    }

    /// Pop next message from a node's inbox.
    fn pop(&self, node: NodeId) -> Option<MeshMessage> {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        inner.inboxes.get_mut(&node)?.pop_front()
    }

    /// Get all registered nodes.
    fn all_nodes(&self) -> Vec<NodeId> {
        let inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        inner.inboxes.keys().copied().collect()
    }
}

impl Default for SimulatedBus {
    fn default() -> Self {
        Self::new()
    }
}

/// A simulated communication channel for one node.
#[derive(Debug, Clone)]
pub struct SimulatedChannel {
    /// This node's identity.
    node_id: NodeId,
    /// Shared bus.
    bus: SimulatedBus,
    /// Simulation mode.
    mode: SimulationMode,
    /// Simulated bandwidth (bytes/sec).
    bw: u32,
}

impl SimulatedChannel {
    /// Create a simulated channel for a node.
    pub fn new(node_id: NodeId, bus: SimulatedBus, mode: SimulationMode, bandwidth: u32) -> Self {
        bus.register(node_id);
        Self {
            node_id,
            bus,
            mode,
            bw: bandwidth,
        }
    }

    /// Should this message be dropped? (for lossy mode)
    fn should_drop(&self) -> bool {
        match self.mode {
            SimulationMode::Lossy { drop_rate } => rand::random::<f64>() < drop_rate,
            _ => false,
        }
    }
}

impl CommChannel for SimulatedChannel {
    fn send(&self, target: NodeId, msg: &MeshMessage) -> Result<(), CommError> {
        // Check partition.
        if matches!(self.mode, SimulationMode::Partitioned)
            && self.bus.is_partitioned(self.node_id, target)
        {
            return Err(CommError::Partitioned(format!(
                "{} → {}",
                self.node_id, target
            )));
        }
        // Check lossy.
        if self.should_drop() {
            return Ok(()); // silently dropped
        }
        self.bus.deliver(target, msg.clone())
    }

    fn broadcast(&self, msg: &MeshMessage) -> Result<(), CommError> {
        let nodes = self.bus.all_nodes();
        for node in nodes {
            if node != self.node_id {
                // Best-effort: ignore per-node partition errors on broadcast.
                let _ = self.send(node, msg);
            }
        }
        Ok(())
    }

    fn receive(&self) -> Result<Option<MeshMessage>, CommError> {
        Ok(self.bus.pop(self.node_id))
    }

    fn link_quality(&self, target: NodeId) -> f64 {
        match self.mode {
            SimulationMode::Perfect => 1.0,
            SimulationMode::Lossy { drop_rate } => 1.0 - drop_rate,
            SimulationMode::Partitioned => {
                if self.bus.is_partitioned(self.node_id, target) {
                    0.0
                } else {
                    1.0
                }
            }
        }
    }

    fn bandwidth(&self) -> u32 {
        self.bw
    }
}

// ---------------------------------------------------------------------------
// Bandwidth manager
// ---------------------------------------------------------------------------

/// Priority-aware entry for the send queue.
#[derive(Debug, Clone)]
struct PriorityMessage {
    /// Message priority (lower = more important).
    priority: u8,
    /// Insertion sequence for FIFO within same priority.
    sequence: u64,
    /// Target node.
    target: Option<NodeId>,
    /// The message.
    message: MeshMessage,
}

impl PartialEq for PriorityMessage {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority && self.sequence == other.sequence
    }
}

impl Eq for PriorityMessage {}

impl PartialOrd for PriorityMessage {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PriorityMessage {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // BinaryHeap is a max-heap, so we reverse: lower priority number = higher importance.
        other
            .priority
            .cmp(&self.priority)
            .then_with(|| other.sequence.cmp(&self.sequence))
    }
}

/// Manages outbound message queue with bandwidth-aware prioritization.
///
/// Priority order (highest to lowest):
/// 1. `ThreatAlert` (priority 0)
/// 2. `TaskAssignment` (priority 1)
/// 3. `StateUpdate` (priority 2)
/// 4. `PheromoneDeposit` (priority 3)
/// 5. `Heartbeat` (priority 4)
///
/// Under low bandwidth, only `ThreatAlert` and `Heartbeat` get through.
#[derive(Debug, Clone)]
pub struct BandwidthManager {
    /// Available bandwidth in bytes/sec.
    bandwidth: u32,
    /// Bytes already consumed this period.
    bytes_used: u32,
    /// Outbound priority queue.
    queue: BinaryHeap<PriorityMessage>,
    /// Monotonic sequence counter.
    sequence: u64,
    /// Estimated bytes per message (conservative).
    bytes_per_message: u32,
    /// Low bandwidth threshold (bytes/sec) — below this, only critical messages pass.
    low_bandwidth_threshold: u32,
}

impl BandwidthManager {
    /// Create a new bandwidth manager.
    ///
    /// - `bandwidth`: available bytes/sec.
    /// - `bytes_per_message`: estimated size of a serialized message.
    pub fn new(bandwidth: u32, bytes_per_message: u32) -> Self {
        Self {
            bandwidth,
            bytes_used: 0,
            queue: BinaryHeap::new(),
            sequence: 0,
            bytes_per_message,
            low_bandwidth_threshold: bandwidth / 4,
        }
    }

    /// Enqueue a message for sending.
    pub fn enqueue(&mut self, target: Option<NodeId>, msg: MeshMessage) {
        let priority = msg.priority();
        self.sequence += 1;
        self.queue.push(PriorityMessage {
            priority,
            sequence: self.sequence,
            target,
            message: msg,
        });
    }

    /// Drain messages that fit within the current bandwidth budget.
    ///
    /// Returns a list of `(target, message)` pairs to be sent.
    /// `target = None` means broadcast.
    pub fn drain(&mut self) -> Vec<(Option<NodeId>, MeshMessage)> {
        let mut result = Vec::new();
        let remaining_bw = self.bandwidth.saturating_sub(self.bytes_used);
        let is_low_bw = remaining_bw < self.low_bandwidth_threshold;

        while let Some(entry) = self.queue.peek() {
            // Check bandwidth.
            if self.bytes_used + self.bytes_per_message > self.bandwidth {
                break;
            }

            // Under low bandwidth, only critical messages pass.
            if is_low_bw && !is_critical(entry.priority) {
                // Skip non-critical but don't pop — they'll be retried next period.
                // Actually, pop and discard to prevent queue bloat.
                self.queue.pop();
                continue;
            }

            // Safe: peek() above confirmed the queue is non-empty.
            let entry = self.queue.pop().expect("queue non-empty: just peeked");
            self.bytes_used += self.bytes_per_message;
            result.push((entry.target, entry.message));
        }

        result
    }

    /// Reset the bandwidth counter for a new period.
    pub fn reset_period(&mut self) {
        self.bytes_used = 0;
    }

    /// Number of messages queued.
    pub fn queued_count(&self) -> usize {
        self.queue.len()
    }

    /// Update available bandwidth.
    pub fn set_bandwidth(&mut self, bw: u32) {
        self.bandwidth = bw;
        self.low_bandwidth_threshold = bw / 4;
    }
}

/// Returns true for message priorities that should always get through.
fn is_critical(priority: u8) -> bool {
    // ThreatAlert (0) and Heartbeat (4) are always critical.
    priority == 0 || priority == 4
}

// ---------------------------------------------------------------------------
// Link quality tracker
// ---------------------------------------------------------------------------

/// Tracks per-peer link quality over a sliding window.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinkQualityTracker {
    /// Per-peer sliding window of success/failure observations.
    history: HashMap<NodeId, VecDeque<bool>>,
    /// Maximum window size.
    window_size: usize,
}

impl LinkQualityTracker {
    /// Create a new tracker.
    pub fn new(window_size: usize) -> Self {
        Self {
            history: HashMap::new(),
            window_size: window_size.max(1),
        }
    }

    /// Record a successful communication with a peer.
    pub fn record_success(&mut self, peer: NodeId) {
        self.push(peer, true);
    }

    /// Record a failed communication with a peer.
    pub fn record_failure(&mut self, peer: NodeId) {
        self.push(peer, false);
    }

    /// Get the current link quality estimate for a peer (0.0–1.0).
    pub fn quality(&self, peer: NodeId) -> f64 {
        match self.history.get(&peer) {
            None => 0.5, // unknown → assume medium
            Some(window) => {
                if window.is_empty() {
                    return 0.5;
                }
                let successes = window.iter().filter(|&&s| s).count();
                successes as f64 / window.len() as f64
            }
        }
    }

    /// Remove all tracking for a peer.
    pub fn remove(&mut self, peer: NodeId) {
        self.history.remove(&peer);
    }

    fn push(&mut self, peer: NodeId, success: bool) {
        let window = self.history.entry(peer).or_default();
        window.push_back(success);
        while window.len() > self.window_size {
            window.pop_front();
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Position3D;

    fn hb(id: u32) -> MeshMessage {
        MeshMessage::Heartbeat {
            sender: NodeId(id),
            timestamp: 0.0,
        }
    }

    fn threat(id: u32) -> MeshMessage {
        MeshMessage::ThreatAlert {
            reporter: NodeId(id),
            position: Position3D::origin(),
            threat_level: 1.0,
            description: "test".into(),
            timestamp: 0.0,
        }
    }

    #[test]
    fn simulated_perfect_send_receive() {
        let bus = SimulatedBus::new();
        let ch0 = SimulatedChannel::new(NodeId(0), bus.clone(), SimulationMode::Perfect, 10000);
        let _ch1 = SimulatedChannel::new(NodeId(1), bus.clone(), SimulationMode::Perfect, 10000);

        ch0.send(NodeId(1), &hb(0)).unwrap();
        // Receive from node 1's perspective — need a channel for node 1.
        let msg = bus.pop(NodeId(1));
        assert!(msg.is_some());
    }

    #[test]
    fn simulated_broadcast() {
        let bus = SimulatedBus::new();
        let ch0 = SimulatedChannel::new(NodeId(0), bus.clone(), SimulationMode::Perfect, 10000);
        let _ch1 = SimulatedChannel::new(NodeId(1), bus.clone(), SimulationMode::Perfect, 10000);
        let _ch2 = SimulatedChannel::new(NodeId(2), bus.clone(), SimulationMode::Perfect, 10000);

        ch0.broadcast(&hb(0)).unwrap();
        assert!(bus.pop(NodeId(1)).is_some());
        assert!(bus.pop(NodeId(2)).is_some());
        // Sender doesn't get its own broadcast.
        assert!(bus.pop(NodeId(0)).is_none());
    }

    #[test]
    fn simulated_partition() {
        let bus = SimulatedBus::new();
        let ch0 = SimulatedChannel::new(NodeId(0), bus.clone(), SimulationMode::Partitioned, 10000);
        let _ch1 =
            SimulatedChannel::new(NodeId(1), bus.clone(), SimulationMode::Partitioned, 10000);

        bus.partition(NodeId(0), NodeId(1));
        let result = ch0.send(NodeId(1), &hb(0));
        assert!(result.is_err());
        assert_eq!(ch0.link_quality(NodeId(1)), 0.0);

        bus.heal_partition(NodeId(0), NodeId(1));
        let result = ch0.send(NodeId(1), &hb(0));
        assert!(result.is_ok());
    }

    #[test]
    fn bandwidth_manager_priority_ordering() {
        let mut bm = BandwidthManager::new(10000, 100);
        bm.enqueue(Some(NodeId(1)), hb(0)); // priority 4
        bm.enqueue(Some(NodeId(1)), threat(0)); // priority 0

        let drained = bm.drain();
        assert_eq!(drained.len(), 2);
        // Threat should come first.
        assert_eq!(drained[0].1.priority(), 0);
        assert_eq!(drained[1].1.priority(), 4);
    }

    #[test]
    fn bandwidth_manager_respects_limit() {
        let mut bm = BandwidthManager::new(150, 100); // room for 1 message
        bm.enqueue(Some(NodeId(1)), hb(0));
        bm.enqueue(Some(NodeId(1)), hb(1));

        let drained = bm.drain();
        assert_eq!(drained.len(), 1); // only 1 fits
    }

    #[test]
    fn bandwidth_manager_reset() {
        let mut bm = BandwidthManager::new(150, 100);
        bm.enqueue(Some(NodeId(1)), hb(0));
        let _ = bm.drain();

        bm.reset_period();
        bm.enqueue(Some(NodeId(1)), hb(1));
        let drained = bm.drain();
        assert_eq!(drained.len(), 1);
    }

    #[test]
    fn link_quality_tracker() {
        let mut tracker = LinkQualityTracker::new(10);
        assert!((tracker.quality(NodeId(0)) - 0.5).abs() < 1e-10); // unknown

        for _ in 0..8 {
            tracker.record_success(NodeId(0));
        }
        for _ in 0..2 {
            tracker.record_failure(NodeId(0));
        }
        // 8/10 = 0.8
        assert!((tracker.quality(NodeId(0)) - 0.8).abs() < 1e-10);
    }

    #[test]
    fn link_quality_sliding_window() {
        let mut tracker = LinkQualityTracker::new(4);
        // Fill window: S S S S → 1.0
        for _ in 0..4 {
            tracker.record_success(NodeId(0));
        }
        assert!((tracker.quality(NodeId(0)) - 1.0).abs() < 1e-10);

        // Add failures → F S S S → then F F S S etc.
        tracker.record_failure(NodeId(0)); // window: S S S F → 0.75
        assert!((tracker.quality(NodeId(0)) - 0.75).abs() < 1e-10);
    }

    #[test]
    fn comm_error_display() {
        let err = CommError::Unreachable("node 5".into());
        assert!(format!("{err}").contains("unreachable"));
    }

    #[test]
    fn simulated_channel_receive() {
        let bus = SimulatedBus::new();
        let ch0 = SimulatedChannel::new(NodeId(0), bus.clone(), SimulationMode::Perfect, 10000);
        let ch1 = SimulatedChannel::new(NodeId(1), bus.clone(), SimulationMode::Perfect, 10000);

        ch0.send(NodeId(1), &hb(0)).unwrap();
        let received = ch1.receive().unwrap();
        assert!(received.is_some());
        assert_eq!(received.unwrap().sender(), NodeId(0));
    }
}
