//! Anti-Fragile Loss Recovery — the swarm gets *stronger* after losses (Taleb).
//!
//! When a drone is destroyed, we don't just grieve — we *learn*:
//!
//! 1. **Record** the circumstances: position, altitude, heading, sensor readings, threat bearing.
//! 2. **Classify** the loss: SAM, small arms, collision, EW, unknown.
//! 3. **Adapt**: expand threat contours, penalise bids near kill zones, update regime transitions.
//! 4. **Re-auction** the destroyed drone's tasks instantly.
//!
//! After N losses the swarm should be measurably better at avoiding the same threat.

use serde::{Deserialize, Serialize};

use crate::{Position, Regime, ThreatType};

// ────────────────────────────────────────────────────────────────────────────────
// Types
// ────────────────────────────────────────────────────────────────────────────────

/// Classification of how a drone was lost.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LossClassification {
    /// Surface-to-air missile.
    Sam,
    /// Small arms / anti-aircraft fire.
    SmallArms,
    /// Mid-air collision.
    Collision,
    /// Electronic warfare (jammed / spoofed).
    ElectronicWarfare,
    /// Unable to determine cause.
    Unknown,
}

impl LossClassification {
    /// Map from [`ThreatType`] to a loss classification.
    pub fn from_threat_type(tt: ThreatType) -> Self {
        match tt {
            ThreatType::Sam => LossClassification::Sam,
            ThreatType::SmallArms => LossClassification::SmallArms,
            ThreatType::ElectronicWarfare => LossClassification::ElectronicWarfare,
            ThreatType::Unknown => LossClassification::Unknown,
        }
    }

    /// Default kill-zone radius for this class of threat (metres).
    pub fn default_kill_zone_radius(&self) -> f64 {
        match self {
            LossClassification::Sam => 2000.0,
            LossClassification::SmallArms => 500.0,
            LossClassification::Collision => 200.0,
            LossClassification::ElectronicWarfare => 1500.0,
            LossClassification::Unknown => 1000.0,
        }
    }

    /// Penalty weight applied to bids near this kill zone.
    pub fn penalty_weight(&self) -> f64 {
        match self {
            LossClassification::Sam => 0.8,
            LossClassification::SmallArms => 0.4,
            LossClassification::Collision => 0.2,
            LossClassification::ElectronicWarfare => 0.6,
            LossClassification::Unknown => 0.5,
        }
    }
}

/// Detailed record of a drone loss event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LossRecord {
    /// ID of the destroyed drone.
    pub drone_id: u32,
    /// Position at time of loss.
    pub position: Position,
    /// Altitude (z component, repeated for clarity).
    pub altitude: f64,
    /// Heading in radians (0 = north, π/2 = east).
    pub heading: f64,
    /// Last known velocity.
    pub velocity: [f64; 3],
    /// Bearing to suspected threat (radians).
    pub threat_bearing: Option<f64>,
    /// What regime the drone was in when lost.
    pub regime_at_loss: Regime,
    /// Classification of the loss.
    pub classification: LossClassification,
    /// Tasks that were assigned to this drone and need re-auction.
    pub orphaned_tasks: Vec<u32>,
    /// Timestamp (mission-relative seconds).
    pub timestamp: f64,
}

/// Kill zone: a region where drones have been destroyed — avoid or penalise.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KillZone {
    /// Centre of the kill zone.
    pub center: Position,
    /// Radius (metres).
    pub radius: f64,
    /// Bid penalty weight.
    pub penalty: f64,
    /// What caused the kill zone.
    pub classification: LossClassification,
    /// How many losses have occurred here (reinforces the zone).
    pub loss_count: u32,
}

/// Regime transition adjustment: how loss data shifts the probability of
/// entering EVADE near kill zones.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeAdjustment {
    /// Position near which the adjustment applies.
    pub near: Position,
    /// Radius of influence.
    pub radius: f64,
    /// Added probability of transitioning to EVADE when inside this radius.
    pub evade_bias: f64,
}

/// The loss analyser — central component of the anti-fragile system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LossAnalyzer {
    /// All recorded loss events.
    pub loss_records: Vec<LossRecord>,
    /// Active kill zones (updated after each loss).
    pub kill_zones: Vec<KillZone>,
    /// Regime-transition adjustments derived from losses.
    pub regime_adjustments: Vec<RegimeAdjustment>,
    /// Growth factor: how much the kill-zone radius expands with each
    /// additional loss at the same location.
    pub zone_growth_factor: f64,
    /// Merge distance: losses within this range are treated as the same zone.
    pub merge_distance: f64,
}

// ────────────────────────────────────────────────────────────────────────────────
// Implementation
// ────────────────────────────────────────────────────────────────────────────────

impl Default for LossAnalyzer {
    fn default() -> Self {
        Self {
            loss_records: Vec::new(),
            kill_zones: Vec::new(),
            regime_adjustments: Vec::new(),
            zone_growth_factor: 1.2,
            merge_distance: 500.0,
        }
    }
}

impl LossAnalyzer {
    /// Create a new loss analyser with default parameters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder: set zone growth factor.
    pub fn with_growth_factor(mut self, factor: f64) -> Self {
        self.zone_growth_factor = factor.max(1.0);
        self
    }

    /// Builder: set merge distance.
    pub fn with_merge_distance(mut self, distance: f64) -> Self {
        self.merge_distance = distance.max(0.0);
        self
    }

    /// Record a loss and trigger adaptation.
    ///
    /// Returns the list of orphaned task IDs that need immediate re-auction.
    pub fn record_loss(&mut self, record: LossRecord) -> Vec<u32> {
        let orphans = record.orphaned_tasks.clone();
        self.loss_records.push(record.clone());
        self.adapt_from_loss(&record);
        orphans
    }

    /// Classify a loss from context clues.
    ///
    /// Simple heuristic: if a threat bearing is known and the loss happened
    /// in ENGAGE regime, likely SAM/SmallArms. Otherwise unknown.
    pub fn classify_loss(
        regime: Regime,
        threat_bearing: Option<f64>,
        altitude: f64,
    ) -> LossClassification {
        match (regime, threat_bearing) {
            (Regime::Engage, Some(_)) if altitude > 200.0 => LossClassification::Sam,
            (Regime::Engage, Some(_)) => LossClassification::SmallArms,
            (Regime::Patrol, None) => LossClassification::Collision,
            (_, Some(_)) => LossClassification::ElectronicWarfare,
            _ => LossClassification::Unknown,
        }
    }

    /// Core adaptation logic — called after every loss.
    fn adapt_from_loss(&mut self, record: &LossRecord) {
        // 1. Update or create kill zone.
        let merged = self.merge_into_existing_zone(record);
        if !merged {
            let kz = KillZone {
                center: record.position,
                radius: record.classification.default_kill_zone_radius(),
                penalty: record.classification.penalty_weight(),
                classification: record.classification,
                loss_count: 1,
            };
            self.kill_zones.push(kz);
        }

        // 2. Add regime adjustment — increase EVADE probability near kill zone.
        self.regime_adjustments.push(RegimeAdjustment {
            near: record.position,
            radius: record.classification.default_kill_zone_radius() * 1.5,
            evade_bias: 0.15 * self.loss_count_near(record.position) as f64,
        });
    }

    /// Try to merge the loss into an existing kill zone. Returns `true` if merged.
    fn merge_into_existing_zone(&mut self, record: &LossRecord) -> bool {
        for kz in &mut self.kill_zones {
            let dist = kz.center.distance_to(&record.position);
            if dist < self.merge_distance {
                kz.loss_count += 1;
                // Expand the radius with each additional loss (anti-fragile growth).
                kz.radius *= self.zone_growth_factor;
                // Increase penalty (more dangerous than we thought).
                kz.penalty = (kz.penalty * 1.1).min(1.0);
                // Shift centre towards the new loss (weighted average).
                let w = 1.0 / kz.loss_count as f64;
                kz.center = Position::new(
                    kz.center.x * (1.0 - w) + record.position.x * w,
                    kz.center.y * (1.0 - w) + record.position.y * w,
                    kz.center.z * (1.0 - w) + record.position.z * w,
                );
                return true;
            }
        }
        false
    }

    /// Count total losses near a position (within merge distance).
    fn loss_count_near(&self, pos: Position) -> u32 {
        self.kill_zones
            .iter()
            .filter(|kz| kz.center.distance_to(&pos) < self.merge_distance)
            .map(|kz| kz.loss_count)
            .sum()
    }

    /// Get all kill zone penalties in the format expected by [`Bidder`](crate::bidder::Bidder).
    ///
    /// Returns `Vec<(center, radius, penalty_weight)>`.
    pub fn kill_zone_penalties(&self) -> Vec<(Position, f64, f64)> {
        self.kill_zones
            .iter()
            .map(|kz| (kz.center, kz.radius, kz.penalty))
            .collect()
    }

    /// Fear-amplified kill zone penalties.
    ///
    /// Higher fear amplifies penalty weights: SAM 0.8→2.0 at F=1.0.
    /// The amplified weight can exceed 1.0 — clamping happens at the
    /// `risk_exposure` level in the bidder.
    pub fn kill_zone_penalties_with_fear(&self, fear: f64) -> Vec<(Position, f64, f64)> {
        let f = fear.clamp(0.0, 1.0);
        let multiplier = 1.0 + f * 1.5; // 1.0→2.5 (SAM: 0.8*2.5=2.0)
        self.kill_zones
            .iter()
            .map(|kz| (kz.center, kz.radius, kz.penalty * multiplier))
            .collect()
    }

    /// Anti-fragility score: how much the swarm has learned from losses.
    ///
    /// Returns a value >= 0.0. Higher means more adapted.
    ///
    /// Formula: `sum over kill zones of (loss_count * ln(1 + loss_count) * radius_growth)`.
    /// The logarithm ensures diminishing returns — the first loss at a location
    /// teaches the most, subsequent losses reinforce.
    pub fn antifragile_score(&self) -> f64 {
        self.kill_zones
            .iter()
            .map(|kz| {
                let base_radius = kz.classification.default_kill_zone_radius();
                let growth = kz.radius / base_radius;
                kz.loss_count as f64 * (1.0 + kz.loss_count as f64).ln() * growth
            })
            .sum()
    }

    /// Total number of losses recorded.
    pub fn total_losses(&self) -> usize {
        self.loss_records.len()
    }

    /// Number of active kill zones.
    pub fn active_kill_zones(&self) -> usize {
        self.kill_zones.len()
    }

    /// Check whether a position is inside any kill zone.
    pub fn is_in_kill_zone(&self, pos: &Position) -> bool {
        self.kill_zones
            .iter()
            .any(|kz| kz.center.distance_to(pos) < kz.radius)
    }

    /// Get the evade bias at a given position (sum of all regime adjustments
    /// whose radius covers this position).
    pub fn evade_bias_at(&self, pos: &Position) -> f64 {
        self.regime_adjustments
            .iter()
            .filter(|ra| ra.near.distance_to(pos) < ra.radius)
            .map(|ra| ra.evade_bias)
            .sum()
    }
}

// ────────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_loss(drone_id: u32, x: f64, y: f64, classification: LossClassification) -> LossRecord {
        LossRecord {
            drone_id,
            position: Position::new(x, y, 500.0),
            altitude: 500.0,
            heading: 0.0,
            velocity: [10.0, 0.0, 0.0],
            threat_bearing: Some(1.57),
            regime_at_loss: Regime::Engage,
            classification,
            orphaned_tasks: vec![100 + drone_id],
            timestamp: drone_id as f64 * 10.0,
        }
    }

    #[test]
    fn test_record_loss_creates_kill_zone() {
        let mut analyzer = LossAnalyzer::new();
        let record = make_loss(1, 100.0, 200.0, LossClassification::Sam);
        let orphans = analyzer.record_loss(record);

        assert_eq!(orphans, vec![101]);
        assert_eq!(analyzer.active_kill_zones(), 1);
        assert_eq!(analyzer.total_losses(), 1);
    }

    #[test]
    fn test_kill_zone_merge() {
        let mut analyzer = LossAnalyzer::new().with_merge_distance(600.0);

        // Two losses close together — should merge into one kill zone.
        analyzer.record_loss(make_loss(1, 100.0, 200.0, LossClassification::Sam));
        analyzer.record_loss(make_loss(2, 120.0, 220.0, LossClassification::Sam));

        assert_eq!(analyzer.active_kill_zones(), 1, "should merge into 1 zone");
        assert_eq!(analyzer.kill_zones[0].loss_count, 2);
    }

    #[test]
    fn test_kill_zone_no_merge_when_far() {
        let mut analyzer = LossAnalyzer::new().with_merge_distance(100.0);

        analyzer.record_loss(make_loss(1, 0.0, 0.0, LossClassification::Sam));
        analyzer.record_loss(make_loss(2, 5000.0, 5000.0, LossClassification::SmallArms));

        assert_eq!(analyzer.active_kill_zones(), 2, "should be separate zones");
    }

    #[test]
    fn test_kill_zone_grows_with_losses() {
        let mut analyzer = LossAnalyzer::new()
            .with_merge_distance(600.0)
            .with_growth_factor(1.5);

        let initial_radius = LossClassification::Sam.default_kill_zone_radius();
        analyzer.record_loss(make_loss(1, 100.0, 200.0, LossClassification::Sam));
        assert!((analyzer.kill_zones[0].radius - initial_radius).abs() < 1e-6);

        analyzer.record_loss(make_loss(2, 110.0, 210.0, LossClassification::Sam));
        assert!(
            analyzer.kill_zones[0].radius > initial_radius,
            "radius should grow after second loss: {}",
            analyzer.kill_zones[0].radius,
        );
    }

    #[test]
    fn test_antifragile_score_increases() {
        let mut analyzer = LossAnalyzer::new().with_merge_distance(600.0);

        analyzer.record_loss(make_loss(1, 100.0, 200.0, LossClassification::Sam));
        let score_1 = analyzer.antifragile_score();

        analyzer.record_loss(make_loss(2, 110.0, 210.0, LossClassification::Sam));
        let score_2 = analyzer.antifragile_score();

        assert!(
            score_2 > score_1,
            "antifragile score should increase: {score_1} → {score_2}"
        );
    }

    #[test]
    fn test_is_in_kill_zone() {
        let mut analyzer = LossAnalyzer::new();
        analyzer.record_loss(make_loss(1, 0.0, 0.0, LossClassification::Sam));

        // SAM kill zone has radius 2000m.
        assert!(analyzer.is_in_kill_zone(&Position::new(0.0, 0.0, 500.0)));
        assert!(analyzer.is_in_kill_zone(&Position::new(500.0, 0.0, 500.0)));
        assert!(!analyzer.is_in_kill_zone(&Position::new(5000.0, 5000.0, 500.0)));
    }

    #[test]
    fn test_evade_bias_increases_near_kill_zone() {
        let mut analyzer = LossAnalyzer::new().with_merge_distance(600.0);

        analyzer.record_loss(make_loss(1, 100.0, 200.0, LossClassification::Sam));
        let bias_1 = analyzer.evade_bias_at(&Position::new(100.0, 200.0, 500.0));

        analyzer.record_loss(make_loss(2, 110.0, 210.0, LossClassification::Sam));
        let bias_2 = analyzer.evade_bias_at(&Position::new(100.0, 200.0, 500.0));

        assert!(
            bias_2 > bias_1,
            "evade bias should increase with more losses: {bias_1} → {bias_2}"
        );
    }

    #[test]
    fn test_classify_loss_sam() {
        let cls = LossAnalyzer::classify_loss(Regime::Engage, Some(1.0), 500.0);
        assert_eq!(cls, LossClassification::Sam);
    }

    #[test]
    fn test_classify_loss_small_arms() {
        let cls = LossAnalyzer::classify_loss(Regime::Engage, Some(1.0), 100.0);
        assert_eq!(cls, LossClassification::SmallArms);
    }

    #[test]
    fn test_classify_loss_collision() {
        let cls = LossAnalyzer::classify_loss(Regime::Patrol, None, 500.0);
        assert_eq!(cls, LossClassification::Collision);
    }

    #[test]
    fn test_kill_zone_penalties_format() {
        let mut analyzer = LossAnalyzer::new();
        analyzer.record_loss(make_loss(1, 100.0, 200.0, LossClassification::Sam));
        let penalties = analyzer.kill_zone_penalties();
        assert_eq!(penalties.len(), 1);
        let (center, radius, weight) = &penalties[0];
        assert!((center.x - 100.0).abs() < 1e-6);
        assert!(*radius > 0.0);
        assert!(*weight > 0.0);
    }

    #[test]
    fn test_orphaned_tasks_returned() {
        let mut analyzer = LossAnalyzer::new();
        let record = LossRecord {
            drone_id: 5,
            position: Position::new(0.0, 0.0, 100.0),
            altitude: 100.0,
            heading: 0.0,
            velocity: [0.0; 3],
            threat_bearing: None,
            regime_at_loss: Regime::Patrol,
            classification: LossClassification::Unknown,
            orphaned_tasks: vec![10, 20, 30],
            timestamp: 42.0,
        };
        let orphans = analyzer.record_loss(record);
        assert_eq!(orphans, vec![10, 20, 30]);
    }

    #[test]
    fn test_swarm_learns_from_repeated_losses() {
        // Simulate 5 losses in the same area — the kill zone should expand significantly.
        let mut analyzer = LossAnalyzer::new()
            .with_merge_distance(600.0)
            .with_growth_factor(1.3);

        let base_radius = LossClassification::Sam.default_kill_zone_radius();
        for i in 0..5 {
            let x = 100.0 + i as f64 * 10.0;
            analyzer.record_loss(make_loss(i, x, 200.0, LossClassification::Sam));
        }

        assert_eq!(analyzer.active_kill_zones(), 1);
        let final_radius = analyzer.kill_zones[0].radius;
        // After 4 merges with growth 1.3: base * 1.3^4 ≈ base * 2.86.
        assert!(
            final_radius > base_radius * 2.0,
            "zone should have grown significantly: {final_radius} vs base {base_radius}"
        );

        let score = analyzer.antifragile_score();
        assert!(
            score > 5.0,
            "antifragile score should be substantial: {score}"
        );
    }
}
