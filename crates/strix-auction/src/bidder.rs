//! Drone Bidding Engine — sealed-bid task evaluation.
//!
//! Each drone independently evaluates its fitness for a set of tasks and submits
//! sealed bids. Drones never see each other's bids — the [`Auctioneer`](crate::auctioneer::Auctioneer)
//! resolves assignments.
//!
//! ## Scoring Function
//!
//! ```text
//! total = urgency*10 + capability_match*3 + proximity*5 + energy*2 - risk*4
//! ```

use serde::{Deserialize, Serialize};

use crate::{Capabilities, DroneState, Position, Task, ThreatState};

// ────────────────────────────────────────────────────────────────────────────────
// Types
// ────────────────────────────────────────────────────────────────────────────────

/// Individual score components that make up a bid.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BidComponents {
    /// Inverse-distance to the task location (higher = closer).
    pub proximity: f64,
    /// Sensor/weapon match score [0, 1].
    pub capability: f64,
    /// Remaining battery/fuel [0, 1].
    pub energy: f64,
    /// Current threat level at the drone's position [0, 1].
    pub risk_exposure: f64,
    /// Bonus for time-sensitive tasks.
    pub urgency_bonus: f64,
}

/// A sealed bid from one drone for one task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bid {
    pub drone_id: u32,
    pub task_id: u32,
    /// Composite bid score.
    pub score: f64,
    /// Breakdown of the score.
    pub components: BidComponents,
}

/// Bidding engine for a single drone.
#[derive(Debug, Clone)]
pub struct Bidder {
    /// The drone whose bids we are computing.
    pub drone: DroneState,
    /// Optional sub-swarm membership (for dark pool filtering).
    pub sub_swarm_id: Option<u32>,
    /// Kill-zone penalty map: `(center, radius, penalty_weight)`.
    /// Populated by the antifragile module after losses.
    pub kill_zone_penalties: Vec<(Position, f64, f64)>,
    /// Fear level F ∈ [0,1] — modulates risk aversion in bid scoring.
    pub fear: f64,
    /// Optional phi-sim scenario context for scenario-enriched bidding.
    pub scenario_context: Option<ScenarioContext>,
}

/// Per-drone scenario data from phi-sim (doom/upside/confidence).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioContext {
    /// Doom-case expected value (typically negative).
    pub doom_value: f64,
    /// Upside-case expected value (typically positive).
    pub upside_value: f64,
    /// Decision confidence [0, 1].
    pub confidence: f64,
}

// ────────────────────────────────────────────────────────────────────────────────
// Implementation
// ────────────────────────────────────────────────────────────────────────────────

impl Bidder {
    /// Create a new bidder for the given drone.
    pub fn new(drone: DroneState) -> Self {
        Self {
            drone,
            sub_swarm_id: None,
            kill_zone_penalties: Vec::new(),
            fear: 0.0,
            scenario_context: None,
        }
    }

    /// Create a bidder with sub-swarm membership (enables dark-pool task visibility).
    pub fn with_sub_swarm(mut self, sub_swarm_id: u32) -> Self {
        self.sub_swarm_id = Some(sub_swarm_id);
        self
    }

    /// Add kill-zone penalties (from antifragile loss analysis).
    pub fn with_kill_zone_penalties(mut self, penalties: Vec<(Position, f64, f64)>) -> Self {
        self.kill_zone_penalties = penalties;
        self
    }

    /// Set fear level for risk-adjusted bidding.
    pub fn with_fear(mut self, fear: f64) -> Self {
        self.fear = fear;
        self
    }

    /// Set scenario context for scenario-enriched bid scoring.
    pub fn with_scenario_context(mut self, ctx: ScenarioContext) -> Self {
        self.scenario_context = Some(ctx);
        self
    }

    /// Evaluate a single task and produce a sealed [`Bid`].
    ///
    /// Returns `None` if the drone is ineligible (dead, dark-pool filtered, etc.).
    pub fn evaluate_task(&self, task: &Task, threats: &[ThreatState]) -> Option<Bid> {
        // Dead drones don't bid.
        if !self.drone.alive {
            return None;
        }

        // Dark pool filtering: if the task is restricted to a sub-swarm the drone
        // doesn't belong to, skip it.
        if let Some(required_sub_swarm) = task.dark_pool {
            if self.sub_swarm_id != Some(required_sub_swarm) {
                return None;
            }
        }

        let components = self.compute_components(task, threats);
        let score = if let Some(ref ctx) = self.scenario_context {
            calculate_bid_with_scenarios(
                &components,
                self.fear,
                ctx.doom_value,
                ctx.upside_value,
                ctx.confidence,
            )
        } else {
            calculate_bid_with_fear(&components, self.fear)
        };

        Some(Bid {
            drone_id: self.drone.id,
            task_id: task.id,
            score,
            components,
        })
    }

    /// Evaluate all eligible tasks and return sealed bids (sorted by score descending).
    pub fn bid_on_tasks(&self, tasks: &[Task], threats: &[ThreatState]) -> Vec<Bid> {
        let mut bids: Vec<Bid> = tasks
            .iter()
            .filter_map(|t| self.evaluate_task(t, threats))
            .collect();
        bids.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        bids
    }

    // ── Private helpers ─────────────────────────────────────────────────────

    fn compute_components(&self, task: &Task, threats: &[ThreatState]) -> BidComponents {
        let distance = self.drone.position.distance_to(&task.location);
        let proximity = if distance > 0.0 { 1.0 / distance } else { 1.0 };

        let capability = capability_match(&self.drone.capabilities, &task.required_capabilities);
        let energy = self.drone.energy.clamp(0.0, 1.0);
        let risk_exposure = compute_risk_exposure(&self.drone.position, threats);
        let urgency_bonus = task.urgency * task.priority;

        // Apply kill-zone penalty: increase perceived risk near known kill zones.
        let kill_zone_risk = self.kill_zone_risk(&task.location);

        BidComponents {
            proximity,
            capability,
            energy,
            risk_exposure: (risk_exposure + kill_zone_risk).clamp(0.0, 1.0),
            urgency_bonus,
        }
    }

    /// Compute extra risk contribution from proximity to known kill zones.
    fn kill_zone_risk(&self, task_location: &Position) -> f64 {
        self.kill_zone_penalties
            .iter()
            .map(|(center, radius, weight)| {
                let dist = task_location.distance_to(center);
                if dist < *radius {
                    // Inside the kill zone — full penalty scaled by weight.
                    *weight * (1.0 - dist / radius)
                } else {
                    0.0
                }
            })
            .sum()
    }
}

/// Composite bid scoring function (fear-neutral, F=0).
///
/// ```text
/// total = urgency*10 + capability*3 + proximity*5 + energy*2 - risk*4
/// ```
pub fn calculate_bid(c: &BidComponents) -> f64 {
    calculate_bid_with_fear(c, 0.0)
}

/// Fear-modulated bid scoring.
///
/// Higher fear → risk weight increases (4→7), proximity bonus decreases (5→2.5).
/// Captures the trading intuition: scared traders demand more risk premium
/// and value distance-to-safety over proximity-to-task.
pub fn calculate_bid_with_fear(c: &BidComponents, fear: f64) -> f64 {
    let f = if fear.is_nan() || fear.is_infinite() {
        0.0
    } else {
        fear.clamp(0.0, 1.0)
    };
    let risk_weight = 4.0 + f * 3.0; // 4→7
    let proximity_weight = 5.0 - f * 2.5; // 5→2.5
    c.urgency_bonus * 10.0 + c.capability * 3.0 + c.proximity * proximity_weight + c.energy * 2.0
        - c.risk_exposure * risk_weight
}

/// Scenario-enriched bid scoring.
///
/// When phi-sim scenarios are available, incorporate doom/upside/confidence:
/// - doom_value (negative) → additional risk penalty
/// - upside_value (positive) → opportunity bonus
/// - confidence → score multiplier
pub fn calculate_bid_with_scenarios(
    c: &BidComponents,
    fear: f64,
    doom_value: f64,
    upside_value: f64,
    confidence: f64,
) -> f64 {
    let base = calculate_bid_with_fear(c, fear);
    let scenario_adjust = upside_value * 0.3 + doom_value * fear.clamp(0.0, 1.0) * 0.5;
    base * confidence.clamp(0.3, 1.0) + scenario_adjust
}

/// Compute capability match as fraction of required capabilities that the drone has.
///
/// Returns a value in [0, 1]. A drone that meets ALL requirements scores 1.0.
pub fn capability_match(drone_caps: &Capabilities, required: &Capabilities) -> f64 {
    let mut total = 0u32;
    let mut matched = 0u32;

    let checks: &[(bool, bool)] = &[
        (required.has_sensor, drone_caps.has_sensor),
        (required.has_weapon, drone_caps.has_weapon),
        (required.has_ew, drone_caps.has_ew),
        (required.has_relay, drone_caps.has_relay),
    ];

    for &(req, has) in checks {
        if req {
            total += 1;
            if has {
                matched += 1;
            }
        }
    }

    if total == 0 {
        1.0 // no requirements — any drone qualifies
    } else {
        matched as f64 / total as f64
    }
}

/// Compute the aggregated threat exposure at a given position.
///
/// Returns a value in [0, 1] where 1 means the position is inside lethal range of many threats.
fn compute_risk_exposure(pos: &Position, threats: &[ThreatState]) -> f64 {
    if threats.is_empty() {
        return 0.0;
    }

    let total_risk: f64 = threats
        .iter()
        .map(|t| {
            let dist = pos.distance_to(&t.position);
            if dist <= 0.0 {
                1.0
            } else if dist < t.lethal_radius {
                1.0 - (dist / t.lethal_radius)
            } else {
                0.0
            }
        })
        .sum();

    // Saturate at 1.0.
    total_risk.min(1.0)
}

// ────────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Capabilities, DroneState, Position, Regime, Task, ThreatState, ThreatType};

    fn sample_drone(id: u32, x: f64, y: f64) -> DroneState {
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
            energy: 0.8,
            alive: true,
        }
    }

    fn sample_task(id: u32, x: f64, y: f64) -> Task {
        Task {
            id,
            location: Position::new(x, y, 50.0),
            required_capabilities: Capabilities {
                has_sensor: true,
                has_weapon: false,
                has_ew: false,
                has_relay: false,
            },
            priority: 0.7,
            urgency: 0.5,
            bundle_id: None,
            dark_pool: None,
        }
    }

    #[test]
    fn test_calculate_bid_deterministic() {
        let c = BidComponents {
            proximity: 0.1,
            capability: 1.0,
            energy: 0.8,
            risk_exposure: 0.2,
            urgency_bonus: 0.5,
        };
        let expected = 0.5 * 10.0 + 1.0 * 3.0 + 0.1 * 5.0 + 0.8 * 2.0 - 0.2 * 4.0;
        let result = calculate_bid(&c);
        assert!(
            (result - expected).abs() < 1e-12,
            "got {result}, expected {expected}"
        );
    }

    #[test]
    fn test_capability_match_full() {
        let d = Capabilities {
            has_sensor: true,
            has_weapon: true,
            has_ew: true,
            has_relay: true,
        };
        let r = Capabilities {
            has_sensor: true,
            has_weapon: true,
            has_ew: false,
            has_relay: false,
        };
        assert!((capability_match(&d, &r) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_capability_match_partial() {
        let d = Capabilities::default(); // sensor only
        let r = Capabilities {
            has_sensor: true,
            has_weapon: true,
            has_ew: false,
            has_relay: false,
        };
        assert!((capability_match(&d, &r) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_capability_match_no_requirements() {
        let d = Capabilities::default();
        let r = Capabilities {
            has_sensor: false,
            has_weapon: false,
            has_ew: false,
            has_relay: false,
        };
        assert!((capability_match(&d, &r) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_dead_drone_no_bid() {
        let mut drone = sample_drone(1, 0.0, 0.0);
        drone.alive = false;
        let bidder = Bidder::new(drone);
        let task = sample_task(1, 10.0, 10.0);
        assert!(bidder.evaluate_task(&task, &[]).is_none());
    }

    #[test]
    fn test_dark_pool_filtering() {
        let drone = sample_drone(1, 0.0, 0.0);
        let bidder = Bidder::new(drone).with_sub_swarm(42);
        let mut task = sample_task(1, 10.0, 10.0);

        // Task restricted to sub-swarm 42 — bidder belongs, should get a bid.
        task.dark_pool = Some(42);
        assert!(bidder.evaluate_task(&task, &[]).is_some());

        // Task restricted to sub-swarm 99 — bidder excluded.
        task.dark_pool = Some(99);
        assert!(bidder.evaluate_task(&task, &[]).is_none());
    }

    #[test]
    fn test_closer_drone_scores_higher_proximity() {
        let near = sample_drone(1, 5.0, 5.0);
        let far = sample_drone(2, 500.0, 500.0);
        let task = sample_task(1, 10.0, 10.0);

        let b_near = Bidder::new(near).evaluate_task(&task, &[]).unwrap();
        let b_far = Bidder::new(far).evaluate_task(&task, &[]).unwrap();

        assert!(
            b_near.components.proximity > b_far.components.proximity,
            "near proximity {} should exceed far {}",
            b_near.components.proximity,
            b_far.components.proximity,
        );
    }

    #[test]
    fn test_threat_increases_risk() {
        let drone = sample_drone(1, 10.0, 10.0);
        let task = sample_task(1, 15.0, 15.0);

        let threat = ThreatState {
            id: 1,
            position: Position::new(10.0, 10.0, 100.0),
            lethal_radius: 500.0,
            threat_type: ThreatType::Sam,
        };

        let bid_safe = Bidder::new(drone.clone())
            .evaluate_task(&task, &[])
            .unwrap();
        let bid_risky = Bidder::new(drone).evaluate_task(&task, &[threat]).unwrap();

        assert!(
            bid_risky.components.risk_exposure > bid_safe.components.risk_exposure,
            "risk with threat {} should exceed risk without {}",
            bid_risky.components.risk_exposure,
            bid_safe.components.risk_exposure,
        );
    }

    #[test]
    fn test_kill_zone_penalty() {
        let drone = sample_drone(1, 0.0, 0.0);
        let task = sample_task(1, 5.0, 5.0);
        let kill_zone = (Position::new(5.0, 5.0, 50.0), 100.0, 0.5);

        let bidder_clean = Bidder::new(drone.clone());
        let bidder_kz = Bidder::new(drone).with_kill_zone_penalties(vec![kill_zone]);

        let bid_clean = bidder_clean.evaluate_task(&task, &[]).unwrap();
        let bid_kz = bidder_kz.evaluate_task(&task, &[]).unwrap();

        assert!(
            bid_kz.score < bid_clean.score,
            "bid near kill zone ({}) should be lower than clean bid ({})",
            bid_kz.score,
            bid_clean.score,
        );
    }

    #[test]
    fn test_bid_on_multiple_tasks_sorted() {
        let drone = sample_drone(1, 0.0, 0.0);
        let tasks = vec![
            sample_task(1, 100.0, 100.0),
            sample_task(2, 10.0, 10.0),
            sample_task(3, 50.0, 50.0),
        ];
        let bids = Bidder::new(drone).bid_on_tasks(&tasks, &[]);
        assert_eq!(bids.len(), 3);
        // Verify descending order.
        for w in bids.windows(2) {
            assert!(w[0].score >= w[1].score);
        }
    }

    #[test]
    fn test_fear_modulated_bid_scoring() {
        let c = BidComponents {
            proximity: 0.1,
            capability: 1.0,
            energy: 0.8,
            risk_exposure: 0.2,
            urgency_bonus: 0.5,
        };
        // At F=0, should equal calculate_bid
        let score_f0 = calculate_bid_with_fear(&c, 0.0);
        let score_base = calculate_bid(&c);
        assert!((score_f0 - score_base).abs() < 1e-12);

        // At F=1: risk_weight=7, proximity_weight=2.5
        let score_f1 = calculate_bid_with_fear(&c, 1.0);
        let expected_f1 = 0.5 * 10.0 + 1.0 * 3.0 + 0.1 * 2.5 + 0.8 * 2.0 - 0.2 * 7.0;
        assert!((score_f1 - expected_f1).abs() < 1e-12);

        // Fear should reduce score when there's risk exposure
        assert!(
            score_f1 < score_f0,
            "higher fear should reduce score when risk > 0"
        );
    }

    #[test]
    fn test_calculate_bid_with_scenarios_no_doom() {
        let c = BidComponents {
            proximity: 0.1,
            capability: 1.0,
            energy: 0.8,
            risk_exposure: 0.2,
            urgency_bonus: 0.5,
        };
        let base = calculate_bid_with_fear(&c, 0.3);
        let scenario = calculate_bid_with_scenarios(&c, 0.3, 0.0, 0.0, 1.0);
        assert!(
            (scenario - base).abs() < 1e-9,
            "zero doom/upside with confidence=1 should equal base"
        );
    }

    #[test]
    fn test_calculate_bid_with_scenarios_doom_reduces() {
        let c = BidComponents {
            proximity: 0.1,
            capability: 1.0,
            energy: 0.8,
            risk_exposure: 0.2,
            urgency_bonus: 0.5,
        };
        let no_doom = calculate_bid_with_scenarios(&c, 0.5, 0.0, 0.0, 1.0);
        let with_doom = calculate_bid_with_scenarios(&c, 0.5, -2.0, 0.0, 1.0);
        assert!(with_doom < no_doom, "doom should reduce bid score");
    }

    #[test]
    fn test_calculate_bid_with_scenarios_upside_increases() {
        let c = BidComponents {
            proximity: 0.1,
            capability: 1.0,
            energy: 0.8,
            risk_exposure: 0.2,
            urgency_bonus: 0.5,
        };
        let no_upside = calculate_bid_with_scenarios(&c, 0.5, 0.0, 0.0, 1.0);
        let with_upside = calculate_bid_with_scenarios(&c, 0.5, 0.0, 2.0, 1.0);
        assert!(with_upside > no_upside, "upside should increase bid score");
    }

    #[test]
    fn test_calculate_bid_with_scenarios_low_confidence_dampens() {
        let c = BidComponents {
            proximity: 0.1,
            capability: 1.0,
            energy: 0.8,
            risk_exposure: 0.0,
            urgency_bonus: 0.5,
        };
        let high_conf = calculate_bid_with_scenarios(&c, 0.0, 0.0, 0.0, 1.0);
        let low_conf = calculate_bid_with_scenarios(&c, 0.0, 0.0, 0.0, 0.3);
        assert!(
            low_conf < high_conf,
            "low confidence should dampen bid score"
        );
    }

    #[test]
    fn test_bidder_with_scenario_context() {
        let drone = sample_drone(1, 0.0, 0.0);
        let task = sample_task(1, 10.0, 10.0);
        let ctx = ScenarioContext {
            doom_value: -1.0,
            upside_value: 0.5,
            confidence: 0.8,
        };
        let bidder = Bidder::new(drone).with_scenario_context(ctx);
        let bid = bidder.evaluate_task(&task, &[]).unwrap();
        assert!(
            bid.score > 0.0,
            "bid with scenario context should produce positive score"
        );
    }

    // ── NaN/Inf fear safety guards ────────────────────────────────────────

    #[test]
    fn test_nan_fear_bid_equals_zero_fear() {
        let c = BidComponents {
            proximity: 0.5,
            capability: 1.0,
            energy: 0.8,
            risk_exposure: 0.3,
            urgency_bonus: 0.5,
        };
        let score_nan = calculate_bid_with_fear(&c, f64::NAN);
        let score_zero = calculate_bid_with_fear(&c, 0.0);
        assert!(
            (score_nan - score_zero).abs() < 1e-12,
            "NaN fear should be treated as F=0: nan={score_nan}, zero={score_zero}"
        );
        assert!(
            score_nan.is_finite(),
            "NaN fear must produce finite bid score"
        );
    }

    #[test]
    fn test_inf_fear_bid_equals_f1_bid() {
        let c = BidComponents {
            proximity: 0.5,
            capability: 1.0,
            energy: 0.8,
            risk_exposure: 0.3,
            urgency_bonus: 0.5,
        };
        let score_inf = calculate_bid_with_fear(&c, f64::INFINITY);
        let score_zero = calculate_bid_with_fear(&c, 0.0);
        assert!(
            score_inf.is_finite(),
            "Inf fear must produce finite bid score: got {score_inf}"
        );
        // Inf treated as 0.0 — same as F=0
        assert!(
            (score_inf - score_zero).abs() < 1e-12,
            "Inf fear should be treated as F=0: inf={score_inf}, zero={score_zero}"
        );
    }
}
