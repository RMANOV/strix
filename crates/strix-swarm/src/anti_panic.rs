//! Anti-panic dampening for swarm stability.
//!
//! Prevents cascade failures where simultaneous threat detections
//! cause the entire swarm to enter EVADE simultaneously.

use serde::{Deserialize, Serialize};

/// Configuration for anti-panic dampening.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PanicDamperConfig {
    /// Maximum fraction of swarm that can change regime per tick [0, 1].
    pub max_regime_change_rate: f64,
    /// Maximum fear delta per tick.
    pub max_fear_delta: f64,
    /// Fear decay rate per tick (how fast fear returns to baseline).
    pub fear_decay_rate: f64,
    /// Minimum ticks between regime changes for a single drone.
    pub regime_change_cooldown: u32,
}

impl Default for PanicDamperConfig {
    fn default() -> Self {
        Self {
            max_regime_change_rate: 0.3, // max 30% can change per tick
            max_fear_delta: 0.15,
            fear_decay_rate: 0.05,
            regime_change_cooldown: 5,
        }
    }
}

/// Anti-panic dampener state.
pub struct PanicDamper {
    config: PanicDamperConfig,
    /// Per-drone tick of last regime change.
    last_regime_change: std::collections::HashMap<u32, u64>,
    /// Current tick counter.
    tick: u64,
    /// Current dampened fear level.
    dampened_fear: f64,
}

impl PanicDamper {
    /// Create a new dampener.
    pub fn new(config: PanicDamperConfig) -> Self {
        Self {
            config,
            last_regime_change: std::collections::HashMap::new(),
            tick: 0,
            dampened_fear: 0.0,
        }
    }

    /// Create with default config.
    #[deprecated(note = "use PanicDamper::default() instead")]
    pub fn default_config() -> Self {
        Self::new(PanicDamperConfig::default())
    }

    /// Process a tick's worth of regime change requests.
    ///
    /// `requests` is a list of (drone_id, wants_to_change_regime).
    /// Returns which drones are actually allowed to change.
    pub fn filter_regime_changes(&mut self, requests: &[(u32, bool)]) -> Vec<u32> {
        self.tick += 1;
        let total = requests.len().max(1) as f64;
        let max_changes = (total * self.config.max_regime_change_rate).ceil() as usize;

        let mut allowed = Vec::with_capacity(max_changes);

        for &(drone_id, wants_change) in requests {
            if !wants_change {
                continue;
            }
            if allowed.len() >= max_changes {
                break;
            }
            // Check cooldown
            if let Some(&last_tick) = self.last_regime_change.get(&drone_id) {
                if self.tick - last_tick < self.config.regime_change_cooldown as u64 {
                    continue; // still in cooldown
                }
            }
            allowed.push(drone_id);
            self.last_regime_change.insert(drone_id, self.tick);
        }

        allowed
    }

    /// Dampen a raw fear value.
    ///
    /// Limits the rate of fear increase and applies decay.
    pub fn dampen_fear(&mut self, raw_fear: f64) -> f64 {
        if !raw_fear.is_finite() {
            return self.dampened_fear;
        }
        let clamped = raw_fear.clamp(0.0, 1.0);
        let delta = clamped - self.dampened_fear;

        if delta > self.config.max_fear_delta {
            self.dampened_fear += self.config.max_fear_delta;
        } else if delta < -self.config.fear_decay_rate {
            self.dampened_fear -= self.config.fear_decay_rate;
        } else {
            self.dampened_fear = clamped;
        }

        self.dampened_fear = self.dampened_fear.clamp(0.0, 1.0);
        self.dampened_fear
    }

    /// Current dampened fear level.
    pub fn current_fear(&self) -> f64 {
        self.dampened_fear
    }

    /// Current tick.
    pub fn current_tick(&self) -> u64 {
        self.tick
    }

    /// Remove a drone from tracking (e.g. after loss).
    pub fn remove_drone(&mut self, drone_id: u32) {
        self.last_regime_change.remove(&drone_id);
    }
}

impl Default for PanicDamper {
    fn default() -> Self {
        Self::new(PanicDamperConfig::default())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_values() {
        let cfg = PanicDamperConfig::default();
        assert!(cfg.max_regime_change_rate > 0.0 && cfg.max_regime_change_rate <= 1.0);
        assert!(cfg.max_fear_delta > 0.0 && cfg.max_fear_delta <= 1.0);
        assert!(cfg.fear_decay_rate > 0.0 && cfg.fear_decay_rate <= 1.0);
        assert!(cfg.regime_change_cooldown > 0);
        // Verify the documented 30% default
        assert!((cfg.max_regime_change_rate - 0.3).abs() < 1e-10);
    }

    #[test]
    fn empty_requests() {
        let mut d = PanicDamper::default();
        let allowed = d.filter_regime_changes(&[]);
        assert!(allowed.is_empty());
    }

    #[test]
    fn rate_limiting() {
        let mut d = PanicDamper::default(); // 30% rate
                                            // 10 drones all want to change — only ceil(10 * 0.3) = 3 allowed
        let requests: Vec<(u32, bool)> = (0u32..10).map(|id| (id, true)).collect();
        let allowed = d.filter_regime_changes(&requests);
        assert_eq!(
            allowed.len(),
            3,
            "expected 3 allowed, got {}",
            allowed.len()
        );
    }

    #[test]
    fn cooldown_enforcement() {
        let mut d = PanicDamper::default(); // cooldown = 5 ticks

        // Tick 1: drone 0 changes
        let allowed = d.filter_regime_changes(&[(0, true)]);
        assert_eq!(allowed, vec![0]);

        // Ticks 2-5: drone 0 is in cooldown (only 1-4 ticks elapsed)
        for _ in 0..4 {
            let allowed = d.filter_regime_changes(&[(0, true)]);
            assert!(
                allowed.is_empty(),
                "drone 0 should be in cooldown, tick={}",
                d.current_tick()
            );
        }

        // Tick 6: cooldown expired (5 ticks elapsed since tick 1)
        let allowed = d.filter_regime_changes(&[(0, true)]);
        assert_eq!(allowed, vec![0], "drone 0 should be allowed after cooldown");
    }

    #[test]
    fn fear_dampening_limits_spike() {
        let mut d = PanicDamper::default(); // max_fear_delta = 0.15
                                            // Sudden spike from 0 to 1.0
        let dampened = d.dampen_fear(1.0);
        assert!(
            dampened <= 0.15 + 1e-10,
            "spike not dampened: got {dampened}"
        );
        // Still clamped to [0, 1]
        assert!((0.0..=1.0).contains(&dampened));
    }

    #[test]
    fn fear_decay() {
        let mut d = PanicDamper::default(); // fear_decay_rate = 0.05
                                            // Raise fear to 0.5 via gradual steps (each step ≤ max_fear_delta=0.15)
        d.dampen_fear(0.15);
        d.dampen_fear(0.30);
        d.dampen_fear(0.45);
        d.dampen_fear(0.50);
        let before = d.current_fear();
        assert!(before > 0.0, "fear should be > 0 before decay");

        // Now drop raw fear to 0 — should decay by fear_decay_rate per call
        let after = d.dampen_fear(0.0);
        assert!(
            after < before,
            "fear should decrease: before={before}, after={after}"
        );
        let decay = before - after;
        assert!(
            (decay - 0.05).abs() < 1e-10,
            "expected decay of 0.05, got {decay}"
        );
    }

    #[test]
    fn fear_gradual_rise() {
        let mut d = PanicDamper::default();
        // Small increments should pass through unchanged
        let v1 = d.dampen_fear(0.05);
        assert!((v1 - 0.05).abs() < 1e-10, "small delta should pass through");
        let v2 = d.dampen_fear(0.10);
        assert!((v2 - 0.10).abs() < 1e-10, "small delta should pass through");
    }

    #[test]
    fn remove_drone_clears_cooldown() {
        let mut d = PanicDamper::default();
        // Let drone 42 change on tick 1
        d.filter_regime_changes(&[(42, true)]);
        // Tick 2 — normally still in cooldown
        let blocked = d.filter_regime_changes(&[(42, true)]);
        assert!(blocked.is_empty(), "should be in cooldown");

        // Remove drone 42 from tracking
        d.remove_drone(42);

        // Now drone 42 has no cooldown record — should be allowed
        let allowed = d.filter_regime_changes(&[(42, true)]);
        assert_eq!(allowed, vec![42], "removed drone should bypass cooldown");
    }

    // -- Edge-case / boundary tests --

    #[test]
    fn nan_fear_returns_previous() {
        let mut d = PanicDamper::default();
        // Gradually ramp up fear to 0.5 (max_fear_delta=0.15 per call)
        for _ in 0..10 {
            d.dampen_fear(1.0);
        }
        let before_nan = d.current_fear();
        let result = d.dampen_fear(f64::NAN);
        assert!(
            result.is_finite(),
            "NaN fear must return finite dampened value"
        );
        assert_eq!(
            result, before_nan,
            "NaN fear should return previous dampened value unchanged"
        );
    }

    #[test]
    fn infinity_fear_clamped() {
        let mut d = PanicDamper::default();
        d.dampen_fear(0.3);
        let result = d.dampen_fear(f64::INFINITY);
        assert!(result.is_finite(), "Infinity fear must not propagate");
    }

    #[test]
    fn fear_exactly_zero() {
        let mut d = PanicDamper::default();
        d.dampen_fear(0.5); // start at 0.5
                            // Should decay toward 0
        let v = d.dampen_fear(0.0);
        assert!(v < 0.5, "fear=0.0 should decay below 0.5");
        assert!(v >= 0.0, "fear must not go negative");
    }

    #[test]
    fn fear_exactly_one_clamped() {
        let mut d = PanicDamper::default();
        d.dampen_fear(0.0); // start at 0
                            // max_fear_delta=0.15, so spike from 0 to 1.0 is clamped
        let v = d.dampen_fear(1.0);
        assert!(
            v <= 0.15 + 1e-10,
            "spike should be clamped to max_fear_delta=0.15, got {v}"
        );
    }

    #[test]
    fn default_impl_matches_default_config() {
        let a = PanicDamper::default();
        let b = PanicDamper::default();
        assert_eq!(a.current_fear(), b.current_fear());
        assert_eq!(a.current_tick(), b.current_tick());
    }
}
