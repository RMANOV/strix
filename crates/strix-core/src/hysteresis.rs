//! Regime hysteresis gate — prevents thrashing between regimes.
//!
//! Without hysteresis, noisy CUSUM signals cause rapid
//! PATROL↔ENGAGE↔EVADE oscillations. This gate enforces three
//! temporal constraints before allowing a regime transition:
//!
//! 1. **Min dwell**: must stay in current regime for N seconds
//! 2. **Cooldown**: minimum gap between consecutive flips
//! 3. **Rate limit**: max transitions per sliding window
//!
//! Design: timing-only gates (no signal quality). Signal quality
//! is the intent pipeline's job — separation of concerns.

use crate::state::Regime;

/// Configuration for the hysteresis gate.
#[derive(Debug, Clone)]
pub struct HysteresisConfig {
    /// Minimum time (seconds) a regime must be held before transitioning.
    pub min_dwell_secs: f64,
    /// Cooldown (seconds) after a transition before another is allowed.
    pub cooldown_secs: f64,
    /// Maximum number of transitions allowed in `rate_window_secs`.
    pub max_transitions_per_window: u32,
    /// Sliding window duration (seconds) for rate limiting.
    pub rate_window_secs: f64,
}

impl Default for HysteresisConfig {
    fn default() -> Self {
        Self {
            min_dwell_secs: 2.0,
            cooldown_secs: 1.0,
            max_transitions_per_window: 3,
            rate_window_secs: 10.0,
        }
    }
}

/// Per-drone hysteresis gate state.
#[derive(Debug, Clone)]
pub struct HysteresisGate {
    /// Current confirmed regime.
    current: Regime,
    /// Timestamp when the current regime was entered.
    entered_at: f64,
    /// Timestamps of recent transitions (for rate limiting).
    transition_times: Vec<f64>,
    /// Configuration.
    config: HysteresisConfig,
}

impl HysteresisGate {
    /// Create a new gate starting in the given regime at `now`.
    pub fn new(initial_regime: Regime, now: f64, config: HysteresisConfig) -> Self {
        Self {
            current: initial_regime,
            entered_at: now,
            transition_times: Vec::new(),
            config,
        }
    }

    /// Current confirmed regime.
    pub fn current(&self) -> Regime {
        self.current
    }

    /// Propose a regime transition. Returns the approved regime
    /// (which may be the current regime if the proposal is rejected).
    pub fn propose(&mut self, proposed: Regime, now: f64) -> Regime {
        // Same regime — no transition needed.
        if proposed == self.current {
            return self.current;
        }

        // Gate 1: minimum dwell time.
        let dwell = now - self.entered_at;
        if dwell < self.config.min_dwell_secs {
            return self.current;
        }

        // Gate 2: cooldown since last flip.
        if let Some(&last) = self.transition_times.last() {
            if now - last < self.config.cooldown_secs {
                return self.current;
            }
        }

        // Gate 3: rate limit — count transitions in sliding window.
        let window_start = now - self.config.rate_window_secs;
        let recent_count = self
            .transition_times
            .iter()
            .filter(|&&t| t >= window_start)
            .count() as u32;
        if recent_count >= self.config.max_transitions_per_window {
            return self.current;
        }

        // All gates passed — approve transition.
        self.apply_transition(proposed, now);
        proposed
    }

    /// Force a regime transition, bypassing all gates.
    ///
    /// Used for emergency overrides (e.g., risk level Retreat/Survival).
    /// Does NOT consume a rate-limit slot — emergency transitions must not
    /// deplete the budget for subsequent normal transitions.
    pub fn force_transition(&mut self, regime: Regime, now: f64) {
        if regime != self.current {
            self.current = regime;
            self.entered_at = now;
            // Intentionally skip transition_times.push() — no rate-limit slot consumed.
        }
    }

    /// How long the drone has been in the current regime.
    pub fn dwell_time(&self, now: f64) -> f64 {
        now - self.entered_at
    }

    /// Number of transitions in the current rate window.
    pub fn recent_transition_count(&self, now: f64) -> u32 {
        let window_start = now - self.config.rate_window_secs;
        self.transition_times
            .iter()
            .filter(|&&t| t >= window_start)
            .count() as u32
    }

    fn apply_transition(&mut self, regime: Regime, now: f64) {
        self.current = regime;
        self.entered_at = now;
        self.transition_times.push(now);

        // Prune old entries beyond 2x the rate window to bound memory.
        let cutoff = now - self.config.rate_window_secs * 2.0;
        self.transition_times.retain(|&t| t >= cutoff);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn gate(regime: Regime) -> HysteresisGate {
        HysteresisGate::new(regime, 0.0, HysteresisConfig::default())
    }

    #[test]
    fn same_regime_always_passes() {
        let mut g = gate(Regime::Patrol);
        assert_eq!(g.propose(Regime::Patrol, 0.5), Regime::Patrol);
    }

    #[test]
    fn min_dwell_blocks_early_transition() {
        let mut g = gate(Regime::Patrol);
        // Only 1s has passed, min_dwell is 2s.
        assert_eq!(g.propose(Regime::Engage, 1.0), Regime::Patrol);
    }

    #[test]
    fn min_dwell_allows_after_threshold() {
        let mut g = gate(Regime::Patrol);
        assert_eq!(g.propose(Regime::Engage, 2.5), Regime::Engage);
    }

    #[test]
    fn cooldown_blocks_rapid_flips() {
        let mut g = gate(Regime::Patrol);
        // First transition at t=2.5.
        assert_eq!(g.propose(Regime::Engage, 2.5), Regime::Engage);
        // Try to flip back immediately — blocked by cooldown (1s).
        assert_eq!(g.propose(Regime::Patrol, 2.8), Regime::Engage);
        // After cooldown + dwell.
        assert_eq!(g.propose(Regime::Patrol, 5.0), Regime::Patrol);
    }

    #[test]
    fn rate_limit_caps_transitions() {
        let config = HysteresisConfig {
            min_dwell_secs: 0.1,
            cooldown_secs: 0.1,
            max_transitions_per_window: 3,
            rate_window_secs: 10.0,
        };
        let mut g = HysteresisGate::new(Regime::Patrol, 0.0, config);

        // 3 transitions should be allowed.
        assert_eq!(g.propose(Regime::Engage, 0.5), Regime::Engage);
        assert_eq!(g.propose(Regime::Patrol, 1.0), Regime::Patrol);
        assert_eq!(g.propose(Regime::Evade, 1.5), Regime::Evade);

        // 4th should be blocked.
        assert_eq!(g.propose(Regime::Patrol, 2.0), Regime::Evade);
    }

    #[test]
    fn rate_limit_resets_after_window() {
        let config = HysteresisConfig {
            min_dwell_secs: 0.1,
            cooldown_secs: 0.1,
            max_transitions_per_window: 2,
            rate_window_secs: 5.0,
        };
        let mut g = HysteresisGate::new(Regime::Patrol, 0.0, config);

        assert_eq!(g.propose(Regime::Engage, 0.5), Regime::Engage);
        assert_eq!(g.propose(Regime::Patrol, 1.0), Regime::Patrol);
        // Blocked — 2 transitions in 5s window.
        assert_eq!(g.propose(Regime::Evade, 1.5), Regime::Patrol);
        // After window expires.
        assert_eq!(g.propose(Regime::Evade, 6.0), Regime::Evade);
    }

    #[test]
    fn force_transition_bypasses_gates() {
        let mut g = gate(Regime::Patrol);
        g.force_transition(Regime::Evade, 0.1);
        assert_eq!(g.current(), Regime::Evade);
    }

    #[test]
    fn force_same_regime_is_noop() {
        let mut g = gate(Regime::Patrol);
        g.force_transition(Regime::Patrol, 0.1);
        assert_eq!(g.recent_transition_count(0.1), 0);
    }

    #[test]
    fn dwell_time_tracks_correctly() {
        let mut g = gate(Regime::Patrol);
        assert!((g.dwell_time(5.0) - 5.0).abs() < 1e-12);
        g.force_transition(Regime::Engage, 3.0);
        assert!((g.dwell_time(5.0) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn noisy_proposals_stabilize() {
        // Simulate 200 noisy proposals at 10Hz — should get <= 5 transitions.
        let mut g = gate(Regime::Patrol);
        let regimes = [Regime::Patrol, Regime::Engage, Regime::Evade];

        for i in 0..200 {
            let t = i as f64 * 0.1; // 10 Hz
            let proposed = regimes[i % 3];
            g.propose(proposed, t);
        }
        let transitions = g.recent_transition_count(20.0);
        assert!(
            transitions <= 5,
            "expected <= 5 transitions in 20s, got {transitions}"
        );
    }
}
