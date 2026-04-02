use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CriticalitySignals {
    pub gossip_convergence: f64,
    pub uncertainty: f64,
    pub dispersion: f64,
    pub consensus_collapse: f64,
    pub fear: f64,
    pub threat_pressure: f64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CriticalityConfig {
    pub target_edge: f64,
    pub gain: f64,
    pub min_exploration_noise: f64,
    pub max_exploration_noise: f64,
    pub min_pheromone_decay_multiplier: f64,
    pub max_pheromone_decay_multiplier: f64,
    pub min_bid_aggression: f64,
    pub max_bid_aggression: f64,
}

impl Default for CriticalityConfig {
    fn default() -> Self {
        Self {
            target_edge: 0.55,
            gain: 0.45,
            min_exploration_noise: 0.75,
            max_exploration_noise: 1.35,
            min_pheromone_decay_multiplier: 0.70,
            max_pheromone_decay_multiplier: 1.35,
            min_bid_aggression: 0.45,
            max_bid_aggression: 1.15,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CriticalityAdjustment {
    pub criticality: f64,
    pub exploration_noise: f64,
    pub pheromone_decay_multiplier: f64,
    pub bid_aggression: f64,
}

pub struct CriticalityScheduler {
    config: CriticalityConfig,
    last: CriticalityAdjustment,
}

impl CriticalityScheduler {
    pub fn new(config: CriticalityConfig) -> Self {
        Self {
            config,
            last: CriticalityAdjustment {
                criticality: config.target_edge,
                exploration_noise: 1.0,
                pheromone_decay_multiplier: 1.0,
                bid_aggression: 1.0,
            },
        }
    }

    pub fn last(&self) -> CriticalityAdjustment {
        self.last
    }

    pub fn evaluate(&mut self, signals: CriticalitySignals) -> CriticalityAdjustment {
        let order = clamp01(
            0.40 * clamp01(signals.gossip_convergence)
                + 0.20 * (1.0 - clamp01(signals.dispersion))
                + 0.20 * (1.0 - clamp01(signals.consensus_collapse))
                + 0.20 * (1.0 - clamp01(signals.fear)),
        );
        let disorder = clamp01(
            0.35 * clamp01(signals.uncertainty)
                + 0.25 * clamp01(signals.consensus_collapse)
                + 0.20 * clamp01(signals.dispersion)
                + 0.20 * clamp01(signals.threat_pressure),
        );
        let edge = clamp01(1.0 - (order - disorder).abs());
        let imbalance = edge - self.config.target_edge;
        let control = imbalance * self.config.gain;

        let exploration_noise = interpolate(
            self.config.min_exploration_noise,
            self.config.max_exploration_noise,
            clamp01(0.5 - control),
        );
        let pheromone_decay_multiplier = interpolate(
            self.config.min_pheromone_decay_multiplier,
            self.config.max_pheromone_decay_multiplier,
            clamp01(0.5 + control),
        );
        let bid_aggression = interpolate(
            self.config.min_bid_aggression,
            self.config.max_bid_aggression,
            clamp01(0.5 + 0.35 * signals.threat_pressure - 0.25 * signals.consensus_collapse + 0.15 * edge),
        );

        self.last = CriticalityAdjustment {
            criticality: edge,
            exploration_noise,
            pheromone_decay_multiplier,
            bid_aggression,
        };
        self.last
    }
}

fn clamp01(value: f64) -> f64 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn interpolate(low: f64, high: f64, alpha: f64) -> f64 {
    let alpha = clamp01(alpha);
    low + (high - low) * alpha
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordered_state_pushes_more_exploration() {
        let mut scheduler = CriticalityScheduler::new(CriticalityConfig::default());
        let ordered = scheduler.evaluate(CriticalitySignals {
            gossip_convergence: 0.95,
            uncertainty: 0.10,
            dispersion: 0.10,
            consensus_collapse: 0.05,
            fear: 0.10,
            threat_pressure: 0.20,
        });
        let disordered = scheduler.evaluate(CriticalitySignals {
            gossip_convergence: 0.30,
            uncertainty: 0.85,
            dispersion: 0.80,
            consensus_collapse: 0.75,
            fear: 0.60,
            threat_pressure: 0.80,
        });
        assert!(ordered.exploration_noise > disordered.exploration_noise);
        assert!(ordered.pheromone_decay_multiplier > disordered.pheromone_decay_multiplier);
    }

    #[test]
    fn threat_pressure_increases_bid_aggression() {
        let mut scheduler = CriticalityScheduler::new(CriticalityConfig::default());
        let calm = scheduler.evaluate(CriticalitySignals {
            gossip_convergence: 0.6,
            uncertainty: 0.3,
            dispersion: 0.3,
            consensus_collapse: 0.2,
            fear: 0.2,
            threat_pressure: 0.1,
        });
        let hot = scheduler.evaluate(CriticalitySignals {
            gossip_convergence: 0.6,
            uncertainty: 0.3,
            dispersion: 0.3,
            consensus_collapse: 0.2,
            fear: 0.2,
            threat_pressure: 0.9,
        });
        assert!(hot.bid_aggression > calm.bid_aggression);
    }
}
