//! Scenario runner and doctrine-aware objective extraction for the STRIX optimizer.
//!
//! `Evaluator::evaluate()` runs a battery of Playground scenarios with a given
//! `SwarmConfig` (derived from `ParamVec`) and returns a weighted-average
//! three-objective vector:
//!
//!   F0 = survival depth     (maximize) — survivability plus safety margin
//!   F1 = mission continuity (maximize) — stability under disruption
//!   F2 = reserve efficiency (maximize) — energy and coordination discipline
//!
//! The exact blend remains doctrine-aware so the same optimizer can be aimed at
//! survivability-first, persistent ISR, or comms-denied operating concepts.

use std::str::FromStr;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use strix_playground::{BattleReport, Playground};

use crate::param_space::{ParamSpace, ParamVec};

// ---------------------------------------------------------------------------
// Doctrine profile
// ---------------------------------------------------------------------------

/// Doctrine-level scoring profile for optimization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum DoctrineProfile {
    #[default]
    Balanced,
    SurvivalFirst,
    PersistentIsr,
    CommunicationsDenied,
    AggressiveStrike,
}

impl DoctrineProfile {
    /// Canonical snake_case name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Balanced => "balanced",
            Self::SurvivalFirst => "survival_first",
            Self::PersistentIsr => "persistent_isr",
            Self::CommunicationsDenied => "communications_denied",
            Self::AggressiveStrike => "aggressive_strike",
        }
    }

    /// Human-readable labels for the three doctrine-weighted objective slots.
    pub fn objective_labels(self) -> [&'static str; 3] {
        match self {
            Self::Balanced => ["survival_depth", "mission_continuity", "reserve_efficiency"],
            Self::SurvivalFirst => ["force_survival", "cohesion_under_fire", "reserve_depth"],
            Self::PersistentIsr => [
                "mission_survival",
                "coverage_continuity",
                "endurance_efficiency",
            ],
            Self::CommunicationsDenied => [
                "survival_depth",
                "degraded_coordination",
                "reserve_efficiency",
            ],
            Self::AggressiveStrike => ["shock_survival", "tempo_continuity", "mission_efficiency"],
        }
    }

    fn scenario_multiplier(self, name: &str) -> f64 {
        match self {
            Self::Balanced => 1.0,
            Self::SurvivalFirst => match name {
                "ambush" => 1.35,
                "attrition" => 1.45,
                "stress" | "stress_test" => 1.10,
                "gps" | "gps_denied" => 0.90,
                _ => 1.0,
            },
            Self::PersistentIsr => match name {
                "gps" | "gps_denied" => 1.30,
                "stress" | "stress_test" => 1.20,
                "attrition" => 1.05,
                "ambush" => 0.85,
                _ => 1.0,
            },
            Self::CommunicationsDenied => match name {
                "gps" | "gps_denied" => 1.55,
                "stress" | "stress_test" => 1.25,
                "attrition" => 1.05,
                "ambush" => 0.85,
                _ => 1.0,
            },
            Self::AggressiveStrike => match name {
                "ambush" => 1.40,
                "attrition" => 1.25,
                "stress" | "stress_test" => 1.05,
                "gps" | "gps_denied" => 0.80,
                _ => 1.0,
            },
        }
    }

    fn scoring(self) -> DoctrineScoring {
        match self {
            Self::Balanced => DoctrineScoring {
                objective_0: ObjectiveBlend {
                    survival_rate: 0.48,
                    stability: 0.14,
                    battery_mean: 0.0,
                    battery_floor: 0.10,
                    safety_margin: 0.28,
                    coordination_margin: 0.0,
                    recovery_margin: 0.0,
                },
                objective_1: ObjectiveBlend {
                    survival_rate: 0.0,
                    stability: 0.45,
                    battery_mean: 0.0,
                    battery_floor: 0.0,
                    safety_margin: 0.20,
                    coordination_margin: 0.20,
                    recovery_margin: 0.15,
                },
                objective_2: ObjectiveBlend {
                    survival_rate: 0.0,
                    stability: 0.0,
                    battery_mean: 0.45,
                    battery_floor: 0.25,
                    safety_margin: 0.15,
                    coordination_margin: 0.15,
                    recovery_margin: 0.0,
                },
            },
            Self::SurvivalFirst => DoctrineScoring {
                objective_0: ObjectiveBlend {
                    survival_rate: 0.50,
                    stability: 0.05,
                    battery_mean: 0.0,
                    battery_floor: 0.10,
                    safety_margin: 0.35,
                    coordination_margin: 0.0,
                    recovery_margin: 0.0,
                },
                objective_1: ObjectiveBlend {
                    survival_rate: 0.0,
                    stability: 0.30,
                    battery_mean: 0.0,
                    battery_floor: 0.0,
                    safety_margin: 0.30,
                    coordination_margin: 0.15,
                    recovery_margin: 0.25,
                },
                objective_2: ObjectiveBlend {
                    survival_rate: 0.0,
                    stability: 0.0,
                    battery_mean: 0.30,
                    battery_floor: 0.40,
                    safety_margin: 0.20,
                    coordination_margin: 0.10,
                    recovery_margin: 0.0,
                },
            },
            Self::PersistentIsr => DoctrineScoring {
                objective_0: ObjectiveBlend {
                    survival_rate: 0.40,
                    stability: 0.20,
                    battery_mean: 0.0,
                    battery_floor: 0.0,
                    safety_margin: 0.20,
                    coordination_margin: 0.20,
                    recovery_margin: 0.0,
                },
                objective_1: ObjectiveBlend {
                    survival_rate: 0.0,
                    stability: 0.40,
                    battery_mean: 0.10,
                    battery_floor: 0.0,
                    safety_margin: 0.10,
                    coordination_margin: 0.30,
                    recovery_margin: 0.10,
                },
                objective_2: ObjectiveBlend {
                    survival_rate: 0.0,
                    stability: 0.0,
                    battery_mean: 0.50,
                    battery_floor: 0.25,
                    safety_margin: 0.05,
                    coordination_margin: 0.20,
                    recovery_margin: 0.0,
                },
            },
            Self::CommunicationsDenied => DoctrineScoring {
                objective_0: ObjectiveBlend {
                    survival_rate: 0.45,
                    stability: 0.10,
                    battery_mean: 0.0,
                    battery_floor: 0.0,
                    safety_margin: 0.25,
                    coordination_margin: 0.20,
                    recovery_margin: 0.0,
                },
                objective_1: ObjectiveBlend {
                    survival_rate: 0.0,
                    stability: 0.25,
                    battery_mean: 0.0,
                    battery_floor: 0.0,
                    safety_margin: 0.20,
                    coordination_margin: 0.40,
                    recovery_margin: 0.15,
                },
                objective_2: ObjectiveBlend {
                    survival_rate: 0.0,
                    stability: 0.0,
                    battery_mean: 0.30,
                    battery_floor: 0.25,
                    safety_margin: 0.20,
                    coordination_margin: 0.25,
                    recovery_margin: 0.0,
                },
            },
            Self::AggressiveStrike => DoctrineScoring {
                objective_0: ObjectiveBlend {
                    survival_rate: 0.50,
                    stability: 0.15,
                    battery_mean: 0.0,
                    battery_floor: 0.0,
                    safety_margin: 0.20,
                    coordination_margin: 0.0,
                    recovery_margin: 0.15,
                },
                objective_1: ObjectiveBlend {
                    survival_rate: 0.0,
                    stability: 0.35,
                    battery_mean: 0.0,
                    battery_floor: 0.0,
                    safety_margin: 0.20,
                    coordination_margin: 0.20,
                    recovery_margin: 0.25,
                },
                objective_2: ObjectiveBlend {
                    survival_rate: 0.25,
                    stability: 0.20,
                    battery_mean: 0.15,
                    battery_floor: 0.05,
                    safety_margin: 0.10,
                    coordination_margin: 0.10,
                    recovery_margin: 0.15,
                },
            },
        }
    }
}

impl FromStr for DoctrineProfile {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().replace('-', "_").as_str() {
            "balanced" => Ok(Self::Balanced),
            "survival_first" => Ok(Self::SurvivalFirst),
            "persistent_isr" => Ok(Self::PersistentIsr),
            "communications_denied" => Ok(Self::CommunicationsDenied),
            "aggressive_strike" => Ok(Self::AggressiveStrike),
            other => Err(format!(
                "unsupported doctrine '{other}' (expected balanced, survival_first, persistent_isr, communications_denied, or aggressive_strike)"
            )),
        }
    }
}

// ---------------------------------------------------------------------------
// Objective shaping
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
struct ObjectiveComponents {
    survival_rate: f64,
    stability: f64,
    battery_mean: f64,
    battery_floor: f64,
    safety_margin: f64,
    coordination_margin: f64,
    recovery_margin: f64,
}

impl ObjectiveComponents {
    fn from_report(report: &BattleReport) -> Self {
        let agg = &report.aggregates;
        let n_init = report.n_drones_initial.max(1) as f64;
        let total_ticks = agg.total_ticks.max(1) as f64;

        let survival_rate = agg.drones_survived as f64 / n_init;
        let churn_rate = agg.regime_changes as f64 / (n_init * total_ticks);
        let stability = (1.0 - churn_rate * 50.0).clamp(0.0, 1.0);
        let battery_mean = agg.battery_mean.clamp(0.0, 1.0);
        let battery_floor = agg.battery_min.clamp(0.0, 1.0);
        let violation_rate = agg.cbf_violations as f64 / n_init;
        let safety_margin =
            (1.0 - violation_rate - 0.35 * agg.cbf_burden_mean.clamp(0.0, 1.0)).clamp(0.0, 1.0);
        let coordination_round_rate = agg.auction_rounds as f64 / total_ticks;
        let coordination_peak = (agg.coordination_churn_peak as f64 / n_init).clamp(0.0, 1.0);
        let coordination_margin = (1.0
            - 0.65 * agg.coordination_burden_mean.clamp(0.0, 1.0)
            - 0.20 * coordination_round_rate.clamp(0.0, 1.0)
            - 0.15 * coordination_peak)
            .clamp(0.0, 1.0);
        let recovery_pressure =
            (agg.forced_evade_count as f64 + 0.5 * agg.kill_zones_created as f64) / n_init;
        let recovery_margin = (1.0 - recovery_pressure).clamp(0.0, 1.0);

        Self {
            survival_rate,
            stability,
            battery_mean,
            battery_floor,
            safety_margin,
            coordination_margin,
            recovery_margin,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct ObjectiveBlend {
    survival_rate: f64,
    stability: f64,
    battery_mean: f64,
    battery_floor: f64,
    safety_margin: f64,
    coordination_margin: f64,
    recovery_margin: f64,
}

impl ObjectiveBlend {
    fn score(self, components: ObjectiveComponents) -> f64 {
        (self.survival_rate * components.survival_rate
            + self.stability * components.stability
            + self.battery_mean * components.battery_mean
            + self.battery_floor * components.battery_floor
            + self.safety_margin * components.safety_margin
            + self.coordination_margin * components.coordination_margin
            + self.recovery_margin * components.recovery_margin)
            .clamp(0.0, 1.0)
    }
}

#[derive(Debug, Clone, Copy)]
struct DoctrineScoring {
    objective_0: ObjectiveBlend,
    objective_1: ObjectiveBlend,
    objective_2: ObjectiveBlend,
}

impl DoctrineScoring {
    fn apply(self, components: ObjectiveComponents) -> [f64; 3] {
        [
            self.objective_0.score(components),
            self.objective_1.score(components),
            self.objective_2.score(components),
        ]
    }
}

// ---------------------------------------------------------------------------
// ScenarioSpec
// ---------------------------------------------------------------------------

/// Describes a single named scenario and its relative weight in the aggregate.
#[derive(Debug, Clone)]
pub struct ScenarioSpec {
    pub name: String,
    pub weight: f64,
}

impl ScenarioSpec {
    pub fn new(name: impl Into<String>, weight: f64) -> Self {
        Self {
            name: name.into(),
            weight,
        }
    }
}

// ---------------------------------------------------------------------------
// Evaluator
// ---------------------------------------------------------------------------

/// Runs scenario batteries and aggregates multi-objective scores.
pub struct Evaluator {
    pub scenarios: Vec<ScenarioSpec>,
    /// Simulation timestep shared across all scenarios.
    pub dt: f64,
    /// Default scenario duration (used for single-scenario quick eval).
    pub quick_duration_secs: f64,
    /// Doctrine profile controlling scenario weights and objective shaping.
    pub doctrine: DoctrineProfile,
}

impl Evaluator {
    /// Build an Evaluator with all 4 default preset scenarios and equal weights.
    pub fn default_scenarios() -> Self {
        Self {
            scenarios: vec![
                ScenarioSpec::new("ambush", 1.0),
                ScenarioSpec::new("gps", 1.0),
                ScenarioSpec::new("attrition", 1.0),
                ScenarioSpec::new("stress", 1.0),
            ],
            dt: 0.1,
            quick_duration_secs: 60.0,
            doctrine: DoctrineProfile::Balanced,
        }
    }

    /// Build an Evaluator from a list of scenario names (equal weights).
    pub fn from_names(names: &[String], dt: f64, quick_duration_secs: f64) -> Self {
        let scenarios = names
            .iter()
            .map(|n| ScenarioSpec::new(n.as_str(), 1.0))
            .collect();
        Self {
            scenarios,
            dt,
            quick_duration_secs,
            doctrine: DoctrineProfile::Balanced,
        }
    }

    /// Apply a doctrine profile to both objective shaping and scenario weighting.
    pub fn with_doctrine(mut self, doctrine: DoctrineProfile) -> Self {
        self.doctrine = doctrine;
        for spec in &mut self.scenarios {
            spec.weight *= doctrine.scenario_multiplier(&spec.name);
        }
        self
    }

    /// Human-readable labels for the current objective slots.
    pub fn objective_labels(&self) -> [&'static str; 3] {
        self.doctrine.objective_labels()
    }

    // ── Core evaluation ─────────────────────────────────────────────────────

    /// Run all scenarios, weighted-average the objectives.
    /// Returns a doctrine-shaped `[objective0, objective1, objective2]` vector.
    pub fn evaluate(&self, space: &ParamSpace, params: &ParamVec) -> [f64; 3] {
        let per_scenario = self.evaluate_detailed(space, params);
        let total_weight: f64 = self
            .scenarios
            .iter()
            .map(|s| s.weight)
            .sum::<f64>()
            .max(1e-9);
        let mut acc = [0.0f64; 3];
        for (spec, scores) in self.scenarios.iter().zip(&per_scenario) {
            let w = spec.weight / total_weight;
            acc[0] += w * scores[0];
            acc[1] += w * scores[1];
            acc[2] += w * scores[2];
        }
        acc
    }

    /// Run all scenarios, return per-scenario objective vectors (same order as `self.scenarios`).
    pub fn evaluate_detailed(&self, space: &ParamSpace, params: &ParamVec) -> Vec<[f64; 3]> {
        let swarm_cfg = space.to_swarm_config(params);
        let dt = self.dt;
        let doctrine = self.doctrine;

        self.scenarios
            .par_iter()
            .map(|spec| {
                let sc = swarm_cfg.clone();
                let report = run_named_scenario(&spec.name, sc, dt);
                extract_objectives(&report, doctrine)
            })
            .collect()
    }

    /// Evaluate a single lightweight scenario (ambush only, quick duration).
    /// Used for initial population seeding.
    pub fn evaluate_quick(&self, space: &ParamSpace, params: &ParamVec) -> [f64; 3] {
        let swarm_cfg = space.to_swarm_config(params);
        let report = Playground::ambush()
            .swarm_config(swarm_cfg)
            .dt(self.dt)
            .run_for(self.quick_duration_secs)
            .run();
        extract_objectives(&report, self.doctrine)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn run_named_scenario(
    name: &str,
    sc: strix_swarm::tick::SwarmConfig,
    dt: f64,
) -> strix_playground::BattleReport {
    match name {
        "ambush" => Playground::ambush()
            .swarm_config(sc)
            .dt(dt)
            .run_for(90.0)
            .run(),
        "gps" | "gps_denied" => Playground::gps_denied()
            .swarm_config(sc)
            .dt(dt)
            .run_for(90.0)
            .run(),
        "attrition" => Playground::attrition()
            .swarm_config(sc)
            .dt(dt)
            .run_for(90.0)
            .run(),
        "stress" | "stress_test" => Playground::stress_test()
            .swarm_config(sc)
            .dt(dt)
            .run_for(60.0)
            .run(),
        // Fallback: basic patrol (no threats) — tests baseline stability/battery
        _ => Playground::new()
            .name("patrol")
            .drones(10)
            .swarm_config(sc)
            .dt(dt)
            .run_for(60.0)
            .run(),
    }
}

fn extract_objectives(report: &BattleReport, doctrine: DoctrineProfile) -> [f64; 3] {
    let components = ObjectiveComponents::from_report(report);
    doctrine.scoring().apply(components)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use strix_playground::report::Aggregates;

    fn synthetic_report(aggregates: Aggregates) -> BattleReport {
        BattleReport {
            scenario_name: "synthetic".to_string(),
            duration: 60.0,
            n_drones_initial: 10,
            n_threats_initial: 0,
            timeline: Vec::new(),
            aggregates,
            per_drone: HashMap::new(),
            tick_data: None,
        }
    }

    #[test]
    fn doctrine_profile_parses_snake_and_kebab_case() {
        assert_eq!(
            "balanced".parse::<DoctrineProfile>().unwrap(),
            DoctrineProfile::Balanced
        );
        assert_eq!(
            "communications-denied".parse::<DoctrineProfile>().unwrap(),
            DoctrineProfile::CommunicationsDenied
        );
    }

    #[test]
    fn communications_denied_reweights_gps_above_ambush() {
        let eval =
            Evaluator::default_scenarios().with_doctrine(DoctrineProfile::CommunicationsDenied);
        let gps_weight = eval
            .scenarios
            .iter()
            .find(|spec| spec.name == "gps")
            .unwrap()
            .weight;
        let ambush_weight = eval
            .scenarios
            .iter()
            .find(|spec| spec.name == "ambush")
            .unwrap()
            .weight;

        assert!(
            gps_weight > ambush_weight,
            "gps={gps_weight} ambush={ambush_weight}"
        );
    }

    #[test]
    fn survival_first_penalizes_safety_breakage_more_than_balanced() {
        let clean = synthetic_report(Aggregates {
            total_ticks: 100,
            drones_survived: 8,
            regime_changes: 4,
            battery_min: 0.4,
            battery_mean: 0.7,
            auction_rounds: 6,
            cbf_violations: 0,
            forced_evade_count: 1,
            kill_zones_created: 1,
            ..Default::default()
        });
        let unsafe_report = synthetic_report(Aggregates {
            total_ticks: 100,
            drones_survived: 8,
            regime_changes: 4,
            battery_min: 0.4,
            battery_mean: 0.7,
            auction_rounds: 6,
            cbf_violations: 3,
            forced_evade_count: 1,
            kill_zones_created: 1,
            ..Default::default()
        });

        let balanced_delta = extract_objectives(&clean, DoctrineProfile::Balanced)[0]
            - extract_objectives(&unsafe_report, DoctrineProfile::Balanced)[0];
        let survival_delta = extract_objectives(&clean, DoctrineProfile::SurvivalFirst)[0]
            - extract_objectives(&unsafe_report, DoctrineProfile::SurvivalFirst)[0];

        assert!(
            survival_delta > balanced_delta,
            "survival_delta={survival_delta} balanced_delta={balanced_delta}"
        );
    }

    #[test]
    fn persistent_isr_rewards_reserve_efficiency_more_than_aggressive_strike() {
        let report = synthetic_report(Aggregates {
            total_ticks: 100,
            drones_survived: 6,
            regime_changes: 8,
            battery_min: 0.9,
            battery_mean: 0.95,
            auction_rounds: 2,
            cbf_violations: 0,
            coordination_burden_mean: 0.05,
            forced_evade_count: 0,
            kill_zones_created: 0,
            ..Default::default()
        });

        let isr = extract_objectives(&report, DoctrineProfile::PersistentIsr);
        let strike = extract_objectives(&report, DoctrineProfile::AggressiveStrike);

        assert!(
            isr[2] > strike[2],
            "persistent_isr reserve objective should exceed aggressive_strike: {:?} vs {:?}",
            isr,
            strike
        );
    }
}
