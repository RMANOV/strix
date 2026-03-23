//! Scenario runner and objective extraction for the STRIX optimizer.
//!
//! `Evaluator::evaluate()` runs a battery of Playground scenarios with a given
//! `SwarmConfig` (derived from `ParamVec`) and returns a weighted-average
//! three-objective vector:
//!
//!   F0 = survival_rate   (maximize) — fraction of drones alive at end
//!   F1 = stability_score (maximize) — low regime-churn proxy
//!   F2 = battery_mean    (maximize) — mean battery at end of mission

use rayon::prelude::*;
use strix_playground::Playground;

use crate::param_space::{ParamSpace, ParamVec};

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
        }
    }

    // ── Core evaluation ─────────────────────────────────────────────────────

    /// Run all scenarios, weighted-average the objectives.
    /// Returns `[survival, stability, battery_mean]`.
    pub fn evaluate(&self, space: &ParamSpace, params: &ParamVec) -> [f64; 3] {
        let per_scenario = self.evaluate_detailed(space, params);
        let total_weight: f64 = self.scenarios.iter().map(|s| s.weight).sum();
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

        self.scenarios
            .par_iter()
            .map(|spec| {
                let sc = swarm_cfg.clone();
                let report = run_named_scenario(&spec.name, sc, dt);
                extract_objectives(&report)
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
        extract_objectives(&report)
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

fn extract_objectives(report: &strix_playground::BattleReport) -> [f64; 3] {
    let agg = &report.aggregates;
    let n_init = report.n_drones_initial.max(1) as f64;
    let total_ticks = agg.total_ticks.max(1) as f64;

    // F0: survival rate [0..1]
    let survival = agg.drones_survived as f64 / n_init;

    // F1: stability — inverse of normalised regime-churn [0..1]
    let churn_rate = agg.regime_changes as f64 / (n_init * total_ticks);
    let stability = (1.0 - churn_rate * 50.0).clamp(0.0, 1.0);

    // F2: battery mean [0..1] — already normalised by simulator
    let battery = agg.battery_mean.clamp(0.0, 1.0);

    [survival, stability, battery]
}
