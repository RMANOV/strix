//! Regression tests for critical audit findings.
//!
//! One test per critical/high finding from the 2026-03-29 audit.
//! Prevents re-introduction of fixed bugs.

use nalgebra::Vector3;
use strix_adapters::simulator::{SimulatorConfig, SimulatorFleet};
use strix_adapters::traits::PlatformAdapter;
use strix_auction::{LossClassification, Position};
use strix_swarm::{SwarmConfig, SwarmOrchestrator};

fn collect_telemetry(fleet: &SimulatorFleet) -> Vec<(u32, strix_adapters::traits::Telemetry)> {
    fleet
        .drones
        .iter()
        .filter_map(|d| d.get_telemetry().ok().map(|t| (d.id(), t)))
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// H1: NaN propagation cascade — single NaN should not poison entire swarm
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn regression_h1_nan_containment() {
    let ids: Vec<u32> = (0..5).collect();
    let mut orch = SwarmOrchestrator::new(&ids, SwarmConfig::default());

    let fleet = SimulatorFleet::new_grid(5, 15.0, SimulatorConfig::default());
    fleet.arm_all().unwrap();
    fleet.step_all_n(10);

    // Baseline: all positions finite.
    let telem = collect_telemetry(&fleet);
    let decision = orch.tick(&telem, &[], 0.1);
    let baseline_valid = decision
        .positions
        .values()
        .filter(|p| p.iter().all(|v| v.is_finite()))
        .count();
    assert_eq!(
        baseline_valid, 5,
        "baseline should have all valid positions"
    );

    // Inject NaN into drone 0.
    fleet.step_all_n(1);
    let mut telem = collect_telemetry(&fleet);
    if let Some((_, ref mut t)) = telem.iter_mut().find(|(id, _)| *id == 0) {
        t.position = [f64::NAN, 0.0, 0.0];
        t.velocity = [f64::NAN, f64::NAN, f64::NAN];
    }
    let decision = orch.tick(&telem, &[], 0.1);

    // At least drones 1-4 should still be valid.
    let still_valid = decision
        .positions
        .iter()
        .filter(|(id, p)| **id > 0 && p.iter().all(|v| v.is_finite()))
        .count();
    assert!(
        still_valid >= 4,
        "NaN in drone 0 should not cascade to others: {still_valid}/4 valid"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Regression: system remains functional after rapid drone additions/removals
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn regression_fleet_churn_stability() {
    let ids: Vec<u32> = (0..10).collect();
    let mut orch = SwarmOrchestrator::new(&ids, SwarmConfig::default());

    let fleet = SimulatorFleet::new_grid(10, 15.0, SimulatorConfig::default());
    fleet.arm_all().unwrap();
    fleet.step_all_n(5);

    // Run 10 ticks.
    for _ in 0..10 {
        fleet.step_all_n(1);
        let telem = collect_telemetry(&fleet);
        orch.tick(&telem, &[], 0.1);
    }

    // Rapidly remove and re-add drones.
    for id in 0..5u32 {
        orch.handle_drone_loss(id, Position::new(0.0, 0.0, 50.0), LossClassification::Sam);
    }
    for id in 10..15u32 {
        orch.register_drone(id, Vector3::new(id as f64 * 15.0, 0.0, 50.0));
    }

    // Should still produce valid decisions.
    fleet.step_all_n(1);
    let telem: Vec<_> = collect_telemetry(&fleet)
        .into_iter()
        .filter(|(id, _)| *id >= 5)
        .collect();
    let decision = orch.tick(&telem, &[], 0.1);
    assert!(
        decision.criticality.is_finite(),
        "system should handle fleet churn"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Regression: trust_mean and quarantine_fraction are always valid
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn regression_health_fields_always_valid() {
    let ids: Vec<u32> = (0..5).collect();
    let mut orch = SwarmOrchestrator::new(&ids, SwarmConfig::default());

    let fleet = SimulatorFleet::new_grid(5, 15.0, SimulatorConfig::default());
    fleet.arm_all().unwrap();
    fleet.step_all_n(5);

    for _ in 0..20 {
        fleet.step_all_n(1);
        let telem = collect_telemetry(&fleet);
        let decision = orch.tick(&telem, &[], 0.1);

        assert!(
            decision.trust_mean.is_finite() && decision.trust_mean >= 0.0,
            "trust_mean invalid: {}",
            decision.trust_mean
        );
        assert!(
            decision.quarantine_fraction.is_finite()
                && decision.quarantine_fraction >= 0.0
                && decision.quarantine_fraction <= 1.0,
            "quarantine_fraction invalid: {}",
            decision.quarantine_fraction
        );
    }
}
