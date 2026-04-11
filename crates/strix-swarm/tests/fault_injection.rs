//! Fault injection tests — exercises dormant fault infrastructure.
//!
//! STRIX has simulated communication partitions, Byzantine validation,
//! and NaN sanitization, but no integration test ever injects faults.
//! These tests fill that gap.

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
// Test: NaN injection into telemetry does not cascade
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn nan_injection_does_not_cascade() {
    let ids: Vec<u32> = (0..5).collect();
    let mut orch = SwarmOrchestrator::new(&ids, SwarmConfig::default());

    let fleet = SimulatorFleet::new_grid(5, 15.0, SimulatorConfig::default());
    fleet.arm_all().unwrap();
    fleet.step_all_n(5);

    // Run 5 normal ticks.
    for _ in 0..5 {
        fleet.step_all_n(1);
        let telem = collect_telemetry(&fleet);
        orch.tick(&telem, &[], 0.1);
    }

    // Inject NaN into one drone's telemetry.
    fleet.step_all_n(1);
    let mut telem = collect_telemetry(&fleet);
    if let Some((_, ref mut t)) = telem.first_mut() {
        t.position = [f64::NAN, f64::NAN, f64::NAN];
    }

    let decision = orch.tick(&telem, &[], 0.1);

    // Other drones should still have valid positions.
    let valid_positions = decision
        .positions
        .values()
        .filter(|p| p.iter().all(|v| v.is_finite()))
        .count();
    assert!(
        valid_positions >= 3,
        "NaN should not cascade: {valid_positions}/5 drones have valid positions"
    );

    // System should remain operational after NaN injection.
    fleet.step_all_n(1);
    let clean_telem = collect_telemetry(&fleet);
    let recovery_decision = orch.tick(&clean_telem, &[], 0.1);
    assert!(
        recovery_decision.fear_level.is_finite(),
        "system should recover after NaN injection"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: Stale telemetry timestamps are handled gracefully
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn stale_timestamps_handled_gracefully() {
    let ids: Vec<u32> = (0..5).collect();
    let mut orch = SwarmOrchestrator::new(&ids, SwarmConfig::default());

    let fleet = SimulatorFleet::new_grid(5, 15.0, SimulatorConfig::default());
    fleet.arm_all().unwrap();
    fleet.step_all_n(5);

    // Run 10 normal ticks.
    for _ in 0..10 {
        fleet.step_all_n(1);
        let telem = collect_telemetry(&fleet);
        orch.tick(&telem, &[], 0.1);
    }

    // Send telemetry with timestamp stuck at 0 (very stale).
    fleet.step_all_n(1);
    let mut telem = collect_telemetry(&fleet);
    for (_, t) in &mut telem {
        t.timestamp = 0.0;
    }

    // Should not panic or produce NaN.
    let decision = orch.tick(&telem, &[], 0.1);
    assert!(decision.criticality.is_finite());
    assert!(decision.gossip_convergence.is_finite());
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: Drone removal mid-run does not crash
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn drone_removal_mid_run_is_safe() {
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

    // Remove 5 drones (50% fleet loss).
    for id in 0..5u32 {
        orch.handle_drone_loss(id, Position::new(0.0, 0.0, 50.0), LossClassification::Sam);
    }

    // Run 20 more ticks with reduced fleet.
    for tick in 0..20 {
        fleet.step_all_n(1);
        let telem: Vec<_> = collect_telemetry(&fleet)
            .into_iter()
            .filter(|(id, _)| *id >= 5)
            .collect();
        let decision = orch.tick(&telem, &[], 0.1);
        assert!(
            decision.criticality.is_finite(),
            "system crashed after drone removal at tick {tick}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: Empty telemetry does not crash
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn empty_telemetry_is_safe() {
    let ids: Vec<u32> = (0..5).collect();
    let mut orch = SwarmOrchestrator::new(&ids, SwarmConfig::default());

    // Run 5 ticks with no telemetry at all.
    for _ in 0..5 {
        let decision = orch.tick(&[], &[], 0.1);
        assert!(decision.fear_level.is_finite());
        assert!(decision.criticality.is_finite());
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: Extreme dt values do not cause divergence
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn extreme_dt_does_not_diverge() {
    let ids: Vec<u32> = (0..5).collect();
    let mut orch = SwarmOrchestrator::new(&ids, SwarmConfig::default());

    let fleet = SimulatorFleet::new_grid(5, 15.0, SimulatorConfig::default());
    fleet.arm_all().unwrap();
    fleet.step_all_n(5);

    // Very small dt.
    fleet.step_all_n(1);
    let telem = collect_telemetry(&fleet);
    let decision = orch.tick(&telem, &[], 0.001);
    assert!(
        decision.criticality.is_finite(),
        "small dt caused divergence"
    );

    // Very large dt.
    fleet.step_all_n(1);
    let telem = collect_telemetry(&fleet);
    let decision = orch.tick(&telem, &[], 10.0);
    assert!(
        decision.criticality.is_finite(),
        "large dt caused divergence"
    );
}
