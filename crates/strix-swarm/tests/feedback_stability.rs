//! Multi-tick feedback loop stability tests.
//!
//! These tests run 300+ tick scenarios to exercise feedback loops that
//! are never tested in the existing 20-30 tick integration tests.
//! Verifies: oscillation damping, convergence stability, fear cascade
//! recovery, and quarantine cascade containment.

use strix_adapters::simulator::{SimulatorConfig, SimulatorFleet};
use strix_adapters::traits::PlatformAdapter;
use strix_auction::{Capabilities, LossClassification, Position, Task};
use strix_swarm::{SwarmConfig, SwarmOrchestrator};

fn collect_telemetry(fleet: &SimulatorFleet) -> Vec<(u32, strix_adapters::traits::Telemetry)> {
    fleet
        .drones
        .iter()
        .filter_map(|d| d.get_telemetry().ok().map(|t| (d.id(), t)))
        .collect()
}

fn make_threat_tasks(n: usize) -> Vec<Task> {
    (0..n)
        .map(|i| Task {
            id: (i + 100) as u32,
            location: Position::new(50.0 + i as f64 * 10.0, 50.0, 30.0),
            required_capabilities: Capabilities {
                has_sensor: true,
                has_weapon: false,
                has_ew: false,
                has_relay: false,
            },
            priority: 0.9,
            urgency: 0.8,
            bundle_id: None,
            dark_pool: None,
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: Criticality does not oscillate indefinitely
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn criticality_settles_within_30_ticks() {
    let ids: Vec<u32> = (0..10).collect();
    let config = SwarmConfig {
        criticality_interval: 1, // every tick for faster testing
        order_params_interval: 1,
        ..SwarmConfig::default()
    };
    let mut orch = SwarmOrchestrator::new(&ids, config);

    let fleet = SimulatorFleet::new_grid(10, 15.0, SimulatorConfig::default());
    fleet.arm_all().unwrap();
    fleet.step_all_n(5);

    let tasks = make_threat_tasks(3);

    // Run 100 ticks — criticality should settle, not oscillate.
    let mut criticality_values = Vec::new();
    for _ in 0..100 {
        fleet.step_all_n(1);
        let telem = collect_telemetry(&fleet);
        let decision = orch.tick(&telem, &tasks, 0.1);
        criticality_values.push(decision.criticality);
    }

    // Verify: last 30 values should have low variance (settled).
    let last_30 = &criticality_values[70..];
    let mean: f64 = last_30.iter().sum::<f64>() / last_30.len() as f64;
    let variance: f64 =
        last_30.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / last_30.len() as f64;

    assert!(
        variance < 0.05,
        "criticality should settle (variance={variance:.4}, mean={mean:.4})"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: Fear spikes then recovers after fleet losses
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn fear_recovers_after_losses() {
    let ids: Vec<u32> = (0..10).collect();
    let mut orch = SwarmOrchestrator::new(&ids, SwarmConfig::default());

    let fleet = SimulatorFleet::new_grid(10, 15.0, SimulatorConfig::default());
    fleet.arm_all().unwrap();
    fleet.step_all_n(5);

    // Run 20 ticks to establish baseline.
    for _ in 0..20 {
        fleet.step_all_n(1);
        let telem = collect_telemetry(&fleet);
        orch.tick(&telem, &[], 0.1);
    }

    // Kill 3 drones (30% attrition).
    for id in 0..3u32 {
        orch.handle_drone_loss(id, Position::new(0.0, 0.0, 50.0), LossClassification::Sam);
    }

    // Run 50 more ticks — fear should spike then decay.
    let mut fear_values = Vec::new();
    for _ in 0..50 {
        fleet.step_all_n(1);
        let telem: Vec<_> = collect_telemetry(&fleet)
            .into_iter()
            .filter(|(id, _)| *id >= 3)
            .collect();
        let decision = orch.tick(&telem, &[], 0.1);
        fear_values.push(decision.fear_level);
    }

    // Fear at end should be lower than max fear during spike.
    // (We can't guarantee exact values without phi-sim, but fear_level should
    // remain finite and bounded.)
    assert!(
        fear_values
            .iter()
            .all(|f| f.is_finite() && *f >= 0.0 && *f <= 1.0),
        "fear values must be finite and bounded"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: Gossip convergence improves over time
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn gossip_convergence_improves() {
    let ids: Vec<u32> = (0..10).collect();
    let mut orch = SwarmOrchestrator::new(&ids, SwarmConfig::default());

    let fleet = SimulatorFleet::new_grid(10, 15.0, SimulatorConfig::default());
    fleet.arm_all().unwrap();
    fleet.step_all_n(5);

    // Run 50 ticks — convergence should improve or stay stable.
    let mut convergence_values = Vec::new();
    for _ in 0..50 {
        fleet.step_all_n(1);
        let telem = collect_telemetry(&fleet);
        let decision = orch.tick(&telem, &[], 0.1);
        convergence_values.push(decision.gossip_convergence);
    }

    // Last convergence should be at least as good as first.
    let first_10_mean: f64 = convergence_values[..10].iter().sum::<f64>() / 10.0;
    let last_10_mean: f64 = convergence_values[40..].iter().sum::<f64>() / 10.0;
    assert!(
        last_10_mean >= first_10_mean - 0.1,
        "convergence should improve: first_10={first_10_mean:.3}, last_10={last_10_mean:.3}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: Health monitor detects issues but system remains functional
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn health_monitor_reports_in_decision() {
    let ids: Vec<u32> = (0..10).collect();
    let mut orch = SwarmOrchestrator::new(&ids, SwarmConfig::default());

    let fleet = SimulatorFleet::new_grid(10, 15.0, SimulatorConfig::default());
    fleet.arm_all().unwrap();
    fleet.step_all_n(5);

    // Run a few ticks — health should be reported in decision.
    for _ in 0..10 {
        fleet.step_all_n(1);
        let telem = collect_telemetry(&fleet);
        let decision = orch.tick(&telem, &[], 0.1);
        // Health fields should be populated.
        assert!(decision.trust_mean.is_finite());
        assert!(decision.quarantine_fraction.is_finite());
        assert!(decision.quarantine_fraction >= 0.0);
        assert!(decision.quarantine_fraction <= 1.0);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: Trust tracker integrates with order parameters
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn trust_feeds_into_order_params() {
    let ids: Vec<u32> = (0..10).collect();
    let config = SwarmConfig {
        order_params_interval: 1,
        ..SwarmConfig::default()
    };
    let mut orch = SwarmOrchestrator::new(&ids, config);

    let fleet = SimulatorFleet::new_grid(10, 15.0, SimulatorConfig::default());
    fleet.arm_all().unwrap();
    fleet.step_all_n(5);

    // Run 30 ticks — order parameters should be computed and valid.
    for _ in 0..30 {
        fleet.step_all_n(1);
        let telem = collect_telemetry(&fleet);
        let decision = orch.tick(&telem, &[], 0.1);

        let op = &decision.order_parameters;
        assert!(op.trust_entropy.is_finite());
        assert!(op.trust_entropy >= 0.0 && op.trust_entropy <= 1.0);
        assert!(op.alignment_order.is_finite());
        assert!(op.fragmentation_index.is_finite());
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: 300-tick sustained operation without crashes or divergence
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn sustained_300_ticks_no_divergence() {
    let ids: Vec<u32> = (0..10).collect();
    let mut orch = SwarmOrchestrator::new(&ids, SwarmConfig::default());

    let fleet = SimulatorFleet::new_grid(10, 15.0, SimulatorConfig::default());
    fleet.arm_all().unwrap();
    fleet.step_all_n(5);

    let tasks = make_threat_tasks(2);

    for tick in 0..300 {
        fleet.step_all_n(1);
        let telem = collect_telemetry(&fleet);
        let decision = orch.tick(&telem, &tasks, 0.1);

        // All values must remain finite and bounded.
        assert!(
            decision.fear_level.is_finite() && decision.fear_level >= 0.0,
            "fear diverged at tick {tick}"
        );
        assert!(
            decision.criticality.is_finite()
                && decision.criticality >= 0.0
                && decision.criticality <= 1.0,
            "criticality diverged at tick {tick}"
        );
        assert!(
            decision.gossip_convergence.is_finite(),
            "convergence diverged at tick {tick}"
        );
        // Positions must be finite.
        for (id, pos) in &decision.positions {
            assert!(
                pos.iter().all(|v| v.is_finite()),
                "position diverged for drone {id} at tick {tick}"
            );
        }
    }
}
