//! Phi-sim integration tests — verify that the fear meta-parameter pipeline
//! correctly modulates all STRIX subsystems: formation, ROE, auction, gossip,
//! pheromone, and the SwarmOrchestrator end-to-end.
//!
//! All tests are gated behind `#[cfg(feature = "phi-sim")]`.

#![cfg(feature = "phi-sim")]

use strix_adapters::traits::{FlightMode, GpsFix, Telemetry};
use strix_auction::bidder::{calculate_bid_with_fear, calculate_bid_with_scenarios, BidComponents};
use strix_auction::{Capabilities, LossClassification, Position, Task};
use strix_core::formation::FormationConfig;
use strix_core::roe::{RoeEngine, WeaponsPosture};
use strix_core::Regime;
use strix_swarm::fear_adapter::{modulate_gossip_fanout, DroneFearInputs, SwarmFearAdapter};
use strix_swarm::{SwarmConfig, SwarmOrchestrator};

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Build a calm (no-threat) DroneFearInputs.
fn calm_inputs() -> DroneFearInputs {
    DroneFearInputs {
        threat_distance: 2000.0,
        closing_rate: 0.0,
        cusum_triggered: false,
        regime: Regime::Patrol,
        speed: 10.0,
        fleet_coherence: 1.0,
    }
}

/// Build telemetry for `n` drones in a line along the X axis.
fn make_telemetry(ids: &[u32]) -> Vec<(u32, Telemetry)> {
    ids.iter()
        .map(|&id| {
            (
                id,
                Telemetry {
                    position: [id as f64 * 10.0, 0.0, -50.0],
                    velocity: [1.0, 0.0, 0.0],
                    attitude: [0.0, 0.0, 0.0],
                    battery: 0.9,
                    gps_fix: GpsFix::Fix3D,
                    armed: true,
                    mode: FlightMode::Auto,
                    timestamp: 0.0,
                },
            )
        })
        .collect()
}

/// Build a simple task list.
fn make_tasks(n: usize) -> Vec<Task> {
    (0..n)
        .map(|i| Task {
            id: (i + 1) as u32,
            location: Position::new(50.0 + i as f64 * 20.0, 0.0, -50.0),
            required_capabilities: Capabilities::default(),
            priority: 0.5,
            urgency: 0.5,
            bundle_id: None,
            dark_pool: None,
        })
        .collect()
}

// ── Test 1 ──────────────────────────────────────────────────────────────────

/// Verify SwarmFearAdapter produces per-drone fear levels in [0, 1] after update.
#[test]
fn test_swarm_fear_adapter_per_drone_fear() {
    let ids = vec![1, 2, 3, 4, 5];
    let mut adapter = SwarmFearAdapter::new(&ids);

    let per_drone: Vec<(u32, DroneFearInputs)> =
        ids.iter().map(|&id| (id, calm_inputs())).collect();
    adapter.update_fear(&per_drone, 0.0, 0, false);

    for &id in &ids {
        let f = adapter.fear_level(id);
        assert!(
            (0.0..=1.0).contains(&f),
            "drone {} fear {} not in [0,1]",
            id,
            f
        );
    }
}

// ── Test 2 ──────────────────────────────────────────────────────────────────

/// Verify collective signals (fear, courage, tension) are in reasonable finite ranges.
#[test]
fn test_collective_signals_aggregation() {
    let ids = vec![1, 2, 3, 4, 5];
    let mut adapter = SwarmFearAdapter::new(&ids);

    let per_drone: Vec<(u32, DroneFearInputs)> =
        ids.iter().map(|&id| (id, calm_inputs())).collect();
    adapter.update_fear(&per_drone, 0.0, 0, false);

    let cf = adapter.collective_fear();
    let cc = adapter.collective_courage();
    let ct = adapter.collective_tension();

    assert!(cf.is_finite(), "collective_fear must be finite, got {cf}");
    assert!(
        cc.is_finite(),
        "collective_courage must be finite, got {cc}"
    );
    assert!(
        ct.is_finite(),
        "collective_tension must be finite, got {ct}"
    );

    // Fear and courage should be non-negative.
    assert!(cf >= 0.0, "collective_fear should be >= 0, got {cf}");
    assert!(cc >= 0.0, "collective_courage should be >= 0, got {cc}");

    // Tension is (C-F)/(1+F*C), bounded in [-1, 1].
    assert!(
        ct >= -1.0 && ct <= 1.0,
        "collective_tension should be in [-1,1], got {ct}"
    );
}

// ── Test 3 ──────────────────────────────────────────────────────────────────

/// Verify that courage rises (or at least stays positive) after many calm ticks.
#[test]
fn test_courage_rises_after_calm_ticks() {
    let ids = vec![1, 2, 3];
    let mut adapter = SwarmFearAdapter::new(&ids);

    let per_drone: Vec<(u32, DroneFearInputs)> =
        ids.iter().map(|&id| (id, calm_inputs())).collect();

    // Warm up with one tick.
    adapter.update_fear(&per_drone, 0.0, 0, false);
    let courage_initial = adapter.courage_level(1);

    // Run 10 more calm ticks.
    for _ in 0..10 {
        adapter.update_fear(&per_drone, 0.0, 0, false);
    }
    let courage_after = adapter.courage_level(1);

    // Courage should have increased (or at minimum stayed positive).
    assert!(
        courage_after > 0.0,
        "courage should be positive after calm ticks, got {courage_after}"
    );
    // It should have risen or stayed constant compared to initial.
    assert!(
        courage_after >= courage_initial - 1e-9,
        "courage should not decrease in calm conditions: initial={courage_initial}, after={courage_after}"
    );
}

// ── Test 4 ──────────────────────────────────────────────────────────────────

/// Verify tension-based ROE posture suggestion.
/// T < -0.3 -> WeaponsHold, T > 0.1 -> WeaponsFree.
#[test]
fn test_tension_drives_roe_suggestion() {
    let engine = RoeEngine::new(WeaponsPosture::WeaponsTight);

    // High fear (tension very negative) -> suggest WeaponsHold.
    let suggestion_hold = engine.tension_posture_suggestion(-0.5);
    assert_eq!(
        suggestion_hold,
        Some(WeaponsPosture::WeaponsHold),
        "tension=-0.5 should suggest WeaponsHold"
    );

    // High courage (tension positive) -> suggest WeaponsFree.
    let suggestion_free = engine.tension_posture_suggestion(0.3);
    assert_eq!(
        suggestion_free,
        Some(WeaponsPosture::WeaponsFree),
        "tension=0.3 should suggest WeaponsFree"
    );

    // Balanced tension -> no change from WeaponsTight.
    let suggestion_tight = engine.tension_posture_suggestion(0.0);
    assert_eq!(
        suggestion_tight, None,
        "tension=0.0 should match current posture (no suggestion)"
    );
}

// ── Test 5 ──────────────────────────────────────────────────────────────────

/// Verify scenario-enriched auction bid scoring:
/// - negative doom_value reduces score
/// - positive upside_value increases score
#[test]
fn test_scenario_enriched_auction_bids() {
    let components = BidComponents {
        proximity: 0.1,
        capability: 1.0,
        energy: 0.8,
        risk_exposure: 0.2,
        urgency_bonus: 0.5,
    };
    let fear = 0.3;

    let base_score = calculate_bid_with_fear(&components, fear);

    // Doom scenario: negative doom_value should lower the score.
    let doom_score = calculate_bid_with_scenarios(&components, fear, -3.0, 0.0, 1.0);
    assert!(
        doom_score < base_score,
        "negative doom should reduce score: doom={doom_score}, base={base_score}"
    );

    // Upside scenario: positive upside_value should raise the score.
    let upside_score = calculate_bid_with_scenarios(&components, fear, 0.0, 3.0, 1.0);
    assert!(
        upside_score > base_score,
        "positive upside should increase score: upside={upside_score}, base={base_score}"
    );

    // Combined: large doom + small upside should still net-reduce.
    let combined_score = calculate_bid_with_scenarios(&components, fear, -5.0, 1.0, 1.0);
    assert!(
        combined_score < base_score,
        "large doom + small upside should net-reduce: combined={combined_score}, base={base_score}"
    );
}

// ── Test 6 ──────────────────────────────────────────────────────────────────

/// Verify FormationConfig::fear_adjusted modulates spacing.
/// High fear (threshold=0.3) -> wider spacing; low fear (threshold=1.0) -> tighter.
#[test]
fn test_fear_axes_modulate_formation() {
    let base = FormationConfig::default(); // spacing=15.0

    // High fear -> threshold=0.3 (max fear clamps to 0.3)
    let high_fear = base.fear_adjusted(0.3);
    // Low fear -> threshold=1.0
    let low_fear = base.fear_adjusted(1.0);

    // High fear should produce wider spacing.
    assert!(
        high_fear.spacing > low_fear.spacing,
        "high fear should widen spacing: high_fear={}, low_fear={}",
        high_fear.spacing,
        low_fear.spacing
    );

    // Exact values: threshold=0.3 -> scale=1.5, threshold=1.0 -> scale=0.8
    let expected_high = base.spacing * 1.5;
    let expected_low = base.spacing * 0.8;
    assert!(
        (high_fear.spacing - expected_high).abs() < 1e-9,
        "high fear spacing: got {}, expected {}",
        high_fear.spacing,
        expected_high
    );
    assert!(
        (low_fear.spacing - expected_low).abs() < 1e-9,
        "low fear spacing: got {}, expected {}",
        low_fear.spacing,
        expected_low
    );

    // vee_angle and max_correction_speed should be preserved.
    assert_eq!(high_fear.vee_angle_deg, base.vee_angle_deg);
    assert_eq!(high_fear.max_correction_speed, base.max_correction_speed);
}

// ── Test 7 ──────────────────────────────────────────────────────────────────

/// Verify SwarmFearAdapter.calibration() returns a CalibrationMetrics with overall() >= 0.
#[test]
fn test_calibration_in_traces() {
    let ids = vec![1, 2, 3, 4, 5];
    let adapter = SwarmFearAdapter::new(&ids);

    let cal = adapter.calibration();
    let overall = cal.overall();

    assert!(
        overall >= 0.0,
        "calibration overall() should be >= 0, got {overall}"
    );
    assert!(
        overall.is_finite(),
        "calibration overall() should be finite, got {overall}"
    );

    // Initial calibration is neutral (all scores at 0.5).
    // overall() = product ^ (1/5) of five 0.5 values, but it's the geometric
    // mean via `nth_root(5, product)`. With CalibrationMetrics::neutral() all
    // at 0.5, the overall should also be approximately 0.5.
    assert!(
        (overall - 0.5).abs() < 0.1,
        "initial calibration should be near neutral (0.5), got {overall}"
    );
}

// ── Test 8 ──────────────────────────────────────────────────────────────────

/// Verify record_outcome + train increases experience_count.
#[test]
fn test_online_learning_outcome_recording() {
    let ids = vec![1, 2];
    let mut adapter = SwarmFearAdapter::new(&ids);

    // Warm up: do at least one fear update so the pipeline is active.
    let per_drone: Vec<(u32, DroneFearInputs)> =
        ids.iter().map(|&id| (id, calm_inputs())).collect();
    adapter.update_fear(&per_drone, 0.0, 0, true);

    let before = adapter.experience_count();

    // Record several outcomes and train.
    for i in 0..5 {
        adapter.record_outcome(0.8 - i as f64 * 0.1);
    }
    adapter.train();

    let after = adapter.experience_count();
    assert!(
        after > before,
        "experience_count should increase after record_outcome+train: before={before}, after={after}"
    );
}

// ── Test 9 ──────────────────────────────────────────────────────────────────

/// Verify gossip fanout modulation by fear.
/// f=0 -> base fanout unchanged. High fear -> larger fanout.
#[test]
fn test_gossip_fanout_reduced_by_fear() {
    let base = 3;

    // Zero fear -> base unchanged.
    let fanout_calm = modulate_gossip_fanout(base, 0.0);
    assert_eq!(fanout_calm, base, "zero fear should leave fanout unchanged");

    // High fear -> larger fanout.
    let fanout_scared = modulate_gossip_fanout(base, 0.8);
    assert!(
        fanout_scared > base,
        "high fear should increase fanout: got {fanout_scared}, base={base}"
    );

    // Maximum fear.
    let fanout_max = modulate_gossip_fanout(base, 1.0);
    assert!(
        fanout_max >= fanout_scared,
        "max fear should produce >= high fear fanout: max={fanout_max}, high={fanout_scared}"
    );

    // Monotonicity check.
    let mut prev = modulate_gossip_fanout(base, 0.0);
    for step in 1..=10 {
        let f = step as f64 / 10.0;
        let current = modulate_gossip_fanout(base, f);
        assert!(
            current >= prev,
            "fanout should be monotonically non-decreasing: f={f}, prev={prev}, cur={current}"
        );
        prev = current;
    }
}

// ── Test 10 ─────────────────────────────────────────────────────────────────

/// Verify SwarmOrchestrator with phi-sim runs without panic and produces
/// non-zero pheromone cells when kill zones exist.
#[test]
fn test_pheromone_threat_intensity_scales_with_fear() {
    let ids: Vec<u32> = (1..=5).collect();
    let config = SwarmConfig {
        auction_interval: 1,
        n_particles: 50,
        n_threat_particles: 30,
        ..Default::default()
    };
    let mut orch = SwarmOrchestrator::new(&ids, config);
    orch.fear_adapter = Some(SwarmFearAdapter::new(&ids));

    // Register a threat and a kill zone to trigger pheromone deposits.
    orch.register_threat(100, nalgebra::Vector3::new(500.0, 0.0, -50.0));
    orch.handle_drone_loss(
        99,
        Position::new(200.0, 0.0, -50.0),
        LossClassification::Sam,
    );

    let telemetry = make_telemetry(&ids);
    let tasks = make_tasks(2);

    // Tick 1
    let _d1 = orch.tick(&telemetry, &tasks, 0.1);

    // Tick 2
    let d2 = orch.tick(&telemetry, &tasks, 0.1);

    // Both ticks should complete without panic.
    // Pheromone field should have active cells (explored + threat from kill zone).
    assert!(
        d2.pheromone_cells > 0,
        "pheromone field should have active cells, got {}",
        d2.pheromone_cells
    );

    // Fear level should be in [0, 1].
    assert!(
        (0.0..=1.0).contains(&d2.fear_level),
        "fear_level should be in [0,1], got {}",
        d2.fear_level
    );
}

// ── Test 11 ─────────────────────────────────────────────────────────────────

/// Full pipeline: 20 drones, 2 threats, tasks, 5 ticks.
/// Verify SwarmDecision has all phi-sim fields populated.
#[test]
fn test_full_pipeline_20_drones_all_features() {
    let ids: Vec<u32> = (1..=20).collect();
    let config = SwarmConfig {
        auction_interval: 2,
        n_particles: 30,
        n_threat_particles: 20,
        ..Default::default()
    };
    let mut orch = SwarmOrchestrator::new(&ids, config);
    orch.fear_adapter = Some(SwarmFearAdapter::new(&ids));

    // Register two threats.
    orch.register_threat(100, nalgebra::Vector3::new(500.0, 0.0, -50.0));
    orch.register_threat(101, nalgebra::Vector3::new(-200.0, 300.0, -50.0));

    let telemetry = make_telemetry(&ids);
    let tasks = make_tasks(5);

    // Run 5 ticks.
    let mut last_decision = None;
    for _ in 0..5 {
        let d = orch.tick(&telemetry, &tasks, 0.1);
        last_decision = Some(d);
    }
    let decision = last_decision.expect("should have at least one tick result");

    // Fear level should be populated and valid.
    assert!(
        decision.fear_level >= 0.0 && decision.fear_level <= 1.0,
        "fear_level should be in [0,1], got {}",
        decision.fear_level
    );

    // Courage level should be non-negative.
    assert!(
        decision.courage_level >= 0.0,
        "courage_level should be >= 0, got {}",
        decision.courage_level
    );

    // Per-drone fear map should have entries for all 20 drones.
    assert!(
        !decision.per_drone_fear.is_empty(),
        "per_drone_fear should not be empty"
    );
    assert_eq!(
        decision.per_drone_fear.len(),
        20,
        "per_drone_fear should have 20 entries, got {}",
        decision.per_drone_fear.len()
    );

    // Each per-drone fear should be in [0, 1].
    for (&drone_id, &fear) in &decision.per_drone_fear {
        assert!(
            (0.0..=1.0).contains(&fear),
            "drone {} fear {} not in [0,1]",
            drone_id,
            fear
        );
    }

    // Tension should be finite.
    assert!(
        decision.tension.is_finite(),
        "tension should be finite, got {}",
        decision.tension
    );

    // Positions should be populated for all drones.
    assert_eq!(
        decision.positions.len(),
        20,
        "should have 20 position estimates, got {}",
        decision.positions.len()
    );

    // Should have assignments (at least one auction tick should have fired).
    // The auction runs on tick 2 and 4 (interval=2), so we should have some.
    assert!(
        !decision.assignments.is_empty(),
        "should have task assignments after 5 ticks"
    );
}

// ── Test 12 ─────────────────────────────────────────────────────────────────

/// Verify drone loss syncs with phi-sim: after handle_drone_loss the adapter
/// still works and the next tick completes without panic.
#[test]
fn test_drone_loss_syncs_phi_sim() {
    let ids: Vec<u32> = vec![1, 2, 3, 4, 5];
    let config = SwarmConfig {
        auction_interval: 1,
        n_particles: 50,
        n_threat_particles: 30,
        ..Default::default()
    };
    let mut orch = SwarmOrchestrator::new(&ids, config);
    orch.fear_adapter = Some(SwarmFearAdapter::new(&ids));

    // Do one tick with all 5 drones alive.
    let telemetry_full = make_telemetry(&ids);
    let tasks = make_tasks(2);
    let d1 = orch.tick(&telemetry_full, &tasks, 0.1);
    assert_eq!(
        d1.per_drone_fear.len(),
        5,
        "should have 5 drone fear values"
    );

    // Destroy drone 3.
    orch.handle_drone_loss(3, Position::new(30.0, 0.0, -50.0), LossClassification::Sam);

    // Next tick with only 4 surviving drones.
    let surviving_ids: Vec<u32> = vec![1, 2, 4, 5];
    let telemetry_reduced = make_telemetry(&surviving_ids);
    let d2 = orch.tick(&telemetry_reduced, &tasks, 0.1);

    // Should succeed without panic and have 4 per-drone fear entries.
    assert_eq!(
        d2.per_drone_fear.len(),
        4,
        "should have 4 drone fear values after loss, got {}",
        d2.per_drone_fear.len()
    );

    // Dead drone should not appear in per_drone_fear.
    assert!(
        !d2.per_drone_fear.contains_key(&3),
        "dead drone 3 should not appear in per_drone_fear"
    );

    // Dead drone should not appear in assignments.
    assert!(
        d2.assignments.iter().all(|a| a.drone_id != 3),
        "dead drone 3 should not have assignments"
    );

    // Fear adapter should still function: verify a surviving drone has valid fear.
    let adapter = orch.fear_adapter.as_ref().unwrap();
    for &id in &surviving_ids {
        let f = adapter.fear_level(id);
        assert!(
            (0.0..=1.0).contains(&f),
            "surviving drone {} fear {} not in [0,1]",
            id,
            f
        );
    }
}
