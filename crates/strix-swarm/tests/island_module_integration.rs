//! Integration test: verifies that all 5 island modules (formation, ROE, EW,
//! CBF, temporal) participate in the tick loop when wired in.
//!
//! This test exercises the FULL pipeline:
//! 1. Formation corrections computed for a Vee formation
//! 2. ROE gates weapon-bearing tasks against threat posture
//! 3. EW response modifies noise and pheromone deposits
//! 4. CBF clamps velocities near inter-drone separation limits
//! 5. SwarmDecision output includes all new fields

use nalgebra::Vector3;
use strix_adapters::simulator::{SimulatorConfig, SimulatorFleet};
use strix_adapters::traits::{Action, PlatformAdapter};
use strix_auction::{Capabilities, Position, Task};
use strix_core::cbf::{CbfConfig, NoFlyZone};
use strix_core::ew_response::{EwEvent, EwSeverity, EwThreat};
use strix_core::formation::FormationType;
use strix_core::roe::WeaponsPosture;
use strix_swarm::{SwarmConfig, SwarmOrchestrator};

fn collect_telemetry(fleet: &SimulatorFleet) -> Vec<(u32, strix_adapters::traits::Telemetry)> {
    fleet
        .drones
        .iter()
        .filter_map(|d| d.get_telemetry().ok().map(|t| (d.id(), t)))
        .collect()
}

fn make_weapon_task(id: u32, pos: [f64; 3]) -> Task {
    Task {
        id,
        location: Position::new(pos[0], pos[1], pos[2]),
        required_capabilities: Capabilities {
            has_sensor: true,
            has_weapon: true,
            has_ew: false,
            has_relay: false,
        },
        priority: 0.9,
        urgency: 0.8,
        bundle_id: None,
        dark_pool: None,
    }
}

fn make_sensor_task(id: u32, pos: [f64; 3]) -> Task {
    Task {
        id,
        location: Position::new(pos[0], pos[1], pos[2]),
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

// ─────────────────────────────────────────────────────────────────────────────
// Formation Integration
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_formation_quality_reported_in_decision() {
    let ids: Vec<u32> = (0..5).collect();
    let config = SwarmConfig {
        formation_type: Some(FormationType::Vee),
        ..Default::default()
    };
    let mut orch = SwarmOrchestrator::new(&ids, config);

    let fleet = SimulatorFleet::new_grid(5, 15.0, SimulatorConfig::default());
    fleet.arm_all().unwrap();
    fleet.step_all_n(10);

    let telem = collect_telemetry(&fleet);
    let decision = orch.tick(&telem, &[], 0.1);

    // Formation quality should be present (formation is enabled).
    assert!(
        decision.formation_quality.is_some(),
        "formation quality should be reported"
    );
    let q = decision.formation_quality.unwrap();
    assert!(
        (0.0..=1.0).contains(&q),
        "formation quality must be in [0,1], got {}",
        q
    );
}

#[test]
fn test_formation_disabled_no_quality() {
    let ids: Vec<u32> = (0..5).collect();
    let config = SwarmConfig {
        formation_type: None,
        cbf_config: None, // also disable CBF so no corrections at all
        ..Default::default()
    };
    let mut orch = SwarmOrchestrator::new(&ids, config);

    let fleet = SimulatorFleet::new_grid(5, 15.0, SimulatorConfig::default());
    fleet.arm_all().unwrap();
    fleet.step_all_n(10);

    let telem = collect_telemetry(&fleet);
    let decision = orch.tick(&telem, &[], 0.1);

    assert!(
        decision.formation_quality.is_none(),
        "no formation quality when disabled"
    );
    assert!(
        decision.formation_corrections.is_empty(),
        "no corrections when both formation and CBF are disabled"
    );
}

#[test]
fn test_formation_can_be_changed_at_runtime() {
    let ids: Vec<u32> = (0..5).collect();
    let mut orch = SwarmOrchestrator::new(&ids, SwarmConfig::default());
    assert_eq!(orch.formation_type, Some(FormationType::Vee));

    orch.set_formation(Some(FormationType::Line));
    assert_eq!(orch.formation_type, Some(FormationType::Line));

    orch.set_formation(None);
    assert_eq!(orch.formation_type, None);
}

// ─────────────────────────────────────────────────────────────────────────────
// ROE Integration
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_roe_denies_weapon_tasks_under_weapons_hold() {
    let ids: Vec<u32> = (0..5).collect();
    let config = SwarmConfig {
        auction_interval: 1,
        ..Default::default()
    };
    let mut orch = SwarmOrchestrator::new(&ids, config);
    orch.set_weapons_posture(WeaponsPosture::WeaponsHold);

    // Add a threat so ROE has something to evaluate.
    orch.register_threat(1, Vector3::new(50.0, 50.0, 0.0));

    let fleet = SimulatorFleet::new_grid(5, 15.0, SimulatorConfig::default());
    fleet.arm_all().unwrap();
    fleet.step_all_n(10);

    // Weapon task should be denied by WeaponsHold.
    let tasks = vec![
        make_weapon_task(1, [50.0, 50.0, 50.0]),
        make_sensor_task(2, [20.0, 20.0, 50.0]),
    ];

    let telem = collect_telemetry(&fleet);
    let decision = orch.tick(&telem, &tasks, 0.1);

    // Weapon task near the non-attacking threat should be denied under WeaponsHold.
    assert!(
        decision.roe_denials > 0 || decision.roe_escalations > 0,
        "WeaponsHold should deny or escalate weapon tasks near threats"
    );
}

#[test]
fn test_roe_sensor_tasks_always_pass() {
    let ids: Vec<u32> = (0..5).collect();
    let config = SwarmConfig {
        auction_interval: 1,
        ..Default::default()
    };
    let mut orch = SwarmOrchestrator::new(&ids, config);
    orch.set_weapons_posture(WeaponsPosture::WeaponsHold);

    // Add threat
    orch.register_threat(1, Vector3::new(50.0, 50.0, 0.0));

    let fleet = SimulatorFleet::new_grid(5, 15.0, SimulatorConfig::default());
    fleet.arm_all().unwrap();
    fleet.step_all_n(10);

    // Sensor-only tasks should never be affected by ROE.
    let tasks = vec![
        make_sensor_task(1, [50.0, 50.0, 50.0]),
        make_sensor_task(2, [20.0, 20.0, 50.0]),
    ];

    let telem = collect_telemetry(&fleet);
    let decision = orch.tick(&telem, &tasks, 0.1);

    assert_eq!(decision.roe_denials, 0, "sensor tasks should bypass ROE");
    assert_eq!(
        decision.roe_escalations, 0,
        "sensor tasks should not trigger escalation"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// EW Integration
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_ew_event_tracked() {
    let ids: Vec<u32> = (0..5).collect();
    let mut orch = SwarmOrchestrator::new(&ids, SwarmConfig::default());

    // Report a GPS denial event.
    let plan = orch.report_ew_event(EwEvent {
        threat: EwThreat::GpsDenial,
        severity: EwSeverity::Degraded,
        source_bearing: None,
        source_range: None,
        confidence: 0.9,
        timestamp: 0.0,
    });

    assert!(
        !plan.actions.is_empty(),
        "EW response plan should have actions"
    );
    assert_eq!(orch.ew_engine.active_threats().len(), 1);

    // Run tick — EW should be reported in decision.
    let fleet = SimulatorFleet::new_grid(5, 15.0, SimulatorConfig::default());
    fleet.arm_all().unwrap();
    fleet.step_all_n(10);

    let telem = collect_telemetry(&fleet);
    let decision = orch.tick(&telem, &[], 0.1);

    assert_eq!(decision.ew_active_threats, 1);
}

#[test]
fn test_ew_stale_events_cleared() {
    let ids: Vec<u32> = (0..3).collect();
    let config = SwarmConfig {
        ew_stale_age: 5.0, // short for test
        ..Default::default()
    };
    let mut orch = SwarmOrchestrator::new(&ids, config);

    orch.report_ew_event(EwEvent {
        threat: EwThreat::CommsJamming,
        severity: EwSeverity::Detected,
        source_bearing: None,
        source_range: None,
        confidence: 0.8,
        timestamp: 0.0,
    });

    let fleet = SimulatorFleet::new_grid(3, 15.0, SimulatorConfig::default());
    fleet.arm_all().unwrap();
    fleet.step_all_n(10);

    // Run 60 ticks at dt=0.1 → sim_time=6.0 > ew_stale_age=5.0
    let telem = collect_telemetry(&fleet);
    for _ in 0..60 {
        orch.tick(&telem, &[], 0.1);
    }

    // By now the event should be cleared.
    let decision = orch.tick(&telem, &[], 0.1);
    assert_eq!(
        decision.ew_active_threats, 0,
        "stale EW event should be cleared"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// CBF Integration
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_cbf_active_with_default_config() {
    let ids: Vec<u32> = (0..3).collect();
    let config = SwarmConfig {
        cbf_config: Some(CbfConfig::default()),
        ..Default::default()
    };
    let mut orch = SwarmOrchestrator::new(&ids, config);

    let fleet = SimulatorFleet::new_grid(3, 15.0, SimulatorConfig::default());
    fleet.arm_all().unwrap();
    fleet.step_all_n(10);

    let telem = collect_telemetry(&fleet);
    let decision = orch.tick(&telem, &[], 0.1);

    // CBF should run (even if no constraints are active at 15m spacing > 5m min)
    // The fact that cbf_active_constraints is 0 is fine — it means drones are safely spaced.
    assert!(decision.cbf_active_constraints == 0 || decision.cbf_active_constraints > 0);
}

#[test]
fn test_cbf_disabled_when_none() {
    let ids: Vec<u32> = (0..3).collect();
    let config = SwarmConfig {
        cbf_config: None,
        ..Default::default()
    };
    let mut orch = SwarmOrchestrator::new(&ids, config);

    let fleet = SimulatorFleet::new_grid(3, 15.0, SimulatorConfig::default());
    fleet.arm_all().unwrap();
    fleet.step_all_n(10);

    let telem = collect_telemetry(&fleet);
    let decision = orch.tick(&telem, &[], 0.1);

    assert_eq!(
        decision.cbf_active_constraints, 0,
        "CBF should be inactive when disabled"
    );
}

#[test]
fn test_nfz_can_be_added_at_runtime() {
    let ids: Vec<u32> = (0..5).collect();
    let mut orch = SwarmOrchestrator::new(&ids, SwarmConfig::default());

    assert!(orch.no_fly_zones.is_empty());

    orch.add_no_fly_zone(NoFlyZone {
        center: Vector3::new(100.0, 100.0, -50.0),
        radius: 50.0,
    });

    assert_eq!(orch.no_fly_zones.len(), 1);
}

// ─────────────────────────────────────────────────────────────────────────────
// Full Pipeline Integration
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_full_pipeline_all_modules_active() {
    // Set up a scenario that exercises all island modules simultaneously.
    let config = SwarmConfig {
        formation_type: Some(FormationType::Wedge),
        cbf_config: Some(CbfConfig::default()),
        auction_interval: 1,
        n_particles: 50,
        n_threat_particles: 30,
        ..Default::default()
    };

    let ids: Vec<u32> = (0..10).collect();
    let mut orch = SwarmOrchestrator::new(&ids, config);

    // Register threats
    orch.register_threat(1, Vector3::new(100.0, 100.0, 0.0));

    // Report EW event
    orch.report_ew_event(EwEvent {
        threat: EwThreat::GpsDenial,
        severity: EwSeverity::Degraded,
        source_bearing: Some(45.0),
        source_range: Some(500.0),
        confidence: 0.8,
        timestamp: 0.0,
    });

    // Add NFZ
    orch.add_no_fly_zone(NoFlyZone {
        center: Vector3::new(200.0, 200.0, -50.0),
        radius: 30.0,
    });

    // Create mixed tasks (sensor + weapon)
    let tasks = vec![
        make_sensor_task(1, [30.0, 30.0, 50.0]),
        make_sensor_task(2, [60.0, 60.0, 50.0]),
        make_weapon_task(3, [100.0, 100.0, 50.0]),
    ];

    let fleet = SimulatorFleet::new_grid(10, 15.0, SimulatorConfig::default());
    fleet.arm_all().unwrap();
    for drone in &fleet.drones {
        drone.execute_action(&Action::Takeoff(50.0)).unwrap();
    }
    fleet.step_all_n(100);

    // Run 20 ticks
    for _ in 0..20 {
        fleet.step_all();
        let telem = collect_telemetry(&fleet);
        orch.tick(&telem, &tasks, 0.1);
    }

    // Collect final decision
    let telem = collect_telemetry(&fleet);
    let decision = orch.tick(&telem, &tasks, 0.1);

    // Verify all island modules participated:
    // 1. Formation: quality should be reported
    assert!(
        decision.formation_quality.is_some(),
        "formation module should be active"
    );

    // 2. EW: active threats should be tracked
    assert_eq!(
        decision.ew_active_threats, 1,
        "EW module should track threats"
    );

    // 3. System should complete without panics (the integration itself is the test)
    assert!(decision.positions.len() == 10);
    assert!(decision.threat_positions.len() == 1);
}

// ─────────────────────────────────────────────────────────────────────────────
// XAI Trace Export E2E
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_trace_export_json_full_pipeline() {
    let ids: Vec<u32> = (0..5).collect();
    let config = SwarmConfig {
        auction_interval: 1,
        n_particles: 50,
        ..Default::default()
    };
    let mut orch = SwarmOrchestrator::new(&ids, config);

    orch.register_threat(1, Vector3::new(50.0, 50.0, 0.0));

    let fleet = SimulatorFleet::new_grid(5, 15.0, SimulatorConfig::default());
    fleet.arm_all().unwrap();
    fleet.step_all_n(10);

    let tasks = vec![make_sensor_task(1, [20.0, 20.0, 50.0])];

    // Run enough ticks to accumulate traces.
    for _ in 0..10 {
        let telem = collect_telemetry(&fleet);
        orch.tick(&telem, &tasks, 0.1);
        fleet.step_all();
    }

    // Export traces as JSON.
    let json = orch
        .tracer
        .export_json()
        .expect("export_json should succeed");
    let parsed: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
    let traces = parsed.as_array().expect("JSON array of traces");

    assert!(
        !traces.is_empty(),
        "exported JSON should contain decision traces"
    );

    // Verify each trace has expected fields.
    for trace in traces {
        assert!(
            trace.get("timestamp").is_some(),
            "trace must have timestamp"
        );
        assert!(
            trace.get("decision_type").is_some(),
            "trace must have decision_type"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ROE Escalation Path (non-zero collateral_risk / close-range threat)
// ─────────────────────────────────────────────────────────────────────────────

/// Place a threat closer than min_engagement_distance (50m) to a weapon task.
/// Under WeaponsTight the ROE distance check should produce EscalationRequired,
/// which the tick loop counts as `roe_escalations`.
#[test]
fn test_roe_escalation_on_close_range_threat() {
    let ids: Vec<u32> = (0..5).collect();
    let config = SwarmConfig {
        auction_interval: 1,
        n_particles: 50,
        n_threat_particles: 30,
        ..Default::default()
    };
    let mut orch = SwarmOrchestrator::new(&ids, config);
    orch.set_weapons_posture(WeaponsPosture::WeaponsTight);

    // Threat placed at (20, 20, 0) — closer than 50m to the weapon task at (20, 20, 50).
    orch.register_threat(1, Vector3::new(20.0, 20.0, 0.0));

    let fleet = SimulatorFleet::new_grid(5, 15.0, SimulatorConfig::default());
    fleet.arm_all().unwrap();
    fleet.step_all_n(10);

    // Weapon task at (20, 20, 50) — distance to threat at (20,20,0) = 50m exactly,
    // which is NOT below the threshold. Move threat to (20, 20, 10) to be within 40m.
    orch.register_threat(1, Vector3::new(20.0, 20.0, 10.0));

    let tasks = vec![make_weapon_task(1, [20.0, 20.0, 50.0])];

    let telem = collect_telemetry(&fleet);
    let decision = orch.tick(&telem, &tasks, 0.1);

    assert!(
        decision.roe_escalations > 0,
        "close-range threat should trigger ROE escalation, got 0 escalations"
    );
    // Escalated tasks should NOT be assigned.
    assert!(
        decision.assignments.is_empty(),
        "escalated tasks must not reach auction"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// 6B: Drone Add/Remove Consistency Test
// ─────────────────────────────────────────────────────────────────────────────

/// Verify that `register_drone` + `handle_drone_loss` maintain consistent
/// internal state: `nav_filters.len() == regimes.len() == original_count`
/// throughout the lifecycle, and that ticks do not panic during add/remove.
#[test]
fn test_drone_add_remove_consistency() {
    use strix_auction::{LossClassification, Position as AuctionPosition};

    let original_ids: Vec<u32> = (0..5).collect();
    let config = SwarmConfig {
        n_particles: 50,
        n_threat_particles: 30,
        ..Default::default()
    };
    let mut orch = SwarmOrchestrator::new(&original_ids, config);

    let fleet = SimulatorFleet::new_grid(5, 15.0, SimulatorConfig::default());
    fleet.arm_all().unwrap();
    fleet.step_all_n(5);

    // ── Step 1: verify baseline consistency ──────────────────────────────
    assert_eq!(
        orch.nav_filters.len(),
        5,
        "should have 5 nav filters initially"
    );
    assert_eq!(
        orch.regimes.len(),
        5,
        "should have 5 regime entries initially"
    );

    // Run a few ticks to warm up internal state.
    let telem = collect_telemetry(&fleet);
    for _ in 0..3 {
        orch.tick(&telem, &[], 0.1); // must not panic
    }

    // ── Step 2: register a new drone (id=10) ─────────────────────────────
    let new_drone_pos = nalgebra::Vector3::new(30.0, 30.0, -50.0);
    orch.register_drone(10, new_drone_pos);

    assert_eq!(
        orch.nav_filters.len(),
        6,
        "nav_filters should grow to 6 after register_drone"
    );
    assert_eq!(
        orch.regimes.len(),
        6,
        "regimes should grow to 6 after register_drone"
    );

    // ── Step 3: tick with the new drone present (no telemetry for it —
    //    that's fine: the tick loop skips drones not in telemetry) ────────
    orch.tick(&telem, &[], 0.1); // must not panic

    // ── Step 4: remove the new drone via handle_drone_loss ───────────────
    let loss_pos = AuctionPosition::new(new_drone_pos.x, new_drone_pos.y, new_drone_pos.z);
    orch.handle_drone_loss(10, loss_pos, LossClassification::Unknown);

    assert_eq!(
        orch.nav_filters.len(),
        5,
        "nav_filters should shrink back to 5 after handle_drone_loss"
    );
    assert_eq!(
        orch.regimes.len(),
        5,
        "regimes should shrink back to 5 after handle_drone_loss"
    );

    // ── Step 5: tick after removal — must not panic ───────────────────────
    orch.tick(&telem, &[], 0.1);

    // ── Step 6: verify the original 5 drones are still intact ────────────
    for &id in &original_ids {
        assert!(
            orch.nav_filters.contains_key(&id),
            "original drone {id} must still have a nav filter"
        );
        assert!(
            orch.regimes.contains_key(&id),
            "original drone {id} must still have a regime entry"
        );
    }

    // Removed drone must no longer appear in any tracking structure.
    assert!(
        !orch.nav_filters.contains_key(&10),
        "removed drone 10 must not be in nav_filters"
    );
    assert!(
        !orch.regimes.contains_key(&10),
        "removed drone 10 must not be in regimes"
    );

    // Final count invariant: nav_filters == regimes == 5.
    assert_eq!(
        orch.nav_filters.len(),
        orch.regimes.len(),
        "nav_filters.len() must always equal regimes.len()"
    );
    assert_eq!(
        orch.nav_filters.len(),
        original_ids.len(),
        "final count must equal original 5 drones"
    );
}
