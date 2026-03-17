use strix_playground::report::TimelineEventType;
use strix_playground::Playground;

#[test]
fn ambush_runs_without_panic() {
    let report = Playground::ambush().run_for(60.0).run();
    assert!(report.aggregates.total_ticks > 0);
    assert!(report.aggregates.cbf_violations == 0);
    // With 3 approaching threats, we expect regime activity
    assert!(
        report.aggregates.auction_rounds > 0,
        "Expected auction rounds in ambush scenario"
    );
    println!("{}", report);
}

#[test]
fn gps_denied_has_jam_event() {
    let report = Playground::gps_denied().run_for(90.0).run();
    assert!(report.aggregates.total_ticks > 0);
    assert!(
        report
            .timeline
            .iter()
            .any(|e| matches!(e.event_type, TimelineEventType::GpsJammed)),
        "Expected GPS jammed event in timeline"
    );
    assert!(
        report
            .timeline
            .iter()
            .any(|e| matches!(e.event_type, TimelineEventType::GpsRestored)),
        "Expected GPS restored event in timeline"
    );
    println!("{}", report);
}

#[test]
fn attrition_loses_drones() {
    let report = Playground::attrition().run_for(100.0).run();
    assert!(
        report.aggregates.drones_lost >= 3,
        "Expected at least 3 drone losses, got {}",
        report.aggregates.drones_lost
    );
    assert!(
        report
            .timeline
            .iter()
            .any(|e| matches!(e.event_type, TimelineEventType::DroneLost { .. })),
        "Expected drone loss events"
    );
    println!("{}", report);
}

#[test]
fn stress_test_survives_chaos() {
    let report = Playground::stress_test().with_json().run_for(120.0).run();
    assert!(report.aggregates.total_ticks > 0);
    assert!(report.aggregates.cbf_violations == 0);
    assert!(report.tick_data.is_some(), "Expected JSON tick data");
    let ticks = report.tick_data.as_ref().unwrap();
    assert!(ticks.len() > 100, "Expected many tick snapshots");
    println!("{}", report);
}

#[test]
fn custom_scenario_works() {
    use strix_playground::ThreatSpec;

    let report = Playground::new()
        .name("Custom")
        .drones(5)
        .altitude(-30.0)
        .add_threat(ThreatSpec::approaching(200.0, 5.0))
        .run_for(10.0)
        .run();

    assert!(report.aggregates.total_ticks > 0);
    assert_eq!(report.scenario_name, "Custom");
    assert_eq!(report.n_drones_initial, 5);
    assert_eq!(report.n_threats_initial, 1);
}

#[cfg(feature = "temporal")]
#[test]
fn temporal_horizons_runs() {
    let report = Playground::temporal().run_for(180.0).run();
    assert!(report.aggregates.total_ticks > 0);
    assert!(report.aggregates.cbf_violations == 0);
    // With 3 different-speed threats, expect regime activity.
    assert!(
        report.aggregates.regime_changes > 0,
        "Expected regime changes with multi-speed threats"
    );
    // Temporal anomaly count should be non-negative (smoke check).
    assert!(report.aggregates.temporal_anomaly_count >= 0);
}
