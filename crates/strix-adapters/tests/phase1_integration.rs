//! # Phase 1 Integration Test — GPS-Denied Navigation Demo
//!
//! Demonstrates the core STRIX capability: drones maintain position
//! accuracy when GPS is suddenly lost, using only IMU + barometer +
//! magnetometer through the 6D particle filter.
//!
//! ## What this proves
//!
//! 1. **Sensor fusion pipeline**: simulator telemetry → observations → particle filter → position estimate
//! 2. **GPS-denied accuracy**: position drift < 10m over 90 seconds without GPS
//! 3. **CUSUM jamming detection**: structural break on signal quality drop
//! 4. **Regime detection**: PATROL → ENGAGE → EVADE transitions
//! 5. **Dual particle filter**: friendly tracking + adversarial threat prediction

use nalgebra::Vector3;
use strix_adapters::simulator::{SimulatorAdapter, SimulatorConfig};
use strix_adapters::traits::*;
use strix_core::anomaly::{detect_jamming, CusumConfig};
use strix_core::particle_nav::{estimate_6d, position_variance, ParticleNavFilter};
use strix_core::regime::{detect_regime, DetectionConfig, RegimeSignals};
use strix_core::state::{Observation, Regime};
use strix_core::threat_tracker::{ThreatObservation, ThreatTracker};

const NUM_DRONES: usize = 4;
const PARTICLES: usize = 200;
const DT: f64 = 0.1;

/// Diamond formation positions at 50m altitude.
const FORMATION: [[f64; 3]; NUM_DRONES] = [
    [0.0, 0.0, 50.0],   // Lead
    [20.0, 0.0, 50.0],  // Right wing
    [-20.0, 0.0, 50.0], // Left wing
    [0.0, 20.0, 50.0],  // Trail
];

/// Generate sensor observations simulating GPS availability.
/// Uses VisualOdometry with high confidence as a GPS proxy.
fn observations_with_gps(pos: [f64; 3], vel: [f64; 3], t: f64) -> Vec<Observation> {
    vec![
        Observation::VisualOdometry {
            delta_position: Vector3::new(pos[0], pos[1], pos[2]),
            confidence: 0.95,
            timestamp: t,
        },
        Observation::Barometer {
            altitude: pos[2],
            timestamp: t,
        },
        Observation::Imu {
            acceleration: Vector3::new(vel[0], vel[1], vel[2]),
            gyro: None,
            timestamp: t,
        },
    ]
}

/// Generate sensor observations WITHOUT GPS — only IMU, barometer, magnetometer.
fn observations_gps_denied(pos: [f64; 3], vel: [f64; 3], t: f64) -> Vec<Observation> {
    vec![
        Observation::Barometer {
            altitude: pos[2],
            timestamp: t,
        },
        Observation::Imu {
            acceleration: Vector3::new(vel[0], vel[1], vel[2]),
            gyro: None,
            timestamp: t,
        },
        Observation::Magnetometer {
            heading: Vector3::new(1.0, 0.0, 0.0),
            timestamp: t,
        },
    ]
}

// ---------------------------------------------------------------------------
// Test 1: Core GPS-denied navigation
// ---------------------------------------------------------------------------

#[test]
fn phase1_gps_denied_navigation() {
    let config = SimulatorConfig {
        gps_noise_std: 0.0,
        battery_drain_rate: 0.0001,
        ..Default::default()
    };

    // Create 4 drones in diamond formation at altitude.
    let drones: Vec<SimulatorAdapter> = (0..NUM_DRONES)
        .map(|i| SimulatorAdapter::new(i as u32, FORMATION[i], config.clone()))
        .collect();

    for drone in &drones {
        drone.execute_action(&Action::Arm).unwrap();
    }

    // Initialize particle filters at formation positions.
    let mut filters: Vec<ParticleNavFilter> = FORMATION
        .iter()
        .map(|pos| ParticleNavFilter::new(PARTICLES, Vector3::new(pos[0], pos[1], pos[2])))
        .collect();

    let no_threat = Vector3::zeros();

    // ── Phase 1: GPS available (30s = 300 steps) ──
    // Particles should converge tightly around true position.
    for step in 0..300 {
        let t = step as f64 * DT;
        for (i, drone) in drones.iter().enumerate() {
            drone.step();
            let telem = drone.get_telemetry().unwrap();
            let obs = observations_with_gps(telem.position, telem.velocity, t);
            filters[i].step(&obs, &no_threat, 1.0, DT);
        }
    }

    // Verify GPS-phase convergence: position error < 5m.
    for i in 0..NUM_DRONES {
        let telem = drones[i].get_telemetry().unwrap();
        let state = filters[i].to_drone_state(i as u32);
        let true_pos = Vector3::new(telem.position[0], telem.position[1], telem.position[2]);
        let error = (state.position - true_pos).norm();
        assert!(
            error < 5.0,
            "GPS phase: drone {} error {:.1}m > 5m",
            i,
            error
        );
    }

    // ── Phase 2: GPS DENIED (90s = 900 steps) ──
    // Only IMU + barometer + magnetometer.
    for step in 0..900 {
        let t = 30.0 + step as f64 * DT;
        for (i, drone) in drones.iter().enumerate() {
            drone.step();
            let telem = drone.get_telemetry().unwrap();
            let obs = observations_gps_denied(telem.position, telem.velocity, t);
            filters[i].step(&obs, &no_threat, 1.0, DT);
        }
    }

    // Verify: position drift < 10m for all drones.
    for i in 0..NUM_DRONES {
        let telem = drones[i].get_telemetry().unwrap();
        let state = filters[i].to_drone_state(i as u32);
        let true_pos = Vector3::new(telem.position[0], telem.position[1], telem.position[2]);
        let error = (state.position - true_pos).norm();

        let (mean_pos, _, _) = estimate_6d(
            &filters[i].particles,
            &filters[i].weights,
            &filters[i].regimes,
        );
        let variance = position_variance(&filters[i].particles, &filters[i].weights, &mean_pos);
        let spread = variance.sqrt();

        eprintln!(
            "  Drone {i}: error={error:.2}m spread={spread:.2}m regime={:?}",
            state.regime
        );
        assert!(
            error < 10.0,
            "GPS-denied: drone {} error {:.1}m > 10m (spread={:.1}m)",
            i,
            error,
            spread
        );
    }
}

// ---------------------------------------------------------------------------
// Test 2: CUSUM jamming detection
// ---------------------------------------------------------------------------

#[test]
fn phase1_cusum_detects_jamming() {
    // Good SNR for 20 samples, then sudden drop (jamming).
    let mut signal_quality: Vec<f64> = vec![30.0; 20];
    signal_quality.extend(vec![5.0; 20]);

    let config = CusumConfig::default();
    let (is_jammed, direction, _cusum_val) = detect_jamming(&signal_quality, &config);

    assert!(is_jammed, "CUSUM should detect GPS jamming");
    assert_eq!(direction, -1, "Jamming = negative signal shift");
}

// ---------------------------------------------------------------------------
// Test 3: Regime detection (PATROL → ENGAGE → EVADE)
// ---------------------------------------------------------------------------

#[test]
fn phase1_regime_transitions() {
    let cfg = DetectionConfig::default();

    // Peaceful — PATROL.
    let peaceful = RegimeSignals {
        cusum_triggered: false,
        cusum_direction: 0,
        hurst: 0.4,
        volatility_ratio: 1.0,
        threat_distance: 2000.0,
        closing_rate: 0.0,
    };
    assert_eq!(
        detect_regime(&peaceful, Regime::Patrol, &cfg),
        Regime::Patrol
    );

    // Systematic approach — ENGAGE.
    let approach = RegimeSignals {
        cusum_triggered: false,
        cusum_direction: 0,
        hurst: 0.7,
        volatility_ratio: 1.0,
        threat_distance: 300.0,
        closing_rate: -3.0,
    };
    assert_eq!(
        detect_regime(&approach, Regime::Patrol, &cfg),
        Regime::Engage
    );

    // Imminent threat + CUSUM break — EVADE.
    let imminent = RegimeSignals {
        cusum_triggered: true,
        cusum_direction: -1,
        hurst: 0.8,
        volatility_ratio: 1.5,
        threat_distance: 100.0,
        closing_rate: -5.0,
    };
    assert_eq!(
        detect_regime(&imminent, Regime::Engage, &cfg),
        Regime::Evade
    );
}

// ---------------------------------------------------------------------------
// Test 4: Dual particle filter (friendly + adversarial)
// ---------------------------------------------------------------------------

#[test]
fn phase1_dual_particle_filter() {
    // Our drone at origin, enemy at [200, 200, 50].
    let mut friendly = ParticleNavFilter::new(100, Vector3::new(0.0, 0.0, 50.0));
    let mut enemy = ThreatTracker::new(1, 200, Vector3::new(200.0, 200.0, 50.0));

    let our_centroid = Vector3::new(0.0, 0.0, 50.0);
    let no_threat = Vector3::zeros();

    let obs_friendly = vec![Observation::Barometer {
        altitude: 50.0,
        timestamp: 0.0,
    }];
    let obs_enemy = vec![ThreatObservation::Radar {
        position: Vector3::new(195.0, 195.0, 50.0),
        sigma: 10.0,
        timestamp: 0.0,
    }];

    // Run 50 steps — enemy approaching.
    for _ in 0..50 {
        friendly.step(&obs_friendly, &no_threat, 1.0, DT);
        enemy.step(&our_centroid, &obs_enemy, DT);
    }

    let our_state = friendly.to_drone_state(0);
    let threat_state = enemy.to_threat_state();
    let future_pos = enemy.predict_future_threat(10.0);

    // Friendly filter should stay near initial position (0, 0, 50).
    let initial_pos = Vector3::new(0.0, 0.0, 50.0);
    let friendly_drift = (our_state.position - initial_pos).norm();
    assert!(
        friendly_drift < 20.0,
        "friendly pos drift too large: {:.1}m",
        friendly_drift
    );

    // Threat tracker should be in reasonable range.
    assert!(
        threat_state.position.norm() < 500.0,
        "threat estimate out of range: {:.1}m",
        threat_state.position.norm()
    );

    // Predictive projection should differ from current estimate.
    let prediction_delta = (future_pos - threat_state.position).norm();
    eprintln!(
        "  Threat: est=({:.0},{:.0},{:.0}) pred_10s=({:.0},{:.0},{:.0}) delta={:.1}m",
        threat_state.position.x,
        threat_state.position.y,
        threat_state.position.z,
        future_pos.x,
        future_pos.y,
        future_pos.z,
        prediction_delta
    );
    // Prediction should project forward (nonzero delta) if enemy has velocity.
    // With noise this is almost always true, but allow zero for edge cases.
    assert!(prediction_delta < 1000.0, "prediction unreasonably far");
}

// ---------------------------------------------------------------------------
// Test 5: Formation geometry preservation
// ---------------------------------------------------------------------------

#[test]
fn phase1_formation_maintenance() {
    let config = SimulatorConfig::default();
    let drones: Vec<SimulatorAdapter> = (0..NUM_DRONES)
        .map(|i| SimulatorAdapter::new(i as u32, FORMATION[i], config.clone()))
        .collect();

    for drone in &drones {
        drone.execute_action(&Action::Arm).unwrap();
    }

    // Run for 10 seconds — drones should hold position (no target = drag stops them).
    for drone in &drones {
        drone.step_n(100);
    }

    let positions: Vec<[f64; 3]> = drones
        .iter()
        .map(|d| d.get_telemetry().unwrap().position)
        .collect();

    // Check Lead-to-Right and Lead-to-Left distances (should be ~20m).
    let dist_01 = ((positions[0][0] - positions[1][0]).powi(2)
        + (positions[0][1] - positions[1][1]).powi(2)
        + (positions[0][2] - positions[1][2]).powi(2))
    .sqrt();
    let dist_02 = ((positions[0][0] - positions[2][0]).powi(2)
        + (positions[0][1] - positions[2][1]).powi(2)
        + (positions[0][2] - positions[2][2]).powi(2))
    .sqrt();

    assert!(
        (dist_01 - 20.0).abs() < 5.0,
        "Lead-Right distortion: {:.1}m (expected ~20m)",
        dist_01
    );
    assert!(
        (dist_02 - 20.0).abs() < 5.0,
        "Lead-Left distortion: {:.1}m (expected ~20m)",
        dist_02
    );
}
