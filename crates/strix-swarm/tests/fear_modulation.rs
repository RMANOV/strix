//! Integration tests for the fear-modulation helpers (A3 bug-hunt).
//!
//! These exercise the deterministic, non-feature-gated public functions
//! `modulate_detection_config` and `modulate_gossip_fanout` from
//! `strix_swarm::fear_adapter` through the crate's PUBLIC API only.
//!
//! Properties asserted:
//! - `evade_distance` strictly INCREASES as fear F rises.
//! - `closing_rate_threshold` strictly DECREASES as fear F rises.
//! - `gossip_fanout` is non-decreasing in F and never exceeds `3 * base`.
//! - Clamping/sanitization: F = 1.5 is identical to F = 1.0; F = NaN and
//!   F = +Inf each collapse to the same safe output as F = 0.0.
//!
//! NOTE (signature reality vs. task brief): the modulation functions take
//! `f64`, not `f32`, so the NaN/Inf sentinels here are `f64::NAN` /
//! `f64::INFINITY`. These are pure deterministic functions — no RNG, no async.
//! They are NOT behind the `phi-sim` feature, so this test runs under the
//! crate's default features.

use strix_swarm::fear_adapter::{modulate_detection_config, modulate_gossip_fanout};

/// Helper: build a fresh default detection config for each assertion so no
/// mutation can leak between checks.
fn base_config() -> strix_core::regime::DetectionConfig {
    strix_core::regime::DetectionConfig::default()
}

// ── Monotonicity of detection-config modulation ──────────────────────────────

#[test]
fn evade_distance_strictly_increases_with_fear() {
    let base = base_config();
    let evade_at = |f: f64| modulate_detection_config(&base, f).evade_distance;

    let e0 = evade_at(0.0);
    let e_half = evade_at(0.5);
    let e1 = evade_at(1.0);

    assert!(
        e0 < e_half,
        "evade_distance must strictly increase 0.0 -> 0.5: {e0} !< {e_half}"
    );
    assert!(
        e_half < e1,
        "evade_distance must strictly increase 0.5 -> 1.0: {e_half} !< {e1}"
    );
}

#[test]
fn closing_rate_threshold_strictly_decreases_with_fear() {
    let base = base_config();
    let closing_at = |f: f64| modulate_detection_config(&base, f).closing_rate_threshold;

    let c0 = closing_at(0.0);
    let c_half = closing_at(0.5);
    let c1 = closing_at(1.0);

    assert!(
        c0 > c_half,
        "closing_rate_threshold must strictly decrease 0.0 -> 0.5: {c0} !> {c_half}"
    );
    assert!(
        c_half > c1,
        "closing_rate_threshold must strictly decrease 0.5 -> 1.0: {c_half} !> {c1}"
    );
}

// ── Monotonicity + hard cap of gossip fanout ─────────────────────────────────

#[test]
fn gossip_fanout_non_decreasing_and_capped_at_3x_base() {
    let base: usize = 3;
    let cap = 3 * base;

    // Sweep F across [0, 1] in fine steps: non-decreasing AND within the cap.
    let mut prev = modulate_gossip_fanout(base, 0.0);
    assert!(
        prev <= cap,
        "fanout at F=0 must be <= 3*base: {prev} > {cap}"
    );
    for step in 1..=20 {
        let f = step as f64 / 20.0;
        let current = modulate_gossip_fanout(base, f);
        assert!(
            current >= prev,
            "fanout must be non-decreasing in F: F={f}, prev={prev}, cur={current}"
        );
        assert!(
            current <= cap,
            "fanout must never exceed 3*base: F={f}, cur={current}, cap={cap}"
        );
        prev = current;
    }

    // The canonical 0.0 / 0.5 / 1.0 sample points, explicitly.
    let f0 = modulate_gossip_fanout(base, 0.0);
    let f_half = modulate_gossip_fanout(base, 0.5);
    let f1 = modulate_gossip_fanout(base, 1.0);
    assert!(f0 <= f_half, "fanout(0.0) <= fanout(0.5): {f0} > {f_half}");
    assert!(f_half <= f1, "fanout(0.5) <= fanout(1.0): {f_half} > {f1}");
    assert!(f1 <= cap, "fanout(1.0) within cap: {f1} > {cap}");
}

// ── Clamping above 1.0 ───────────────────────────────────────────────────────

#[test]
fn fear_above_one_clamps_to_one() {
    let base = base_config();

    let at_1_5 = modulate_detection_config(&base, 1.5);
    let at_1_0 = modulate_detection_config(&base, 1.0);

    assert_eq!(
        at_1_5.evade_distance, at_1_0.evade_distance,
        "evade_distance: F=1.5 must equal F=1.0 (clamped)"
    );
    assert_eq!(
        at_1_5.closing_rate_threshold, at_1_0.closing_rate_threshold,
        "closing_rate_threshold: F=1.5 must equal F=1.0 (clamped)"
    );
    assert_eq!(
        at_1_5.engage_distance, at_1_0.engage_distance,
        "engage_distance: F=1.5 must equal F=1.0 (clamped)"
    );

    // Gossip fanout clamps the same way.
    let base_fanout: usize = 4;
    assert_eq!(
        modulate_gossip_fanout(base_fanout, 1.5),
        modulate_gossip_fanout(base_fanout, 1.0),
        "gossip_fanout: F=1.5 must equal F=1.0 (clamped)"
    );
}

// ── Non-finite sanitization (NaN / +Inf collapse to F = 0.0) ─────────────────

#[test]
fn nan_fear_collapses_to_zero() {
    let base = base_config();

    let at_nan = modulate_detection_config(&base, f64::NAN);
    let at_zero = modulate_detection_config(&base, 0.0);

    assert_eq!(
        at_nan.evade_distance, at_zero.evade_distance,
        "evade_distance: F=NaN must equal F=0.0 (safe default)"
    );
    assert_eq!(
        at_nan.closing_rate_threshold, at_zero.closing_rate_threshold,
        "closing_rate_threshold: F=NaN must equal F=0.0 (safe default)"
    );
    assert_eq!(
        at_nan.engage_distance, at_zero.engage_distance,
        "engage_distance: F=NaN must equal F=0.0 (safe default)"
    );

    assert_eq!(
        modulate_gossip_fanout(3, f64::NAN),
        modulate_gossip_fanout(3, 0.0),
        "gossip_fanout: F=NaN must equal F=0.0 (safe default)"
    );
}

#[test]
fn infinite_fear_collapses_to_zero() {
    let base = base_config();

    let at_inf = modulate_detection_config(&base, f64::INFINITY);
    let at_zero = modulate_detection_config(&base, 0.0);

    assert_eq!(
        at_inf.evade_distance, at_zero.evade_distance,
        "evade_distance: F=+Inf must equal F=0.0 (safe default)"
    );
    assert_eq!(
        at_inf.closing_rate_threshold, at_zero.closing_rate_threshold,
        "closing_rate_threshold: F=+Inf must equal F=0.0 (safe default)"
    );
    assert_eq!(
        at_inf.engage_distance, at_zero.engage_distance,
        "engage_distance: F=+Inf must equal F=0.0 (safe default)"
    );

    assert_eq!(
        modulate_gossip_fanout(3, f64::INFINITY),
        modulate_gossip_fanout(3, 0.0),
        "gossip_fanout: F=+Inf must equal F=0.0 (safe default)"
    );
}
