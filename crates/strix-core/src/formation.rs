//! Formation control — geometry computation and correction vectors for drone swarms.
//!
//! This module provides the enforcement loop that the Python geometry code lacked:
//! it computes ideal slot positions for each tactical formation type and derives
//! per-drone correction velocity vectors that guide the fleet back into shape.
//!
//! ## Formation types
//!
//! | Type          | Use case                                           |
//! |---------------|----------------------------------------------------|
//! | Vee           | Aerodynamic efficiency + wide sensor coverage      |
//! | Line          | Maximum sensor sweep width, abreast                |
//! | Wedge         | Balanced offense/defense, tighter than Vee         |
//! | Column        | Minimum cross-section, maximum stealth             |
//! | EchelonLeft   | Flanking approach from left                        |
//! | EchelonRight  | Flanking approach from right                       |
//! | Spread        | Maximum area coverage, grid pattern                |
//!
//! ## Control law
//!
//! The correction velocity uses proportional control with speed clamping:
//!
//! ```text
//! if ||delta|| < deadband  →  v_corr = 0
//! else                     →  v_corr = (delta / ||delta||) * min(||delta||, v_max)
//! ```
//!
//! This saturates correction speed at `max_correction_speed` while pointing
//! directly towards the target slot.

use nalgebra::Vector3;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Formation type
// ---------------------------------------------------------------------------

/// Supported tactical formation types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FormationType {
    /// V-formation: aerodynamic efficiency + wide sensor coverage.
    Vee,
    /// Line abreast: maximum sensor sweep width.
    Line,
    /// Wedge: balanced offense/defense (tighter Vee at 15°).
    Wedge,
    /// Column: minimum cross-section, maximum stealth.
    Column,
    /// Echelon left: flanking approach, all drones offset left-rear.
    EchelonLeft,
    /// Echelon right: flanking approach, all drones offset right-rear.
    EchelonRight,
    /// Spread: maximum area coverage, grid pattern.
    Spread,
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for formation geometry and control law.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormationConfig {
    /// Inter-drone spacing (meters).
    pub spacing: f64,
    /// V-formation half-angle (degrees). Converted to radians internally.
    /// Wedge overrides this with 15° regardless of this setting.
    pub vee_angle_deg: f64,
    /// Maximum correction velocity (m/s) — saturates proportional law.
    pub max_correction_speed: f64,
    /// Deadband radius (meters) — no correction if drone is already within
    /// this distance of its target slot.
    pub deadband: f64,
}

impl Default for FormationConfig {
    fn default() -> Self {
        Self {
            spacing: 15.0,
            vee_angle_deg: 30.0,
            max_correction_speed: 5.0,
            deadband: 1.0,
        }
    }
}

impl FormationConfig {
    /// Return a copy with spacing/deadband modulated by FearAxes.threshold.
    /// threshold ∈ [0.3, 1.0]:
    ///   threshold=1.0 → spacing×0.8, deadband×0.5 (tight, disciplined)
    ///   threshold=0.3 → spacing×1.5, deadband×2.0 (loose, survival-oriented)
    pub fn fear_adjusted(&self, threshold: f64) -> Self {
        let t = threshold.clamp(0.3, 1.0);
        let spacing_scale = 1.8 - t; // 1.0→0.8, 0.3→1.5
        let deadband_scale = 2.5 - 2.0 * t; // 1.0→0.5, 0.3→1.9
        Self {
            spacing: self.spacing * spacing_scale,
            deadband: self.deadband * deadband_scale,
            ..self.clone()
        }
    }
}

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------

/// Build a unit vector in the XY plane from `direction`, ignoring the Z component.
///
/// Falls back to the X axis if `direction` has no XY component (vertical direction).
fn xy_unit(direction: &Vector3<f64>) -> Vector3<f64> {
    let flat = Vector3::new(direction.x, direction.y, 0.0);
    let norm = flat.norm();
    if norm < 1e-12 {
        Vector3::new(1.0, 0.0, 0.0)
    } else {
        flat / norm
    }
}

/// Compute the right-perpendicular to a forward direction in the XY plane.
///
/// `forward` must already be a unit vector in the XY plane.
/// Returns a unit vector pointing 90° clockwise (right of the heading).
fn right_of(forward: &Vector3<f64>) -> Vector3<f64> {
    // Rotate 90° CW in XY: (fx, fy, 0) → (fy, -fx, 0)
    Vector3::new(forward.y, -forward.x, 0.0)
}

// ---------------------------------------------------------------------------
// Formation position computation
// ---------------------------------------------------------------------------

/// Compute target positions for a formation centred at `center`, heading `direction`.
///
/// Returns `Vec<(drone_index, target_position)>` where drone index 0 is the
/// leader and is always placed at `center`. Altitude (Z) is preserved from
/// `center` for all drones so the formation lies in a horizontal plane.
///
/// # Arguments
/// * `formation`  — tactical shape
/// * `n_drones`   — number of drones (must be ≥ 1)
/// * `center`     — formation centroid / leader position (NED, meters)
/// * `direction`  — heading unit vector (XY plane; Z is ignored and normalised away)
/// * `config`     — spacing, angles, and control parameters
///
/// # Panics
/// Does not panic. Returns an empty Vec when `n_drones == 0`.
pub fn compute_formation_positions(
    formation: FormationType,
    n_drones: usize,
    center: &Vector3<f64>,
    direction: &Vector3<f64>,
    config: &FormationConfig,
) -> Vec<(usize, Vector3<f64>)> {
    if n_drones == 0 {
        return Vec::new();
    }

    let fwd = xy_unit(direction);
    let right = right_of(&fwd);
    let z = center.z;
    let s = config.spacing;

    match formation {
        FormationType::Vee => {
            compute_vee(n_drones, center, &fwd, &right, z, s, config.vee_angle_deg)
        }
        FormationType::Wedge => compute_vee(n_drones, center, &fwd, &right, z, s, 15.0),
        FormationType::Line => compute_line(n_drones, center, &right, z, s),
        FormationType::Column => compute_column(n_drones, center, &fwd, z, s),
        FormationType::EchelonLeft => compute_echelon(n_drones, center, &fwd, &right, z, s, -1.0),
        FormationType::EchelonRight => compute_echelon(n_drones, center, &fwd, &right, z, s, 1.0),
        FormationType::Spread => compute_spread(n_drones, center, &fwd, &right, z, s),
    }
}

/// V-formation (and Wedge, same algorithm with different angle).
///
/// Leader at `center`. Wingmen alternate left/right in increasing ranks.
/// Each rank is `spacing * cos(angle_rad)` behind the previous and
/// `spacing * sin(angle_rad)` laterally outward.
fn compute_vee(
    n_drones: usize,
    center: &Vector3<f64>,
    fwd: &Vector3<f64>,
    right: &Vector3<f64>,
    z: f64,
    spacing: f64,
    angle_deg: f64,
) -> Vec<(usize, Vector3<f64>)> {
    let angle_rad = angle_deg.to_radians();
    let back_step = spacing * angle_rad.cos(); // longitudinal offset per rank
    let side_step = spacing * angle_rad.sin(); // lateral offset per rank

    let mut positions = Vec::with_capacity(n_drones);
    // Leader at index 0 — at center.
    positions.push((0, *center));

    // Remaining drones pair up: (1,2), (3,4), (5,6), …
    // Odd index → right wing, even index → left wing (1-based pairing).
    for i in 1..n_drones {
        let rank = i.div_ceil(2) as f64; // 1,1,2,2,3,3,…
        let side = if i % 2 == 1 { 1.0 } else { -1.0 }; // right / left

        let pos = center
            - fwd * (back_step * rank)    // move backward in formation
            + right * (side_step * rank * side); // spread outward
        let pos = Vector3::new(pos.x, pos.y, z);
        positions.push((i, pos));
    }
    positions
}

/// Line abreast — all drones in a row perpendicular to `fwd`, centred on `center`.
fn compute_line(
    n_drones: usize,
    center: &Vector3<f64>,
    right: &Vector3<f64>,
    z: f64,
    spacing: f64,
) -> Vec<(usize, Vector3<f64>)> {
    let mut positions = Vec::with_capacity(n_drones);
    // Total span = (n-1) * spacing; leftmost offset = -(n-1)/2 * spacing.
    let half_span = (n_drones as f64 - 1.0) * spacing / 2.0;

    for i in 0..n_drones {
        let lateral = -half_span + i as f64 * spacing;
        let pos = center + right * lateral;
        let pos = Vector3::new(pos.x, pos.y, z);
        positions.push((i, pos));
    }
    positions
}

/// Column — all drones in a single file behind the leader.
fn compute_column(
    n_drones: usize,
    center: &Vector3<f64>,
    fwd: &Vector3<f64>,
    z: f64,
    spacing: f64,
) -> Vec<(usize, Vector3<f64>)> {
    let mut positions = Vec::with_capacity(n_drones);
    for i in 0..n_drones {
        // Drone 0 at center, each subsequent drone one spacing behind.
        let pos = center - fwd * (i as f64 * spacing);
        let pos = Vector3::new(pos.x, pos.y, z);
        positions.push((i, pos));
    }
    positions
}

/// Echelon — all drones offset to one side, each one spacing behind and
/// one spacing to the side of the previous.
///
/// `side_sign`: +1.0 = right (EchelonRight), -1.0 = left (EchelonLeft).
fn compute_echelon(
    n_drones: usize,
    center: &Vector3<f64>,
    fwd: &Vector3<f64>,
    right: &Vector3<f64>,
    z: f64,
    spacing: f64,
    side_sign: f64,
) -> Vec<(usize, Vector3<f64>)> {
    let mut positions = Vec::with_capacity(n_drones);
    for i in 0..n_drones {
        let rank = i as f64;
        let pos = center - fwd * (spacing * rank) + right * (spacing * rank * side_sign);
        let pos = Vector3::new(pos.x, pos.y, z);
        positions.push((i, pos));
    }
    positions
}

/// Spread — drones distributed in a grid centred on `center`.
///
/// Grid is oriented with columns along `fwd` and rows along `right`.
/// Layout: square-ish grid, `ceil(sqrt(n))` columns, enough rows to fit all drones.
fn compute_spread(
    n_drones: usize,
    center: &Vector3<f64>,
    fwd: &Vector3<f64>,
    right: &Vector3<f64>,
    z: f64,
    spacing: f64,
) -> Vec<(usize, Vector3<f64>)> {
    let cols = (n_drones as f64).sqrt().ceil() as usize;
    let rows = n_drones.div_ceil(cols);

    // Grid origin: top-left corner of the grid so the grid is centred on `center`.
    let col_span = (cols as f64 - 1.0) * spacing;
    let row_span = (rows as f64 - 1.0) * spacing;

    let mut positions = Vec::with_capacity(n_drones);
    for i in 0..n_drones {
        let col = i % cols;
        let row = i / cols;

        // Center the grid: offset by half the total span in each axis.
        let lateral = col as f64 * spacing - col_span / 2.0;
        let longitudinal = row as f64 * spacing - row_span / 2.0;

        let pos = center + right * lateral - fwd * longitudinal;
        let pos = Vector3::new(pos.x, pos.y, z);
        positions.push((i, pos));
    }
    positions
}

// ---------------------------------------------------------------------------
// Correction velocity
// ---------------------------------------------------------------------------

/// Compute correction velocity for a single drone to reach its formation slot.
///
/// Uses proportional control with speed saturation and deadband:
/// - If `||target - current|| < deadband`, returns zero vector (drone is close enough).
/// - Otherwise returns a velocity pointing toward the target, clamped to
///   `max_correction_speed`.
///
/// # Arguments
/// * `current_pos` — drone's current position (NED, meters)
/// * `target_pos`  — desired formation slot position (NED, meters)
/// * `config`      — deadband and max correction speed
pub fn formation_correction(
    current_pos: &Vector3<f64>,
    target_pos: &Vector3<f64>,
    config: &FormationConfig,
) -> Vector3<f64> {
    let delta = target_pos - current_pos;
    let dist = delta.norm();
    if dist <= config.deadband {
        return Vector3::zeros();
    }
    let direction = delta / dist;
    let speed = dist.min(config.max_correction_speed);
    direction * speed
}

// ---------------------------------------------------------------------------
// Formation quality metric
// ---------------------------------------------------------------------------

/// Compute a formation quality score ∈ [0, 1].
///
/// 1.0 means every drone is exactly at its target slot. 0.0 means the
/// average positional error equals or exceeds `spacing` (completely scattered).
///
/// # Algorithm
/// For each drone, find the nearest unassigned target slot (greedy nearest-
/// neighbour assignment), measure the distance, and average across all drones.
/// Quality = 1 − clamp(avg_error / spacing, 0, 1).
///
/// Greedy assignment is O(n²) and sufficient for the typical swarm size
/// (≤ 64 drones). A full Hungarian algorithm would be O(n³) but the
/// greedy result is identical when drones are near their correct slots.
///
/// # Arguments
/// * `current_positions` — slice of `(drone_id, position)` tuples
/// * `target_positions`  — slice of `(slot_index, position)` tuples (from
///   `compute_formation_positions`)
/// * `config`            — provides `spacing` as the normalising distance
pub fn formation_quality(
    current_positions: &[(u32, Vector3<f64>)],
    target_positions: &[(usize, Vector3<f64>)],
    config: &FormationConfig,
) -> f64 {
    if current_positions.is_empty() || target_positions.is_empty() {
        return 1.0; // vacuously perfect
    }

    // Collect target positions into a mutable pool for greedy assignment.
    let mut unmatched: Vec<&Vector3<f64>> = target_positions.iter().map(|(_, p)| p).collect();

    let mut total_error = 0.0_f64;
    let n = current_positions.len();

    for (_, drone_pos) in current_positions {
        if unmatched.is_empty() {
            // More drones than slots — remaining drones contribute full spacing error.
            total_error += config.spacing;
            continue;
        }

        // Find the nearest unassigned slot.
        let (best_idx, best_dist) = unmatched
            .iter()
            .enumerate()
            .map(|(i, slot)| (i, (drone_pos - *slot).norm()))
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap(); // safe: unmatched is non-empty

        unmatched.swap_remove(best_idx);
        total_error += best_dist;
    }

    let avg_error = total_error / n as f64;
    let normalised = avg_error / config.spacing;
    1.0 - normalised.clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-9;

    fn default_config() -> FormationConfig {
        FormationConfig::default()
    }

    // ── Vee formation ──────────────────────────────────────────────────────

    #[test]
    fn test_vee_formation_geometry() {
        let config = default_config(); // spacing=15, angle=30°
        let center = Vector3::new(0.0, 0.0, -50.0);
        let direction = Vector3::new(1.0, 0.0, 0.0); // heading north (+X)

        let positions =
            compute_formation_positions(FormationType::Vee, 5, &center, &direction, &config);

        assert_eq!(positions.len(), 5);

        // Drone 0 (leader) must be at center.
        let (idx0, pos0) = &positions[0];
        assert_eq!(*idx0, 0);
        assert!(
            (pos0 - center).norm() < EPS,
            "leader not at center: {:?}",
            pos0
        );

        // All drones at the same altitude.
        for (_, pos) in &positions {
            assert!(
                (pos.z - center.z).abs() < EPS,
                "drone not at formation altitude: {:?}",
                pos
            );
        }

        let angle_rad = 30_f64.to_radians();
        let back_step = 15.0 * angle_rad.cos(); // ≈ 12.99 m
        let side_step = 15.0 * angle_rad.sin(); // ≈ 7.5 m

        // Drone 1 (right wing, rank 1): back by back_step, right by side_step.
        let (_, pos1) = &positions[1];
        assert!(
            (pos1.x - (center.x - back_step)).abs() < EPS,
            "drone1 x wrong"
        );
        assert!(
            (pos1.y - (center.y - side_step)).abs() < EPS,
            "drone1 y wrong"
        );
        // NB: right of (+X forward) is (0, -1, 0) direction.

        // Drone 2 (left wing, rank 1): back by back_step, left by side_step.
        let (_, pos2) = &positions[2];
        assert!(
            (pos2.x - (center.x - back_step)).abs() < EPS,
            "drone2 x wrong"
        );
        assert!(
            (pos2.y - (center.y + side_step)).abs() < EPS,
            "drone2 y wrong"
        );

        // Drone 3 (right wing, rank 2): back by 2*back_step, right by 2*side_step.
        let (_, pos3) = &positions[3];
        assert!(
            (pos3.x - (center.x - 2.0 * back_step)).abs() < EPS,
            "drone3 x wrong"
        );
        assert!(
            (pos3.y - (center.y - 2.0 * side_step)).abs() < EPS,
            "drone3 y wrong"
        );

        // Drone 4 (left wing, rank 2).
        let (_, pos4) = &positions[4];
        assert!(
            (pos4.x - (center.x - 2.0 * back_step)).abs() < EPS,
            "drone4 x wrong"
        );
        assert!(
            (pos4.y - (center.y + 2.0 * side_step)).abs() < EPS,
            "drone4 y wrong"
        );
    }

    #[test]
    fn test_vee_single_drone() {
        let config = default_config();
        let center = Vector3::new(5.0, 3.0, -20.0);
        let direction = Vector3::new(0.0, 1.0, 0.0);
        let positions =
            compute_formation_positions(FormationType::Vee, 1, &center, &direction, &config);
        assert_eq!(positions.len(), 1);
        assert!((positions[0].1 - center).norm() < EPS);
    }

    // ── Line formation ────────────────────────────────────────────────────

    #[test]
    fn test_line_formation_geometry() {
        let config = default_config(); // spacing = 15
        let center = Vector3::new(0.0, 0.0, -50.0);
        let direction = Vector3::new(1.0, 0.0, 0.0); // +X heading

        let positions =
            compute_formation_positions(FormationType::Line, 5, &center, &direction, &config);

        assert_eq!(positions.len(), 5);

        // All drones at same Z and same X (perpendicular to heading = Y axis).
        for (_, pos) in &positions {
            assert!(
                (pos.x - center.x).abs() < EPS,
                "line drones must share X with center: {:?}",
                pos
            );
            assert!(
                (pos.z - center.z).abs() < EPS,
                "line drone altitude mismatch"
            );
        }

        // Y positions must span symmetrically, total span = (n-1)*spacing = 60m.
        // NB: right of (+X heading) is (0, -1, 0), so "leftmost" in formation
        // perspective corresponds to positive Y values.
        let half_span = 2.0 * 15.0; // (5-1)/2 * 15
        let y_vals: Vec<f64> = positions.iter().map(|(_, p)| p.y).collect();

        // First drone is at left wing (positive Y, since right is -Y).
        assert!(
            (y_vals[0] - half_span).abs() < EPS,
            "first drone Y={}, expected {}",
            y_vals[0],
            half_span
        );
        assert!(
            (y_vals[4] - (-half_span)).abs() < EPS,
            "last drone Y={}, expected {}",
            y_vals[4],
            -half_span
        );

        // Spacing between consecutive drones exactly `config.spacing`.
        for i in 1..5 {
            let gap = (y_vals[i] - y_vals[i - 1]).abs();
            assert!(
                (gap - 15.0).abs() < EPS,
                "gap between drones {} and {}: {}",
                i - 1,
                i,
                gap
            );
        }
    }

    #[test]
    fn test_line_centred_on_center() {
        let config = FormationConfig {
            spacing: 10.0,
            ..Default::default()
        };
        let center = Vector3::new(100.0, 200.0, -30.0);
        let direction = Vector3::new(0.0, 1.0, 0.0); // +Y heading → right is +X

        let positions =
            compute_formation_positions(FormationType::Line, 3, &center, &direction, &config);
        // 3 drones: offsets -10, 0, +10 along right (right of +Y is +X)
        let x_vals: Vec<f64> = positions.iter().map(|(_, p)| p.x).collect();
        assert!((x_vals[0] - (100.0 - 10.0)).abs() < EPS);
        assert!((x_vals[1] - 100.0).abs() < EPS);
        assert!((x_vals[2] - (100.0 + 10.0)).abs() < EPS);
    }

    // ── Column formation ──────────────────────────────────────────────────

    #[test]
    fn test_column_formation() {
        let config = default_config(); // spacing = 15
        let center = Vector3::new(0.0, 0.0, -50.0);
        let direction = Vector3::new(1.0, 0.0, 0.0); // +X heading

        let positions =
            compute_formation_positions(FormationType::Column, 4, &center, &direction, &config);

        assert_eq!(positions.len(), 4);

        // All at same Y and Z; X decreases by spacing per drone.
        for (i, (idx, pos)) in positions.iter().enumerate() {
            assert_eq!(*idx, i);
            assert!(
                (pos.y - center.y).abs() < EPS,
                "column drone has non-zero Y offset"
            );
            assert!((pos.z - center.z).abs() < EPS, "column altitude mismatch");
            let expected_x = center.x - i as f64 * 15.0;
            assert!(
                (pos.x - expected_x).abs() < EPS,
                "drone {} X: expected {}, got {}",
                i,
                expected_x,
                pos.x
            );
        }
    }

    // ── Echelon formation ─────────────────────────────────────────────────

    #[test]
    fn test_echelon_right_offset() {
        let config = default_config(); // spacing = 15
        let center = Vector3::new(0.0, 0.0, -50.0);
        let direction = Vector3::new(1.0, 0.0, 0.0); // +X heading, right is -Y

        let positions = compute_formation_positions(
            FormationType::EchelonRight,
            3,
            &center,
            &direction,
            &config,
        );

        assert_eq!(positions.len(), 3);
        // Drone 0 at center, drone 1 one step back-right, drone 2 two steps back-right.
        let (_, p0) = &positions[0];
        let (_, p1) = &positions[1];
        let (_, p2) = &positions[2];
        assert!((p0 - center).norm() < EPS);
        // back by 15 along X, right (−Y) by 15
        assert!((p1.x - (center.x - 15.0)).abs() < EPS);
        assert!((p1.y - (center.y - 15.0)).abs() < EPS);
        assert!((p2.x - (center.x - 30.0)).abs() < EPS);
        assert!((p2.y - (center.y - 30.0)).abs() < EPS);
    }

    #[test]
    fn test_echelon_left_offset() {
        let config = default_config();
        let center = Vector3::new(0.0, 0.0, -50.0);
        let direction = Vector3::new(1.0, 0.0, 0.0); // right is -Y, left is +Y

        let positions = compute_formation_positions(
            FormationType::EchelonLeft,
            3,
            &center,
            &direction,
            &config,
        );

        let (_, p1) = &positions[1];
        // back by 15, left (+Y) by 15
        assert!((p1.x - (center.x - 15.0)).abs() < EPS);
        assert!((p1.y - (center.y + 15.0)).abs() < EPS);
    }

    // ── Spread formation ──────────────────────────────────────────────────

    #[test]
    fn test_spread_formation_count() {
        let config = default_config();
        let center = Vector3::new(0.0, 0.0, -50.0);
        let direction = Vector3::new(1.0, 0.0, 0.0);

        for n in [1, 2, 4, 5, 9, 10] {
            let positions =
                compute_formation_positions(FormationType::Spread, n, &center, &direction, &config);
            assert_eq!(positions.len(), n, "spread count mismatch for n={}", n);
        }
    }

    #[test]
    fn test_spread_altitude_preserved() {
        let config = default_config();
        let center = Vector3::new(10.0, 20.0, -75.0);
        let direction = Vector3::new(1.0, 0.0, 0.0);
        let positions =
            compute_formation_positions(FormationType::Spread, 9, &center, &direction, &config);
        for (_, pos) in &positions {
            assert!(
                (pos.z - center.z).abs() < EPS,
                "spread Z mismatch: {:?}",
                pos
            );
        }
    }

    // ── Correction vector ─────────────────────────────────────────────────

    #[test]
    fn test_formation_correction_deadband() {
        let config = default_config(); // deadband = 1.0
        let current = Vector3::new(0.0, 0.0, -50.0);

        // Exactly inside deadband.
        let target_near = Vector3::new(0.5, 0.0, -50.0); // 0.5 m < 1.0 m
        let v = formation_correction(&current, &target_near, &config);
        assert!(
            v.norm() < EPS,
            "should return zero inside deadband: {:?}",
            v
        );

        // Right on the boundary (norm == deadband) — should also return zero.
        let target_boundary = Vector3::new(1.0, 0.0, -50.0); // exactly 1.0 m
        let v2 = formation_correction(&current, &target_boundary, &config);
        assert!(
            v2.norm() < EPS,
            "should return zero at deadband boundary: {:?}",
            v2
        );
    }

    #[test]
    fn test_formation_correction_direction() {
        let config = default_config(); // max_correction_speed = 5
        let current = Vector3::new(0.0, 0.0, -50.0);
        let target = Vector3::new(3.0, 4.0, -50.0); // 5 m away

        let v = formation_correction(&current, &target, &config);
        // Should point exactly toward target (unit vector * speed).
        let expected_dir = (target - current).normalize();
        let actual_dir = v.normalize();
        assert!(
            (actual_dir - expected_dir).norm() < 1e-6,
            "direction mismatch"
        );
    }

    #[test]
    fn test_formation_correction_speed_limit() {
        let config = default_config(); // max_correction_speed = 5
        let current = Vector3::new(0.0, 0.0, -50.0);
        // Target very far away — speed must be clamped.
        let target = Vector3::new(1000.0, 0.0, -50.0);

        let v = formation_correction(&current, &target, &config);
        assert!(
            (v.norm() - config.max_correction_speed).abs() < 1e-9,
            "speed not clamped: got {}, expected {}",
            v.norm(),
            config.max_correction_speed
        );
    }

    #[test]
    fn test_formation_correction_proportional_below_limit() {
        let config = default_config(); // max = 5, deadband = 1
        let current = Vector3::new(0.0, 0.0, -50.0);
        // 3 m away, which is less than max_correction_speed=5 — speed should equal distance.
        let target = Vector3::new(3.0, 0.0, -50.0);

        let v = formation_correction(&current, &target, &config);
        assert!(
            (v.norm() - 3.0).abs() < 1e-9,
            "speed should be proportional (3 m/s): {}",
            v.norm()
        );
    }

    // ── Formation quality ─────────────────────────────────────────────────

    #[test]
    fn test_formation_quality_perfect() {
        let config = default_config();
        let center = Vector3::new(0.0, 0.0, -50.0);
        let direction = Vector3::new(1.0, 0.0, 0.0);

        let targets =
            compute_formation_positions(FormationType::Vee, 4, &center, &direction, &config);

        // Drones exactly at their target positions.
        let current: Vec<(u32, Vector3<f64>)> = targets
            .iter()
            .map(|(idx, pos)| (*idx as u32, *pos))
            .collect();

        let q = formation_quality(&current, &targets, &config);
        assert!(
            (q - 1.0).abs() < 1e-9,
            "perfect formation quality should be 1.0, got {}",
            q
        );
    }

    #[test]
    fn test_formation_quality_scattered() {
        let config = default_config(); // spacing = 15
        let center = Vector3::new(0.0, 0.0, -50.0);
        let direction = Vector3::new(1.0, 0.0, 0.0);

        let targets =
            compute_formation_positions(FormationType::Line, 4, &center, &direction, &config);

        // Place drones 50 m (>spacing) away from their slots → quality near 0.
        let current: Vec<(u32, Vector3<f64>)> = vec![
            (0, Vector3::new(100.0, 200.0, -50.0)),
            (1, Vector3::new(-100.0, -200.0, -50.0)),
            (2, Vector3::new(50.0, -150.0, -50.0)),
            (3, Vector3::new(-50.0, 150.0, -50.0)),
        ];

        let q = formation_quality(&current, &targets, &config);
        assert!(
            q < 0.5,
            "scattered drones should yield low quality, got {}",
            q
        );
    }

    #[test]
    fn test_formation_quality_partial() {
        let config = default_config();
        let center = Vector3::new(0.0, 0.0, -50.0);
        let direction = Vector3::new(1.0, 0.0, 0.0);

        let targets =
            compute_formation_positions(FormationType::Column, 4, &center, &direction, &config);

        // First two drones exactly in position, last two off by half spacing.
        let half = config.spacing / 2.0; // 7.5 m
        let mut current: Vec<(u32, Vector3<f64>)> = targets
            .iter()
            .take(2)
            .map(|(idx, pos)| (*idx as u32, *pos))
            .collect();
        for (idx, pos) in targets.iter().skip(2) {
            current.push((*idx as u32, pos + Vector3::new(half, 0.0, 0.0)));
        }

        let q = formation_quality(&current, &targets, &config);
        // avg_error = (0+0+7.5+7.5)/4 = 3.75; normalised = 3.75/15 = 0.25 → quality = 0.75
        assert!((q - 0.75).abs() < 1e-6, "expected quality ~0.75, got {}", q);
    }

    #[test]
    fn test_formation_quality_empty() {
        let config = default_config();
        let q = formation_quality(&[], &[], &config);
        assert!((q - 1.0).abs() < EPS, "empty inputs should yield 1.0");
    }

    // ── Zero drones edge case ─────────────────────────────────────────────

    #[test]
    fn test_zero_drones_returns_empty() {
        let config = default_config();
        let center = Vector3::new(0.0, 0.0, -50.0);
        let direction = Vector3::new(1.0, 0.0, 0.0);
        for formation in [
            FormationType::Vee,
            FormationType::Line,
            FormationType::Wedge,
            FormationType::Column,
            FormationType::EchelonLeft,
            FormationType::EchelonRight,
            FormationType::Spread,
        ] {
            let positions = compute_formation_positions(formation, 0, &center, &direction, &config);
            assert!(
                positions.is_empty(),
                "n=0 should return empty for {:?}",
                formation
            );
        }
    }

    // ── Altitude preservation ─────────────────────────────────────────────

    #[test]
    fn test_all_formations_preserve_altitude() {
        let config = default_config();
        let center = Vector3::new(10.0, 20.0, -123.45);
        let direction = Vector3::new(1.0, 1.0, 5.0); // diagonal + vertical component

        for formation in [
            FormationType::Vee,
            FormationType::Line,
            FormationType::Wedge,
            FormationType::Column,
            FormationType::EchelonLeft,
            FormationType::EchelonRight,
            FormationType::Spread,
        ] {
            let positions = compute_formation_positions(formation, 6, &center, &direction, &config);
            for (_, pos) in &positions {
                assert!(
                    (pos.z - center.z).abs() < EPS,
                    "altitude not preserved for {:?}: expected {}, got {}",
                    formation,
                    center.z,
                    pos.z
                );
            }
        }
    }

    // ── Wedge is tighter than Vee ──────────────────────────────────────────

    // ── Fear-adjusted formation ─────────────────────────────────────────

    #[test]
    fn test_fear_adjusted_high_threshold() {
        let config = default_config();
        let adjusted = config.fear_adjusted(1.0);
        assert!((adjusted.spacing - config.spacing * 0.8).abs() < 1e-9);
        assert!((adjusted.deadband - config.deadband * 0.5).abs() < 1e-9);
        assert_eq!(adjusted.vee_angle_deg, config.vee_angle_deg); // unchanged
        assert_eq!(adjusted.max_correction_speed, config.max_correction_speed); // unchanged
    }

    #[test]
    fn test_fear_adjusted_low_threshold() {
        let config = default_config();
        let adjusted = config.fear_adjusted(0.3);
        assert!((adjusted.spacing - config.spacing * 1.5).abs() < 1e-9);
        assert!((adjusted.deadband - config.deadband * 1.9).abs() < 1e-9);
    }

    #[test]
    fn test_fear_adjusted_clamps() {
        let config = default_config();
        let a1 = config.fear_adjusted(0.0); // clamps to 0.3
        let a2 = config.fear_adjusted(0.3);
        assert!((a1.spacing - a2.spacing).abs() < 1e-9);
    }

    // ── Wedge is tighter than Vee ──────────────────────────────────────────

    #[test]
    fn test_wedge_tighter_than_vee() {
        let config = default_config();
        let center = Vector3::new(0.0, 0.0, -50.0);
        let direction = Vector3::new(1.0, 0.0, 0.0);
        let n = 5;

        let vee = compute_formation_positions(FormationType::Vee, n, &center, &direction, &config);
        let wedge =
            compute_formation_positions(FormationType::Wedge, n, &center, &direction, &config);

        // Compute total lateral spread (max Y extent) for each.
        let vee_span = vee.iter().map(|(_, p)| p.y.abs()).fold(0.0_f64, f64::max);
        let wedge_span = wedge.iter().map(|(_, p)| p.y.abs()).fold(0.0_f64, f64::max);

        assert!(
            wedge_span < vee_span,
            "wedge lateral span ({}) should be tighter than vee ({})",
            wedge_span,
            vee_span
        );
    }
}
