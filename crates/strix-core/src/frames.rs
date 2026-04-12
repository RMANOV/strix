//! Frame-safe coordinate types for compile-time frame confusion prevention.
//!
//! STRIX internally uses NED (North-East-Down) but adapters may report in
//! ENU, WGS84, or body-fixed frames. These newtypes make frame mismatches
//! a compile error rather than a runtime bug.

use nalgebra::Vector3;
use serde::{Deserialize, Serialize};
use std::fmt;

// ---------------------------------------------------------------------------
// Coordinate frame enum
// ---------------------------------------------------------------------------

/// Named coordinate frames used throughout STRIX.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Frame {
    /// North-East-Down — STRIX internal convention.
    Ned,
    /// East-North-Up — common in ROS/geodesy.
    Enu,
    /// WGS-84 geodetic (lat/lon/alt).
    Wgs84,
    /// Body-fixed (forward-right-down relative to vehicle).
    BodyFixed,
}

// ---------------------------------------------------------------------------
// NED position — the canonical internal frame
// ---------------------------------------------------------------------------

/// Position in the local NED (North-East-Down) frame, metres.
///
/// This is the canonical internal representation. All core algorithms
/// (particle filter, CBF, formation) operate in NED.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct NedPosition(pub Vector3<f64>);

impl NedPosition {
    /// Create from components.
    #[inline]
    pub fn new(north: f64, east: f64, down: f64) -> Self {
        Self(Vector3::new(north, east, down))
    }

    /// Origin.
    pub const fn origin() -> Self {
        Self(Vector3::new(0.0, 0.0, 0.0))
    }

    /// North component (metres).
    #[inline]
    pub fn north(&self) -> f64 {
        self.0.x
    }

    /// East component (metres).
    #[inline]
    pub fn east(&self) -> f64 {
        self.0.y
    }

    /// Down component (metres, positive = below origin).
    #[inline]
    pub fn down(&self) -> f64 {
        self.0.z
    }

    /// Altitude above ground (positive up) = -down.
    #[inline]
    pub fn altitude_agl(&self) -> f64 {
        -self.0.z
    }

    /// Convert to ENU.
    #[inline]
    pub fn to_enu(&self) -> EnuPosition {
        EnuPosition(Vector3::new(self.0.y, self.0.x, -self.0.z))
    }

    /// Euclidean distance to another NED position.
    #[inline]
    pub fn distance(&self, other: &Self) -> f64 {
        (self.0 - other.0).norm()
    }

    /// Inner vector (for math operations).
    #[inline]
    pub fn as_vector(&self) -> &Vector3<f64> {
        &self.0
    }
}

impl fmt::Display for NedPosition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "NED({:.1}N, {:.1}E, {:.1}D)",
            self.0.x, self.0.y, self.0.z
        )
    }
}

impl From<Vector3<f64>> for NedPosition {
    #[inline]
    fn from(v: Vector3<f64>) -> Self {
        Self(v)
    }
}

impl From<NedPosition> for Vector3<f64> {
    #[inline]
    fn from(p: NedPosition) -> Self {
        p.0
    }
}

impl From<&NedPosition> for Vector3<f64> {
    #[inline]
    fn from(p: &NedPosition) -> Self {
        p.0
    }
}

// ---------------------------------------------------------------------------
// ENU position
// ---------------------------------------------------------------------------

/// Position in East-North-Up frame (metres).
///
/// Common in ROS2 and geodetic software. Convert to NED for internal use.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EnuPosition(pub Vector3<f64>);

impl EnuPosition {
    /// Create from components.
    #[inline]
    pub fn new(east: f64, north: f64, up: f64) -> Self {
        Self(Vector3::new(east, north, up))
    }

    /// Convert to NED.
    #[inline]
    pub fn to_ned(&self) -> NedPosition {
        NedPosition(Vector3::new(self.0.y, self.0.x, -self.0.z))
    }
}

impl fmt::Display for EnuPosition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ENU({:.1}E, {:.1}N, {:.1}U)",
            self.0.x, self.0.y, self.0.z
        )
    }
}

// ---------------------------------------------------------------------------
// WGS-84 geodetic position
// ---------------------------------------------------------------------------

/// WGS-84 geodetic position (degrees + metres).
///
/// Used at the adapter boundary (GPS, MAVLink). Must be converted to NED
/// via an explicit origin before internal use.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct WgsPosition {
    /// Latitude in degrees (positive = north).
    pub lat_deg: f64,
    /// Longitude in degrees (positive = east).
    pub lon_deg: f64,
    /// Altitude above WGS-84 ellipsoid in metres.
    pub alt_m: f64,
}

impl WgsPosition {
    /// Create a new WGS-84 position.
    #[inline]
    pub fn new(lat_deg: f64, lon_deg: f64, alt_m: f64) -> Self {
        Self {
            lat_deg,
            lon_deg,
            alt_m,
        }
    }

    /// Convert to local NED relative to an origin point.
    ///
    /// Uses a flat-earth approximation valid for distances < ~10 km.
    /// For battlefield-scale operations this is sufficient; for
    /// continent-scale, use a proper geodetic library.
    pub fn to_ned_relative(&self, origin: &WgsPosition) -> NedPosition {
        // WGS-84 constants
        const A: f64 = 6_378_137.0; // semi-major axis
        const F: f64 = 1.0 / 298.257_223_563; // flattening
        const E2: f64 = 2.0 * F - F * F; // eccentricity squared

        let lat_rad = origin.lat_deg.to_radians();
        let sin_lat = lat_rad.sin();
        let cos_lat = lat_rad.cos();

        // Radii of curvature
        let r_n = A / (1.0 - E2 * sin_lat * sin_lat).sqrt();
        let r_m = r_n * (1.0 - E2) / (1.0 - E2 * sin_lat * sin_lat);

        let d_lat = (self.lat_deg - origin.lat_deg).to_radians();
        let d_lon = (self.lon_deg - origin.lon_deg).to_radians();
        let d_alt = self.alt_m - origin.alt_m;

        let north = d_lat * r_m;
        let east = d_lon * r_n * cos_lat;
        let down = -d_alt; // NED: down is positive

        NedPosition::new(north, east, down)
    }

    /// Convert from local NED back to WGS-84 given the same origin.
    pub fn from_ned_relative(ned: &NedPosition, origin: &WgsPosition) -> Self {
        const A: f64 = 6_378_137.0;
        const F: f64 = 1.0 / 298.257_223_563;
        const E2: f64 = 2.0 * F - F * F;

        let lat_rad = origin.lat_deg.to_radians();
        let sin_lat = lat_rad.sin();
        let cos_lat = lat_rad.cos();

        let r_n = A / (1.0 - E2 * sin_lat * sin_lat).sqrt();
        let r_m = r_n * (1.0 - E2) / (1.0 - E2 * sin_lat * sin_lat);

        let lat_deg = origin.lat_deg + (ned.north() / r_m).to_degrees();
        let lon_deg = origin.lon_deg + (ned.east() / (r_n * cos_lat)).to_degrees();
        let alt_m = origin.alt_m - ned.down();

        Self {
            lat_deg,
            lon_deg,
            alt_m,
        }
    }
}

impl fmt::Display for WgsPosition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "WGS84({:.6}\u{00b0}, {:.6}\u{00b0}, {:.1}m)",
            self.lat_deg, self.lon_deg, self.alt_m
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ned_enu_roundtrip() {
        let ned = NedPosition::new(100.0, 200.0, -50.0);
        let enu = ned.to_enu();
        let back = enu.to_ned();
        assert!((back.north() - ned.north()).abs() < 1e-10);
        assert!((back.east() - ned.east()).abs() < 1e-10);
        assert!((back.down() - ned.down()).abs() < 1e-10);
    }

    #[test]
    fn ned_enu_components() {
        let ned = NedPosition::new(10.0, 20.0, -5.0);
        let enu = ned.to_enu();
        // ENU: east=NED.east, north=NED.north, up=-NED.down
        assert!((enu.0.x - 20.0).abs() < 1e-10); // east
        assert!((enu.0.y - 10.0).abs() < 1e-10); // north
        assert!((enu.0.z - 5.0).abs() < 1e-10); // up = -(-5) = 5
    }

    #[test]
    fn wgs84_ned_roundtrip() {
        let origin = WgsPosition::new(42.6977, 23.3219, 550.0); // Sofia
        let point = WgsPosition::new(42.6987, 23.3229, 560.0); // ~100m away

        let ned = point.to_ned_relative(&origin);
        let back = WgsPosition::from_ned_relative(&ned, &origin);

        assert!(
            (back.lat_deg - point.lat_deg).abs() < 1e-8,
            "lat: {} vs {}",
            back.lat_deg,
            point.lat_deg
        );
        assert!(
            (back.lon_deg - point.lon_deg).abs() < 1e-8,
            "lon: {} vs {}",
            back.lon_deg,
            point.lon_deg
        );
        assert!(
            (back.alt_m - point.alt_m).abs() < 1e-4,
            "alt: {} vs {}",
            back.alt_m,
            point.alt_m
        );
    }

    #[test]
    fn wgs84_ned_distance_sanity() {
        let origin = WgsPosition::new(0.0, 0.0, 0.0);
        // ~1 degree latitude ≈ 111 km
        let point = WgsPosition::new(1.0, 0.0, 0.0);
        let ned = point.to_ned_relative(&origin);
        let dist_km = ned.north() / 1000.0;
        assert!(
            (dist_km - 111.0).abs() < 1.0,
            "1 deg lat should be ~111 km, got {dist_km}"
        );
    }

    #[test]
    fn altitude_agl() {
        let pos = NedPosition::new(0.0, 0.0, -50.0);
        assert!((pos.altitude_agl() - 50.0).abs() < 1e-10);
    }

    #[test]
    fn ned_distance() {
        let a = NedPosition::new(0.0, 0.0, 0.0);
        let b = NedPosition::new(3.0, 4.0, 0.0);
        assert!((a.distance(&b) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn vector3_conversions() {
        let v = Vector3::new(1.0, 2.0, 3.0);
        let ned: NedPosition = v.into();
        let back: Vector3<f64> = ned.into();
        assert_eq!(v, back);
    }

    #[test]
    fn display_formats() {
        let ned = NedPosition::new(100.5, 200.3, -50.0);
        assert!(format!("{ned}").contains("NED"));

        let enu = EnuPosition::new(200.3, 100.5, 50.0);
        assert!(format!("{enu}").contains("ENU"));

        let wgs = WgsPosition::new(42.6977, 23.3219, 550.0);
        assert!(format!("{wgs}").contains("WGS84"));
    }

    #[test]
    fn serde_roundtrip() {
        let ned = NedPosition::new(1.0, 2.0, 3.0);
        let json = serde_json::to_string(&ned).unwrap();
        let back: NedPosition = serde_json::from_str(&json).unwrap();
        assert_eq!(ned, back);

        let wgs = WgsPosition::new(42.0, 23.0, 550.0);
        let json = serde_json::to_string(&wgs).unwrap();
        let back: WgsPosition = serde_json::from_str(&json).unwrap();
        assert_eq!(wgs, back);
    }
}
