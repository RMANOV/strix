//! Semantic newtypes for compile-time safety.
//!
//! Raw `f64` is used for trust, fear, convergence, confidence, severity,
//! and many other [0,1] quantities throughout STRIX. They are all type-
//! compatible, meaning `trust + convergence` compiles without warning.
//!
//! This module introduces **opt-in** newtypes that catch semantic mixing
//! at compile time. Applied incrementally to new and modified code only.

use serde::{Deserialize, Serialize};
use std::fmt;

// ---------------------------------------------------------------------------
// UnitInterval — any quantity clamped to [0, 1]
// ---------------------------------------------------------------------------

/// A value guaranteed to be in [0, 1] and finite.
///
/// Construction validates the invariant; NaN and infinity map to 0.0.
/// This is the foundational newtype — trust, fear, convergence, confidence,
/// and criticality are all `UnitInterval` semantically, but they should
/// NOT be mixed at call sites without explicit conversion.
#[derive(Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct UnitInterval(f64);

impl UnitInterval {
    /// Zero.
    pub const ZERO: Self = Self(0.0);
    /// One.
    pub const ONE: Self = Self(1.0);
    /// Neutral midpoint.
    pub const HALF: Self = Self(0.5);

    /// Create a new `UnitInterval`, clamping to [0, 1].
    /// NaN / infinity → 0.0.
    #[inline]
    pub fn new(v: f64) -> Self {
        if v.is_finite() {
            Self(v.clamp(0.0, 1.0))
        } else {
            Self(0.0)
        }
    }

    /// Create without clamping — caller guarantees `v ∈ [0, 1]`.
    ///
    /// # Safety (logical)
    /// Panics in debug if the invariant is violated; silently clamps in release.
    #[inline]
    pub fn new_unchecked(v: f64) -> Self {
        debug_assert!(
            v.is_finite() && (0.0..=1.0).contains(&v),
            "UnitInterval::new_unchecked called with {v}"
        );
        Self::new(v)
    }

    /// Inner value.
    #[inline]
    pub fn get(self) -> f64 {
        self.0
    }

    /// Complement: 1 - self.
    #[inline]
    pub fn complement(self) -> Self {
        Self(1.0 - self.0)
    }

    /// Linear interpolation: self * (1 - t) + other * t.
    #[inline]
    pub fn lerp(self, other: Self, t: Self) -> Self {
        Self::new(self.0 * (1.0 - t.0) + other.0 * t.0)
    }

    /// Scale by a raw factor, re-clamping the result.
    #[inline]
    pub fn scale(self, factor: f64) -> Self {
        Self::new(self.0 * factor)
    }
}

impl Default for UnitInterval {
    fn default() -> Self {
        Self::HALF
    }
}

impl fmt::Debug for UnitInterval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "UnitInterval({:.4})", self.0)
    }
}

impl fmt::Display for UnitInterval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.4}", self.0)
    }
}

impl From<UnitInterval> for f64 {
    #[inline]
    fn from(u: UnitInterval) -> f64 {
        u.0
    }
}

// ---------------------------------------------------------------------------
// RiskContext — unified multi-dimensional risk
// ---------------------------------------------------------------------------

/// Unified risk context that names each risk dimension explicitly.
///
/// Replaces the ambiguous "risk" vocabulary:
/// - `attrition`: fleet-level loss ratio (from `strix_auction::risk::RiskLevel`)
/// - `positional`: drone-level threat exposure (from bidder scoring)
/// - `collateral`: engagement-level civilian harm estimate (from ROE)
/// - `environmental`: spatial threat density (from pheromone field)
///
/// Each dimension is a [`UnitInterval`] — they share a scale but are
/// semantically distinct and must not be mixed without explicit intent.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct RiskContext {
    /// Fleet attrition rate — fraction of drones lost.
    pub attrition: UnitInterval,
    /// Positional threat exposure at drone's location.
    pub positional: UnitInterval,
    /// Collateral damage risk for engagement decisions.
    pub collateral: UnitInterval,
    /// Environmental threat density from pheromone/sensor field.
    pub environmental: UnitInterval,
}

impl RiskContext {
    /// Worst-case risk across all dimensions.
    pub fn max_risk(&self) -> UnitInterval {
        let m = self
            .attrition
            .get()
            .max(self.positional.get())
            .max(self.collateral.get())
            .max(self.environmental.get());
        UnitInterval::new(m)
    }

    /// Weighted aggregate risk.
    pub fn weighted(
        &self,
        w_attrition: f64,
        w_positional: f64,
        w_collateral: f64,
        w_environmental: f64,
    ) -> UnitInterval {
        let total = w_attrition + w_positional + w_collateral + w_environmental;
        if total < 1e-12 {
            return UnitInterval::ZERO;
        }
        let v = (w_attrition * self.attrition.get()
            + w_positional * self.positional.get()
            + w_collateral * self.collateral.get()
            + w_environmental * self.environmental.get())
            / total;
        UnitInterval::new(v)
    }
}

// ---------------------------------------------------------------------------
// Semantic newtypes over UnitInterval
// ---------------------------------------------------------------------------

macro_rules! unit_newtype {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
        #[serde(transparent)]
        pub struct $name(UnitInterval);

        impl $name {
            /// Zero.
            pub const ZERO: Self = Self(UnitInterval::ZERO);
            /// One.
            pub const ONE: Self = Self(UnitInterval::ONE);
            /// Neutral midpoint.
            pub const HALF: Self = Self(UnitInterval::HALF);

            /// Create from raw f64, clamping to [0, 1].
            #[inline]
            pub fn new(v: f64) -> Self {
                Self(UnitInterval::new(v))
            }

            /// Inner value as f64.
            #[inline]
            pub fn get(self) -> f64 {
                self.0.get()
            }

            /// Underlying UnitInterval.
            #[inline]
            pub fn unit(self) -> UnitInterval {
                self.0
            }

            /// Complement: 1 - self.
            #[inline]
            pub fn complement(self) -> Self {
                Self(self.0.complement())
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self(UnitInterval::default())
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{:.4}", self.0.get())
            }
        }

        impl From<$name> for f64 {
            #[inline]
            fn from(v: $name) -> f64 {
                v.0.get()
            }
        }

        impl From<UnitInterval> for $name {
            #[inline]
            fn from(u: UnitInterval) -> Self {
                Self(u)
            }
        }

        impl From<$name> for UnitInterval {
            #[inline]
            fn from(v: $name) -> UnitInterval {
                v.0
            }
        }
    };
}

unit_newtype!(
    /// Peer trust score — how much we believe a peer's data.
    Trust
);

unit_newtype!(
    /// Fear level — meta-parameter controlling risk aversion.
    Fear
);

unit_newtype!(
    /// Confidence in a measurement or estimate.
    Confidence
);

unit_newtype!(
    /// Gossip convergence — fraction of peers with consistent state.
    Convergence
);

// ---------------------------------------------------------------------------
// Timestamp
// ---------------------------------------------------------------------------

/// Monotonic timestamp in seconds.
///
/// Wraps f64 to prevent mixing timestamps with durations or other scalars.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Timestamp(f64);

impl Timestamp {
    /// Create a timestamp from seconds.
    #[inline]
    pub fn from_secs(s: f64) -> Self {
        Self(if s.is_finite() { s } else { 0.0 })
    }

    /// Zero timestamp.
    pub const ZERO: Self = Self(0.0);

    /// Inner value in seconds.
    #[inline]
    pub fn as_secs(self) -> f64 {
        self.0
    }

    /// Seconds elapsed since this timestamp, given a current time.
    #[inline]
    pub fn elapsed_since(self, now: f64) -> f64 {
        (now - self.0).max(0.0)
    }

    /// Whether this timestamp is older than `max_age_s` relative to `now`.
    #[inline]
    pub fn is_stale(self, now: f64, max_age_s: f64) -> bool {
        self.elapsed_since(now) > max_age_s
    }
}

impl Default for Timestamp {
    fn default() -> Self {
        Self::ZERO
    }
}

impl fmt::Display for Timestamp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.3}s", self.0)
    }
}

impl From<f64> for Timestamp {
    #[inline]
    fn from(s: f64) -> Self {
        Self::from_secs(s)
    }
}

impl From<Timestamp> for f64 {
    #[inline]
    fn from(t: Timestamp) -> f64 {
        t.0
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_interval_clamps() {
        assert_eq!(UnitInterval::new(1.5).get(), 1.0);
        assert_eq!(UnitInterval::new(-0.3).get(), 0.0);
        assert_eq!(UnitInterval::new(0.7).get(), 0.7);
    }

    #[test]
    fn unit_interval_nan_is_zero() {
        assert_eq!(UnitInterval::new(f64::NAN).get(), 0.0);
        assert_eq!(UnitInterval::new(f64::INFINITY).get(), 0.0);
        assert_eq!(UnitInterval::new(f64::NEG_INFINITY).get(), 0.0);
    }

    #[test]
    fn complement() {
        let u = UnitInterval::new(0.3);
        assert!((u.complement().get() - 0.7).abs() < 1e-10);
    }

    #[test]
    fn lerp_endpoints() {
        let a = UnitInterval::ZERO;
        let b = UnitInterval::ONE;
        assert!((a.lerp(b, UnitInterval::ZERO).get()).abs() < 1e-10);
        assert!((a.lerp(b, UnitInterval::ONE).get() - 1.0).abs() < 1e-10);
        assert!((a.lerp(b, UnitInterval::HALF).get() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn scale_reclamps() {
        let u = UnitInterval::new(0.8);
        assert_eq!(u.scale(2.0).get(), 1.0); // 1.6 clamped to 1.0
        assert!((u.scale(0.5).get() - 0.4).abs() < 1e-10);
    }

    #[test]
    fn serde_roundtrip() {
        let u = UnitInterval::new(0.42);
        let json = serde_json::to_string(&u).unwrap();
        let back: UnitInterval = serde_json::from_str(&json).unwrap();
        assert!((back.get() - 0.42).abs() < 1e-10);
    }

    #[test]
    fn risk_context_max() {
        let rc = RiskContext {
            attrition: UnitInterval::new(0.1),
            positional: UnitInterval::new(0.9),
            collateral: UnitInterval::new(0.3),
            environmental: UnitInterval::new(0.5),
        };
        assert!((rc.max_risk().get() - 0.9).abs() < 1e-10);
    }

    #[test]
    fn risk_context_weighted() {
        let rc = RiskContext {
            attrition: UnitInterval::new(1.0),
            positional: UnitInterval::ZERO,
            collateral: UnitInterval::ZERO,
            environmental: UnitInterval::ZERO,
        };
        // Only attrition has weight
        let w = rc.weighted(1.0, 0.0, 0.0, 0.0);
        assert!((w.get() - 1.0).abs() < 1e-10);

        // Equal weights
        let w2 = rc.weighted(1.0, 1.0, 1.0, 1.0);
        assert!((w2.get() - 0.25).abs() < 1e-10);
    }

    #[test]
    #[should_panic(expected = "UnitInterval::new_unchecked")]
    #[cfg(debug_assertions)]
    fn unchecked_panics_in_debug() {
        let _ = UnitInterval::new_unchecked(1.5);
    }

    #[test]
    fn default_is_half() {
        assert!((UnitInterval::default().get() - 0.5).abs() < 1e-10);
    }

    // --- Semantic newtypes ---

    #[test]
    fn trust_and_fear_are_distinct_types() {
        let t = Trust::new(0.8);
        let f = Fear::new(0.3);
        // They have the same inner value type but are NOT interchangeable:
        // Trust + Fear does NOT compile — that's the point.
        assert!((t.get() - 0.8).abs() < 1e-10);
        assert!((f.get() - 0.3).abs() < 1e-10);
    }

    #[test]
    fn semantic_newtype_clamps() {
        assert_eq!(Trust::new(1.5).get(), 1.0);
        assert_eq!(Fear::new(-0.1).get(), 0.0);
        assert_eq!(Confidence::new(f64::NAN).get(), 0.0);
    }

    #[test]
    fn semantic_newtype_complement() {
        let t = Trust::new(0.3);
        assert!((t.complement().get() - 0.7).abs() < 1e-10);
    }

    #[test]
    fn semantic_newtype_conversions() {
        let u = UnitInterval::new(0.6);
        let t: Trust = u.into();
        let back: UnitInterval = t.into();
        assert!((back.get() - 0.6).abs() < 1e-10);

        let f64_val: f64 = t.into();
        assert!((f64_val - 0.6).abs() < 1e-10);
    }

    #[test]
    fn semantic_newtype_serde() {
        let c = Confidence::new(0.95);
        let json = serde_json::to_string(&c).unwrap();
        let back: Confidence = serde_json::from_str(&json).unwrap();
        assert!((back.get() - 0.95).abs() < 1e-10);
    }

    // --- Timestamp ---

    #[test]
    fn timestamp_elapsed() {
        let t = Timestamp::from_secs(10.0);
        assert!((t.elapsed_since(15.0) - 5.0).abs() < 1e-10);
        // Negative elapsed is clamped to 0
        assert!((t.elapsed_since(5.0)).abs() < 1e-10);
    }

    #[test]
    fn timestamp_is_stale() {
        let t = Timestamp::from_secs(10.0);
        assert!(!t.is_stale(12.0, 5.0)); // 2s old, max 5s
        assert!(t.is_stale(20.0, 5.0)); // 10s old, max 5s
        assert!(!t.is_stale(15.0, 5.0)); // exactly 5s, not stale (> not >=)
    }

    #[test]
    fn timestamp_nan_handled() {
        let t = Timestamp::from_secs(f64::NAN);
        assert_eq!(t.as_secs(), 0.0);
    }

    #[test]
    fn timestamp_conversions() {
        let t: Timestamp = 42.0_f64.into();
        let back: f64 = t.into();
        assert!((back - 42.0).abs() < 1e-10);
    }

    #[test]
    fn convergence_default() {
        assert!((Convergence::default().get() - 0.5).abs() < 1e-10);
    }
}
