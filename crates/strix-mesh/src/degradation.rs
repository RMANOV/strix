//! Graceful degradation profiles for swarm operations under stress.

use serde::{Deserialize, Serialize};

/// How much of the swarm has been lost.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttritionLevel {
    /// Less than 20% lost — nominal operations.
    Nominal,
    /// 20-40% lost — reduced operations.
    Light,
    /// 40-60% lost — minimal operations.
    Heavy,
    /// Over 60% lost — survival mode.
    Critical,
}

impl AttritionLevel {
    /// Classify from a loss fraction [0, 1].
    pub fn from_loss_fraction(fraction: f64) -> Self {
        match fraction {
            f if f < 0.20 => Self::Nominal,
            f if f < 0.40 => Self::Light,
            f if f < 0.60 => Self::Heavy,
            _ => Self::Critical,
        }
    }
}

/// What the swarm should do at each attrition level.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DegradationProfile {
    /// Current attrition level.
    pub level: AttritionLevel,
    /// Mission scope adjustment.
    pub mission_scope: MissionScope,
    /// Whether to consolidate formations.
    pub consolidate_formations: bool,
    /// Whether to increase gossip frequency (for faster recovery).
    pub boost_gossip: bool,
    /// Fear multiplier (higher = more conservative bidding).
    pub fear_multiplier: f64,
    /// Maximum concurrent tasks (fraction of original).
    pub task_capacity_fraction: f64,
}

/// Mission scope under degradation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MissionScope {
    /// Full mission — all objectives active.
    Full,
    /// Reduced — drop low-priority objectives.
    Reduced { dropped_priorities_below: u8 },
    /// Minimal — only highest-priority objective.
    Minimal,
    /// Survival — retreat and preserve remaining assets.
    Survival,
}

impl DegradationProfile {
    /// Get the default profile for a given attrition level.
    pub fn for_level(level: AttritionLevel) -> Self {
        match level {
            AttritionLevel::Nominal => Self {
                level,
                mission_scope: MissionScope::Full,
                consolidate_formations: false,
                boost_gossip: false,
                fear_multiplier: 1.0,
                task_capacity_fraction: 1.0,
            },
            AttritionLevel::Light => Self {
                level,
                mission_scope: MissionScope::Reduced {
                    dropped_priorities_below: 3,
                },
                consolidate_formations: true,
                boost_gossip: true,
                fear_multiplier: 1.3,
                task_capacity_fraction: 0.7,
            },
            AttritionLevel::Heavy => Self {
                level,
                mission_scope: MissionScope::Minimal,
                consolidate_formations: true,
                boost_gossip: true,
                fear_multiplier: 1.8,
                task_capacity_fraction: 0.4,
            },
            AttritionLevel::Critical => Self {
                level,
                mission_scope: MissionScope::Survival,
                consolidate_formations: true,
                boost_gossip: false, // conserve bandwidth
                fear_multiplier: 2.5,
                task_capacity_fraction: 0.1,
            },
        }
    }

    /// Get the profile for a given loss fraction.
    pub fn from_loss_fraction(fraction: f64) -> Self {
        Self::for_level(AttritionLevel::from_loss_fraction(fraction))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- AttritionLevel classification ----

    #[test]
    fn attrition_levels_boundaries() {
        // Below 0.20 → Nominal
        assert_eq!(
            AttritionLevel::from_loss_fraction(0.0),
            AttritionLevel::Nominal
        );
        assert_eq!(
            AttritionLevel::from_loss_fraction(0.19),
            AttritionLevel::Nominal
        );
        // Exactly 0.20 → Light
        assert_eq!(
            AttritionLevel::from_loss_fraction(0.20),
            AttritionLevel::Light
        );
        assert_eq!(
            AttritionLevel::from_loss_fraction(0.39),
            AttritionLevel::Light
        );
        // Exactly 0.40 → Heavy
        assert_eq!(
            AttritionLevel::from_loss_fraction(0.40),
            AttritionLevel::Heavy
        );
        assert_eq!(
            AttritionLevel::from_loss_fraction(0.59),
            AttritionLevel::Heavy
        );
        // Exactly 0.60 → Critical
        assert_eq!(
            AttritionLevel::from_loss_fraction(0.60),
            AttritionLevel::Critical
        );
        assert_eq!(
            AttritionLevel::from_loss_fraction(1.0),
            AttritionLevel::Critical
        );
    }

    // ---- DegradationProfile per-level checks ----

    #[test]
    fn nominal_profile() {
        let p = DegradationProfile::for_level(AttritionLevel::Nominal);
        assert_eq!(p.level, AttritionLevel::Nominal);
        assert_eq!(p.mission_scope, MissionScope::Full);
        assert!(!p.consolidate_formations);
        assert!(!p.boost_gossip);
        assert!((p.fear_multiplier - 1.0).abs() < f64::EPSILON);
        assert!((p.task_capacity_fraction - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn light_profile() {
        let p = DegradationProfile::for_level(AttritionLevel::Light);
        assert_eq!(p.level, AttritionLevel::Light);
        assert_eq!(
            p.mission_scope,
            MissionScope::Reduced {
                dropped_priorities_below: 3
            }
        );
        assert!(p.consolidate_formations);
        assert!(p.boost_gossip);
        assert!((p.fear_multiplier - 1.3).abs() < 1e-10);
        assert!((p.task_capacity_fraction - 0.7).abs() < 1e-10);
    }

    #[test]
    fn heavy_profile() {
        let p = DegradationProfile::for_level(AttritionLevel::Heavy);
        assert_eq!(p.level, AttritionLevel::Heavy);
        assert_eq!(p.mission_scope, MissionScope::Minimal);
        assert!(p.consolidate_formations);
        assert!(p.boost_gossip);
        assert!((p.fear_multiplier - 1.8).abs() < 1e-10);
        assert!((p.task_capacity_fraction - 0.4).abs() < 1e-10);
    }

    #[test]
    fn critical_profile() {
        let p = DegradationProfile::for_level(AttritionLevel::Critical);
        assert_eq!(p.level, AttritionLevel::Critical);
        assert_eq!(p.mission_scope, MissionScope::Survival);
        assert!(p.consolidate_formations);
        assert!(!p.boost_gossip); // conserve bandwidth
        assert!((p.fear_multiplier - 2.5).abs() < 1e-10);
        assert!((p.task_capacity_fraction - 0.1).abs() < 1e-10);
    }

    // ---- from_loss_fraction end-to-end ----

    #[test]
    fn from_loss_fraction_end_to_end() {
        let p = DegradationProfile::from_loss_fraction(0.05);
        assert_eq!(p.level, AttritionLevel::Nominal);

        let p = DegradationProfile::from_loss_fraction(0.25);
        assert_eq!(p.level, AttritionLevel::Light);

        let p = DegradationProfile::from_loss_fraction(0.50);
        assert_eq!(p.level, AttritionLevel::Heavy);

        let p = DegradationProfile::from_loss_fraction(0.75);
        assert_eq!(p.level, AttritionLevel::Critical);
    }

    // ---- serde round-trips ----

    #[test]
    fn serde_roundtrip_attrition_level() {
        let levels = [
            AttritionLevel::Nominal,
            AttritionLevel::Light,
            AttritionLevel::Heavy,
            AttritionLevel::Critical,
        ];
        for &level in &levels {
            let json = serde_json::to_string(&level).unwrap();
            let back: AttritionLevel = serde_json::from_str(&json).unwrap();
            assert_eq!(level, back);
        }
    }

    #[test]
    fn serde_roundtrip_mission_scope() {
        let scopes = [
            MissionScope::Full,
            MissionScope::Reduced {
                dropped_priorities_below: 3,
            },
            MissionScope::Minimal,
            MissionScope::Survival,
        ];
        for scope in &scopes {
            let json = serde_json::to_string(scope).unwrap();
            let back: MissionScope = serde_json::from_str(&json).unwrap();
            assert_eq!(*scope, back);
        }
    }

    #[test]
    fn serde_roundtrip_degradation_profile() {
        for &level in &[
            AttritionLevel::Nominal,
            AttritionLevel::Light,
            AttritionLevel::Heavy,
            AttritionLevel::Critical,
        ] {
            let profile = DegradationProfile::for_level(level);
            let json = serde_json::to_string(&profile).unwrap();
            let back: DegradationProfile = serde_json::from_str(&json).unwrap();
            assert_eq!(back.level, level);
            assert_eq!(back.mission_scope, profile.mission_scope);
            assert_eq!(back.consolidate_formations, profile.consolidate_formations);
            assert_eq!(back.boost_gossip, profile.boost_gossip);
            assert!((back.fear_multiplier - profile.fear_multiplier).abs() < 1e-10);
            assert!((back.task_capacity_fraction - profile.task_capacity_fraction).abs() < 1e-10);
        }
    }
}
