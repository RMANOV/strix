//! **Rules of Engagement (ROE)** — go/no-go gate before any autonomous engagement action.
//!
//! This module implements a standards-compliant ROE engine for STRIX drone swarm
//! orchestration. Every engagement request must pass through [`RoeEngine::authorize_engagement`]
//! before any weapon effect is applied. The engine enforces:
//!
//! - **Weapons posture**: Hold / Tight / Free
//! - **Self-defense override**: hostile act always permits return fire
//! - **PID (Positive Identification)**: required under WeaponsTight
//! - **Collateral damage cap**: configurable threshold triggers human escalation
//! - **Minimum engagement distance**: safety floor for autonomous fire
//!
//! # Usage
//!
//! ```rust
//! use strix_core::roe::{RoeEngine, WeaponsPosture, EngagementContext, ThreatClassification};
//! use strix_core::roe::EngagementAuth;
//! use strix_core::Regime;
//!
//! let engine = RoeEngine::new(WeaponsPosture::WeaponsTight);
//! let ctx = EngagementContext {
//!     threat_class: ThreatClassification::ConfirmedHostile,
//!     threat_distance: 200.0,
//!     hostile_act: false,
//!     hostile_intent: true,
//!     collateral_risk: 0.1,
//!     friendlies_at_risk: 0,
//!     regime: Regime::Engage,
//! };
//! assert!(matches!(engine.authorize_engagement(&ctx), EngagementAuth::Authorized { .. }));
//! ```

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Weapons Posture
// ---------------------------------------------------------------------------

/// Weapons posture — standard military ROE states.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WeaponsPosture {
    /// Engage only if fired upon. Maximum restraint.
    WeaponsHold,
    /// Engage only positively identified threats. Standard posture.
    WeaponsTight,
    /// Engage any target not positively identified as friendly. Maximum aggression.
    WeaponsFree,
}

// ---------------------------------------------------------------------------
// Engagement Authorization
// ---------------------------------------------------------------------------

/// Engagement authorization result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EngagementAuth {
    /// Engagement authorized with specified conditions.
    Authorized { conditions: Vec<String> },
    /// Engagement denied with reason.
    Denied { reason: String },
    /// Escalation required — human-in-the-loop decision needed.
    EscalationRequired {
        reason: String,
        urgency: EscalationUrgency,
    },
}

/// Urgency level for a human-in-the-loop escalation request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EscalationUrgency {
    /// Routine — no immediate time pressure.
    Routine,
    /// Priority — decision needed within minutes.
    Priority,
    /// Immediate — decision needed within seconds.
    Immediate,
}

// ---------------------------------------------------------------------------
// Threat Classification
// ---------------------------------------------------------------------------

/// Threat classification for ROE purposes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ThreatClassification {
    /// Confirmed hostile — positively identified as enemy.
    ConfirmedHostile,
    /// Suspected hostile — behavior matches threat patterns.
    SuspectedHostile,
    /// Unknown — insufficient data for classification.
    Unknown,
    /// Friendly — positively identified as friendly.
    Friendly,
    /// Civilian/Non-combatant.
    Civilian,
}

// ---------------------------------------------------------------------------
// Engagement Context
// ---------------------------------------------------------------------------

/// Context for an engagement decision.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngagementContext {
    /// Threat classification.
    pub threat_class: ThreatClassification,
    /// Distance to threat (meters).
    pub threat_distance: f64,
    /// Whether the threat has demonstrated hostile intent (fired/attacked).
    pub hostile_act: bool,
    /// Whether the threat is demonstrating hostile intent (aiming/approaching aggressively).
    pub hostile_intent: bool,
    /// Estimated collateral damage risk [0, 1].
    pub collateral_risk: f64,
    /// Number of friendly assets at risk if engagement is denied.
    pub friendlies_at_risk: u32,
    /// Current operational regime.
    pub regime: crate::Regime,
}

// ---------------------------------------------------------------------------
// ROE Engine
// ---------------------------------------------------------------------------

/// Rules of Engagement engine — evaluates engagement requests against current posture.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoeEngine {
    /// Current weapons posture.
    pub posture: WeaponsPosture,
    /// Maximum acceptable collateral risk [0, 1] before requiring escalation.
    pub max_collateral_risk: f64,
    /// Minimum threat distance for autonomous engagement (meters).
    /// Closer than this always requires escalation.
    pub min_engagement_distance: f64,
    /// Whether self-defense override is enabled (always allows defense if under attack).
    pub self_defense_override: bool,
}

impl Default for RoeEngine {
    fn default() -> Self {
        Self {
            posture: WeaponsPosture::WeaponsTight,
            max_collateral_risk: 0.3,
            min_engagement_distance: 50.0,
            self_defense_override: true,
        }
    }
}

impl RoeEngine {
    /// Construct a new [`RoeEngine`] with the given posture and all other
    /// fields at their safe defaults.
    pub fn new(posture: WeaponsPosture) -> Self {
        Self {
            posture,
            ..Default::default()
        }
    }

    /// Evaluate an engagement request against current ROE.
    ///
    /// Decision priority (highest to lowest):
    /// 1. Protected classification — friendly/civilian → **Denied** (absolute, overrides everything)
    /// 2. Self-defense override — hostile act + override enabled → **Authorized**
    /// 3. Collateral risk cap exceeded → **EscalationRequired**
    /// 4. Below minimum engagement distance → **EscalationRequired**
    /// 5. Posture-specific rules
    pub fn authorize_engagement(&self, ctx: &EngagementContext) -> EngagementAuth {
        // 1. ABSOLUTE GUARD: Never engage friendlies or civilians, even in self-defense.
        // This check MUST come before self-defense override to prevent fratricide.
        if matches!(
            ctx.threat_class,
            ThreatClassification::Friendly | ThreatClassification::Civilian
        ) {
            return EngagementAuth::Denied {
                reason: format!(
                    "Target classified as {:?} — engagement prohibited",
                    ctx.threat_class
                ),
            };
        }

        // 2. Self-defense override: hostile act + self_defense_override → authorized.
        if self.self_defense_override && ctx.hostile_act {
            return EngagementAuth::Authorized {
                conditions: vec!["Self-defense: hostile act confirmed".into()],
            };
        }

        // 3. Collateral risk check.
        if ctx.collateral_risk > self.max_collateral_risk {
            return EngagementAuth::EscalationRequired {
                reason: format!(
                    "Collateral risk {:.0}% exceeds maximum {:.0}%",
                    ctx.collateral_risk * 100.0,
                    self.max_collateral_risk * 100.0
                ),
                urgency: if ctx.hostile_intent {
                    EscalationUrgency::Immediate
                } else {
                    EscalationUrgency::Priority
                },
            };
        }

        // 4. Distance check.
        if ctx.threat_distance < self.min_engagement_distance {
            return EngagementAuth::EscalationRequired {
                reason: format!(
                    "Target at {:.0}m — below minimum autonomous engagement distance ({:.0}m)",
                    ctx.threat_distance, self.min_engagement_distance
                ),
                urgency: EscalationUrgency::Immediate,
            };
        }

        // 5. Posture-specific rules.
        match self.posture {
            WeaponsPosture::WeaponsHold => {
                // Only engage if under direct attack.
                if ctx.hostile_act {
                    EngagementAuth::Authorized {
                        conditions: vec!["Weapons-hold: return fire authorized".into()],
                    }
                } else {
                    EngagementAuth::Denied {
                        reason: "Weapons-hold: no hostile act observed".into(),
                    }
                }
            }

            WeaponsPosture::WeaponsTight => {
                // Engage only positively identified threats.
                match ctx.threat_class {
                    ThreatClassification::ConfirmedHostile => EngagementAuth::Authorized {
                        conditions: vec!["Weapons-tight: confirmed hostile".into()],
                    },
                    ThreatClassification::SuspectedHostile if ctx.hostile_intent => {
                        EngagementAuth::EscalationRequired {
                            reason: "Suspected hostile with intent — confirm PID before engagement"
                                .into(),
                            urgency: EscalationUrgency::Priority,
                        }
                    }
                    _ => EngagementAuth::Denied {
                        reason: format!(
                            "Weapons-tight: target classified as {:?} — PID required",
                            ctx.threat_class
                        ),
                    },
                }
            }

            WeaponsPosture::WeaponsFree => {
                // Engage anything not identified as friendly (already handled above).
                match ctx.threat_class {
                    ThreatClassification::Friendly | ThreatClassification::Civilian => {
                        EngagementAuth::Denied {
                            reason: "Target identified as non-hostile".into(),
                        }
                    }
                    _ => EngagementAuth::Authorized {
                        conditions: vec![format!("Weapons-free: target {:?}", ctx.threat_class)],
                    },
                }
            }
        }
    }

    /// Transition weapons posture. Returns the new posture.
    pub fn set_posture(&mut self, new_posture: WeaponsPosture) -> WeaponsPosture {
        self.posture = new_posture;
        self.posture
    }

    /// Check if a posture transition is valid.
    ///
    /// Currently always returns `true`; provides a hook for future
    /// escalation-only transition constraints (e.g., WeaponsFree requires
    /// explicit commander authorization).
    pub fn can_transition(&self, _to: WeaponsPosture) -> bool {
        true
    }

    /// Suggest a weapons posture based on phi-sim's opponent process tension.
    ///
    /// T = (C-F)/(1+F*C):
    ///   T < -0.3  → WeaponsHold  (fear dominates → be cautious)
    ///   T ∈ [-0.3, 0.1] → WeaponsTight (balanced)
    ///   T > 0.1   → WeaponsFree  (courage dominates → be aggressive)
    ///
    /// Returns None if the suggested posture matches the current one.
    /// This is advisory only — the tick loop logs the suggestion but does NOT auto-switch.
    pub fn tension_posture_suggestion(&self, tension: f64) -> Option<WeaponsPosture> {
        let suggested = if tension < -0.3 {
            WeaponsPosture::WeaponsHold
        } else if tension > 0.1 {
            WeaponsPosture::WeaponsFree
        } else {
            WeaponsPosture::WeaponsTight
        };
        if suggested != self.posture {
            Some(suggested)
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Regime;

    /// Convenience builder — returns a "safe" baseline context that passes all
    /// guards except the ones we intentionally override in each test.
    fn baseline_ctx() -> EngagementContext {
        EngagementContext {
            threat_class: ThreatClassification::ConfirmedHostile,
            threat_distance: 200.0,
            hostile_act: false,
            hostile_intent: false,
            collateral_risk: 0.1,
            friendlies_at_risk: 0,
            regime: Regime::Engage,
        }
    }

    // -----------------------------------------------------------------------
    // Self-defense override
    // -----------------------------------------------------------------------

    #[test]
    fn test_self_defense_override() {
        // Even WeaponsHold must authorize when under direct attack.
        let engine = RoeEngine::new(WeaponsPosture::WeaponsHold);
        let ctx = EngagementContext {
            hostile_act: true,
            ..baseline_ctx()
        };
        let auth = engine.authorize_engagement(&ctx);
        assert!(
            matches!(auth, EngagementAuth::Authorized { .. }),
            "self-defense override should authorize regardless of posture"
        );
    }

    #[test]
    fn test_self_defense_override_disabled() {
        // With override disabled and WeaponsHold, hostile act does NOT auto-authorize
        // at the top-level; falls through to posture logic which allows return fire.
        let mut engine = RoeEngine::new(WeaponsPosture::WeaponsHold);
        engine.self_defense_override = false;
        let ctx = EngagementContext {
            hostile_act: true,
            ..baseline_ctx()
        };
        // WeaponsHold posture logic also allows return fire, so still Authorized.
        let auth = engine.authorize_engagement(&ctx);
        assert!(matches!(auth, EngagementAuth::Authorized { .. }));
    }

    // -----------------------------------------------------------------------
    // Protected classifications
    // -----------------------------------------------------------------------

    #[test]
    fn test_never_engage_friendly() {
        let engine = RoeEngine::new(WeaponsPosture::WeaponsFree);
        let ctx = EngagementContext {
            threat_class: ThreatClassification::Friendly,
            hostile_act: false, // no self-defense override path
            ..baseline_ctx()
        };
        let auth = engine.authorize_engagement(&ctx);
        assert!(
            matches!(auth, EngagementAuth::Denied { .. }),
            "friendly must always be denied"
        );
    }

    #[test]
    fn test_never_engage_civilian() {
        let engine = RoeEngine::new(WeaponsPosture::WeaponsFree);
        let ctx = EngagementContext {
            threat_class: ThreatClassification::Civilian,
            hostile_act: false,
            ..baseline_ctx()
        };
        let auth = engine.authorize_engagement(&ctx);
        assert!(
            matches!(auth, EngagementAuth::Denied { .. }),
            "civilian must always be denied"
        );
    }

    // -----------------------------------------------------------------------
    // WeaponsHold
    // -----------------------------------------------------------------------

    #[test]
    fn test_weapons_hold_denies_without_hostile_act() {
        let engine = RoeEngine::new(WeaponsPosture::WeaponsHold);
        let ctx = EngagementContext {
            hostile_act: false,
            ..baseline_ctx()
        };
        let auth = engine.authorize_engagement(&ctx);
        assert!(
            matches!(auth, EngagementAuth::Denied { .. }),
            "WeaponsHold must deny without hostile act"
        );
    }

    #[test]
    fn test_weapons_hold_allows_return_fire() {
        let mut engine = RoeEngine::new(WeaponsPosture::WeaponsHold);
        // Disable the top-level override to exercise the posture-branch logic.
        engine.self_defense_override = false;
        let ctx = EngagementContext {
            hostile_act: true,
            ..baseline_ctx()
        };
        let auth = engine.authorize_engagement(&ctx);
        assert!(
            matches!(auth, EngagementAuth::Authorized { .. }),
            "WeaponsHold must allow return fire when hostile act observed"
        );
    }

    // -----------------------------------------------------------------------
    // WeaponsTight
    // -----------------------------------------------------------------------

    #[test]
    fn test_weapons_tight_allows_confirmed_hostile() {
        let engine = RoeEngine::new(WeaponsPosture::WeaponsTight);
        let ctx = EngagementContext {
            threat_class: ThreatClassification::ConfirmedHostile,
            ..baseline_ctx()
        };
        let auth = engine.authorize_engagement(&ctx);
        assert!(
            matches!(auth, EngagementAuth::Authorized { .. }),
            "WeaponsTight must authorize confirmed hostile"
        );
    }

    #[test]
    fn test_weapons_tight_denies_unknown() {
        let engine = RoeEngine::new(WeaponsPosture::WeaponsTight);
        let ctx = EngagementContext {
            threat_class: ThreatClassification::Unknown,
            hostile_intent: false,
            ..baseline_ctx()
        };
        let auth = engine.authorize_engagement(&ctx);
        assert!(
            matches!(auth, EngagementAuth::Denied { .. }),
            "WeaponsTight must deny unknown target without hostile intent"
        );
    }

    #[test]
    fn test_weapons_tight_escalates_suspected_with_intent() {
        let engine = RoeEngine::new(WeaponsPosture::WeaponsTight);
        let ctx = EngagementContext {
            threat_class: ThreatClassification::SuspectedHostile,
            hostile_intent: true,
            ..baseline_ctx()
        };
        let auth = engine.authorize_engagement(&ctx);
        assert!(
            matches!(
                auth,
                EngagementAuth::EscalationRequired {
                    urgency: EscalationUrgency::Priority,
                    ..
                }
            ),
            "WeaponsTight must escalate suspected hostile with hostile intent"
        );
    }

    // -----------------------------------------------------------------------
    // WeaponsFree
    // -----------------------------------------------------------------------

    #[test]
    fn test_weapons_free_allows_unknown() {
        let engine = RoeEngine::new(WeaponsPosture::WeaponsFree);
        let ctx = EngagementContext {
            threat_class: ThreatClassification::Unknown,
            ..baseline_ctx()
        };
        let auth = engine.authorize_engagement(&ctx);
        assert!(
            matches!(auth, EngagementAuth::Authorized { .. }),
            "WeaponsFree must authorize unknown target"
        );
    }

    #[test]
    fn test_weapons_free_denies_friendly() {
        let engine = RoeEngine::new(WeaponsPosture::WeaponsFree);
        let ctx = EngagementContext {
            threat_class: ThreatClassification::Friendly,
            hostile_act: false,
            ..baseline_ctx()
        };
        let auth = engine.authorize_engagement(&ctx);
        assert!(
            matches!(auth, EngagementAuth::Denied { .. }),
            "WeaponsFree must still deny friendly"
        );
    }

    // -----------------------------------------------------------------------
    // Safety guards: collateral risk and distance
    // -----------------------------------------------------------------------

    #[test]
    fn test_collateral_risk_escalation() {
        let engine = RoeEngine::new(WeaponsPosture::WeaponsFree);
        let ctx = EngagementContext {
            // 60% risk > 30% default cap
            collateral_risk: 0.6,
            hostile_intent: false,
            ..baseline_ctx()
        };
        let auth = engine.authorize_engagement(&ctx);
        assert!(
            matches!(
                auth,
                EngagementAuth::EscalationRequired {
                    urgency: EscalationUrgency::Priority,
                    ..
                }
            ),
            "high collateral risk must trigger escalation"
        );
    }

    #[test]
    fn test_collateral_risk_escalation_immediate_when_hostile_intent() {
        let engine = RoeEngine::new(WeaponsPosture::WeaponsFree);
        let ctx = EngagementContext {
            collateral_risk: 0.6,
            hostile_intent: true,
            ..baseline_ctx()
        };
        let auth = engine.authorize_engagement(&ctx);
        assert!(
            matches!(
                auth,
                EngagementAuth::EscalationRequired {
                    urgency: EscalationUrgency::Immediate,
                    ..
                }
            ),
            "collateral risk + hostile intent must escalate as Immediate"
        );
    }

    #[test]
    fn test_distance_check_escalation() {
        let engine = RoeEngine::new(WeaponsPosture::WeaponsFree);
        let ctx = EngagementContext {
            // 30m < 50m default minimum
            threat_distance: 30.0,
            ..baseline_ctx()
        };
        let auth = engine.authorize_engagement(&ctx);
        assert!(
            matches!(
                auth,
                EngagementAuth::EscalationRequired {
                    urgency: EscalationUrgency::Immediate,
                    ..
                }
            ),
            "target below minimum distance must escalate as Immediate"
        );
    }

    // -----------------------------------------------------------------------
    // Posture transitions
    // -----------------------------------------------------------------------

    #[test]
    fn test_posture_transition() {
        let mut engine = RoeEngine::new(WeaponsPosture::WeaponsHold);
        assert_eq!(engine.posture, WeaponsPosture::WeaponsHold);

        let new_posture = engine.set_posture(WeaponsPosture::WeaponsTight);
        assert_eq!(new_posture, WeaponsPosture::WeaponsTight);
        assert_eq!(engine.posture, WeaponsPosture::WeaponsTight);

        engine.set_posture(WeaponsPosture::WeaponsFree);
        assert_eq!(engine.posture, WeaponsPosture::WeaponsFree);

        assert!(engine.can_transition(WeaponsPosture::WeaponsHold));
        assert!(engine.can_transition(WeaponsPosture::WeaponsTight));
    }

    // -----------------------------------------------------------------------
    // Serialization round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn test_roe_engine_serde_round_trip() {
        let engine = RoeEngine {
            posture: WeaponsPosture::WeaponsFree,
            max_collateral_risk: 0.15,
            min_engagement_distance: 100.0,
            self_defense_override: false,
        };
        let json = serde_json::to_string(&engine).expect("serialize");
        let decoded: RoeEngine = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(decoded.posture, engine.posture);
        assert!((decoded.max_collateral_risk - engine.max_collateral_risk).abs() < 1e-12);
        assert!((decoded.min_engagement_distance - engine.min_engagement_distance).abs() < 1e-12);
        assert_eq!(decoded.self_defense_override, engine.self_defense_override);
    }

    #[test]
    fn test_engagement_auth_serde_round_trip() {
        let auth = EngagementAuth::EscalationRequired {
            reason: "test".into(),
            urgency: EscalationUrgency::Immediate,
        };
        let json = serde_json::to_string(&auth).expect("serialize");
        let decoded: EngagementAuth = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(decoded, auth);
    }

    // -----------------------------------------------------------------------
    // Tension-based posture suggestion
    // -----------------------------------------------------------------------

    #[test]
    fn test_tension_posture_suggestion_hold() {
        let engine = RoeEngine::new(WeaponsPosture::WeaponsTight);
        let suggestion = engine.tension_posture_suggestion(-0.5);
        assert_eq!(suggestion, Some(WeaponsPosture::WeaponsHold));
    }

    #[test]
    fn test_tension_posture_suggestion_free() {
        let engine = RoeEngine::new(WeaponsPosture::WeaponsTight);
        let suggestion = engine.tension_posture_suggestion(0.3);
        assert_eq!(suggestion, Some(WeaponsPosture::WeaponsFree));
    }

    #[test]
    fn test_tension_posture_suggestion_tight() {
        let engine = RoeEngine::new(WeaponsPosture::WeaponsHold);
        let suggestion = engine.tension_posture_suggestion(0.0);
        assert_eq!(suggestion, Some(WeaponsPosture::WeaponsTight));
    }

    #[test]
    fn test_tension_posture_suggestion_no_change() {
        let engine = RoeEngine::new(WeaponsPosture::WeaponsTight);
        let suggestion = engine.tension_posture_suggestion(0.0); // tight zone
        assert_eq!(
            suggestion, None,
            "should return None when suggestion matches current posture"
        );
    }

    #[test]
    fn test_tension_posture_suggestion_boundary() {
        let engine = RoeEngine::new(WeaponsPosture::WeaponsFree);
        // Exactly at -0.3 boundary → should be in tight zone
        let suggestion = engine.tension_posture_suggestion(-0.3);
        assert_eq!(suggestion, Some(WeaponsPosture::WeaponsTight));
        // Exactly at 0.1 → should be in tight zone
        let suggestion2 = engine.tension_posture_suggestion(0.1);
        assert_eq!(suggestion2, Some(WeaponsPosture::WeaponsTight));
    }

    // -----------------------------------------------------------------------
    // Critical regression: self-defense MUST NOT override protected classification
    // -----------------------------------------------------------------------

    #[test]
    fn test_self_defense_does_not_override_friendly_protection() {
        // Even under hostile_act + self_defense_override, engaging a Friendly is DENIED.
        let engine = RoeEngine::new(WeaponsPosture::WeaponsFree);
        let ctx = EngagementContext {
            threat_class: ThreatClassification::Friendly,
            hostile_act: true, // under fire, but target is friendly
            ..baseline_ctx()
        };
        let auth = engine.authorize_engagement(&ctx);
        assert!(
            matches!(auth, EngagementAuth::Denied { .. }),
            "self-defense override MUST NOT authorize engagement of friendly targets"
        );
    }

    #[test]
    fn test_self_defense_does_not_override_civilian_protection() {
        let engine = RoeEngine::new(WeaponsPosture::WeaponsFree);
        let ctx = EngagementContext {
            threat_class: ThreatClassification::Civilian,
            hostile_act: true, // under fire, but target is civilian
            ..baseline_ctx()
        };
        let auth = engine.authorize_engagement(&ctx);
        assert!(
            matches!(auth, EngagementAuth::Denied { .. }),
            "self-defense override MUST NOT authorize engagement of civilian targets"
        );
    }
}
