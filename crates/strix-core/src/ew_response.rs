//! Electronic Warfare (EW) response module for STRIX.
//!
//! Classifies EW threats (GPS denial/spoofing, comms jamming, radar lock,
//! directed energy) and produces automated multi-action response plans.
//!
//! # Response philosophy
//!
//! Each threat type maps to a priority-ordered action set scaled by severity:
//!
//! | Threat           | Primary defence                        |
//! |------------------|----------------------------------------|
//! | GPS denial       | Inertial fallback + noise expansion    |
//! | GPS spoofing     | Inertial fallback + forced evasion     |
//! | Comms jamming    | Gossip protocol degradation            |
//! | Radar lock       | Terrain masking + immediate evade      |
//! | Directed energy  | Immediate evade + terrain masking      |

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Threat classification
// ---------------------------------------------------------------------------

/// Electronic warfare threat classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EwThreat {
    /// GPS signal denial/jamming — navigation degraded.
    GpsDenial,
    /// GPS spoofing — false position data injected.
    GpsSpoofing,
    /// Communications jamming — mesh links degraded.
    CommsJamming,
    /// Radar lock detected — active tracking by enemy.
    RadarLock,
    /// Directed energy weapon detected.
    DirectedEnergy,
}

/// Severity of the EW threat.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum EwSeverity {
    /// Detected but not impacting operations.
    Detected,
    /// Degrading capability but still operational.
    Degraded,
    /// Severe impact — failover required.
    Severe,
    /// Complete denial of affected capability.
    Denied,
}

/// EW detection event with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EwEvent {
    pub threat: EwThreat,
    pub severity: EwSeverity,
    /// Estimated bearing to source (degrees, 0=North, clockwise). None if unknown.
    pub source_bearing: Option<f64>,
    /// Estimated range to source (meters). None if unknown.
    pub source_range: Option<f64>,
    /// Confidence in the detection [0, 1].
    pub confidence: f64,
    /// Timestamp (seconds since mission start).
    pub timestamp: f64,
}

// ---------------------------------------------------------------------------
// Response actions
// ---------------------------------------------------------------------------

/// Automated response action for an EW threat.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EwResponse {
    /// Expand particle filter process noise to account for degraded navigation.
    ExpandNavigationNoise { noise_multiplier: f64 },
    /// Switch gossip protocol to low-bandwidth fallback mode.
    GossipFallback {
        reduced_fanout: usize,
        priority_only: bool,
    },
    /// Immediate regime override to EVADE for affected drones.
    /// An empty `affected_drone_ids` means the entire swarm.
    ForceEvade { affected_drone_ids: Vec<u32> },
    /// Increase pheromone deposit weight near EW source for avoidance.
    MarkEwZone {
        bearing: f64,
        range: f64,
        penalty_weight: f64,
    },
    /// Reduce altitude to terrain-mask from radar.
    TerrainMask { target_altitude: f64 },
    /// Switch to inertial-only navigation (dead reckoning).
    InertialFallback,
    /// No response needed at this severity level.
    Monitor,
}

/// A complete response plan for an EW event (may have multiple actions).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EwResponsePlan {
    pub event: EwEvent,
    pub actions: Vec<EwResponse>,
    /// Human-readable summary of the response.
    pub summary: String,
}

// ---------------------------------------------------------------------------
// Response engine
// ---------------------------------------------------------------------------

/// Electronic warfare response engine.
///
/// Evaluates EW detections and produces automated response plans.
/// Call [`EwEngine::respond`] whenever a new [`EwEvent`] is detected;
/// the engine accumulates history and can be queried for active threats.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EwEngine {
    /// Base noise expansion factor for GPS denial (scaled by severity multiplier).
    pub gps_denial_noise_base: f64,
    /// Gossip fanout to use when comms are jammed.
    pub jammed_gossip_fanout: usize,
    /// Terrain masking altitude (meters AGL) for radar / DE threats.
    pub terrain_mask_altitude: f64,
    /// Pheromone penalty weight applied to detected EW zones.
    pub ew_zone_penalty: f64,
    /// Active EW events being tracked.
    active_events: Vec<EwEvent>,
}

impl Default for EwEngine {
    fn default() -> Self {
        Self {
            gps_denial_noise_base: 2.0,
            jammed_gossip_fanout: 1,
            terrain_mask_altitude: 30.0,
            ew_zone_penalty: 5.0,
            active_events: Vec::new(),
        }
    }
}

impl EwEngine {
    /// Create a new engine with default tuning parameters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Process an EW detection event and return the response plan.
    ///
    /// The event is recorded internally so it can be queried via
    /// [`EwEngine::active_threats`].
    pub fn respond(&mut self, event: EwEvent) -> EwResponsePlan {
        self.active_events.push(event.clone());
        let actions = self.compute_responses(&event);
        let summary = Self::summarize(&event, &actions);
        EwResponsePlan {
            event,
            actions,
            summary,
        }
    }

    /// Currently active EW threats (all events not yet cleared).
    pub fn active_threats(&self) -> &[EwEvent] {
        &self.active_events
    }

    /// Clear resolved events older than `max_age` seconds.
    ///
    /// Call periodically with the current mission clock.
    pub fn clear_stale_events(&mut self, current_time: f64, max_age: f64) {
        self.active_events
            .retain(|e| current_time - e.timestamp < max_age);
    }

    // -----------------------------------------------------------------------
    // Internal dispatch
    // -----------------------------------------------------------------------

    fn compute_responses(&self, event: &EwEvent) -> Vec<EwResponse> {
        match event.threat {
            EwThreat::GpsDenial => self.respond_gps_denial(event),
            EwThreat::GpsSpoofing => self.respond_gps_spoofing(event),
            EwThreat::CommsJamming => self.respond_comms_jamming(event),
            EwThreat::RadarLock => self.respond_radar_lock(event),
            EwThreat::DirectedEnergy => self.respond_directed_energy(event),
        }
    }

    // -----------------------------------------------------------------------
    // Per-threat response builders
    // -----------------------------------------------------------------------

    fn respond_gps_denial(&self, event: &EwEvent) -> Vec<EwResponse> {
        let mut actions = Vec::new();
        let severity_mult = severity_multiplier(event.severity);

        match event.severity {
            EwSeverity::Detected => {
                // Just monitor — no operational impact yet.
                actions.push(EwResponse::Monitor);
            }
            EwSeverity::Degraded => {
                // Widen the particle filter's process noise to compensate.
                actions.push(EwResponse::ExpandNavigationNoise {
                    noise_multiplier: self.gps_denial_noise_base * severity_mult,
                });
            }
            EwSeverity::Severe | EwSeverity::Denied => {
                // Noise expansion + full inertial handover.
                actions.push(EwResponse::ExpandNavigationNoise {
                    noise_multiplier: self.gps_denial_noise_base * severity_mult,
                });
                actions.push(EwResponse::InertialFallback);
                // Mark source zone if geolocation data is available.
                if let (Some(bearing), Some(range)) = (event.source_bearing, event.source_range) {
                    actions.push(EwResponse::MarkEwZone {
                        bearing,
                        range,
                        penalty_weight: self.ew_zone_penalty,
                    });
                }
            }
        }
        actions
    }

    fn respond_gps_spoofing(&self, event: &EwEvent) -> Vec<EwResponse> {
        // Spoofing is more dangerous than pure denial: the drone cannot detect
        // corrupt position data on its own.  Always switch to inertial and
        // inflate uncertainty heavily.
        let mut actions = vec![
            EwResponse::InertialFallback,
            EwResponse::ExpandNavigationNoise {
                noise_multiplier: self.gps_denial_noise_base * 3.0,
            },
        ];
        if event.severity >= EwSeverity::Severe {
            // Evade — spoofing at severe/denied severity may be coordinating an
            // intercept; best to maneuver immediately.
            actions.push(EwResponse::ForceEvade {
                affected_drone_ids: vec![], // empty = entire swarm
            });
        }
        actions
    }

    fn respond_comms_jamming(&self, event: &EwEvent) -> Vec<EwResponse> {
        match event.severity {
            EwSeverity::Detected => {
                // Link quality falling but still acceptable.
                vec![EwResponse::Monitor]
            }
            EwSeverity::Degraded => {
                // Reduce gossip fanout to conserve bandwidth for critical msgs.
                vec![EwResponse::GossipFallback {
                    reduced_fanout: self.jammed_gossip_fanout,
                    priority_only: false,
                }]
            }
            EwSeverity::Severe | EwSeverity::Denied => {
                // Priority-only: health, emergency, regime commands only.
                vec![EwResponse::GossipFallback {
                    reduced_fanout: self.jammed_gossip_fanout,
                    priority_only: true,
                }]
            }
        }
    }

    fn respond_radar_lock(&self, event: &EwEvent) -> Vec<EwResponse> {
        // Radar lock means an enemy fire-control system has acquired the drone.
        // Immediate evasion + terrain masking are non-negotiable.
        let mut actions = vec![
            EwResponse::ForceEvade {
                affected_drone_ids: vec![],
            },
            EwResponse::TerrainMask {
                target_altitude: self.terrain_mask_altitude,
            },
        ];
        if let (Some(bearing), Some(range)) = (event.source_bearing, event.source_range) {
            // Double penalty weight — radar lock zone is a hard-avoidance area.
            actions.push(EwResponse::MarkEwZone {
                bearing,
                range,
                penalty_weight: self.ew_zone_penalty * 2.0,
            });
        }
        actions
    }

    fn respond_directed_energy(&self, event: &EwEvent) -> Vec<EwResponse> {
        // Directed energy (laser / HPM) is potentially lethal in milliseconds.
        // Maximum urgency: evade immediately and go low.
        // Source geolocation is used only when available; we do not block on it.
        let mut actions = vec![
            EwResponse::ForceEvade {
                affected_drone_ids: vec![],
            },
            EwResponse::TerrainMask {
                target_altitude: self.terrain_mask_altitude,
            },
        ];
        if let (Some(bearing), Some(range)) = (event.source_bearing, event.source_range) {
            actions.push(EwResponse::MarkEwZone {
                bearing,
                range,
                penalty_weight: self.ew_zone_penalty * 3.0, // hardest avoidance
            });
        }
        // Even at Detected severity, mark and evade — DE has near-zero latency.
        let _ = event.severity; // severity doesn't soften the DE response
        actions
    }

    // -----------------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------------

    fn summarize(event: &EwEvent, actions: &[EwResponse]) -> String {
        let threat_name = match event.threat {
            EwThreat::GpsDenial => "GPS denial",
            EwThreat::GpsSpoofing => "GPS spoofing",
            EwThreat::CommsJamming => "Comms jamming",
            EwThreat::RadarLock => "Radar lock",
            EwThreat::DirectedEnergy => "Directed energy",
        };
        let severity_name = match event.severity {
            EwSeverity::Detected => "detected",
            EwSeverity::Degraded => "degraded",
            EwSeverity::Severe => "severe",
            EwSeverity::Denied => "denied",
        };
        format!(
            "{} ({}) — {} response action(s)",
            threat_name,
            severity_name,
            actions.len()
        )
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Map a severity level to a numeric multiplier for scaling response intensity.
fn severity_multiplier(severity: EwSeverity) -> f64 {
    match severity {
        EwSeverity::Detected => 1.0,
        EwSeverity::Degraded => 1.5,
        EwSeverity::Severe => 2.5,
        EwSeverity::Denied => 4.0,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn make_event(threat: EwThreat, severity: EwSeverity) -> EwEvent {
        EwEvent {
            threat,
            severity,
            source_bearing: None,
            source_range: None,
            confidence: 0.9,
            timestamp: 0.0,
        }
    }

    fn make_event_with_location(
        threat: EwThreat,
        severity: EwSeverity,
        bearing: f64,
        range: f64,
    ) -> EwEvent {
        EwEvent {
            threat,
            severity,
            source_bearing: Some(bearing),
            source_range: Some(range),
            confidence: 0.9,
            timestamp: 0.0,
        }
    }

    // -----------------------------------------------------------------------
    // GPS denial
    // -----------------------------------------------------------------------

    #[test]
    fn test_gps_denial_detected_monitors() {
        let mut engine = EwEngine::new();
        let plan = engine.respond(make_event(EwThreat::GpsDenial, EwSeverity::Detected));
        assert_eq!(plan.actions.len(), 1);
        assert_eq!(plan.actions[0], EwResponse::Monitor);
    }

    #[test]
    fn test_gps_denial_degraded_expands_noise() {
        let mut engine = EwEngine::new();
        let plan = engine.respond(make_event(EwThreat::GpsDenial, EwSeverity::Degraded));
        assert_eq!(plan.actions.len(), 1);
        match &plan.actions[0] {
            EwResponse::ExpandNavigationNoise { noise_multiplier } => {
                // base=2.0, severity_mult=1.5 → 3.0
                let expected = engine.gps_denial_noise_base * 1.5;
                assert!(
                    (noise_multiplier - expected).abs() < 1e-10,
                    "expected noise_multiplier={expected}, got {noise_multiplier}"
                );
            }
            other => panic!("unexpected action: {other:?}"),
        }
    }

    #[test]
    fn test_gps_denial_severe_inertial_fallback() {
        let mut engine = EwEngine::new();
        // With source location so MarkEwZone is also emitted.
        let event = make_event_with_location(EwThreat::GpsDenial, EwSeverity::Severe, 45.0, 800.0);
        let plan = engine.respond(event);

        assert_eq!(plan.actions.len(), 3);
        assert!(
            matches!(plan.actions[0], EwResponse::ExpandNavigationNoise { .. }),
            "first action must expand noise"
        );
        assert_eq!(plan.actions[1], EwResponse::InertialFallback);
        assert!(
            matches!(plan.actions[2], EwResponse::MarkEwZone { bearing, range, .. } if (bearing - 45.0).abs() < 1e-10 && (range - 800.0).abs() < 1e-10),
            "third action must mark the EW zone at correct coordinates"
        );
    }

    #[test]
    fn test_gps_denial_severe_no_location_skips_zone_mark() {
        let mut engine = EwEngine::new();
        let plan = engine.respond(make_event(EwThreat::GpsDenial, EwSeverity::Severe));
        // Without source location: only noise expansion + inertial
        assert_eq!(plan.actions.len(), 2);
        assert!(matches!(plan.actions[1], EwResponse::InertialFallback));
    }

    // -----------------------------------------------------------------------
    // GPS spoofing
    // -----------------------------------------------------------------------

    #[test]
    fn test_gps_spoofing_always_inertial() {
        for severity in [
            EwSeverity::Detected,
            EwSeverity::Degraded,
            EwSeverity::Severe,
            EwSeverity::Denied,
        ] {
            let mut engine = EwEngine::new();
            let plan = engine.respond(make_event(EwThreat::GpsSpoofing, severity));
            assert!(
                plan.actions.contains(&EwResponse::InertialFallback),
                "spoofing at {severity:?} must always trigger InertialFallback"
            );
            let noise_present = plan
                .actions
                .iter()
                .any(|a| matches!(a, EwResponse::ExpandNavigationNoise { .. }));
            assert!(
                noise_present,
                "spoofing must always expand navigation noise"
            );
        }
    }

    #[test]
    fn test_gps_spoofing_severe_force_evade() {
        for severity in [EwSeverity::Severe, EwSeverity::Denied] {
            let mut engine = EwEngine::new();
            let plan = engine.respond(make_event(EwThreat::GpsSpoofing, severity));
            assert!(
                plan.actions
                    .iter()
                    .any(|a| matches!(a, EwResponse::ForceEvade { .. })),
                "spoofing at {severity:?} must force evade"
            );
        }
    }

    #[test]
    fn test_gps_spoofing_detected_no_forced_evade() {
        let mut engine = EwEngine::new();
        let plan = engine.respond(make_event(EwThreat::GpsSpoofing, EwSeverity::Detected));
        let has_evade = plan
            .actions
            .iter()
            .any(|a| matches!(a, EwResponse::ForceEvade { .. }));
        assert!(!has_evade, "spoofing at Detected must NOT force evade yet");
    }

    // -----------------------------------------------------------------------
    // Comms jamming
    // -----------------------------------------------------------------------

    #[test]
    fn test_comms_jamming_detected_monitors() {
        let mut engine = EwEngine::new();
        let plan = engine.respond(make_event(EwThreat::CommsJamming, EwSeverity::Detected));
        assert_eq!(plan.actions, vec![EwResponse::Monitor]);
    }

    #[test]
    fn test_comms_jamming_gossip_fallback() {
        let mut engine = EwEngine::new();
        let plan = engine.respond(make_event(EwThreat::CommsJamming, EwSeverity::Degraded));
        assert_eq!(plan.actions.len(), 1);
        match &plan.actions[0] {
            EwResponse::GossipFallback {
                reduced_fanout,
                priority_only,
            } => {
                assert_eq!(*reduced_fanout, engine.jammed_gossip_fanout);
                assert!(
                    !priority_only,
                    "Degraded jamming must not set priority_only"
                );
            }
            other => panic!("unexpected action: {other:?}"),
        }
    }

    #[test]
    fn test_comms_jamming_severe_priority_only() {
        for severity in [EwSeverity::Severe, EwSeverity::Denied] {
            let mut engine = EwEngine::new();
            let plan = engine.respond(make_event(EwThreat::CommsJamming, severity));
            assert_eq!(plan.actions.len(), 1);
            match &plan.actions[0] {
                EwResponse::GossipFallback { priority_only, .. } => {
                    assert!(priority_only, "{severity:?} jamming must set priority_only");
                }
                other => panic!("unexpected action: {other:?}"),
            }
        }
    }

    // -----------------------------------------------------------------------
    // Radar lock
    // -----------------------------------------------------------------------

    #[test]
    fn test_radar_lock_immediate_evade() {
        let mut engine = EwEngine::new();
        let plan = engine.respond(make_event(EwThreat::RadarLock, EwSeverity::Detected));
        // Always: ForceEvade + TerrainMask (regardless of severity)
        let has_evade = plan
            .actions
            .iter()
            .any(|a| matches!(a, EwResponse::ForceEvade { .. }));
        let has_mask = plan
            .actions
            .iter()
            .any(|a| matches!(a, EwResponse::TerrainMask { .. }));
        assert!(has_evade, "radar lock must always trigger ForceEvade");
        assert!(has_mask, "radar lock must always trigger TerrainMask");
    }

    #[test]
    fn test_radar_lock_marks_zone_when_located() {
        let mut engine = EwEngine::new();
        let event =
            make_event_with_location(EwThreat::RadarLock, EwSeverity::Severe, 270.0, 2000.0);
        let plan = engine.respond(event);
        let zone = plan.actions.iter().find_map(|a| {
            if let EwResponse::MarkEwZone {
                bearing,
                range,
                penalty_weight,
            } = a
            {
                Some((*bearing, *range, *penalty_weight))
            } else {
                None
            }
        });
        let (b, r, w) = zone.expect("must mark EW zone when source location is known");
        assert!((b - 270.0).abs() < 1e-10);
        assert!((r - 2000.0).abs() < 1e-10);
        // Radar lock zone gets 2× base penalty weight.
        assert!(
            (w - engine.ew_zone_penalty * 2.0).abs() < 1e-10,
            "radar zone penalty must be 2× base"
        );
    }

    // -----------------------------------------------------------------------
    // Directed energy
    // -----------------------------------------------------------------------

    #[test]
    fn test_directed_energy_evade() {
        let mut engine = EwEngine::new();
        let plan = engine.respond(make_event(EwThreat::DirectedEnergy, EwSeverity::Detected));
        let has_evade = plan
            .actions
            .iter()
            .any(|a| matches!(a, EwResponse::ForceEvade { .. }));
        let has_mask = plan
            .actions
            .iter()
            .any(|a| matches!(a, EwResponse::TerrainMask { .. }));
        assert!(has_evade, "DE must always trigger ForceEvade");
        assert!(has_mask, "DE must always trigger TerrainMask");
    }

    #[test]
    fn test_directed_energy_triple_penalty_when_located() {
        let mut engine = EwEngine::new();
        let event =
            make_event_with_location(EwThreat::DirectedEnergy, EwSeverity::Severe, 90.0, 500.0);
        let plan = engine.respond(event);
        let penalty = plan.actions.iter().find_map(|a| {
            if let EwResponse::MarkEwZone { penalty_weight, .. } = a {
                Some(*penalty_weight)
            } else {
                None
            }
        });
        let w = penalty.expect("DE with source location must mark zone");
        assert!(
            (w - engine.ew_zone_penalty * 3.0).abs() < 1e-10,
            "DE zone penalty must be 3× base"
        );
    }

    // -----------------------------------------------------------------------
    // Event tracking & lifecycle
    // -----------------------------------------------------------------------

    #[test]
    fn test_active_threats_tracking() {
        let mut engine = EwEngine::new();
        assert!(engine.active_threats().is_empty());

        engine.respond(make_event(EwThreat::GpsDenial, EwSeverity::Detected));
        assert_eq!(engine.active_threats().len(), 1);

        engine.respond(make_event(EwThreat::CommsJamming, EwSeverity::Degraded));
        assert_eq!(engine.active_threats().len(), 2);

        let threats: Vec<EwThreat> = engine.active_threats().iter().map(|e| e.threat).collect();
        assert!(threats.contains(&EwThreat::GpsDenial));
        assert!(threats.contains(&EwThreat::CommsJamming));
    }

    #[test]
    fn test_clear_stale_events() {
        let mut engine = EwEngine::new();

        // Old event at t=0.
        engine.respond(make_event(EwThreat::GpsDenial, EwSeverity::Detected));

        // Recent event at t=100.
        engine.respond(EwEvent {
            threat: EwThreat::CommsJamming,
            severity: EwSeverity::Degraded,
            source_bearing: None,
            source_range: None,
            confidence: 0.8,
            timestamp: 100.0,
        });

        assert_eq!(engine.active_threats().len(), 2);

        // Clear events older than 60 s relative to t=120.
        // t=0 event age=120 ≥ 60 → removed.
        // t=100 event age=20 < 60 → kept.
        engine.clear_stale_events(120.0, 60.0);
        assert_eq!(engine.active_threats().len(), 1);
        assert_eq!(engine.active_threats()[0].threat, EwThreat::CommsJamming);
    }

    #[test]
    fn test_clear_stale_events_removes_all_when_all_old() {
        let mut engine = EwEngine::new();
        engine.respond(make_event(EwThreat::GpsDenial, EwSeverity::Detected));
        engine.respond(make_event(EwThreat::RadarLock, EwSeverity::Severe));

        engine.clear_stale_events(10000.0, 60.0);
        assert!(engine.active_threats().is_empty());
    }

    #[test]
    fn test_clear_stale_events_keeps_all_when_recent() {
        let mut engine = EwEngine::new();
        engine.respond(make_event(EwThreat::GpsDenial, EwSeverity::Detected));
        engine.clear_stale_events(10.0, 60.0);
        assert_eq!(engine.active_threats().len(), 1);
    }

    // -----------------------------------------------------------------------
    // Summary string
    // -----------------------------------------------------------------------

    #[test]
    fn test_summary_contains_threat_and_severity() {
        let mut engine = EwEngine::new();
        let plan = engine.respond(make_event(EwThreat::RadarLock, EwSeverity::Severe));
        assert!(plan.summary.contains("Radar lock"));
        assert!(plan.summary.contains("severe"));
        assert!(plan.summary.contains("response action"));
    }

    // -----------------------------------------------------------------------
    // Severity ordering
    // -----------------------------------------------------------------------

    #[test]
    fn test_severity_ordering() {
        assert!(EwSeverity::Detected < EwSeverity::Degraded);
        assert!(EwSeverity::Degraded < EwSeverity::Severe);
        assert!(EwSeverity::Severe < EwSeverity::Denied);
    }
}
