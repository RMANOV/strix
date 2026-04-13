use serde::{Deserialize, Serialize};
use strix_auction::ThreatType;

// ---------------------------------------------------------------------------
// Threat behavior models
// ---------------------------------------------------------------------------

/// How a threat actor moves relative to the fleet centroid.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThreatBehavior {
    /// Straight-line approach toward fleet centroid at constant speed (m/s).
    Approaching { speed: f64 },
    /// Arc approach from angular offset (degrees) — oblique attack vector.
    Flanking { speed: f64, angle_deg: f64 },
    /// Circular orbit at fixed radius around fleet centroid.
    Circling { speed: f64, radius: f64 },
    /// Moving directly away from fleet centroid.
    Retreating { speed: f64 },
    /// Fixed position — stationary emitter or barrier.
    Stationary,
}

// ---------------------------------------------------------------------------
// Threat specification (builder input)
// ---------------------------------------------------------------------------

/// Blueprint for spawning a threat actor in the simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatSpec {
    /// Distance from fleet centroid at spawn (meters).
    pub initial_distance: f64,
    /// Compass bearing from centroid: 0=North, 90=East (degrees).
    pub initial_bearing_deg: f64,
    /// NED altitude (negative = above ground).
    pub altitude: f64,
    /// Movement model.
    pub behavior: ThreatBehavior,
    /// Lethal engagement radius (meters).
    pub lethal_radius: f64,
    /// Classification.
    pub threat_type: ThreatType,
}

impl ThreatSpec {
    /// Frontal approach at given distance and speed.
    pub fn approaching(distance: f64, speed: f64) -> Self {
        Self {
            initial_distance: distance,
            initial_bearing_deg: 0.0,
            altitude: -50.0,
            behavior: ThreatBehavior::Approaching { speed },
            lethal_radius: 200.0,
            threat_type: ThreatType::Sam,
        }
    }

    /// Flanking arc from angular offset.
    pub fn flanking(distance: f64, speed: f64, angle_deg: f64) -> Self {
        Self {
            initial_distance: distance,
            initial_bearing_deg: angle_deg,
            altitude: -50.0,
            behavior: ThreatBehavior::Flanking { speed, angle_deg },
            lethal_radius: 200.0,
            threat_type: ThreatType::Sam,
        }
    }

    /// Circling orbit at given radius and speed.
    pub fn circling(distance: f64, speed: f64) -> Self {
        Self {
            initial_distance: distance,
            initial_bearing_deg: 0.0,
            altitude: -50.0,
            behavior: ThreatBehavior::Circling {
                speed,
                radius: distance,
            },
            lethal_radius: 200.0,
            threat_type: ThreatType::Sam,
        }
    }

    /// Retreating threat moving away.
    pub fn retreating(distance: f64, speed: f64) -> Self {
        Self {
            initial_distance: distance,
            initial_bearing_deg: 0.0,
            altitude: -50.0,
            behavior: ThreatBehavior::Retreating { speed },
            lethal_radius: 200.0,
            threat_type: ThreatType::Sam,
        }
    }

    /// Fixed-position threat (e.g. SAM site).
    pub fn stationary(distance: f64) -> Self {
        Self {
            initial_distance: distance,
            initial_bearing_deg: 0.0,
            altitude: -50.0,
            behavior: ThreatBehavior::Stationary,
            lethal_radius: 200.0,
            threat_type: ThreatType::Sam,
        }
    }

    /// Set the compass bearing for this threat.
    pub fn bearing(mut self, deg: f64) -> Self {
        self.initial_bearing_deg = deg;
        self
    }

    /// Set the threat type classification.
    pub fn threat_type(mut self, tt: ThreatType) -> Self {
        self.threat_type = tt;
        self
    }

    /// Set the lethal radius.
    pub fn lethal_radius(mut self, r: f64) -> Self {
        self.lethal_radius = r;
        self
    }

    /// Set the altitude (NED, negative = above ground).
    pub fn altitude(mut self, alt: f64) -> Self {
        self.altitude = alt;
        self
    }
}

// ---------------------------------------------------------------------------
// Scheduled events
// ---------------------------------------------------------------------------

/// Time-triggered battlefield event injected into the simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Event {
    /// Increase GPS noise across all drones (simulates jamming).
    JamGps { noise_multiplier: f64 },
    /// Restore nominal GPS accuracy.
    RestoreGps,
    /// Mark a specific drone as destroyed/failed.
    LoseDrone { drone_id: u32 },
    /// Spawn a new threat actor mid-simulation.
    SpawnThreat(ThreatSpec),
    /// Change the wind vector (NED m/s).
    WindChange([f64; 3]),
    /// Add a new no-fly zone.
    AddNfz { center: [f64; 3], radius: f64 },
}

/// An event bound to a specific simulation time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduledEvent {
    pub time_secs: f64,
    pub event: Event,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn threat_spec_approaching() {
        let s = ThreatSpec::approaching(500.0, 10.0);
        assert_eq!(s.initial_distance, 500.0);
        assert!(matches!(s.behavior, ThreatBehavior::Approaching { speed } if speed == 10.0));
    }

    #[test]
    fn threat_spec_flanking() {
        let s = ThreatSpec::flanking(400.0, 8.0, 45.0);
        assert!(
            matches!(s.behavior, ThreatBehavior::Flanking { speed, angle_deg } if speed == 8.0 && angle_deg == 45.0)
        );
    }

    #[test]
    fn threat_spec_retreating() {
        let s = ThreatSpec::retreating(300.0, 5.0);
        assert!(matches!(s.behavior, ThreatBehavior::Retreating { speed } if speed == 5.0));
    }

    #[test]
    fn threat_spec_stationary() {
        let s = ThreatSpec::stationary(200.0);
        assert!(matches!(s.behavior, ThreatBehavior::Stationary));
    }

    #[test]
    fn threat_spec_circling() {
        let s = ThreatSpec::circling(600.0, 15.0);
        assert!(
            matches!(s.behavior, ThreatBehavior::Circling { speed, radius } if speed == 15.0 && radius == 600.0)
        );
    }

    #[test]
    fn threat_spec_builder_chain() {
        let s = ThreatSpec::approaching(500.0, 10.0)
            .bearing(90.0)
            .altitude(-100.0)
            .lethal_radius(300.0);
        assert_eq!(s.initial_bearing_deg, 90.0);
        assert_eq!(s.altitude, -100.0);
        assert_eq!(s.lethal_radius, 300.0);
    }

    #[test]
    fn scheduled_event_serde_roundtrip() {
        let se = ScheduledEvent {
            time_secs: 30.0,
            event: Event::JamGps {
                noise_multiplier: 5.0,
            },
        };
        let json = serde_json::to_string(&se).expect("serialize");
        let back: ScheduledEvent = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.time_secs, 30.0);
    }

    #[test]
    fn event_all_variants_serde() {
        let events = vec![
            Event::JamGps {
                noise_multiplier: 3.0,
            },
            Event::RestoreGps,
            Event::LoseDrone { drone_id: 5 },
            Event::SpawnThreat(ThreatSpec::stationary(100.0)),
            Event::WindChange([1.0, 2.0, 0.0]),
            Event::AddNfz {
                center: [100.0, 200.0, 0.0],
                radius: 50.0,
            },
        ];
        for e in &events {
            let json = serde_json::to_string(e).expect("serialize");
            let back: Event = serde_json::from_str(&json).expect("deserialize");
            let json2 = serde_json::to_string(&back).expect("re-serialize");
            assert_eq!(json, json2, "roundtrip failed for {e:?}");
        }
    }
}
