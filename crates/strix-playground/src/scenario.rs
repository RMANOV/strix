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
