use std::collections::HashMap;
use std::fmt;
use std::io;
use std::path::Path;

use serde::{Deserialize, Serialize};
use strix_core::Regime;

// ---------------------------------------------------------------------------
// Timeline events
// ---------------------------------------------------------------------------

/// Classification of a timeline entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TimelineEventType {
    RegimeChange {
        drone_id: u32,
        from: Regime,
        to: Regime,
        intent_score: f64,
    },
    HysteresisBlock {
        drone_id: u32,
        proposed: Regime,
        stayed: Regime,
    },
    DroneLost {
        drone_id: u32,
    },
    GpsJammed,
    GpsRestored,
    CusumFired {
        drone_id: u32,
        direction: i32,
    },
    CbfCorrection {
        drone_id: u32,
        magnitude: f64,
    },
    AuctionRound {
        assignments: usize,
        unassigned: usize,
    },
    ThreatSpawned {
        threat_id: u32,
    },
    ForcedEvade {
        reason: String,
    },
    WindChanged,
    NfzAdded,
    #[cfg(feature = "temporal")]
    TemporalAnomaly {
        horizon: String,
        direction: i32,
    },
    #[cfg(feature = "temporal")]
    TemporalConstraint {
        count: usize,
        suggested_regime: String,
    },
}

/// A single timestamped diagnostic event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineEntry {
    pub time: f64,
    pub event_type: TimelineEventType,
}

impl TimelineEntry {
    pub fn new(time: f64, event_type: TimelineEventType) -> Self {
        Self { time, event_type }
    }
}

impl fmt::Display for TimelineEntry {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[{:7.2}s] ", self.time)?;
        match &self.event_type {
            TimelineEventType::RegimeChange {
                drone_id,
                from,
                to,
                intent_score,
            } => write!(
                f,
                "Drone {} regime: {:?} -> {:?} (intent={:.3})",
                drone_id, from, to, intent_score
            ),
            TimelineEventType::HysteresisBlock {
                drone_id,
                proposed,
                stayed,
            } => write!(
                f,
                "Drone {} hysteresis BLOCKED {:?} -> stayed {:?}",
                drone_id, proposed, stayed
            ),
            TimelineEventType::DroneLost { drone_id } => {
                write!(f, "DRONE {} LOST", drone_id)
            }
            TimelineEventType::GpsJammed => write!(f, "GPS JAMMED"),
            TimelineEventType::GpsRestored => write!(f, "GPS restored"),
            TimelineEventType::CusumFired {
                drone_id,
                direction,
            } => write!(f, "CUSUM fired drone {} dir={}", drone_id, direction),
            TimelineEventType::CbfCorrection {
                drone_id,
                magnitude,
            } => write!(f, "CBF correction drone {} mag={:.3}", drone_id, magnitude),
            TimelineEventType::AuctionRound {
                assignments,
                unassigned,
            } => write!(
                f,
                "Auction: {} assigned, {} unassigned",
                assignments, unassigned
            ),
            TimelineEventType::ThreatSpawned { threat_id } => {
                write!(f, "Threat {} spawned", threat_id)
            }
            TimelineEventType::ForcedEvade { reason } => {
                write!(f, "FORCED EVADE: {}", reason)
            }
            TimelineEventType::WindChanged => write!(f, "Wind changed"),
            TimelineEventType::NfzAdded => write!(f, "NFZ added"),
            #[cfg(feature = "temporal")]
            TimelineEventType::TemporalAnomaly { horizon, direction } => {
                write!(f, "TEMPORAL ANOMALY: {} dir={}", horizon, direction)
            }
            #[cfg(feature = "temporal")]
            TimelineEventType::TemporalConstraint {
                count,
                suggested_regime,
            } => {
                write!(
                    f,
                    "Temporal: {} constraints, suggest {}",
                    count, suggested_regime
                )
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Aggregates
// ---------------------------------------------------------------------------

/// Summary statistics over the entire simulation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Aggregates {
    pub total_ticks: usize,
    pub regime_changes: usize,
    pub hysteresis_blocks: usize,
    pub cusum_fires: usize,
    pub cbf_activations: usize,
    pub cbf_violations: usize,
    pub cbf_activation_ticks: usize,
    pub cbf_constraints_total: usize,
    pub cbf_constraints_peak: usize,
    pub cbf_burden_mean: f64,
    pub auction_rounds: usize,
    pub coordination_churn_total: usize,
    pub coordination_churn_peak: usize,
    pub coordination_burden_mean: f64,
    pub drones_lost: usize,
    pub drones_survived: usize,
    pub max_intent_score: f64,
    pub max_intent_class: String,
    pub kill_zones_created: usize,
    pub forced_evade_count: usize,
    pub battery_min: f64,
    pub battery_mean: f64,
    #[cfg(feature = "temporal")]
    pub temporal_anomaly_count: usize,
    #[cfg(feature = "temporal")]
    pub temporal_constraint_count: usize,
}

// ---------------------------------------------------------------------------
// Per-drone summary
// ---------------------------------------------------------------------------

/// End-of-simulation summary for a single drone.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DroneSummary {
    pub id: u32,
    pub final_regime: Regime,
    pub final_battery: f64,
    pub distance_traveled: f64,
    pub regime_changes: usize,
    pub alive: bool,
}

// ---------------------------------------------------------------------------
// Tick snapshot (optional JSON telemetry)
// ---------------------------------------------------------------------------

/// Full simulation state at one instant — used for JSON export / replay.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TickSnapshot {
    pub time: f64,
    pub drone_positions: HashMap<u32, [f64; 3]>,
    pub drone_regimes: HashMap<u32, String>,
    pub threat_positions: HashMap<u32, [f64; 3]>,
    pub intent_score: f64,
    pub assignments: Vec<(u32, u32)>,
}

// ---------------------------------------------------------------------------
// BattleReport
// ---------------------------------------------------------------------------

/// Complete simulation output: timeline, aggregates, per-drone, optional ticks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BattleReport {
    pub scenario_name: String,
    pub duration: f64,
    pub n_drones_initial: usize,
    pub n_threats_initial: usize,
    pub timeline: Vec<TimelineEntry>,
    pub aggregates: Aggregates,
    pub per_drone: HashMap<u32, DroneSummary>,
    pub tick_data: Option<Vec<TickSnapshot>>,
}

impl fmt::Display for BattleReport {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f)?;
        writeln!(
            f,
            "=== SCENARIO: {} ({:.1}s, {} drones, {} threats) ===",
            self.scenario_name, self.duration, self.n_drones_initial, self.n_threats_initial
        )?;
        writeln!(f)?;

        // Timeline (show up to 50 most interesting entries)
        writeln!(f, "--- TIMELINE ({} events) ---", self.timeline.len())?;
        for entry in self.timeline.iter().take(50) {
            writeln!(f, "  {}", entry)?;
        }
        if self.timeline.len() > 50 {
            writeln!(f, "  ... and {} more events", self.timeline.len() - 50)?;
        }
        writeln!(f)?;

        // Aggregates
        let a = &self.aggregates;
        writeln!(f, "--- AGGREGATES ---")?;
        writeln!(f, "  Total ticks:       {}", a.total_ticks)?;
        writeln!(f, "  Regime changes:    {}", a.regime_changes)?;
        writeln!(f, "  Hysteresis blocks: {}", a.hysteresis_blocks)?;
        writeln!(f, "  CUSUM fires:       {}", a.cusum_fires)?;
        writeln!(f, "  CBF activations:   {}", a.cbf_activations)?;
        writeln!(f, "  CBF violations:    {}", a.cbf_violations)?;
        writeln!(f, "  CBF active ticks:  {}", a.cbf_activation_ticks)?;
        writeln!(
            f,
            "  CBF constraints:   total={} peak={} burden={:.3}",
            a.cbf_constraints_total, a.cbf_constraints_peak, a.cbf_burden_mean
        )?;
        writeln!(f, "  Auction rounds:    {}", a.auction_rounds)?;
        writeln!(
            f,
            "  Coordination load: total={} peak={} burden={:.3}",
            a.coordination_churn_total, a.coordination_churn_peak, a.coordination_burden_mean
        )?;
        writeln!(f, "  Drones lost:       {}", a.drones_lost)?;
        writeln!(f, "  Drones survived:   {}", a.drones_survived)?;
        writeln!(
            f,
            "  Max intent score:  {:.4} ({})",
            a.max_intent_score, a.max_intent_class
        )?;
        writeln!(f, "  Kill zones:        {}", a.kill_zones_created)?;
        writeln!(f, "  Forced evades:     {}", a.forced_evade_count)?;
        writeln!(
            f,
            "  Battery min/mean:  {:.3} / {:.3}",
            a.battery_min, a.battery_mean
        )?;
        #[cfg(feature = "temporal")]
        {
            writeln!(f, "  Temporal anomalies: {}", a.temporal_anomaly_count)?;
            writeln!(f, "  Temporal constraints: {}", a.temporal_constraint_count)?;
        }
        writeln!(f)?;

        // Per-drone
        writeln!(f, "--- PER-DRONE ({} total) ---", self.per_drone.len())?;
        let mut drones: Vec<_> = self.per_drone.values().collect();
        drones.sort_by_key(|d| d.id);
        for d in drones {
            let status = if d.alive { "ALIVE" } else { "LOST " };
            writeln!(
                f,
                "  [{}] Drone {:>2}: {:?} | bat={:.2} | dist={:.0}m | regime_chg={}",
                status,
                d.id,
                d.final_regime,
                d.final_battery,
                d.distance_traveled,
                d.regime_changes
            )?;
        }
        writeln!(f)?;

        if let Some(data) = &self.tick_data {
            writeln!(f, "  [JSON tick data: {} snapshots available]", data.len())?;
        }

        writeln!(f, "=== END REPORT ===")
    }
}

impl BattleReport {
    /// Serialize the full report to a JSON value.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or_default()
    }

    /// Write the JSON report to a file.
    pub fn save_json(&self, path: &Path) -> io::Result<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        std::fs::write(path, json)
    }
}
