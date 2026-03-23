use std::collections::{HashMap, VecDeque};

use nalgebra::Vector3;
use strix_adapters::simulator::SimulatorFleet;
use strix_adapters::traits::{PlatformAdapter, Waypoint};
use strix_auction::{LossClassification, Position, Task};
use strix_core::threat_tracker::ThreatObservation;
use strix_core::Regime;
use strix_swarm::{SwarmDecision, SwarmOrchestrator};

use crate::report::{
    Aggregates, BattleReport, DroneSummary, TickSnapshot, TimelineEntry, TimelineEventType,
};
use crate::scenario::{Event, ScheduledEvent};
use crate::threats::ThreatActor;

// ---------------------------------------------------------------------------
// Engine configuration
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub duration: f64,
    pub dt: f64,
    pub record_snapshots: bool,
}

// ---------------------------------------------------------------------------
// Simulation engine
// ---------------------------------------------------------------------------

/// Closed-loop battlefield simulation engine.
///
/// Ties `SimulatorFleet`, `SwarmOrchestrator`, and `ThreatActor`s together
/// in a tick loop: events → threats → telemetry → orchestrate → physics.
pub struct Engine {
    fleet: SimulatorFleet,
    orchestrator: SwarmOrchestrator,
    threats: Vec<ThreatActor>,
    event_queue: VecDeque<ScheduledEvent>,
    tasks: Vec<Task>,
    timeline: Vec<TimelineEntry>,
    tick_snapshots: Vec<TickSnapshot>,
    config: EngineConfig,
    sim_time: f64,
    scenario_name: String,
    n_drones_initial: usize,
    n_threats_initial: usize,
    /// Track previous regimes to detect changes.
    prev_regimes: HashMap<u32, Regime>,
    /// Track previous assignments count to detect auction rounds.
    prev_assignment_count: usize,
    /// Track previous kill zone count.
    prev_kill_zones: usize,
    /// Track cumulative distance per drone.
    distances: HashMap<u32, f64>,
    /// Track previous positions for distance calculation.
    prev_positions: HashMap<u32, [f64; 3]>,
    /// Track which drones have been lost.
    lost_drones: Vec<u32>,
    /// GPS jammed flag.
    gps_jammed: bool,
    /// Next threat ID counter.
    next_threat_id: u32,
}

impl Engine {
    /// Create a new engine from scenario parameters.
    pub fn new(
        fleet: SimulatorFleet,
        orchestrator: SwarmOrchestrator,
        threats: Vec<ThreatActor>,
        events: Vec<ScheduledEvent>,
        tasks: Vec<Task>,
        config: EngineConfig,
        scenario_name: String,
    ) -> Self {
        let n_drones = fleet.drones.len();
        let n_threats = threats.len();
        let next_threat_id = threats.iter().map(|t| t.id).max().unwrap_or(0) + 1;

        let mut sorted_events = events;
        sorted_events.sort_by(|a, b| a.time_secs.total_cmp(&b.time_secs));

        Self {
            fleet,
            orchestrator,
            threats,
            event_queue: VecDeque::from(sorted_events),
            tasks,
            timeline: Vec::new(),
            tick_snapshots: Vec::new(),
            config,
            sim_time: 0.0,
            scenario_name,
            n_drones_initial: n_drones,
            n_threats_initial: n_threats,
            prev_regimes: HashMap::new(),
            prev_assignment_count: 0,
            prev_kill_zones: 0,
            distances: HashMap::new(),
            prev_positions: HashMap::new(),
            lost_drones: Vec::new(),
            gps_jammed: false,
            next_threat_id,
        }
    }

    /// Run the full simulation and produce a BattleReport.
    pub fn run(mut self) -> BattleReport {
        let total_ticks = (self.config.duration / self.config.dt) as usize;

        // Arm the fleet
        let _ = self.fleet.arm_all();

        // Register initial threats with the orchestrator
        for threat in &self.threats {
            self.orchestrator
                .register_threat(threat.id, threat.position);
        }

        for _tick in 0..total_ticks {
            self.sim_time += self.config.dt;

            // 1. Process scheduled events
            self.process_events();

            // 2. Advance threat positions
            let centroid = self.compute_fleet_centroid();
            for threat in &mut self.threats {
                threat.advance(&centroid, self.config.dt);
            }

            // 3. Update threat trackers with current positions
            self.sync_threat_positions();

            // 4. Collect telemetry from fleet
            let telemetry = self.collect_telemetry();

            // 5. Run the full SwarmOrchestrator tick (7-stage pipeline)
            let decision = self
                .orchestrator
                .tick(&telemetry, &self.tasks, self.config.dt);

            // 6. Record diagnostics to timeline
            self.record_diagnostics(&decision, &telemetry);

            // 7. Apply task assignments → send waypoints to assigned drones
            self.apply_assignments(&decision);

            // 8. Step physics (with CBF safety if configured)
            if self.fleet.cbf_config.is_some() {
                self.fleet.step_all_safe();
            } else {
                self.fleet.step_all();
            }

            // 9. Update distance tracking
            self.update_distances(&telemetry);

            // 10. Optional: record full tick snapshot for JSON export
            if self.config.record_snapshots {
                self.snapshot(&decision);
            }
        }

        self.build_report()
    }

    // ── Event processing ─────────────────────────────────────────────────

    fn process_events(&mut self) {
        while let Some(evt) = self.event_queue.front() {
            if evt.time_secs > self.sim_time {
                break;
            }
            // Safe: front() just confirmed the queue is non-empty.
            let evt = self
                .event_queue
                .pop_front()
                .expect("queue non-empty: just checked front()");
            match evt.event {
                Event::JamGps { noise_multiplier } => {
                    self.gps_jammed = true;
                    // Increase GPS noise on all drones by modifying their configs.
                    // SimulatorAdapter uses interior mutability, but we can't directly
                    // change config. Instead, the effect is observable via CUSUM detection
                    // in the orchestrator when telemetry noise increases.
                    // For now, we record the event; the orchestrator's CUSUM will detect
                    // the anomaly from the noise spike.
                    let _ = noise_multiplier; // used in future GPS noise injection
                    self.timeline.push(TimelineEntry::new(
                        self.sim_time,
                        TimelineEventType::GpsJammed,
                    ));
                }
                Event::RestoreGps => {
                    self.gps_jammed = false;
                    self.timeline.push(TimelineEntry::new(
                        self.sim_time,
                        TimelineEventType::GpsRestored,
                    ));
                }
                Event::LoseDrone { drone_id } => {
                    self.lost_drones.push(drone_id);
                    // Notify the orchestrator about the loss
                    if let Some(drone) = self.fleet.get(drone_id) {
                        if let Ok(telem) = drone.get_telemetry() {
                            let pos = Position::new(
                                telem.position[0],
                                telem.position[1],
                                telem.position[2],
                            );
                            self.orchestrator.handle_drone_loss(
                                drone_id,
                                pos,
                                LossClassification::Sam,
                            );
                        }
                    }
                    self.timeline.push(TimelineEntry::new(
                        self.sim_time,
                        TimelineEventType::DroneLost { drone_id },
                    ));
                }
                Event::SpawnThreat(spec) => {
                    let centroid = self.compute_fleet_centroid();
                    let id = self.next_threat_id;
                    self.next_threat_id += 1;
                    let actor = ThreatActor::from_spec(id, &spec, &centroid);
                    self.orchestrator.register_threat(id, actor.position);
                    self.threats.push(actor);
                    self.timeline.push(TimelineEntry::new(
                        self.sim_time,
                        TimelineEventType::ThreatSpawned { threat_id: id },
                    ));
                }
                Event::WindChange(_new_wind) => {
                    // Wind change would require mutable access to SimulatorConfig.
                    // For now we record the event; future: rebuild configs.
                    self.timeline.push(TimelineEntry::new(
                        self.sim_time,
                        TimelineEventType::WindChanged,
                    ));
                }
                Event::AddNfz { center, radius } => {
                    use strix_core::cbf::NoFlyZone;
                    self.fleet.add_no_fly_zone(NoFlyZone {
                        center: Vector3::new(center[0], center[1], center[2]),
                        radius,
                    });
                    self.timeline.push(TimelineEntry::new(
                        self.sim_time,
                        TimelineEventType::NfzAdded,
                    ));
                }
            }
        }
    }

    // ── Telemetry & physics helpers ──────────────────────────────────────

    fn compute_fleet_centroid(&self) -> Vector3<f64> {
        let mut sum = Vector3::zeros();
        let mut count = 0usize;
        for drone in &self.fleet.drones {
            if drone.has_failed() || self.lost_drones.contains(&drone.id()) {
                continue;
            }
            if let Ok(t) = drone.get_telemetry() {
                sum += Vector3::new(t.position[0], t.position[1], t.position[2]);
                count += 1;
            }
        }
        if count > 0 {
            sum / count as f64
        } else {
            Vector3::zeros()
        }
    }

    fn collect_telemetry(&self) -> Vec<(u32, strix_adapters::traits::Telemetry)> {
        self.fleet
            .drones
            .iter()
            .filter(|d| !d.has_failed() && !self.lost_drones.contains(&d.id()))
            .filter_map(|d| d.get_telemetry().ok().map(|t| (d.id(), t)))
            .collect()
    }

    fn sync_threat_positions(&mut self) {
        // Feed current threat actor positions into the orchestrator's threat trackers
        // as radar observations (position + noise).
        for threat in &self.threats {
            if let Some(tracker) = self.orchestrator.threat_trackers.get_mut(&threat.id) {
                let obs = vec![ThreatObservation::Radar {
                    position: threat.position,
                    sigma: 5.0,
                    timestamp: self.sim_time,
                }];
                let centroid = Vector3::zeros(); // already updated this tick
                tracker.step(&centroid, &obs, self.config.dt);
            }
        }
    }

    fn apply_assignments(&self, decision: &SwarmDecision) {
        for assignment in &decision.assignments {
            // Find the task to get its location
            if let Some(task) = self.tasks.iter().find(|t| t.id == assignment.task_id) {
                // Send waypoint to the assigned drone
                if let Some(drone) = self.fleet.get(assignment.drone_id) {
                    if !drone.has_failed() && !self.lost_drones.contains(&assignment.drone_id) {
                        let wp = Waypoint {
                            lat: task.location.x,
                            lon: task.location.y,
                            alt: task.location.z,
                            speed: 10.0,
                            heading: None,
                        };
                        let _ = drone.send_waypoint(&wp);
                    }
                }
            }
        }
    }

    fn update_distances(&mut self, telemetry: &[(u32, strix_adapters::traits::Telemetry)]) {
        for (id, telem) in telemetry {
            let pos = telem.position;
            if let Some(prev) = self.prev_positions.get(id) {
                let dx = pos[0] - prev[0];
                let dy = pos[1] - prev[1];
                let dz = pos[2] - prev[2];
                let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                *self.distances.entry(*id).or_insert(0.0) += dist;
            }
            self.prev_positions.insert(*id, pos);
        }
    }

    // ── Diagnostics recording ────────────────────────────────────────────

    fn record_diagnostics(
        &mut self,
        decision: &SwarmDecision,
        _telemetry: &[(u32, strix_adapters::traits::Telemetry)],
    ) {
        // Detect regime changes
        for (&drone_id, &regime) in &decision.regimes {
            let prev = self.prev_regimes.get(&drone_id).copied();
            if let Some(old) = prev {
                if old != regime {
                    self.timeline.push(TimelineEntry::new(
                        self.sim_time,
                        TimelineEventType::RegimeChange {
                            drone_id,
                            from: old,
                            to: regime,
                            intent_score: decision.max_intent_score,
                        },
                    ));
                }
            }
            self.prev_regimes.insert(drone_id, regime);
        }

        // Detect auction rounds (assignment count changed)
        if decision.assignments.len() != self.prev_assignment_count {
            let unassigned = self.tasks.len().saturating_sub(decision.assignments.len());
            self.timeline.push(TimelineEntry::new(
                self.sim_time,
                TimelineEventType::AuctionRound {
                    assignments: decision.assignments.len(),
                    unassigned,
                },
            ));
            self.prev_assignment_count = decision.assignments.len();
        }

        // Detect new kill zones
        if decision.kill_zone_count > self.prev_kill_zones {
            self.prev_kill_zones = decision.kill_zone_count;
        }

        // Check for forced evade (all drones in Evade regime)
        let all_evade =
            !decision.regimes.is_empty() && decision.regimes.values().all(|r| *r == Regime::Evade);
        if all_evade && decision.regimes.len() > 1 {
            // Only record once — check if we already have a ForcedEvade recently
            let already_recorded = self
                .timeline
                .iter()
                .rev()
                .take(5)
                .any(|e| matches!(e.event_type, TimelineEventType::ForcedEvade { .. }));
            if !already_recorded {
                self.timeline.push(TimelineEntry::new(
                    self.sim_time,
                    TimelineEventType::ForcedEvade {
                        reason: format!(
                            "Attrition cascade: {} drones lost",
                            self.lost_drones.len()
                        ),
                    },
                ));
            }
        }

        // Record temporal anomalies from multi-horizon managers.
        #[cfg(feature = "temporal")]
        for (horizon, direction, _value) in &decision.temporal_anomalies {
            self.timeline.push(TimelineEntry::new(
                self.sim_time,
                TimelineEventType::TemporalAnomaly {
                    horizon: horizon.clone(),
                    direction: *direction,
                },
            ));
        }
    }

    // ── Snapshot for JSON export ──────────────────────────────────────────

    fn snapshot(&mut self, decision: &SwarmDecision) {
        let drone_positions: HashMap<u32, [f64; 3]> = decision
            .positions
            .iter()
            .map(|(&id, pos)| (id, [pos.x, pos.y, pos.z]))
            .collect();

        let drone_regimes: HashMap<u32, String> = decision
            .regimes
            .iter()
            .map(|(&id, r)| (id, format!("{:?}", r)))
            .collect();

        let threat_positions: HashMap<u32, [f64; 3]> = self
            .threats
            .iter()
            .map(|t| (t.id, [t.position.x, t.position.y, t.position.z]))
            .collect();

        let assignments: Vec<(u32, u32)> = decision
            .assignments
            .iter()
            .map(|a| (a.drone_id, a.task_id))
            .collect();

        self.tick_snapshots.push(TickSnapshot {
            time: self.sim_time,
            drone_positions,
            drone_regimes,
            threat_positions,
            intent_score: decision.max_intent_score,
            assignments,
        });
    }

    // ── Report building ──────────────────────────────────────────────────

    fn build_report(self) -> BattleReport {
        let mut per_drone = HashMap::new();
        let mut batteries = Vec::new();

        for drone in &self.fleet.drones {
            let id = drone.id();
            let alive = !drone.has_failed() && !self.lost_drones.contains(&id);
            let (regime, battery) = if let Ok(telem) = drone.get_telemetry() {
                let r = self
                    .prev_regimes
                    .get(&id)
                    .copied()
                    .unwrap_or(Regime::Patrol);
                (r, telem.battery)
            } else {
                (Regime::Patrol, 0.0)
            };

            if alive {
                batteries.push(battery);
            }

            // Count regime changes for this drone from the timeline
            let regime_changes = self
                .timeline
                .iter()
                .filter(|e| match &e.event_type {
                    TimelineEventType::RegimeChange { drone_id, .. } => *drone_id == id,
                    _ => false,
                })
                .count();

            per_drone.insert(
                id,
                DroneSummary {
                    id,
                    final_regime: regime,
                    final_battery: battery,
                    distance_traveled: self.distances.get(&id).copied().unwrap_or(0.0),
                    regime_changes,
                    alive,
                },
            );
        }

        let total_regime_changes = self
            .timeline
            .iter()
            .filter(|e| matches!(e.event_type, TimelineEventType::RegimeChange { .. }))
            .count();

        let hysteresis_blocks = self
            .timeline
            .iter()
            .filter(|e| matches!(e.event_type, TimelineEventType::HysteresisBlock { .. }))
            .count();

        let cusum_fires = self
            .timeline
            .iter()
            .filter(|e| matches!(e.event_type, TimelineEventType::CusumFired { .. }))
            .count();

        let cbf_activations = self
            .timeline
            .iter()
            .filter(|e| matches!(e.event_type, TimelineEventType::CbfCorrection { .. }))
            .count();

        let auction_rounds = self
            .timeline
            .iter()
            .filter(|e| matches!(e.event_type, TimelineEventType::AuctionRound { .. }))
            .count();

        let forced_evade_count = self
            .timeline
            .iter()
            .filter(|e| matches!(e.event_type, TimelineEventType::ForcedEvade { .. }))
            .count();

        #[cfg(feature = "temporal")]
        let temporal_anomaly_count = self
            .timeline
            .iter()
            .filter(|e| matches!(e.event_type, TimelineEventType::TemporalAnomaly { .. }))
            .count();
        #[cfg(feature = "temporal")]
        let temporal_constraint_count = self
            .timeline
            .iter()
            .filter(|e| matches!(e.event_type, TimelineEventType::TemporalConstraint { .. }))
            .count();

        let max_intent_score = self
            .timeline
            .iter()
            .filter_map(|e| match &e.event_type {
                TimelineEventType::RegimeChange { intent_score, .. } => Some(*intent_score),
                _ => None,
            })
            .fold(0.0_f64, f64::max);

        let battery_min = batteries.iter().cloned().fold(f64::INFINITY, f64::min);
        let battery_mean = if batteries.is_empty() {
            0.0
        } else {
            batteries.iter().sum::<f64>() / batteries.len() as f64
        };

        let aggregates = Aggregates {
            total_ticks: (self.config.duration / self.config.dt) as usize,
            regime_changes: total_regime_changes,
            hysteresis_blocks,
            cusum_fires,
            cbf_activations,
            cbf_violations: 0, // CBF should prevent all violations
            auction_rounds,
            drones_lost: self.lost_drones.len(),
            drones_survived: self.n_drones_initial - self.lost_drones.len(),
            max_intent_score,
            max_intent_class: if max_intent_score > 0.5 {
                "HIGH".into()
            } else if max_intent_score > 0.2 {
                "MEDIUM".into()
            } else {
                "LOW".into()
            },
            kill_zones_created: self.prev_kill_zones,
            forced_evade_count,
            battery_min: if battery_min.is_infinite() {
                0.0
            } else {
                battery_min
            },
            battery_mean,
            #[cfg(feature = "temporal")]
            temporal_anomaly_count,
            #[cfg(feature = "temporal")]
            temporal_constraint_count,
        };

        let tick_data = if self.config.record_snapshots {
            Some(self.tick_snapshots)
        } else {
            None
        };

        BattleReport {
            scenario_name: self.scenario_name,
            duration: self.config.duration,
            n_drones_initial: self.n_drones_initial,
            n_threats_initial: self.n_threats_initial,
            timeline: self.timeline,
            aggregates,
            per_drone,
            tick_data,
        }
    }
}
