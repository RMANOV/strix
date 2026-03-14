//! The main tick() loop — one orchestration cycle: sense → think → act.
//!
//! Each tick chains all 5 STRIX crates in sequence:
//! 1. strix-core: particle filter prediction + measurement update
//! 2. strix-core: CUSUM anomaly detection + regime transitions
//! 3. strix-core: threat tracker update
//! 4. strix-auction: combinatorial task auction
//! 5. strix-mesh: gossip state propagation + pheromone update
//! 6. strix-xai: decision trace recording

use std::collections::HashMap;

use nalgebra::Vector3;

use strix_adapters::traits::Telemetry;
use strix_auction::{Assignment, Auctioneer, Capabilities, LossAnalyzer, Task};
use strix_core::anomaly::CusumConfig;
use strix_core::particle_nav::ParticleNavFilter;
use strix_core::regime::{detect_regime, DetectionConfig, RegimeSignals};
use strix_core::threat_tracker::ThreatTracker;
use strix_core::Regime;
use strix_mesh::gossip::GossipEngine;
use strix_mesh::stigmergy::{Pheromone, PheromoneField, PheromoneType};
use strix_mesh::{NodeId, Position3D};
use strix_xai::trace::{DecisionTrace, DecisionType, TraceInputs, TraceRecorder};

use crate::convert;

/// Configuration for the swarm orchestrator.
#[derive(Debug, Clone)]
pub struct SwarmConfig {
    /// Number of particles per drone's navigation filter.
    pub n_particles: usize,
    /// Number of particles per threat tracker.
    pub n_threat_particles: usize,
    /// How often to re-run the auction (in ticks).
    pub auction_interval: u32,
    /// CUSUM anomaly detection config.
    pub cusum_config: CusumConfig,
    /// Regime detection config.
    pub detection_config: DetectionConfig,
    /// Pheromone field resolution (meters per cell).
    pub pheromone_resolution: f64,
    /// Pheromone decay rate (per second).
    pub pheromone_decay_rate: f64,
    /// Gossip fanout.
    pub gossip_fanout: usize,
    /// Default drone capabilities for auction bidding.
    pub default_capabilities: Capabilities,
}

impl Default for SwarmConfig {
    fn default() -> Self {
        Self {
            n_particles: 200,
            n_threat_particles: 100,
            auction_interval: 5,
            cusum_config: CusumConfig::default(),
            detection_config: DetectionConfig::default(),
            pheromone_resolution: 10.0,
            pheromone_decay_rate: 0.05,
            gossip_fanout: 3,
            default_capabilities: Capabilities {
                has_sensor: true,
                has_weapon: false,
                has_ew: false,
                has_relay: false,
            },
        }
    }
}

/// Result of a single orchestration tick.
#[derive(Debug, Clone)]
pub struct SwarmDecision {
    /// Current drone-to-task assignments.
    pub assignments: Vec<Assignment>,
    /// Per-drone regime (drone_id → Regime).
    pub regimes: HashMap<u32, Regime>,
    /// Per-drone estimated position.
    pub positions: HashMap<u32, Vector3<f64>>,
    /// Per-threat estimated position.
    pub threat_positions: HashMap<u32, Vector3<f64>>,
    /// Kill zones from antifragile analysis.
    pub kill_zone_count: usize,
    /// Antifragile score.
    pub antifragile_score: f64,
    /// Number of decision traces recorded this tick.
    pub traces_recorded: u32,
    /// Gossip convergence estimate.
    pub gossip_convergence: f64,
    /// Active pheromone cells.
    pub pheromone_cells: usize,
}

/// The swarm orchestrator — chains all 5 STRIX crates together.
pub struct SwarmOrchestrator {
    /// Per-drone particle filters (GPS-denied navigation).
    pub nav_filters: HashMap<u32, ParticleNavFilter>,
    /// Enemy trackers.
    pub threat_trackers: HashMap<u32, ThreatTracker>,
    /// Task auction engine.
    pub auctioneer: Auctioneer,
    /// Anti-fragile loss analyzer.
    pub loss_analyzer: LossAnalyzer,
    /// Gossip network for state sync.
    pub gossip: GossipEngine,
    /// Digital pheromone field.
    pub pheromones: PheromoneField,
    /// Decision trace recorder.
    pub tracer: TraceRecorder,
    /// Current assignments.
    pub assignments: Vec<Assignment>,
    /// Per-drone regime tracking.
    pub regimes: HashMap<u32, Regime>,
    /// Per-drone previous positions (for visual odometry delta).
    prev_positions: HashMap<u32, Vector3<f64>>,
    /// Per-drone signal histories for CUSUM (drone_id → signal buffer).
    signal_histories: HashMap<u32, Vec<f64>>,
    /// Per-drone threat distance histories for Hurst exponent (drone_id → distance buffer).
    threat_distance_histories: HashMap<u32, Vec<f64>>,
    /// Gossip version counter.
    gossip_version: u64,
    /// Tick counter.
    tick_count: u32,
    /// Simulation clock.
    sim_time: f64,
    /// Config.
    pub config: SwarmConfig,
}

impl SwarmOrchestrator {
    /// Create a new orchestrator for the given drone IDs.
    pub fn new(drone_ids: &[u32], config: SwarmConfig) -> Self {
        let mut nav_filters = HashMap::new();
        let mut regimes = HashMap::new();
        let mut signal_histories = HashMap::new();
        let mut threat_distance_histories = HashMap::new();

        // Initialize gossip network
        let self_id = drone_ids.first().copied().unwrap_or(0);
        let mut gossip = GossipEngine::new(NodeId(self_id), config.gossip_fanout);
        for &id in drone_ids {
            gossip.add_peer(NodeId(id));
        }

        for &id in drone_ids {
            nav_filters.insert(
                id,
                ParticleNavFilter::new(config.n_particles, Vector3::zeros()),
            );
            regimes.insert(id, Regime::Patrol);
            signal_histories.insert(id, Vec::new());
            threat_distance_histories.insert(id, Vec::new());
        }

        Self {
            nav_filters,
            threat_trackers: HashMap::new(),
            auctioneer: Auctioneer::new(),
            loss_analyzer: LossAnalyzer::new(),
            gossip,
            pheromones: PheromoneField::new(
                config.pheromone_resolution,
                config.pheromone_decay_rate,
            ),
            tracer: TraceRecorder::new(),
            assignments: Vec::new(),
            regimes,
            prev_positions: HashMap::new(),
            signal_histories,
            threat_distance_histories,
            gossip_version: 0,
            tick_count: 0,
            sim_time: 0.0,
            config,
        }
    }

    /// Current simulation time.
    pub fn sim_time(&self) -> f64 {
        self.sim_time
    }

    /// Register a new drone mid-simulation.
    pub fn register_drone(&mut self, id: u32, position: Vector3<f64>) {
        self.nav_filters.insert(
            id,
            ParticleNavFilter::new(self.config.n_particles, position),
        );
        self.regimes.insert(id, Regime::Patrol);
        self.signal_histories.insert(id, Vec::new());
        self.threat_distance_histories.insert(id, Vec::new());
        self.gossip.add_peer(NodeId(id));
    }

    /// Register a new threat to track.
    pub fn register_threat(&mut self, threat_id: u32, position: Vector3<f64>) {
        self.threat_trackers.insert(
            threat_id,
            ThreatTracker::new(threat_id, self.config.n_threat_particles, position),
        );
    }

    /// Mark a drone as lost and trigger anti-fragile response.
    pub fn handle_drone_loss(
        &mut self,
        drone_id: u32,
        position: strix_auction::Position,
        classification: strix_auction::LossClassification,
    ) {
        // Find orphaned tasks
        let orphaned: Vec<u32> = self
            .assignments
            .iter()
            .filter(|a| a.drone_id == drone_id)
            .map(|a| a.task_id)
            .collect();

        let record = strix_auction::LossRecord {
            drone_id,
            position,
            altitude: position.z,
            heading: 0.0,
            velocity: [0.0; 3],
            threat_bearing: None,
            regime_at_loss: self
                .regimes
                .get(&drone_id)
                .copied()
                .unwrap_or(Regime::Patrol),
            classification,
            orphaned_tasks: orphaned,
            timestamp: self.sim_time,
        };

        self.loss_analyzer.record_loss(record);

        // Remove drone from active tracking
        self.nav_filters.remove(&drone_id);
        self.regimes.remove(&drone_id);
        self.prev_positions.remove(&drone_id);
        self.signal_histories.remove(&drone_id);
        self.threat_distance_histories.remove(&drone_id);
        self.gossip.remove_peer(NodeId(drone_id));

        // Remove assignments for this drone
        self.assignments.retain(|a| a.drone_id != drone_id);

        // Trigger re-auction
        self.auctioneer.trigger_reauction();

        // Record trace
        let alive_ids: Vec<u32> = self.nav_filters.keys().copied().collect();
        let trace = DecisionTrace::new(self.sim_time, DecisionType::ReAuction)
            .with_inputs(TraceInputs {
                drone_ids: alive_ids,
                regime: format!("mixed"),
                metrics: serde_json::json!({
                    "lost_drone": drone_id,
                    "kill_zones": self.loss_analyzer.active_kill_zones(),
                    "antifragile_score": self.loss_analyzer.antifragile_score(),
                }),
                context: serde_json::json!({"lost_drone_id": drone_id}),
            })
            .with_output(
                &format!("Drone {} lost — re-auctioning tasks", drone_id),
                serde_json::json!({"kill_zones": self.loss_analyzer.active_kill_zones()}),
            )
            .with_confidence(0.9);
        self.tracer.record(trace);
    }

    /// One orchestration cycle: sense → think → act.
    pub fn tick(
        &mut self,
        telemetry: &[(u32, Telemetry)],
        tasks: &[Task],
        dt: f64,
    ) -> SwarmDecision {
        self.tick_count += 1;
        self.sim_time += dt;
        let mut traces_recorded = 0u32;

        // ── 1. Update particle filters from telemetry ─────────────────────
        let mut fleet_centroid = Vector3::zeros();
        let mut alive_count = 0usize;

        // Pre-compute threat bearings (avoids borrow conflict with nav_filters)
        let threat_bearings: HashMap<u32, Vector3<f64>> = telemetry
            .iter()
            .map(|(id, telem)| {
                let drone_pos =
                    Vector3::new(telem.position[0], telem.position[1], telem.position[2]);
                (*id, self.nearest_threat_bearing(&drone_pos))
            })
            .collect();

        for (id, telem) in telemetry {
            if let Some(filter) = self.nav_filters.get_mut(id) {
                let drone_pos =
                    Vector3::new(telem.position[0], telem.position[1], telem.position[2]);

                // ── Multi-sensor fusion: build observations from telemetry ──
                let mut obs = Vec::with_capacity(4);

                // 1. Barometer — constrains vertical position
                obs.push(strix_core::Observation::Barometer {
                    altitude: -telem.position[2], // NED: z down, altitude up
                    timestamp: telem.timestamp,
                });

                // 2. IMU (velocity as pseudo-acceleration) — constrains velocity
                obs.push(strix_core::Observation::Imu {
                    acceleration: Vector3::new(
                        telem.velocity[0],
                        telem.velocity[1],
                        telem.velocity[2],
                    ),
                    gyro: None,
                    timestamp: telem.timestamp,
                });

                // 3. Magnetometer — constrains heading from yaw
                let yaw = telem.attitude[2];
                obs.push(strix_core::Observation::Magnetometer {
                    heading: Vector3::new(yaw.cos(), yaw.sin(), 0.0),
                    timestamp: telem.timestamp,
                });

                // 4. Visual Odometry — position delta from previous tick
                if let Some(prev) = self.prev_positions.get(id) {
                    let delta = drone_pos - prev;
                    // Only use VO if the drone actually moved (avoids noise on stationary)
                    if delta.norm() > 0.01 {
                        obs.push(strix_core::Observation::VisualOdometry {
                            delta_position: delta,
                            confidence: 0.7,
                            timestamp: telem.timestamp,
                        });
                    }
                }
                // ── End multi-sensor fusion ─────────────────────────────────

                let bearing = threat_bearings
                    .get(id)
                    .copied()
                    .unwrap_or_else(Vector3::zeros);

                // Run particle filter step with fused observations
                let (_pos, _vel, _probs) = filter.step(&obs, &bearing, 1.0, dt);

                // Store position for next tick's VO delta
                self.prev_positions.insert(*id, drone_pos);

                fleet_centroid += drone_pos;
                alive_count += 1;

                // Update signal history for CUSUM
                let speed = (telem.velocity[0].powi(2)
                    + telem.velocity[1].powi(2)
                    + telem.velocity[2].powi(2))
                .sqrt();
                if let Some(history) = self.signal_histories.get_mut(id) {
                    history.push(speed);
                    if history.len() > 100 {
                        history.drain(..50);
                    }
                }
            }
        }

        if alive_count > 0 {
            fleet_centroid /= alive_count as f64;
        }

        // ── 2. Detect regimes (CUSUM + signals) ──────────────────────────
        let mut regime_changes = Vec::new();
        for (id, telem) in telemetry {
            let regime = self.regimes.get(id).copied().unwrap_or(Regime::Patrol);
            let drone_pos = Vector3::new(telem.position[0], telem.position[1], telem.position[2]);

            // Compute regime signals
            let (nearest_threat_dist, closing_rate) = self.nearest_threat_metrics(&drone_pos);

            // Update threat distance history for Hurst exponent computation
            if let Some(tdh) = self.threat_distance_histories.get_mut(id) {
                tdh.push(nearest_threat_dist);
                if tdh.len() > 100 {
                    tdh.drain(..50);
                }
            }

            let cusum_triggered = self
                .signal_histories
                .get(id)
                .map(|h| {
                    if h.len() >= self.config.cusum_config.min_samples {
                        let (triggered, _dir, _val) = strix_core::anomaly::cusum_test(
                            h,
                            self.config.cusum_config.threshold_h,
                            self.config.cusum_config.min_samples,
                        );
                        triggered
                    } else {
                        false
                    }
                })
                .unwrap_or(false);

            let signals = RegimeSignals {
                cusum_triggered,
                cusum_direction: 0,
                hurst: self
                    .threat_distance_histories
                    .get(id)
                    .filter(|h| h.len() >= 20)
                    .map(|h| strix_core::uncertainty::hurst_exponent(h, 10, 50).0)
                    .unwrap_or(0.5),
                volatility_ratio: self
                    .signal_histories
                    .get(id)
                    .filter(|h| h.len() >= 20)
                    .map(|h| strix_core::uncertainty::volatility_compression(h, 10, 50).0)
                    .unwrap_or(1.0),
                threat_distance: nearest_threat_dist,
                closing_rate,
            };

            let new_regime = detect_regime(&signals, regime, &self.config.detection_config);

            if new_regime != regime {
                regime_changes.push((*id, regime, new_regime));
                self.regimes.insert(*id, new_regime);
            }
        }

        // Record regime change traces
        for (drone_id, old_regime, new_regime) in &regime_changes {
            let trace = DecisionTrace::new(self.sim_time, DecisionType::RegimeChange)
                .with_inputs(TraceInputs {
                    drone_ids: vec![*drone_id],
                    regime: format!("{:?}", old_regime),
                    metrics: serde_json::json!({}),
                    context: serde_json::Value::Null,
                })
                .with_output(
                    &format!(
                        "Regime {:?} → {:?} for drone {}",
                        old_regime, new_regime, drone_id
                    ),
                    serde_json::json!({"new_regime": format!("{:?}", new_regime)}),
                )
                .with_confidence(0.85);
            self.tracer.record(trace);
            traces_recorded += 1;
        }

        // ── 3. Update threat trackers ────────────────────────────────────
        for tracker in self.threat_trackers.values_mut() {
            tracker.step(&fleet_centroid, &[], dt);
        }

        // ── 4. Run combinatorial auction ─────────────────────────────────
        let should_auction =
            self.tick_count % self.config.auction_interval == 0 || self.auctioneer.needs_reauction;

        if should_auction && !tasks.is_empty() {
            // Build auction drone states from telemetry
            let auction_drones: Vec<strix_auction::DroneState> = telemetry
                .iter()
                .map(|(id, telem)| {
                    let regime = self.regimes.get(id).copied().unwrap_or(Regime::Patrol);
                    convert::telemetry_to_auction_drone(
                        *id,
                        telem,
                        regime,
                        &self.config.default_capabilities,
                    )
                })
                .collect();

            // Build threat states for risk scoring
            let auction_threats: Vec<strix_auction::ThreatState> = self
                .threat_trackers
                .values()
                .map(|t| {
                    let (pos, _vel, _probs) = t.estimate_threat();
                    strix_auction::ThreatState {
                        id: t.threat_id,
                        position: strix_auction::Position::from(pos),
                        lethal_radius: 200.0,
                        threat_type: strix_auction::ThreatType::Unknown,
                    }
                })
                .collect();

            let kill_zone_penalties = self.loss_analyzer.kill_zone_penalties();

            let result = self.auctioneer.run_auction(
                &auction_drones,
                tasks,
                &auction_threats,
                &HashMap::new(),
                &kill_zone_penalties,
            );

            self.assignments = result.assignments.clone();

            // Record auction trace
            let drone_ids: Vec<u32> = telemetry.iter().map(|(id, _)| *id).collect();
            let trace = DecisionTrace::new(self.sim_time, DecisionType::TaskAssignment)
                .with_inputs(TraceInputs {
                    drone_ids,
                    regime: "mixed".into(),
                    metrics: serde_json::json!({
                        "total_welfare": result.total_welfare,
                        "assigned": result.assignments.len(),
                        "unassigned": result.unassigned_tasks.len(),
                    }),
                    context: serde_json::Value::Null,
                })
                .with_output(
                    &format!(
                        "Auction: {} tasks assigned, {} unassigned",
                        result.assignments.len(),
                        result.unassigned_tasks.len()
                    ),
                    serde_json::json!({"assignments": result.assignments.len()}),
                )
                .with_confidence(0.9);
            self.tracer.record(trace);
            traces_recorded += 1;
        }

        // ── 5. Propagate via gossip ──────────────────────────────────────
        self.gossip_version += 1;
        for (id, telem) in telemetry {
            let regime = self.regimes.get(id).copied().unwrap_or(Regime::Patrol);
            self.gossip.update_self_state(
                Position3D(telem.position),
                telem.battery,
                format!("{:?}", regime),
                telem.timestamp,
            );
        }

        // Report threats to gossip
        for (threat_id, tracker) in &self.threat_trackers {
            let (pos, _vel, _probs) = tracker.estimate_threat();
            self.gossip.report_threat(
                *threat_id as u64,
                Position3D([pos.x, pos.y, pos.z]),
                0.8,
                self.sim_time,
            );
        }

        // ── 6. Update pheromone field ────────────────────────────────────
        self.pheromones.evaporate(self.sim_time);

        // Deposit "explored" pheromones at drone positions
        for (id, telem) in telemetry {
            self.pheromones.deposit(&Pheromone {
                position: Position3D(telem.position),
                ptype: PheromoneType::Explored,
                intensity: 1.0,
                timestamp: self.sim_time,
                depositor: NodeId(*id),
            });
        }

        // Deposit "threat" pheromones at kill zones
        for kz in &self.loss_analyzer.kill_zones {
            self.pheromones.deposit(&Pheromone {
                position: Position3D([kz.center.x, kz.center.y, kz.center.z]),
                ptype: PheromoneType::Threat,
                intensity: kz.penalty * 5.0,
                timestamp: self.sim_time,
                depositor: NodeId(0),
            });
        }

        // ── 7. Build decision ────────────────────────────────────────────
        let positions: HashMap<u32, Vector3<f64>> = telemetry
            .iter()
            .map(|(id, t)| {
                (
                    *id,
                    Vector3::new(t.position[0], t.position[1], t.position[2]),
                )
            })
            .collect();

        let threat_positions: HashMap<u32, Vector3<f64>> = self
            .threat_trackers
            .iter()
            .map(|(id, t)| {
                let (pos, _vel, _probs) = t.estimate_threat();
                (*id, pos)
            })
            .collect();

        SwarmDecision {
            assignments: self.assignments.clone(),
            regimes: self.regimes.clone(),
            positions,
            threat_positions,
            kill_zone_count: self.loss_analyzer.active_kill_zones(),
            antifragile_score: self.loss_analyzer.antifragile_score(),
            traces_recorded,
            gossip_convergence: self.gossip.convergence_estimate(),
            pheromone_cells: self.pheromones.active_cells(),
        }
    }

    // ── Helpers ──────────────────────────────────────────────────────────

    fn nearest_threat_bearing(&self, drone_pos: &Vector3<f64>) -> Vector3<f64> {
        self.threat_trackers
            .values()
            .map(|t| {
                let (pos, _vel, _probs) = t.estimate_threat();
                (pos, (pos - drone_pos).norm())
            })
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(pos, _)| convert::threat_bearing(drone_pos, &pos))
            .unwrap_or_else(Vector3::zeros)
    }

    fn nearest_threat_metrics(&self, drone_pos: &Vector3<f64>) -> (f64, f64) {
        self.threat_trackers
            .values()
            .map(|t| {
                let (pos, vel, _probs) = t.estimate_threat();
                let diff = pos - drone_pos;
                let dist = diff.norm();
                // Closing rate: negative means approaching
                let closing = if dist > 1e-6 {
                    vel.dot(&diff) / dist
                } else {
                    0.0
                };
                (dist, closing)
            })
            .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
            .unwrap_or((f64::MAX, 0.0))
    }
}
