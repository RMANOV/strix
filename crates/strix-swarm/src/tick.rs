//! The main tick() loop — one orchestration cycle: sense → think → act.
//!
//! Each tick chains all STRIX crates in sequence:
//! 1. strix-core: EW threat scan + noise/gossip modulation
//! 2. strix-core: particle filter prediction + measurement update
//! 3. strix-core: CUSUM anomaly detection + regime transitions
//! 4. strix-core: formation correction (geometry + deadband control)
//! 5. strix-core: threat tracker update
//! 6. strix-core: ROE authorization gate
//! 7. strix-auction: combinatorial task auction
//! 8. strix-mesh: gossip state propagation + pheromone update
//! 9. strix-core: CBF safety clamp (collision + altitude + NFZ)
//! 10. strix-xai: decision trace recording

use std::collections::HashMap;

use nalgebra::Vector3;

use strix_adapters::traits::Telemetry;
use strix_auction::{Assignment, Auctioneer, Capabilities, LossAnalyzer, Task};
use strix_core::anomaly::CusumConfig;
use strix_core::cbf::{self, CbfConfig, NeighborState, NoFlyZone};
use strix_core::ew_response::{EwEngine, EwEvent, EwResponse, EwResponsePlan};
use strix_core::formation::{self, FormationConfig, FormationType};
use strix_core::hysteresis::{HysteresisConfig, HysteresisGate};
use strix_core::intent::{self, IntentConfig, IntentSignals};
use strix_core::particle_nav::ParticleNavFilter;
use strix_core::regime::{detect_regime, DetectionConfig, RegimeSignals};
use strix_core::roe::{
    EngagementAuth, EngagementContext, RoeEngine, ThreatClassification, WeaponsPosture,
};
use strix_core::threat_tracker::ThreatTracker;
use strix_core::Regime;
use strix_mesh::gossip::GossipEngine;
use strix_mesh::stigmergy::{Pheromone, PheromoneField, PheromoneType};
use strix_mesh::{NodeId, Position3D};
use strix_xai::trace::{DecisionTrace, DecisionType, TraceInputs, TraceRecorder};

use crate::convert;

#[cfg(feature = "temporal")]
use strix_core::temporal::{HorizonConstraint, TemporalManager};

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
    /// Hysteresis gate configuration for regime stability.
    pub hysteresis_config: HysteresisConfig,
    /// Intent detection pipeline configuration.
    pub intent_config: IntentConfig,
    /// Base fear level F ∈ [0,1]. Overridden by FearAdapter when present.
    pub fear: f64,
    /// Formation type for the swarm (None = disabled).
    pub formation_type: Option<FormationType>,
    /// Formation geometry and control law parameters.
    pub formation_config: FormationConfig,
    /// Rules of engagement engine.
    pub roe_engine: RoeEngine,
    /// Control barrier function config (None = disabled).
    pub cbf_config: Option<CbfConfig>,
    /// Static no-fly zones for CBF.
    pub no_fly_zones: Vec<NoFlyZone>,
    /// Maximum age (seconds) before stale EW events are cleared.
    pub ew_stale_age: f64,
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
            hysteresis_config: HysteresisConfig::default(),
            intent_config: IntentConfig::default(),
            fear: 0.0,
            formation_type: Some(FormationType::Vee),
            formation_config: FormationConfig::default(),
            roe_engine: RoeEngine::default(),
            cbf_config: Some(CbfConfig::default()),
            no_fly_zones: Vec::new(),
            ew_stale_age: 120.0,
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
    /// Maximum threat intent score across all drones this tick [-1, 1].
    pub max_intent_score: f64,
    /// Current fear level F ∈ [0,1] (0 = aggressive, 1 = maximum caution).
    pub fear_level: f64,
    /// Formation quality score ∈ [0, 1] (1.0 = perfect formation, 0.0 = scattered).
    /// None if formation is disabled.
    pub formation_quality: Option<f64>,
    /// Per-drone formation correction vectors (formation slot → drone).
    pub formation_corrections: HashMap<u32, Vector3<f64>>,
    /// Number of engagement requests denied by ROE this tick.
    pub roe_denials: u32,
    /// Number of engagement requests that require human escalation.
    pub roe_escalations: u32,
    /// Number of active EW threats being tracked.
    pub ew_active_threats: usize,
    /// Number of CBF safety constraints active this tick.
    pub cbf_active_constraints: u32,
    /// Number of drones pushed into deadlock escape maneuvers this tick.
    pub deadlock_escape_count: usize,
    // ── Phi-sim intelligence fields ─────────────────────────────────────
    /// Collective courage level [0, 1].
    pub courage_level: f64,
    /// Collective tension (C-F)/(1+F*C).
    pub tension: f64,
    /// Per-drone fear levels (drone_id → F).
    pub per_drone_fear: HashMap<u32, f64>,
    /// Phi-sim calibration quality [0, 1] (0 if phi-sim disabled).
    pub calibration_quality: f64,
    /// Multi-horizon temporal anomalies: (horizon_name, direction, cusum_value).
    pub temporal_anomalies: Vec<(String, i32, f64)>,
}

// ── Phase 3 intermediate structs ─────────────────────────────────────────

/// Fear and EW state computed at the start of each tick.
struct TickFearState {
    f: f64,
    collective_c: f64,
    collective_t: f64,
    fear_threshold: f64,
    fear_bias: f64,
    fear_detection_config: DetectionConfig,
    tick_noise: strix_core::particle_nav::ProcessNoiseConfig,
    ew_active_threats: usize,
    ew_force_evade: bool,
    ew_gossip_override: Option<(usize, bool)>,
}

/// Fleet state snapshot after particle filter updates.
struct FleetSnapshot {
    centroid: Vector3<f64>,
    heading: Vector3<f64>,
    #[cfg(feature = "temporal")]
    temporal_constraints: HashMap<u32, HorizonConstraint>,
}

/// Results from regime detection and intent analysis.
struct RegimeDecisions {
    max_intent_score: f64,
    cusum_break_count: u32,
    traces_recorded: u32,
}

/// The swarm orchestrator — chains all STRIX crates together.
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
    /// Per-drone hysteresis gates for regime stability.
    hysteresis_gates: HashMap<u32, HysteresisGate>,
    /// Per-drone previous closing rates for closing acceleration (Item B.3).
    prev_closing_rates: HashMap<u32, f64>,
    /// Optional PhiSim fear adapter for adaptive risk modulation.
    #[cfg(feature = "phi-sim")]
    pub fear_adapter: Option<crate::fear_adapter::SwarmFearAdapter>,
    /// Previous tick's max intent score (for fear computation).
    last_intent_score: f64,
    /// Previous tick's CUSUM break count (for fear computation).
    last_cusum_breaks: u32,
    /// Base process noise config (captured at creation, used as fear-scaling reference).
    base_noise_cfg: strix_core::particle_nav::ProcessNoiseConfig,
    /// Gossip version counter.
    gossip_version: u64,
    /// Tick counter.
    tick_count: u32,
    /// Simulation clock.
    sim_time: f64,
    /// Config.
    pub config: SwarmConfig,

    // ── Island modules ───────────────────────────────────────────────────
    /// Current formation type (None = formation disabled).
    pub formation_type: Option<FormationType>,
    /// Formation geometry config.
    pub formation_config: FormationConfig,
    /// Rules of engagement engine.
    pub roe_engine: RoeEngine,
    /// Electronic warfare response engine.
    pub ew_engine: EwEngine,
    /// CBF config (None = disabled).
    pub cbf_config: Option<CbfConfig>,
    /// GCBF+ neural safety filter (None = use classical CBF).
    #[cfg(feature = "gcbf")]
    pub gcbf_barrier: Option<strix_core::gcbf::NeuralBarrier>,
    /// Active no-fly zones.
    pub no_fly_zones: Vec<NoFlyZone>,
    /// Per-drone multi-horizon temporal managers (replaces nav_filter step when active).
    #[cfg(feature = "temporal")]
    pub temporal_managers: HashMap<u32, TemporalManager>,
}

impl SwarmOrchestrator {
    /// Create a new orchestrator for the given drone IDs.
    pub fn new(drone_ids: &[u32], config: SwarmConfig) -> Self {
        let mut nav_filters = HashMap::new();
        let mut regimes = HashMap::new();
        let mut signal_histories = HashMap::new();
        let mut threat_distance_histories = HashMap::new();
        let mut hysteresis_gates = HashMap::new();

        // Initialize gossip network — exclude self_id from peers (no self-loop).
        let self_id = drone_ids.first().copied().unwrap_or(0);
        let mut gossip = GossipEngine::new(NodeId(self_id), config.gossip_fanout);
        for &id in drone_ids {
            if id != self_id {
                gossip.add_peer(NodeId(id));
            }
        }

        for &id in drone_ids {
            nav_filters.insert(
                id,
                ParticleNavFilter::new(config.n_particles, Vector3::zeros()),
            );
            regimes.insert(id, Regime::Patrol);
            signal_histories.insert(id, Vec::new());
            threat_distance_histories.insert(id, Vec::new());
            hysteresis_gates.insert(
                id,
                HysteresisGate::new(Regime::Patrol, 0.0, config.hysteresis_config.clone()),
            );
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
            hysteresis_gates,
            prev_closing_rates: HashMap::new(),
            #[cfg(feature = "phi-sim")]
            fear_adapter: None,
            last_intent_score: 0.0,
            last_cusum_breaks: 0,
            base_noise_cfg: strix_core::particle_nav::ProcessNoiseConfig::default(),
            gossip_version: 0,
            tick_count: 0,
            sim_time: 0.0,
            // Capture island module configs before moving config
            formation_type: config.formation_type,
            formation_config: config.formation_config.clone(),
            roe_engine: config.roe_engine.clone(),
            ew_engine: EwEngine::new(),
            cbf_config: config.cbf_config.clone(),
            #[cfg(feature = "gcbf")]
            gcbf_barrier: None,
            no_fly_zones: config.no_fly_zones.clone(),
            #[cfg(feature = "temporal")]
            temporal_managers: {
                let mut tm = HashMap::new();
                for &id in drone_ids {
                    tm.insert(id, TemporalManager::new(Vector3::zeros()));
                }
                tm
            },
            config,
        }
    }

    /// Current simulation time.
    pub fn sim_time(&self) -> f64 {
        self.sim_time
    }

    /// Register a new drone mid-simulation.
    ///
    /// # Timing contract
    /// Must be called **between** ticks (not during `tick()`). The new drone
    /// will be included in the next `tick()` call's telemetry iteration.
    pub fn register_drone(&mut self, id: u32, position: Vector3<f64>) {
        self.nav_filters.insert(
            id,
            ParticleNavFilter::new(self.config.n_particles, position),
        );
        self.regimes.insert(id, Regime::Patrol);
        self.signal_histories.insert(id, Vec::new());
        self.threat_distance_histories.insert(id, Vec::new());
        self.hysteresis_gates.insert(
            id,
            HysteresisGate::new(
                Regime::Patrol,
                self.sim_time,
                self.config.hysteresis_config.clone(),
            ),
        );
        self.gossip.add_peer(NodeId(id));

        #[cfg(feature = "temporal")]
        self.temporal_managers
            .insert(id, TemporalManager::new(position));

        #[cfg(feature = "phi-sim")]
        if let Some(adapter) = &mut self.fear_adapter {
            adapter.register_drone(id);
        }
    }

    /// Register a new threat to track.
    pub fn register_threat(&mut self, threat_id: u32, position: Vector3<f64>) {
        self.threat_trackers.insert(
            threat_id,
            ThreatTracker::new(threat_id, self.config.n_threat_particles, position),
        );
    }

    // ── Island module configuration ──────────────────────────────────────

    /// Change the swarm formation type. Pass None to disable formation control.
    pub fn set_formation(&mut self, formation: Option<FormationType>) {
        self.formation_type = formation;
    }

    /// Update the weapons posture. Returns the new posture.
    pub fn set_weapons_posture(
        &mut self,
        posture: strix_core::roe::WeaponsPosture,
    ) -> strix_core::roe::WeaponsPosture {
        self.roe_engine.set_posture(posture)
    }

    /// Report an electronic warfare detection event.
    ///
    /// The response plan is generated immediately and stored. The tick loop
    /// will apply the response actions on the next call to [`tick()`].
    pub fn report_ew_event(&mut self, event: EwEvent) -> EwResponsePlan {
        self.ew_engine.respond(event)
    }

    /// Add a no-fly zone for the CBF safety filter.
    pub fn add_no_fly_zone(&mut self, nfz: NoFlyZone) {
        self.no_fly_zones.push(nfz);
    }

    /// Enable or disable CBF safety filtering.
    pub fn set_cbf(&mut self, config: Option<CbfConfig>) {
        self.cbf_config = config;
    }

    /// Mark a drone as lost and trigger anti-fragile response.
    ///
    /// # Timing contract
    /// Must be called **between** ticks (not during `tick()`). The drone is
    /// removed from all tracking structures and its tasks are orphaned for
    /// re-auction on the next tick.
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
        self.hysteresis_gates.remove(&drone_id);
        self.prev_closing_rates.remove(&drone_id);
        self.gossip.remove_peer(NodeId(drone_id));

        #[cfg(feature = "temporal")]
        self.temporal_managers.remove(&drone_id);

        #[cfg(feature = "phi-sim")]
        if let Some(adapter) = &mut self.fear_adapter {
            adapter.remove_drone(drone_id);
        }

        // Remove assignments for this drone
        self.assignments.retain(|a| a.drone_id != drone_id);

        // Trigger re-auction
        self.auctioneer.trigger_reauction();

        // Record trace
        let mut alive_ids: Vec<u32> = self.nav_filters.keys().copied().collect();
        alive_ids.sort_unstable();
        let trace = DecisionTrace::new(self.sim_time, DecisionType::ReAuction)
            .with_inputs(TraceInputs {
                drone_ids: alive_ids,
                regime: "mixed".to_string(),
                metrics: serde_json::json!({
                    "lost_drone": drone_id,
                    "kill_zones": self.loss_analyzer.active_kill_zones(),
                    "antifragile_score": self.loss_analyzer.antifragile_score(),
                }),
                context: serde_json::json!({"lost_drone_id": drone_id}),
                fear_level: None,
                courage_level: None,
                tension: None,
                calibration_quality: None,
            })
            .with_output(
                &format!("Drone {} lost — re-auctioning tasks", drone_id),
                serde_json::json!({"kill_zones": self.loss_analyzer.active_kill_zones()}),
            )
            .with_confidence(0.9);
        self.tracer.record(trace);
    }

    // ── Phase methods (extracted from tick()) ────────────────────────────

    /// Phase 0: Compute fear level F and EW state for this tick.
    #[cfg_attr(not(feature = "phi-sim"), allow(unused_variables))]
    fn compute_fear_state(&mut self, telemetry: &[(u32, Telemetry)]) -> TickFearState {
        // ── 0. Compute fear level F ────────────────────────────────────────
        #[cfg(feature = "phi-sim")]
        let f = if let Some(adapter) = &mut self.fear_adapter {
            let alive = self.nav_filters.len() as u32;
            let initial = (alive + self.loss_analyzer.total_losses() as u32).max(1);
            let fleet_attrition = 1.0 - (alive as f64 / initial as f64);
            let is_auction_tick = self.tick_count % self.config.auction_interval.max(1) == 0;
            let fleet_velocities: Vec<Vector3<f64>> = telemetry
                .iter()
                .map(|(_, telem)| {
                    Vector3::new(telem.velocity[0], telem.velocity[1], telem.velocity[2])
                })
                .collect();
            let fleet_coherence =
                strix_core::fleet_metrics::velocity_coherence(&fleet_velocities, 0.5);

            // Build per-drone fear inputs from available telemetry.
            let per_drone: Vec<(u32, crate::fear_adapter::DroneFearInputs)> = telemetry
                .iter()
                .map(|(id, telem)| {
                    let speed = (telem.velocity[0].powi(2)
                        + telem.velocity[1].powi(2)
                        + telem.velocity[2].powi(2))
                    .sqrt();
                    let regime = self
                        .regimes
                        .get(id)
                        .copied()
                        .unwrap_or(strix_core::Regime::Patrol);
                    (
                        *id,
                        crate::fear_adapter::DroneFearInputs {
                            threat_distance: 1000.0, // default; refined in later ticks
                            closing_rate: self.last_intent_score.min(0.0),
                            cusum_triggered: self.last_cusum_breaks > 0,
                            regime,
                            speed,
                            fleet_coherence,
                        },
                    )
                })
                .collect();

            adapter.update_fear(
                &per_drone,
                fleet_attrition,
                self.last_cusum_breaks,
                is_auction_tick,
            );
            adapter.collective_fear()
        } else {
            self.config.fear
        };
        #[cfg(not(feature = "phi-sim"))]
        let f = self.config.fear;

        // Extract collective phi-sim signals for module modulation.
        #[cfg(feature = "phi-sim")]
        let (collective_c, collective_t) = if let Some(adapter) = &self.fear_adapter {
            (adapter.collective_courage(), adapter.collective_tension())
        } else {
            (0.0, 0.0)
        };
        #[cfg(not(feature = "phi-sim"))]
        let (collective_c, collective_t) = (0.0, 0.0);

        // Derive fear axes as plain f64s for module modulation (mirrors FearAxes::from_fear).
        let fear_threshold = 1.0 - f * 0.7; // [0.3, 1.0]
        let fear_speed = 1.0 + f * 2.0; // [1.0, 3.0]
        let fear_bias = f; // [0.0, 1.0]

        // Pre-compute fear-modulated configs used throughout this tick.
        let fear_detection_config =
            crate::fear_adapter::modulate_detection_config(&self.config.detection_config, f);
        let fear_noise = self.base_noise_cfg.scaled_by_fear(f);

        // ── 0.5 EW threat scan + response application ─────────────────────
        // Clear stale EW events and apply any active threat responses.
        self.ew_engine
            .clear_stale_events(self.sim_time, self.config.ew_stale_age);
        let ew_active_threats = self.ew_engine.active_threats().len();

        // Compute aggregate EW effects for this tick.
        let mut ew_noise_multiplier = 1.0_f64;
        let mut ew_force_evade = false;
        let mut ew_gossip_override: Option<(usize, bool)> = None;

        for event in self.ew_engine.active_threats() {
            // Re-derive responses from active events (stateless — based on threat type/severity).
            let responses = self.ew_engine.compute_responses_readonly(event);
            for action in &responses {
                match action {
                    EwResponse::ExpandNavigationNoise { noise_multiplier } => {
                        ew_noise_multiplier = ew_noise_multiplier.max(*noise_multiplier);
                    }
                    EwResponse::ForceEvade { .. } => {
                        ew_force_evade = true;
                    }
                    EwResponse::GossipFallback {
                        reduced_fanout,
                        priority_only,
                    } => {
                        ew_gossip_override = Some((*reduced_fanout, *priority_only));
                    }
                    EwResponse::MarkEwZone {
                        bearing,
                        range,
                        penalty_weight,
                    } => {
                        // Deposit threat pheromone in the EW zone direction.
                        // Approximate position from bearing + range relative to fleet centroid.
                        let bearing_rad = bearing.to_radians();
                        let ew_pos =
                            Position3D([bearing_rad.sin() * range, bearing_rad.cos() * range, 0.0]);
                        self.pheromones.deposit(&Pheromone {
                            position: ew_pos,
                            ptype: PheromoneType::Threat,
                            intensity: *penalty_weight,
                            timestamp: self.sim_time,
                            depositor: NodeId(0),
                        });
                    }
                    _ => {} // Monitor, InertialFallback, TerrainMask — handled passively
                }
            }
        }

        // Amplify EW severity by fear (scared swarm reacts more aggressively to EW).
        let fear_ew_multiplier = self
            .ew_engine
            .severity_with_fear(ew_noise_multiplier, fear_speed);

        // Apply EW-modulated noise on top of fear-modulated noise.
        let tick_noise = if fear_ew_multiplier > 1.0 {
            fear_noise.scaled_by_ew(fear_ew_multiplier)
        } else {
            fear_noise.clone()
        };

        TickFearState {
            f,
            collective_c,
            collective_t,
            fear_threshold,
            fear_bias,
            fear_detection_config,
            tick_noise,
            ew_active_threats,
            ew_force_evade,
            ew_gossip_override,
        }
    }

    /// Phase 1: Update particle filters from telemetry and compute fleet snapshot.
    fn update_navigation(
        &mut self,
        telemetry: &[(u32, Telemetry)],
        fear: &TickFearState,
        _dt: f64,
    ) -> FleetSnapshot {
        // ── 1. Update particle filters from telemetry ─────────────────────
        let mut fleet_centroid = Vector3::zeros();
        let mut fleet_heading = Vector3::zeros();
        let mut alive_count = 0usize;

        // Pre-compute threat bearings (avoids borrow conflict with nav_filters)
        let threat_bearings: HashMap<u32, Vector3<f64>> = telemetry
            .iter()
            .filter_map(|(id, telem)| {
                let drone_pos =
                    Vector3::new(telem.position[0], telem.position[1], telem.position[2]);
                self.nearest_threat_bearing(&drone_pos).map(|b| (*id, b))
            })
            .collect();

        #[cfg(feature = "temporal")]
        let mut temporal_constraints: HashMap<u32, HorizonConstraint> = HashMap::new();

        for (id, telem) in telemetry {
            if !self.nav_filters.contains_key(id) {
                continue;
            }

            // Apply fear+EW-scaled noise for this tick.
            if let Some(filter) = self.nav_filters.get_mut(id) {
                filter.noise_cfg = fear.tick_noise.clone();
            }

            let drone_pos = Vector3::new(telem.position[0], telem.position[1], telem.position[2]);

            // ── Multi-sensor fusion: build observations from telemetry ──
            let mut obs = Vec::with_capacity(4);

            // 1. Barometer — constrains vertical position
            obs.push(strix_core::Observation::Barometer {
                altitude: -telem.position[2], // NED: z down, altitude up
                timestamp: telem.timestamp,
            });

            // 2. IMU (velocity as pseudo-acceleration) — constrains velocity
            obs.push(strix_core::Observation::Imu {
                acceleration: Vector3::new(telem.velocity[0], telem.velocity[1], telem.velocity[2]),
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
            #[cfg(feature = "temporal")]
            {
                if let Some(tm) = self.temporal_managers.get_mut(id) {
                    let (_pos, _vel, _probs, constraint) = tm.step(&obs, &bearing, 1.0);
                    if let Some(c) = constraint {
                        temporal_constraints.insert(*id, c);
                    }
                }
            }
            #[cfg(not(feature = "temporal"))]
            {
                if let Some(filter) = self.nav_filters.get_mut(id) {
                    let (_pos, _vel, _probs) = filter.step(&obs, &bearing, 1.0, _dt);
                }
            }

            // Store position for next tick's VO delta
            self.prev_positions.insert(*id, drone_pos);

            let vel = Vector3::new(telem.velocity[0], telem.velocity[1], telem.velocity[2]);
            fleet_centroid += drone_pos;
            fleet_heading += vel;
            alive_count += 1;

            // Update signal history for CUSUM
            let speed = vel.norm();
            if let Some(history) = self.signal_histories.get_mut(id) {
                history.push(speed);
                if history.len() > 100 {
                    history.drain(..50);
                }
            }
        }

        if alive_count > 0 {
            fleet_centroid /= alive_count as f64;
            fleet_heading /= alive_count as f64;
        }
        // Fallback heading if fleet is stationary.
        if fleet_heading.norm() < 1e-6 {
            fleet_heading = Vector3::new(1.0, 0.0, 0.0);
        }

        // Temporal: blend strategic waypoints into fleet centroid.
        #[cfg(feature = "temporal")]
        if !temporal_constraints.is_empty() {
            let mut strategic_waypoint = Vector3::zeros();
            let mut total_confidence = 0.0;
            for c in temporal_constraints.values() {
                strategic_waypoint += c.waypoint * c.confidence;
                total_confidence += c.confidence;
            }
            if total_confidence > 0.0 {
                strategic_waypoint /= total_confidence;
                // Blend capped at 0.3 to prevent strategic override.
                let blend = (total_confidence / temporal_constraints.len() as f64).min(0.3);
                fleet_centroid = fleet_centroid * (1.0 - blend) + strategic_waypoint * blend;
            }
        }

        FleetSnapshot {
            centroid: fleet_centroid,
            heading: fleet_heading,
            #[cfg(feature = "temporal")]
            temporal_constraints,
        }
    }

    /// Phase 2: Detect regimes via CUSUM + intent analysis, return regime decisions.
    #[cfg_attr(not(feature = "temporal"), allow(unused_variables))]
    fn detect_regimes_and_intent(
        &mut self,
        telemetry: &[(u32, Telemetry)],
        fear: &TickFearState,
        fleet: &FleetSnapshot,
        dt: f64,
    ) -> RegimeDecisions {
        // ── 2. Detect regimes (CUSUM + signals) + intent ─────────────────

        // Item D: Compute fleet velocity coherence once per tick.
        let fleet_velocities: Vec<Vector3<f64>> = telemetry
            .iter()
            .map(|(_, t)| Vector3::new(t.velocity[0], t.velocity[1], t.velocity[2]))
            .collect();
        let fleet_coherence = strix_core::fleet_metrics::velocity_coherence(&fleet_velocities, 0.5);

        let mut regime_changes = Vec::new();
        let mut max_intent_score = 0.0_f64;
        let mut cusum_break_count = 0u32;
        for (id, telem) in telemetry {
            let regime = self.regimes.get(id).copied().unwrap_or(Regime::Patrol);
            let drone_pos = Vector3::new(telem.position[0], telem.position[1], telem.position[2]);

            // Compute regime signals
            let nearest_threat_metrics = self.nearest_threat_metrics(&drone_pos);
            let (nearest_threat_dist, closing_rate) =
                nearest_threat_metrics.unwrap_or((f64::MAX, 0.0));

            // Update threat distance history for Hurst exponent computation.
            // Only push real distances — f64::MAX from no-threat cases would
            // pollute the Hurst exponent and momentum score calculations.
            if let (Some(tdh), Some((real_dist, _))) = (
                self.threat_distance_histories.get_mut(id),
                nearest_threat_metrics,
            ) {
                tdh.push(real_dist);
                if tdh.len() > 100 {
                    tdh.drain(..50);
                }
            }

            // Item B.1: Extract actual CUSUM direction instead of hardcoding 0.
            let (cusum_triggered, cusum_direction) = self
                .signal_histories
                .get(id)
                .map(|h| {
                    if h.len() >= self.config.cusum_config.min_samples {
                        let (triggered, dir, _val) = strix_core::anomaly::cusum_test(
                            h,
                            self.config.cusum_config.threshold_h,
                            self.config.cusum_config.min_samples,
                        );
                        (triggered, dir)
                    } else {
                        (false, 0)
                    }
                })
                .unwrap_or((false, 0));

            if cusum_triggered {
                cusum_break_count += 1;
            }

            // Item B.3: Compute closing acceleration = d(closing_rate)/dt.
            let prev_cr = self
                .prev_closing_rates
                .get(id)
                .copied()
                .unwrap_or(closing_rate);
            let closing_acceleration = if dt > 1e-9 {
                (closing_rate - prev_cr) / dt
            } else {
                0.0
            };
            self.prev_closing_rates.insert(*id, closing_rate);

            // Item C: Compute Hurst with uncertainty for intent pipeline.
            // Adaptive max_window: uses all available data up to 50, eliminating
            // the blind spot where ticks 20-49 always returned default (0.5, 0.5).
            let (hurst_val, hurst_unc) = self
                .threat_distance_histories
                .get(id)
                .filter(|h| h.len() >= 20)
                .map(|h| strix_core::uncertainty::hurst_exponent(h, 10, h.len().min(50)))
                .unwrap_or((0.5, 0.5));

            // Intent pipeline uses THREAT distance volatility (threat behavior),
            // while regime detection uses SELF speed volatility (below).
            // Adaptive long_window mirrors the Hurst fix above.
            let intent_vol_ratio = self
                .threat_distance_histories
                .get(id)
                .filter(|h| h.len() >= 20)
                .map(|h| strix_core::uncertainty::volatility_compression(h, 10, h.len().min(50)).0)
                .unwrap_or(1.0);

            // Self-speed volatility for regime detection.
            let self_vol_ratio = self
                .signal_histories
                .get(id)
                .filter(|h| h.len() >= 20)
                .map(|h| strix_core::uncertainty::volatility_compression(h, 10, h.len().min(50)).0)
                .unwrap_or(1.0);

            // Item C: Run intent detection pipeline.
            let intent_signals = IntentSignals {
                hurst: hurst_val,
                hurst_uncertainty: hurst_unc,
                closing_rate,
                closing_acceleration,
                volatility_ratio: intent_vol_ratio,
                threat_distance: nearest_threat_dist,
                fleet_coherence: Some(fleet_coherence),
            };
            let intent_result = intent::detect_intent(&intent_signals, &self.config.intent_config);
            if intent_result.score.abs() > max_intent_score.abs() {
                max_intent_score = intent_result.score;
            }

            let drone_pos_for_bias =
                strix_auction::Position::new(drone_pos.x, drone_pos.y, drone_pos.z);
            let evade_bias = self.loss_analyzer.evade_bias_at(&drone_pos_for_bias);

            let signals = RegimeSignals {
                cusum_triggered,
                cusum_direction,
                hurst: hurst_val,
                volatility_ratio: self_vol_ratio,
                threat_distance: nearest_threat_dist,
                closing_rate,
                evade_bias,
            };

            // Temporal: bias regime detection when constraint suggests EVADE.
            // Only clone the config when temporal constraint modifies it.
            #[cfg(feature = "temporal")]
            let temporal_cfg_override = fleet.temporal_constraints.get(id).and_then(|c| {
                if c.suggested_regime == Regime::Evade && c.confidence > 0.6 {
                    let mut cfg = fear.fear_detection_config.clone();
                    cfg.evade_distance *= 1.0 + c.confidence * 0.5;
                    cfg.closing_rate_threshold *= 1.0 - c.confidence * 0.3;
                    Some(cfg)
                } else {
                    None
                }
            });
            #[cfg(feature = "temporal")]
            let detection_cfg = temporal_cfg_override
                .as_ref()
                .unwrap_or(&fear.fear_detection_config);
            #[cfg(not(feature = "temporal"))]
            let detection_cfg = &fear.fear_detection_config;

            let proposed_regime = detect_regime(&signals, regime, detection_cfg);

            // Item A: Route through hysteresis gate instead of direct apply.
            let approved_regime = if let Some(gate) = self.hysteresis_gates.get_mut(id) {
                gate.propose(proposed_regime, self.sim_time)
            } else {
                proposed_regime
            };

            if approved_regime != regime {
                regime_changes.push((*id, regime, approved_regime));
                self.regimes.insert(*id, approved_regime);
            }
        }

        // EW ForceEvade override: if active EW demands evasion, force all drones to EVADE.
        if fear.ew_force_evade {
            self.force_all_evade(&mut regime_changes);
        }

        // Item B.2: Check attrition risk level — force EVADE if Retreat/Survival.
        {
            let alive = self.nav_filters.len() as u32;
            let initial = (alive + self.loss_analyzer.total_losses() as u32).max(1);
            let attrition_rate = 1.0 - (alive as f64 / initial as f64);
            let risk_level =
                strix_auction::RiskLevel::from_attrition_with_fear(attrition_rate, fear.f);
            if matches!(
                risk_level,
                strix_auction::RiskLevel::Retreat | strix_auction::RiskLevel::Survival
            ) {
                self.force_all_evade(&mut regime_changes);
            }
        }

        // Record regime change traces
        let mut traces_recorded = 0u32;
        for (drone_id, old_regime, new_regime) in &regime_changes {
            let trace = DecisionTrace::new(self.sim_time, DecisionType::RegimeChange)
                .with_inputs(TraceInputs {
                    drone_ids: vec![*drone_id],
                    regime: format!("{:?}", old_regime),
                    metrics: serde_json::json!({}),
                    context: serde_json::Value::Null,
                    fear_level: Some(fear.f),
                    courage_level: Some(fear.collective_c),
                    tension: Some(fear.collective_t),
                    calibration_quality: None,
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

        RegimeDecisions {
            max_intent_score,
            cusum_break_count,
            traces_recorded,
        }
    }

    /// Phase 3: Run ROE authorization gate and combinatorial task auction.
    /// Returns (roe_denials, roe_escalations, traces_recorded).
    #[cfg_attr(not(feature = "temporal"), allow(unused_variables))]
    fn run_roe_and_auction(
        &mut self,
        telemetry: &[(u32, Telemetry)],
        tasks: &[Task],
        fear: &TickFearState,
        fleet: &FleetSnapshot,
        regimes: &RegimeDecisions,
    ) -> (u32, u32, u32) {
        // ── 3.5 ROE authorization gate ───────────────────────────────────
        // Check engagement authorization for tasks requiring weapon capability.
        let mut roe_denials = 0u32;
        let mut roe_escalations = 0u32;
        let mut roe_filtered_tasks: Vec<Task> = tasks.to_vec();
        let mut traces_recorded = 0u32;

        if !tasks.is_empty() {
            let mut denied_task_ids = Vec::new();

            if self.threat_trackers.is_empty() {
                // No threats registered. Under WeaponsHold or WeaponsTight, weapon
                // tasks are denied by default — there is no hostile act that could
                // trigger the self-defense override, so the posture gate applies
                // unconditionally. WeaponsFree passes: engage anything not friendly.
                let deny_weapons_no_threat = matches!(
                    self.roe_engine.posture,
                    WeaponsPosture::WeaponsHold | WeaponsPosture::WeaponsTight
                );
                if deny_weapons_no_threat {
                    for task in tasks {
                        if task.required_capabilities.has_weapon {
                            denied_task_ids.push(task.id);
                            roe_denials += 1;
                            self.record_roe_trace(
                                task.id,
                                f64::MAX,
                                "denied",
                                "no threats registered — weapon task denied under current posture",
                                1.0,
                            );
                            traces_recorded += 1;
                        }
                    }
                }
            } else {
                for task in tasks {
                    if !task.required_capabilities.has_weapon {
                        continue; // ROE only gates weapon-bearing tasks
                    }

                    // Find nearest threat to this task's location.
                    let task_pos = Vector3::new(task.location.x, task.location.y, task.location.z);
                    if let Some((threat_dist, threat_regime)) =
                        self.nearest_threat_to_point(&task_pos)
                    {
                        let threat_class = match threat_regime {
                            strix_core::ThreatRegime::CounterAttack => {
                                ThreatClassification::ConfirmedHostile
                            }
                            strix_core::ThreatRegime::Defend => {
                                ThreatClassification::SuspectedHostile
                            }
                            strix_core::ThreatRegime::Retreat => ThreatClassification::Unknown,
                        };

                        let hostile_act = threat_regime == strix_core::ThreatRegime::CounterAttack
                            && threat_dist < 200.0;
                        let collateral_risk = self.estimate_collateral_risk(&task_pos, telemetry);
                        let friendlies_at_risk =
                            self.count_friendlies_at_risk(&task_pos, telemetry);

                        let ctx = EngagementContext {
                            threat_class,
                            threat_distance: threat_dist,
                            hostile_act,
                            hostile_intent: threat_regime
                                == strix_core::ThreatRegime::CounterAttack,
                            collateral_risk,
                            friendlies_at_risk,
                            regime: Regime::Engage,
                        };

                        match self.roe_engine.authorize_engagement(&ctx) {
                            EngagementAuth::Denied { reason } => {
                                denied_task_ids.push(task.id);
                                roe_denials += 1;
                                self.record_roe_trace(task.id, threat_dist, "denied", &reason, 1.0);
                                traces_recorded += 1;
                            }
                            EngagementAuth::EscalationRequired { reason, .. } => {
                                // Escalated tasks MUST NOT proceed to auction without
                                // human approval — filter them just like denied tasks.
                                denied_task_ids.push(task.id);
                                roe_escalations += 1;
                                self.record_roe_trace(
                                    task.id,
                                    threat_dist,
                                    "escalation_required",
                                    &reason,
                                    0.7,
                                );
                                traces_recorded += 1;
                            }
                            EngagementAuth::Authorized { .. } => {} // pass through
                        }
                    }
                }
            }

            // ROE tension advisory: log posture suggestion based on phi-sim tension.
            if let Some(suggested) = self
                .roe_engine
                .tension_posture_suggestion(fear.collective_t)
            {
                let trace = DecisionTrace::new(self.sim_time, DecisionType::RegimeChange)
                    .with_inputs(TraceInputs {
                        drone_ids: vec![],
                        regime: "ROE_TENSION".to_string(),
                        metrics: serde_json::json!({
                            "tension": fear.collective_t,
                            "current_posture": format!("{:?}", self.roe_engine.posture),
                            "suggested_posture": format!("{:?}", suggested),
                        }),
                        context: serde_json::Value::Null,
                        fear_level: Some(fear.f),
                        courage_level: Some(fear.collective_c),
                        tension: Some(fear.collective_t),
                        calibration_quality: None,
                    })
                    .with_output(
                        &format!("Tension {:.2} suggests {:?}", fear.collective_t, suggested),
                        serde_json::json!({"suggested": format!("{:?}", suggested)}),
                    )
                    .with_confidence(0.6);
                self.tracer.record(trace);
                traces_recorded += 1;
            }

            // Remove ROE-denied tasks from auction pool.
            if !denied_task_ids.is_empty() {
                roe_filtered_tasks.retain(|t| !denied_task_ids.contains(&t.id));
            }
        }

        // ── 4. Run combinatorial auction ─────────────────────────────────
        let auction_interval = self.config.auction_interval.max(1);
        let should_auction =
            self.tick_count.is_multiple_of(auction_interval) || self.auctioneer.needs_reauction;

        if should_auction && !roe_filtered_tasks.is_empty() {
            // Item C: Intent-based urgency boost — high threat intent raises
            // urgency of nearby tasks, creating a market-driven tactical response.
            let mut boosted_tasks: Vec<Task> = roe_filtered_tasks;
            if regimes.max_intent_score > 0.0 {
                for task in &mut boosted_tasks {
                    // Boost urgency proportional to intent score for all tasks.
                    // The auction market naturally reallocates drones toward
                    // high-urgency tasks without requiring regime changes.
                    task.urgency *= 1.0 + regimes.max_intent_score * (1.0 + fear.f);
                }
            }

            // Temporal: boost urgency for tasks near constraint waypoints.
            #[cfg(feature = "temporal")]
            for task in &mut boosted_tasks {
                for c in fleet.temporal_constraints.values() {
                    let task_pos = Vector3::new(task.location.x, task.location.y, task.location.z);
                    let dist = (task_pos - c.waypoint).norm();
                    if dist < 200.0 {
                        let proximity = 1.0 - dist / 200.0;
                        task.urgency *= 1.0 + proximity * c.confidence * 0.5;
                    }
                }
            }

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

            let kill_zone_penalties = self.loss_analyzer.kill_zone_penalties_with_fear(fear.f);

            // Inject per-drone scenario contexts from phi-sim into the auctioneer.
            self.auctioneer.clear_scenario_contexts();
            #[cfg(feature = "phi-sim")]
            if let Some(adapter) = &self.fear_adapter {
                for drone in &auction_drones {
                    if let Some(decision) = adapter.decision_for_drone(drone.id) {
                        self.auctioneer.set_scenario_context(
                            drone.id,
                            strix_auction::bidder::ScenarioContext {
                                doom_value: decision.doom_value,
                                upside_value: decision.upside_value,
                                confidence: decision.confidence,
                            },
                        );
                    }
                }
            }

            self.auctioneer.fear = fear.f;
            let result = self.auctioneer.run_auction(
                &auction_drones,
                &boosted_tasks,
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
                    fear_level: Some(fear.f),
                    courage_level: Some(fear.collective_c),
                    tension: Some(fear.collective_t),
                    calibration_quality: None,
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
        } else if should_auction && roe_filtered_tasks.is_empty() && !tasks.is_empty() {
            // All tasks were ROE-denied — clear assignments.
            self.assignments.clear();
        }

        (roe_denials, roe_escalations, traces_recorded)
    }

    /// Phase 4: Propagate state via gossip, update pheromones, apply CBF safety clamp.
    /// Returns `(cbf_active_constraints, deadlock_escape_count)`.
    fn propagate_and_safety(
        &mut self,
        telemetry: &[(u32, Telemetry)],
        fear: &TickFearState,
        formation_corrections: &mut HashMap<u32, Vector3<f64>>,
    ) -> (u32, usize) {
        // ── 5. Propagate via gossip ──────────────────────────────────────
        self.gossip_version += 1;

        // Apply fear-modulated gossip fanout, then EW override (whichever is lower).
        let fear_fanout =
            crate::fear_adapter::modulate_gossip_fanout(self.config.gossip_fanout, fear.f);
        if let Some((reduced_fanout, _priority_only)) = fear.ew_gossip_override {
            self.gossip.set_fanout(reduced_fanout.min(fear_fanout));
        } else {
            self.gossip.set_fanout(fear_fanout);
        }

        for (id, telem) in telemetry {
            let regime = self.regimes.get(id).copied().unwrap_or(Regime::Patrol);
            let regime_str = match regime {
                Regime::Patrol => "Patrol",
                Regime::Engage => "Engage",
                Regime::Evade => "Evade",
            };
            self.gossip.update_self_state(
                Position3D(telem.position),
                telem.battery,
                regime_str.to_string(),
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

        // Prune stale threats to prevent unbounded gossip memory growth.
        self.gossip.prune_threats(self.sim_time, 300.0);

        // ── 6. Update pheromone field ────────────────────────────────────
        self.pheromones.evaporate(self.sim_time);

        // Deposit "explored" pheromones at drone positions (reduced under fear).
        let explored_intensity = 1.0 - fear.fear_bias * 0.3; // 1.0→0.7 at max fear
        for (id, telem) in telemetry {
            self.pheromones.deposit(&Pheromone {
                position: Position3D(telem.position),
                ptype: PheromoneType::Explored,
                intensity: explored_intensity,
                timestamp: self.sim_time,
                depositor: NodeId(*id),
            });
        }

        // Deposit "threat" pheromones at kill zones (amplified under fear).
        let threat_intensity_scale = 1.0 + fear.fear_bias * 2.0; // 1.0→3.0 at max fear
        for kz in &self.loss_analyzer.kill_zones {
            self.pheromones.deposit(&Pheromone {
                position: Position3D([kz.center.x, kz.center.y, kz.center.z]),
                ptype: PheromoneType::Threat,
                intensity: kz.penalty * 5.0 * threat_intensity_scale,
                timestamp: self.sim_time,
                depositor: NodeId(0),
            });
        }

        // ── 6.5 CBF safety clamp ─────────────────────────────────────────
        // Apply control barrier functions to clamp velocities for safety.
        // With `gcbf` feature: O(n·k) neural barrier via GCBF+ GNN.
        // Without: O(n²) classical pairwise CBF (existing behavior).
        let mut cbf_active_constraints = 0u32;
        let mut deadlock_escape_count = 0usize;

        if let Some(ref base_cbf) = self.cbf_config {
            let cbf_cfg = base_cbf.with_fear(fear.f);

            // Common: collect all drone positions and velocities.
            let all_positions: Vec<(u32, Vector3<f64>)> = telemetry
                .iter()
                .map(|(id, t)| {
                    (
                        *id,
                        Vector3::new(t.position[0], t.position[1], t.position[2]),
                    )
                })
                .collect();

            let vel_map: HashMap<u32, Vector3<f64>> = telemetry
                .iter()
                .map(|(id, t)| {
                    (
                        *id,
                        Vector3::new(t.velocity[0], t.velocity[1], t.velocity[2]),
                    )
                })
                .collect();

            // GCBF+ path: O(n·k) neural barrier (when feature enabled and weights loaded).
            #[cfg(feature = "gcbf")]
            let gcbf_used = if let Some(ref gcbf) = self.gcbf_barrier {
                let desired_map: HashMap<u32, Vector3<f64>> = all_positions
                    .iter()
                    .map(|(id, _)| {
                        let base = vel_map.get(id).copied().unwrap_or_else(Vector3::zeros);
                        let adj = formation_corrections
                            .get(id)
                            .copied()
                            .unwrap_or_else(Vector3::zeros);
                        (*id, base + adj)
                    })
                    .collect();

                let (results, active) = gcbf.filter_all(
                    &all_positions,
                    &vel_map,
                    &desired_map,
                    &self.no_fly_zones,
                    fear.f,
                );
                cbf_active_constraints += active;

                for (drone_id, result) in &results {
                    if result.any_active {
                        let base_vel = vel_map
                            .get(drone_id)
                            .copied()
                            .unwrap_or_else(Vector3::zeros);
                        formation_corrections.insert(*drone_id, result.safe_velocity - base_vel);
                    }
                }
                true
            } else {
                false
            };
            #[cfg(not(feature = "gcbf"))]
            let gcbf_used = false;

            if !gcbf_used {
                Self::classical_cbf_pass(
                    &all_positions,
                    &vel_map,
                    &self.no_fly_zones,
                    &cbf_cfg,
                    formation_corrections,
                    &mut cbf_active_constraints,
                    &mut deadlock_escape_count,
                );
            }
        }

        (cbf_active_constraints, deadlock_escape_count)
    }

    /// Classical O(n²) pairwise CBF pass — extracted so both the
    /// `#[cfg(feature = "gcbf")]` fallback and the default path share one impl.
    fn classical_cbf_pass(
        all_positions: &[(u32, Vector3<f64>)],
        vel_map: &HashMap<u32, Vector3<f64>>,
        no_fly_zones: &[NoFlyZone],
        cbf_cfg: &CbfConfig,
        formation_corrections: &mut HashMap<u32, Vector3<f64>>,
        cbf_active_constraints: &mut u32,
        deadlock_escape_count: &mut usize,
    ) {
        let positions_only: Vec<Vector3<f64>> = all_positions.iter().map(|(_, pos)| *pos).collect();
        let velocities_only: Vec<Vector3<f64>> = all_positions
            .iter()
            .map(|(id, _)| vel_map.get(id).copied().unwrap_or_else(Vector3::zeros))
            .collect();

        let deadlock =
            cbf::detect_deadlock(&positions_only, &velocities_only, no_fly_zones, cbf_cfg);

        if deadlock.is_deadlocked {
            let cluster_positions: Vec<Vector3<f64>> = deadlock
                .involved_indices
                .iter()
                .map(|&idx| positions_only[idx])
                .collect();
            let escape_maneuvers = cbf::generate_escape_maneuvers(&cluster_positions, cbf_cfg);

            for (cluster_idx, escape) in escape_maneuvers.into_iter().enumerate() {
                if let Some(&global_idx) = deadlock.involved_indices.get(cluster_idx) {
                    if let Some((drone_id, _)) = all_positions.get(global_idx) {
                        let entry = formation_corrections
                            .entry(*drone_id)
                            .or_insert_with(Vector3::zeros);
                        let mut combined = *entry + escape;
                        let combined_norm = combined.norm();
                        if combined_norm > cbf_cfg.max_correction {
                            combined *= cbf_cfg.max_correction / combined_norm;
                        }
                        *entry = combined;
                    }
                }
            }

            *cbf_active_constraints += deadlock.involved_count as u32;
            *deadlock_escape_count += deadlock.involved_count;
        }

        let all_neighbors: Vec<NeighborState> = all_positions
            .iter()
            .map(|(id, pos)| NeighborState {
                position: *pos,
                velocity: vel_map.get(id).copied().unwrap_or_else(Vector3::zeros),
            })
            .collect();

        for (local_idx, (drone_id, drone_pos)) in all_positions.iter().enumerate() {
            let neighbors: Vec<NeighborState> = all_neighbors
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != local_idx)
                .map(|(_, n)| n.clone())
                .collect();

            let base_vel = vel_map
                .get(drone_id)
                .copied()
                .unwrap_or_else(Vector3::zeros);

            let formation_adj = formation_corrections
                .get(drone_id)
                .copied()
                .unwrap_or_else(Vector3::zeros);

            let desired_vel = base_vel + formation_adj;

            let result = cbf::cbf_filter_with_neighbor_states(
                drone_pos,
                &desired_vel,
                &neighbors,
                no_fly_zones,
                cbf_cfg,
            );

            if result.any_active {
                *cbf_active_constraints += result.active_count;
                let total_correction = result.safe_velocity - base_vel;
                formation_corrections.insert(*drone_id, total_correction);
            }
        }
    }

    /// One orchestration cycle: sense → think → act.
    pub fn tick(
        &mut self,
        telemetry: &[(u32, Telemetry)],
        tasks: &[Task],
        dt: f64,
    ) -> SwarmDecision {
        self.tick_count += 1;
        if !self.sim_time.is_finite() || self.sim_time < 0.0 {
            self.sim_time = 0.0;
        }
        let dt = convert::sanitize_dt(dt);
        self.sim_time += dt;
        if !self.sim_time.is_finite() {
            self.sim_time = 0.0;
        }

        let telemetry: Vec<(u32, Telemetry)> = telemetry
            .iter()
            .map(|(id, telem)| {
                let fallback_position = self
                    .prev_positions
                    .get(id)
                    .map(|position| [position.x, position.y, position.z])
                    .unwrap_or([0.0; 3]);
                (
                    *id,
                    convert::sanitize_telemetry(telem, fallback_position, self.sim_time),
                )
            })
            .collect();

        let fear = self.compute_fear_state(&telemetry);
        let fleet = self.update_navigation(&telemetry, &fear, dt);
        let regimes = self.detect_regimes_and_intent(&telemetry, &fear, &fleet, dt);

        // Accumulate traces from detect phase
        let mut traces_recorded = regimes.traces_recorded;

        // ── 2.5 Formation correction ──────────────────────────────────────
        // Compute desired formation positions and per-drone correction vectors.
        // Active only in PATROL/ENGAGE regimes — EVADE overrides formation hold.
        let mut formation_corrections: HashMap<u32, Vector3<f64>> = HashMap::new();
        let formation_quality;

        if let Some(formation) = self.formation_type {
            let n_formation_drones: Vec<(u32, Vector3<f64>)> = telemetry
                .iter()
                .filter(|(id, _)| {
                    let regime = self.regimes.get(id).copied().unwrap_or(Regime::Patrol);
                    regime != Regime::Evade
                })
                .map(|(id, t)| {
                    (
                        *id,
                        Vector3::new(t.position[0], t.position[1], t.position[2]),
                    )
                })
                .collect();

            if n_formation_drones.len() >= 2 {
                // Fear-adjusted formation: high fear → wider spacing, larger deadband.
                let fear_formation_config =
                    self.formation_config.fear_adjusted(fear.fear_threshold);

                let target_positions = formation::compute_formation_positions(
                    formation,
                    n_formation_drones.len(),
                    &fleet.centroid,
                    &fleet.heading,
                    &fear_formation_config,
                );

                // Compute correction for each drone in formation.
                for (slot_idx, (drone_id, drone_pos)) in n_formation_drones.iter().enumerate() {
                    if let Some((_, target_pos)) = target_positions.get(slot_idx) {
                        let correction = formation::formation_correction(
                            drone_pos,
                            target_pos,
                            &fear_formation_config,
                        );
                        if correction.norm() > 1e-6 {
                            formation_corrections.insert(*drone_id, correction);
                        }
                    }
                }

                // Measure formation quality.
                formation_quality = Some(formation::formation_quality(
                    &n_formation_drones,
                    &target_positions,
                    &fear_formation_config,
                ));
            } else {
                formation_quality = Some(1.0); // trivial formation with 0-1 drones
            }
        } else {
            formation_quality = None;
        }

        // ── 2.6 EVADE minimum separation repulsion ───────────────────────
        // Drones in EVADE regime that are too close to neighbours receive a
        // repulsive velocity correction. This fires earlier than CBF (which
        // only triggers at very close range) to close the all-EVADE gap.
        {
            let min_sep = self
                .cbf_config
                .as_ref()
                .map(|c| c.min_separation)
                .unwrap_or(10.0);
            let all_evade_positions: Vec<(u32, Vector3<f64>)> = telemetry
                .iter()
                .filter(|(id, _)| self.regimes.get(id).copied() == Some(Regime::Evade))
                .map(|(id, t)| {
                    (
                        *id,
                        Vector3::new(t.position[0], t.position[1], t.position[2]),
                    )
                })
                .collect();
            let all_positions_snap: Vec<(u32, Vector3<f64>)> = telemetry
                .iter()
                .map(|(id, t)| {
                    (
                        *id,
                        Vector3::new(t.position[0], t.position[1], t.position[2]),
                    )
                })
                .collect();
            for (drone_id, drone_pos) in &all_evade_positions {
                let mut repulsion = Vector3::zeros();
                for (other_id, other_pos) in &all_positions_snap {
                    if other_id == drone_id {
                        continue;
                    }
                    let diff = *drone_pos - *other_pos;
                    let dist = diff.norm();
                    if dist < min_sep && dist > 1e-6 {
                        // Repulsive force proportional to penetration depth.
                        let penetration = min_sep - dist;
                        repulsion += diff / dist * penetration;
                    }
                }
                if repulsion.norm() > 1e-6 {
                    let entry = formation_corrections
                        .entry(*drone_id)
                        .or_insert_with(Vector3::zeros);
                    *entry += repulsion;
                }
            }
        }

        // ── 3. Update threat trackers ────────────────────────────────────
        for tracker in self.threat_trackers.values_mut() {
            tracker.step(&fleet.centroid, &[], dt);
        }

        let (roe_denials, roe_escalations, auction_traces) =
            self.run_roe_and_auction(&telemetry, tasks, &fear, &fleet, &regimes);
        traces_recorded += auction_traces;

        let (cbf_active_constraints, deadlock_escape_count) =
            self.propagate_and_safety(&telemetry, &fear, &mut formation_corrections);

        // ── 7. Track fear signals + record outcome ──────────────────────
        self.last_intent_score = regimes.max_intent_score;
        self.last_cusum_breaks = regimes.cusum_break_count;

        #[cfg(feature = "temporal")]
        let temporal_anomalies: Vec<(String, i32, f64)> = self
            .temporal_managers
            .values()
            .flat_map(|tm| {
                tm.check_all_anomalies()
                    .into_iter()
                    .map(|(name, _is_break, dir, val)| (name, dir, val))
            })
            .collect();
        #[cfg(not(feature = "temporal"))]
        let temporal_anomalies: Vec<(String, i32, f64)> = Vec::new();

        #[cfg(feature = "phi-sim")]
        if let Some(adapter) = &mut self.fear_adapter {
            // Outcome = weighted survival rate + task assignment rate.
            let alive = self.nav_filters.len() as f64;
            let initial = (alive + self.loss_analyzer.total_losses() as f64).max(1.0);
            let survival = alive / initial;
            let assignment_rate = if !tasks.is_empty() {
                self.assignments.len() as f64 / tasks.len() as f64
            } else {
                1.0
            };
            adapter.record_outcome(survival * 0.6 + assignment_rate * 0.4);
        }

        // ── 8. Build decision ────────────────────────────────────────────
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

        // Collect per-drone fear levels and calibration from phi-sim.
        #[cfg(feature = "phi-sim")]
        let (per_drone_fear, calibration_quality) = if let Some(adapter) = &self.fear_adapter {
            let pdf: HashMap<u32, f64> = telemetry
                .iter()
                .map(|(id, _)| (*id, adapter.fear_level(*id)))
                .collect();
            let cal = adapter.calibration().overall();
            (pdf, cal)
        } else {
            (HashMap::new(), 0.0)
        };
        #[cfg(not(feature = "phi-sim"))]
        let (per_drone_fear, calibration_quality) = (HashMap::new(), 0.0);

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
            max_intent_score: regimes.max_intent_score,
            fear_level: fear.f,
            formation_quality,
            formation_corrections,
            roe_denials,
            roe_escalations,
            ew_active_threats: fear.ew_active_threats,
            cbf_active_constraints,
            deadlock_escape_count,
            courage_level: fear.collective_c,
            tension: fear.collective_t,
            per_drone_fear,
            calibration_quality,
            temporal_anomalies,
        }
    }

    // ── Helpers ──────────────────────────────────────────────────────────

    /// Record a ROE decision trace (shared by denial and escalation paths).
    fn record_roe_trace(
        &mut self,
        task_id: u32,
        threat_dist: f64,
        action: &str,
        reason: &str,
        confidence: f64,
    ) {
        let trace = DecisionTrace::new(self.sim_time, DecisionType::ThreatResponse)
            .with_inputs(TraceInputs {
                drone_ids: vec![],
                regime: "ROE".to_string(),
                metrics: serde_json::json!({
                    "task_id": task_id,
                    "threat_distance": threat_dist,
                }),
                context: serde_json::Value::Null,
                fear_level: None,
                courage_level: None,
                tension: None,
                calibration_quality: None,
            })
            .with_output(
                &format!("ROE {} task {}: {}", action, task_id, reason),
                serde_json::json!({"action": action}),
            )
            .with_confidence(confidence);
        self.tracer.record(trace);
    }

    /// Force all non-EVADE drones into EVADE regime, bypassing hysteresis.
    fn force_all_evade(&mut self, regime_changes: &mut Vec<(u32, Regime, Regime)>) {
        let mut drone_ids: Vec<u32> = self.regimes.keys().copied().collect();
        drone_ids.sort_unstable();
        for drone_id in drone_ids {
            if self.regimes.get(&drone_id).copied() != Some(Regime::Evade) {
                if let Some(gate) = self.hysteresis_gates.get_mut(&drone_id) {
                    gate.force_transition(Regime::Evade, self.sim_time);
                }
                let old = self
                    .regimes
                    .insert(drone_id, Regime::Evade)
                    .unwrap_or(Regime::Patrol);
                if old != Regime::Evade {
                    regime_changes.push((drone_id, old, Regime::Evade));
                }
            }
        }
    }

    fn nearest_threat_bearing(&self, drone_pos: &Vector3<f64>) -> Option<Vector3<f64>> {
        self.threat_trackers
            .values()
            .map(|t| {
                let (pos, _vel, _probs) = t.estimate_threat();
                (pos, (pos - drone_pos).norm())
            })
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(pos, _)| convert::threat_bearing(drone_pos, &pos))
    }

    fn nearest_threat_metrics(&self, drone_pos: &Vector3<f64>) -> Option<(f64, f64)> {
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
            .min_by(|a, b| a.0.total_cmp(&b.0))
    }

    /// Find nearest threat distance and regime relative to a point.
    fn nearest_threat_to_point(
        &self,
        point: &Vector3<f64>,
    ) -> Option<(f64, strix_core::ThreatRegime)> {
        self.threat_trackers
            .values()
            .map(|t| {
                let (pos, _vel, probs) = t.estimate_threat();
                let dist = (pos - point).norm();
                // Determine dominant regime from probabilities.
                let regime_idx = probs
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.total_cmp(b))
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                let regime = match regime_idx {
                    1 => strix_core::ThreatRegime::CounterAttack,
                    2 => strix_core::ThreatRegime::Retreat,
                    _ => strix_core::ThreatRegime::Defend,
                };
                (dist, regime)
            })
            .min_by(|a, b| a.0.total_cmp(&b.0))
    }

    fn estimate_collateral_risk(
        &self,
        point: &Vector3<f64>,
        telemetry: &[(u32, Telemetry)],
    ) -> f64 {
        let nearby_friendlies = self.count_friendlies_at_risk(point, telemetry) as f64;
        let friendly_pressure = if telemetry.is_empty() {
            0.0
        } else {
            (nearby_friendlies / telemetry.len() as f64).clamp(0.0, 1.0)
        };

        let no_fly_zone_pressure = self
            .config
            .no_fly_zones
            .iter()
            .map(|zone| {
                let dist = (zone.center - point).norm();
                if dist <= zone.radius {
                    1.0
                } else {
                    (1.0 - (dist - zone.radius) / 100.0).clamp(0.0, 1.0)
                }
            })
            .fold(0.0_f64, |max_risk: f64, risk| max_risk.max(risk));

        let threat_pressure = self
            .nearest_threat_to_point(point)
            .map(|(dist, _)| (1.0 - dist / 400.0).clamp(0.0, 1.0))
            .unwrap_or(0.0);

        (friendly_pressure * 0.45 + no_fly_zone_pressure * 0.40 + threat_pressure * 0.15)
            .clamp(0.0, 1.0)
    }

    fn count_friendlies_at_risk(
        &self,
        point: &Vector3<f64>,
        telemetry: &[(u32, Telemetry)],
    ) -> u32 {
        let risk_radius = self
            .config
            .cbf_config
            .as_ref()
            .map(|cfg| (cfg.min_separation * 20.0).max(75.0))
            .unwrap_or(100.0);

        telemetry
            .iter()
            .filter(|(_, telem)| {
                let drone_pos =
                    Vector3::new(telem.position[0], telem.position[1], telem.position[2]);
                (drone_pos - point).norm() <= risk_radius
            })
            .count() as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use strix_adapters::traits::{FlightMode, GpsFix};

    fn invalid_telemetry() -> Telemetry {
        Telemetry {
            position: [f64::NAN, 25.0, f64::INFINITY],
            velocity: [f64::NEG_INFINITY, 4.0, f64::NAN],
            attitude: [f64::NAN, f64::NEG_INFINITY, f64::NAN],
            battery: f64::NAN,
            gps_fix: GpsFix::Fix3D,
            armed: true,
            mode: FlightMode::Guided,
            timestamp: f64::NAN,
        }
    }

    #[test]
    fn tick_sanitizes_invalid_dt_and_telemetry_ingress() {
        let mut orchestrator = SwarmOrchestrator::new(&[1], SwarmConfig::default());
        let telemetry = vec![(1, invalid_telemetry())];
        let tasks: Vec<Task> = Vec::new();

        let decision = orchestrator.tick(&telemetry, &tasks, f64::NAN);

        assert_eq!(orchestrator.sim_time(), 0.0);
        let position = decision.positions.get(&1).unwrap();
        assert!(position.x.is_finite());
        assert!(position.y.is_finite());
        assert!(position.z.is_finite());

        let self_state = orchestrator.gossip.known_states.get(&NodeId(1)).unwrap();
        assert_eq!(self_state.battery, 0.0);
        assert!(self_state.timestamp.is_finite());
    }

    #[test]
    fn force_all_evade_runs_in_sorted_drone_order() {
        let mut orchestrator = SwarmOrchestrator::new(&[3, 1, 2], SwarmConfig::default());
        let mut changes = Vec::new();

        orchestrator.force_all_evade(&mut changes);

        let changed_ids: Vec<u32> = changes.into_iter().map(|(id, _, _)| id).collect();
        assert_eq!(changed_ids, vec![1, 2, 3]);
    }
}
