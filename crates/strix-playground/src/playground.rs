use strix_adapters::simulator::{SimulatorConfig, SimulatorFleet};
use strix_adapters::traits::PlatformAdapter;
use strix_auction::{Capabilities, Position, Task};
use strix_core::cbf::{CbfConfig, NoFlyZone};
use strix_swarm::{SwarmConfig, SwarmOrchestrator};

use nalgebra::Vector3;

use crate::engine::{Engine, EngineConfig};
use crate::scenario::{Event, ScheduledEvent, ThreatSpec};
use crate::threats::ThreatActor;

// ---------------------------------------------------------------------------
// Playground builder
// ---------------------------------------------------------------------------

/// Fluent builder for constructing and running battlefield simulations.
///
/// # Example
/// ```ignore
/// let report = Playground::new()
///     .name("Quick test")
///     .drones(10)
///     .threats(vec![ThreatSpec::approaching(400.0, 8.0)])
///     .cbf(CbfConfig::default())
///     .run_for(60.0)
///     .run();
/// println!("{}", report);
/// ```
pub struct Playground {
    name: String,
    n_drones: usize,
    formation_spacing: f64,
    altitude: f64,
    initial_threats: Vec<ThreatSpec>,
    events: Vec<ScheduledEvent>,
    wind: [f64; 3],
    cbf_config: Option<CbfConfig>,
    no_fly_zones: Vec<NoFlyZone>,
    tasks: Vec<Task>,
    swarm_config: SwarmConfig,
    sim_config: SimulatorConfig,
    duration_secs: f64,
    dt: f64,
    record_json: bool,
}

impl Playground {
    /// Create a playground with sensible defaults.
    pub fn new() -> Self {
        Self {
            name: "Unnamed".into(),
            n_drones: 10,
            formation_spacing: 15.0,
            altitude: -50.0,
            initial_threats: Vec::new(),
            events: Vec::new(),
            wind: [0.0, 0.0, 0.0],
            cbf_config: None,
            no_fly_zones: Vec::new(),
            tasks: Vec::new(),
            swarm_config: SwarmConfig::default(),
            sim_config: SimulatorConfig::default(),
            duration_secs: 60.0,
            dt: 0.1,
            record_json: false,
        }
    }

    // ── Fleet configuration ─────────────────────────────────────────────

    pub fn name(mut self, n: &str) -> Self {
        self.name = n.into();
        self
    }

    pub fn drones(mut self, n: usize) -> Self {
        self.n_drones = n;
        self
    }

    pub fn spacing(mut self, m: f64) -> Self {
        self.formation_spacing = m;
        self
    }

    pub fn altitude(mut self, m: f64) -> Self {
        self.altitude = m;
        self
    }

    // ── Threats ─────────────────────────────────────────────────────────

    pub fn threats(mut self, t: Vec<ThreatSpec>) -> Self {
        self.initial_threats = t;
        self
    }

    pub fn add_threat(mut self, t: ThreatSpec) -> Self {
        self.initial_threats.push(t);
        self
    }

    // ── Environment ─────────────────────────────────────────────────────

    pub fn wind(mut self, w: [f64; 3]) -> Self {
        self.wind = w;
        self
    }

    pub fn cbf(mut self, c: CbfConfig) -> Self {
        self.cbf_config = Some(c);
        self
    }

    pub fn nfz(mut self, center: [f64; 3], radius: f64) -> Self {
        self.no_fly_zones.push(NoFlyZone {
            center: Vector3::new(center[0], center[1], center[2]),
            radius,
        });
        self
    }

    // ── Scheduled events ────────────────────────────────────────────────

    pub fn jam_at_sec(mut self, t: f64) -> Self {
        self.events.push(ScheduledEvent {
            time_secs: t,
            event: Event::JamGps {
                noise_multiplier: 10.0,
            },
        });
        self
    }

    pub fn restore_gps_at(mut self, t: f64) -> Self {
        self.events.push(ScheduledEvent {
            time_secs: t,
            event: Event::RestoreGps,
        });
        self
    }

    pub fn lose_drone_at(mut self, t: f64, id: u32) -> Self {
        self.events.push(ScheduledEvent {
            time_secs: t,
            event: Event::LoseDrone { drone_id: id },
        });
        self
    }

    pub fn spawn_threat_at(mut self, t: f64, spec: ThreatSpec) -> Self {
        self.events.push(ScheduledEvent {
            time_secs: t,
            event: Event::SpawnThreat(spec),
        });
        self
    }

    pub fn wind_change_at(mut self, t: f64, w: [f64; 3]) -> Self {
        self.events.push(ScheduledEvent {
            time_secs: t,
            event: Event::WindChange(w),
        });
        self
    }

    pub fn nfz_at(mut self, t: f64, center: [f64; 3], radius: f64) -> Self {
        self.events.push(ScheduledEvent {
            time_secs: t,
            event: Event::AddNfz { center, radius },
        });
        self
    }

    pub fn event(mut self, t: f64, e: Event) -> Self {
        self.events.push(ScheduledEvent {
            time_secs: t,
            event: e,
        });
        self
    }

    // ── Config overrides ────────────────────────────────────────────────

    pub fn swarm_config(mut self, c: SwarmConfig) -> Self {
        self.swarm_config = c;
        self
    }

    pub fn sim_config(mut self, c: SimulatorConfig) -> Self {
        self.sim_config = c;
        self
    }

    pub fn dt(mut self, dt: f64) -> Self {
        self.dt = dt;
        self
    }

    pub fn with_json(mut self) -> Self {
        self.record_json = true;
        self
    }

    // ── Tasks ───────────────────────────────────────────────────────────

    pub fn tasks(mut self, t: Vec<Task>) -> Self {
        self.tasks = t;
        self
    }

    // ── Execute ─────────────────────────────────────────────────────────

    /// Set the simulation duration and build the Engine.
    pub fn run_for(mut self, secs: f64) -> Engine {
        self.duration_secs = secs;
        self.build_engine()
    }

    // ── Internal ────────────────────────────────────────────────────────

    fn build_engine(self) -> Engine {
        // Destructure to avoid partial-move issues
        let Playground {
            name,
            n_drones,
            formation_spacing,
            altitude,
            initial_threats,
            events,
            wind,
            cbf_config,
            no_fly_zones,
            tasks: user_tasks,
            swarm_config,
            mut sim_config,
            duration_secs,
            dt,
            record_json,
        } = self;

        sim_config.wind = wind;
        sim_config.dt = dt;

        // Create fleet in grid formation
        let mut fleet = SimulatorFleet::new_grid(n_drones, formation_spacing, sim_config);

        // Apply CBF if configured
        if let Some(cbf) = cbf_config {
            fleet = fleet.with_cbf(cbf);
        }

        // Apply NFZs
        for nfz in &no_fly_zones {
            fleet.add_no_fly_zone(nfz.clone());
        }

        // Build drone IDs
        let drone_ids: Vec<u32> = fleet.drones.iter().map(|d| d.id()).collect();

        // Compute fleet centroid for threat placement
        let centroid = {
            let mut sum = Vector3::zeros();
            for drone in &fleet.drones {
                if let Ok(t) = drone.get_telemetry() {
                    sum += Vector3::new(t.position[0], t.position[1], t.position[2]);
                }
            }
            if !fleet.drones.is_empty() {
                sum / fleet.drones.len() as f64
            } else {
                Vector3::zeros()
            }
        };

        // Spawn threat actors relative to fleet centroid
        let threats: Vec<ThreatActor> = initial_threats
            .iter()
            .enumerate()
            .map(|(i, spec)| ThreatActor::from_spec(i as u32, spec, &centroid))
            .collect();

        // Auto-generate patrol tasks if none specified
        let tasks = if user_tasks.is_empty() {
            generate_patrol_tasks(n_drones, formation_spacing, altitude, &drone_ids, &centroid)
        } else {
            user_tasks
        };

        // Create orchestrator
        let orchestrator = SwarmOrchestrator::new(&drone_ids, swarm_config);

        let engine_config = EngineConfig {
            duration: duration_secs,
            dt,
            record_snapshots: record_json,
        };

        Engine::new(
            fleet,
            orchestrator,
            threats,
            events,
            tasks,
            engine_config,
            name,
        )
    }
}

/// Generate patrol waypoints in a grid pattern around the centroid.
fn generate_patrol_tasks(
    _n_drones: usize,
    formation_spacing: f64,
    altitude: f64,
    drone_ids: &[u32],
    centroid: &Vector3<f64>,
) -> Vec<Task> {
    let n = drone_ids.len();
    let cols = (n as f64).sqrt().ceil() as usize;
    let spacing = formation_spacing * 3.0;

    (0..n)
        .map(|i| {
            let row = i / cols;
            let col = i % cols;
            let x = centroid.x + (col as f64 - cols as f64 / 2.0) * spacing;
            let y = centroid.y + (row as f64 - cols as f64 / 2.0) * spacing;

            Task {
                id: (i + 1) as u32,
                location: Position::new(x, y, altitude),
                required_capabilities: Capabilities::default(),
                priority: 0.5,
                urgency: 0.3,
                bundle_id: None,
                dark_pool: None,
            }
        })
        .collect()
}

impl Default for Playground {
    fn default() -> Self {
        Self::new()
    }
}
