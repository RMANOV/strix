use criterion::{black_box, criterion_group, criterion_main, Criterion};
use strix_adapters::traits::{FlightMode, GpsFix, Telemetry};
use strix_auction::{Capabilities, Position, Task};
use strix_swarm::{SwarmConfig, SwarmOrchestrator};

fn make_telemetry(n: usize) -> Vec<(u32, Telemetry)> {
    (0..n)
        .map(|i| {
            let id = i as u32;
            let telem = Telemetry {
                position: [i as f64 * 10.0, (i % 5) as f64 * 10.0, 50.0],
                velocity: [1.0, 0.0, 0.0],
                attitude: [0.0, 0.0, 0.0],
                battery: 0.85,
                gps_fix: GpsFix::Fix3D,
                armed: true,
                mode: FlightMode::Guided,
                timestamp: 0.1,
            };
            (id, telem)
        })
        .collect()
}

fn make_tasks(n: usize) -> Vec<Task> {
    (0..n)
        .map(|i| Task {
            id: i as u32,
            location: Position::new(10.0 * i as f64, 10.0, 50.0),
            required_capabilities: Capabilities {
                has_sensor: true,
                has_weapon: false,
                has_ew: false,
                has_relay: false,
            },
            priority: 0.7,
            urgency: 0.5,
            bundle_id: None,
            dark_pool: None,
        })
        .collect()
}

fn bench_swarm_tick(c: &mut Criterion) {
    let mut group = c.benchmark_group("swarm_tick");

    for n_drones in [5, 10, 20, 50, 100] {
        let label = format!("{}_drones", n_drones);
        group.bench_function(&label, |b| {
            let ids: Vec<u32> = (0..n_drones as u32).collect();
            let config = SwarmConfig {
                n_particles: 50,
                n_threat_particles: 30,
                auction_interval: 5,
                ..Default::default()
            };
            let mut orch = SwarmOrchestrator::new(&ids, config);

            let tasks = make_tasks(5);
            let telem = make_telemetry(n_drones);

            b.iter(|| black_box(orch.tick(black_box(&telem), black_box(&tasks), black_box(0.1))));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_swarm_tick);
criterion_main!(benches);
