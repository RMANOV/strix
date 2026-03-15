use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::collections::HashMap;
use strix_auction::{Auctioneer, Capabilities, DroneState, Position, Regime, Task, ThreatState};

fn make_drones(n: usize) -> Vec<DroneState> {
    (0..n)
        .map(|i| DroneState {
            id: i as u32,
            position: Position::new(i as f64 * 10.0, (i % 5) as f64 * 10.0, 50.0),
            velocity: [1.0, 0.0, 0.0],
            regime: Regime::Patrol,
            capabilities: Capabilities {
                has_sensor: true,
                has_weapon: i % 3 == 0,
                has_ew: i % 5 == 0,
                has_relay: i % 4 == 0,
            },
            energy: 0.8,
            alive: true,
        })
        .collect()
}

fn make_tasks(n: usize) -> Vec<Task> {
    (0..n)
        .map(|i| Task {
            id: i as u32,
            location: Position::new(50.0 + i as f64 * 5.0, 50.0 + (i % 3) as f64 * 5.0, 50.0),
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

fn bench_auction_round(c: &mut Criterion) {
    let mut group = c.benchmark_group("auction");

    for (n_drones, n_tasks) in [(5, 3), (10, 5), (20, 10), (50, 20)] {
        let label = format!("{}d_{}t", n_drones, n_tasks);
        group.bench_function(&label, |b| {
            let drones = make_drones(n_drones);
            let tasks = make_tasks(n_tasks);
            let threats: Vec<ThreatState> = Vec::new();
            let sub_swarms: HashMap<u32, u32> = HashMap::new();
            let kill_zones: Vec<(Position, f64, f64)> = Vec::new();

            b.iter(|| {
                let mut auctioneer = Auctioneer::new();
                black_box(auctioneer.run_auction(
                    black_box(&drones),
                    black_box(&tasks),
                    black_box(&threats),
                    black_box(&sub_swarms),
                    black_box(&kill_zones),
                ))
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_auction_round);
criterion_main!(benches);
