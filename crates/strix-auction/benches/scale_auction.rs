use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::collections::HashMap;
use strix_auction::{Auctioneer, Capabilities, DroneState, Position, Regime, Task, ThreatState};

fn make_drones(n: usize) -> Vec<DroneState> {
    (0..n)
        .map(|i| DroneState {
            id: i as u32,
            position: Position::new(i as f64 * 10.0, (i % 10) as f64 * 10.0, 50.0),
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
            location: Position::new(50.0 + i as f64 * 5.0, 50.0 + (i % 5) as f64 * 5.0, 50.0),
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

fn bench_scale_auction(c: &mut Criterion) {
    let mut group = c.benchmark_group("scale_auction");

    // For each drone count, vary task count: N/2, N, 2*N
    for n_drones in [10usize, 50, 100] {
        for task_multiplier in [0.5f64, 1.0, 2.0] {
            let n_tasks = ((n_drones as f64 * task_multiplier) as usize).max(1);
            let label = format!("{}d_{}t", n_drones, n_tasks);

            group.bench_with_input(
                BenchmarkId::new("auction_round", &label),
                &(n_drones, n_tasks),
                |b, &(nd, nt)| {
                    let drones = make_drones(nd);
                    let tasks = make_tasks(nt);
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
                },
            );
        }
    }

    group.finish();
}

criterion_group!(benches, bench_scale_auction);
criterion_main!(benches);
