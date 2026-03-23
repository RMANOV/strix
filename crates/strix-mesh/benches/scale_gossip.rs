use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use strix_mesh::{
    gossip::{GossipEngine, GossipMessage},
    NodeId, Position3D,
};

/// Build a fully-connected gossip network with `n` nodes.
fn make_network(n: usize, fanout: usize) -> Vec<GossipEngine> {
    let mut engines: Vec<GossipEngine> = (0..n)
        .map(|i| GossipEngine::new(NodeId(i as u32), fanout))
        .collect();

    #[allow(clippy::needless_range_loop)]
    for i in 0..n {
        for j in 0..n {
            if i != j {
                engines[i].add_peer(NodeId(j as u32));
            }
        }
    }

    for (i, engine) in engines.iter_mut().enumerate() {
        let pos = Position3D([i as f64 * 10.0, (i % 5) as f64 * 10.0, 50.0]);
        engine.update_self_state(pos, 0.85, format!("drone_{i}"), 0.0);
    }

    engines
}

/// Simulate gossip rounds. Returns nothing — side effects on engines.
fn run_gossip_rounds(engines: &mut [GossipEngine], n_rounds: usize) {
    let n = engines.len();
    for _round in 0..n_rounds {
        let digests: Vec<GossipMessage> = engines.iter().map(|e| e.build_digest()).collect();

        let mut exchanges: Vec<(usize, GossipMessage)> = Vec::new();
        for (sender_idx, digest) in digests.iter().enumerate() {
            #[allow(clippy::needless_range_loop)]
            for receiver_idx in 0..n {
                if receiver_idx != sender_idx {
                    if let Some(exchange) = engines[receiver_idx].respond_to_digest(digest) {
                        exchanges.push((sender_idx, exchange));
                    }
                }
            }
        }

        for (target_idx, msg) in exchanges {
            engines[target_idx].merge_state(&msg);
        }
    }
}

fn count_converged(engines: &[GossipEngine], n: usize) -> usize {
    engines.iter().filter(|e| e.known_states.len() >= n).count()
}

fn bench_gossip_round(c: &mut Criterion) {
    let mut group = c.benchmark_group("gossip_round");

    for n in [10usize, 50, 100] {
        let fanout = (n as f64).log2().ceil() as usize;
        group.bench_with_input(BenchmarkId::new("single_round", n), &n, |b, &n| {
            b.iter(|| {
                let mut engines = make_network(n, fanout);
                run_gossip_rounds(&mut engines, 1);
                black_box(&engines);
            });
        });
    }

    group.finish();
}

fn bench_gossip_convergence(c: &mut Criterion) {
    let mut group = c.benchmark_group("gossip_convergence");
    group.sample_size(20);

    for n in [10usize, 50, 100] {
        let fanout = (n as f64).log2().ceil() as usize;
        group.bench_with_input(BenchmarkId::new("full_convergence", n), &n, |b, &n| {
            b.iter(|| {
                let mut engines = make_network(n, fanout);
                let max_rounds = 20;
                for round in 0..max_rounds {
                    run_gossip_rounds(&mut engines, 1);
                    if count_converged(&engines, n) == n {
                        return black_box(round + 1);
                    }
                }
                black_box(max_rounds)
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_gossip_round, bench_gossip_convergence);
criterion_main!(benches);
