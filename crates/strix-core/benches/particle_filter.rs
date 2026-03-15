use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use nalgebra::Vector3;
use strix_core::particle_nav::ParticleNavFilter;
use strix_core::Observation;

fn bench_particle_filter_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("particle_filter");

    for n_particles in [50, 100, 200, 500, 1000] {
        group.bench_with_input(
            BenchmarkId::new("step", n_particles),
            &n_particles,
            |b, &n| {
                let mut pf = ParticleNavFilter::new(n, Vector3::new(100.0, 200.0, 50.0));
                let obs = vec![
                    Observation::Barometer {
                        altitude: 50.0,
                        timestamp: 0.1,
                    },
                    Observation::Imu {
                        acceleration: Vector3::new(0.1, 0.2, -9.8),
                        gyro: None,
                        timestamp: 0.1,
                    },
                ];
                let bearing = Vector3::new(1.0, 0.0, 0.0);

                b.iter(|| {
                    black_box(pf.step(
                        black_box(&obs),
                        black_box(&bearing),
                        black_box(1.0),
                        black_box(0.1),
                    ))
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_particle_filter_step);
criterion_main!(benches);
