//! strix-optimize — Multi-objective SMCO parameter search for STRIX drone swarm.

use std::path::Path;
use std::time::Instant;

use clap::Parser;
use rayon::prelude::*;

use strix_optimizer::{
    evaluator::Evaluator,
    param_space::{strix_full, ParamVec},
    pareto::ParetoArchive,
    report::{OptimizationReport, HV_REFERENCE},
    smco::{SmcoConfig, SmcoOptimizer},
};

#[derive(Parser, Debug)]
#[command(
    name = "strix-optimize",
    about = "Multi-objective SMCO optimizer for STRIX"
)]
struct Cli {
    /// Number of SMCO outer iterations
    #[arg(long, default_value_t = 500)]
    iterations: usize,
    /// Candidate population size per iteration
    #[arg(long, default_value_t = 20)]
    population: usize,
    /// Maximum Pareto archive size
    #[arg(long, default_value_t = 100)]
    archive_size: usize,
    /// Output JSON file path
    #[arg(long, default_value = "optimization_results.json")]
    output: String,
    /// RNG seed
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// Perturbation step size (fraction of param range)
    #[arg(long, default_value_t = 0.1)]
    sigma: f64,
    /// Simulation timestep (seconds)
    #[arg(long, default_value_t = 0.1)]
    dt: f64,
}

fn main() {
    let cli = Cli::parse();

    println!(
        "strix-optimize v0.1.0 | iters={} pop={} archive={} seed={}",
        cli.iterations, cli.population, cli.archive_size, cli.seed
    );

    let eval_space = strix_full();
    let evaluator = Evaluator::default_scenarios();
    let t0 = Instant::now();

    let smco_config = SmcoConfig {
        max_iterations: cli.iterations,
        population_size: cli.population,
        step_size: cli.sigma,
        archive_size: cli.archive_size,
        seed: cli.seed,
        ..SmcoConfig::default()
    };
    let archive = ParetoArchive::new(cli.archive_size);
    let mut optimizer = SmcoOptimizer::new(smco_config, strix_full(), archive);
    let mut hv_history: Vec<(usize, f64)> = Vec::with_capacity(cli.iterations);
    let mut total_evaluated: usize = 0;

    println!("Running {} SMCO iterations...", cli.iterations);

    while !optimizer.is_done() {
        let candidates = optimizer.generate_candidates();

        let results: Vec<(ParamVec, [f64; 3])> = candidates
            .into_par_iter()
            .map(|params| {
                let objs = evaluator.evaluate(&eval_space, &params);
                (params, objs)
            })
            .collect();

        total_evaluated += results.len();
        optimizer.report_results(results);

        let iter = optimizer.iteration();
        let hv = optimizer.hypervolume(HV_REFERENCE);
        hv_history.push((iter, hv));

        if iter % 10 == 0 || iter == 1 {
            println!(
                "  iter {:>4}/{}: archive={} hv={:.6} step={:.4} evaluated={}",
                iter,
                cli.iterations,
                optimizer.archive().len(),
                hv,
                optimizer.current_step(),
                total_evaluated
            );
        }
    }

    let elapsed = t0.elapsed().as_secs_f64();
    let report =
        OptimizationReport::from_archive(optimizer.archive(), hv_history, total_evaluated, elapsed);

    report.print_summary();

    if !cli.output.is_empty() {
        if let Err(e) = report.to_json(Path::new(&cli.output)) {
            eprintln!("Failed to write output: {}", e);
        } else {
            println!("Report written to: {}", cli.output);
        }
    }
}
