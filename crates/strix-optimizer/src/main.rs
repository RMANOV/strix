//! strix-optimize — Multi-objective SMCO parameter search for STRIX drone swarm.

use std::path::Path;
use std::time::Instant;

use clap::Parser;
use rayon::prelude::*;

use strix_optimizer::{
    contextual_archive::{ContextualArchive, ContextualArchiveConfig, OptimizationContext},
    evaluator::{DoctrineProfile, Evaluator, ScenarioSpec},
    graph_surrogate::{GraphEdge, GraphNode, GraphSnapshot, GraphSurrogate, ThreatCoupling},
    heterogeneity::{
        decode_heterogeneous_policy, strix_heterogeneous, Echelon, HeterogeneousPolicy, Role,
    },
    param_space::ParamVec,
    pareto::{ParetoArchive, ParetoSolution},
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
    /// Doctrine profile for objective shaping.
    #[arg(long, default_value = "balanced")]
    doctrine: String,
    /// Blend weight for the offline graph surrogate.
    #[arg(long, default_value_t = 0.18)]
    graph_surrogate_weight: f64,
}

fn main() {
    let cli = Cli::parse();

    let doctrine = cli
        .doctrine
        .parse::<DoctrineProfile>()
        .unwrap_or_else(|err| {
            eprintln!("Invalid doctrine '{}': {}", cli.doctrine, err);
            std::process::exit(2);
        });

    let graph_weight = cli.graph_surrogate_weight.clamp(0.0, 0.5);

    println!(
        "strix-optimize v0.1.0 | iters={} pop={} archive={} seed={} doctrine={} graph_weight={:.2}",
        cli.iterations,
        cli.population,
        cli.archive_size,
        cli.seed,
        doctrine.as_str(),
        graph_weight
    );

    let eval_space = strix_heterogeneous();
    let evaluator = Evaluator::default_scenarios().with_doctrine(doctrine);
    let surrogate = GraphSurrogate::default();
    let mut contextual_archive = ContextualArchive::new(ContextualArchiveConfig::default());
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
    let mut optimizer = SmcoOptimizer::new(smco_config, eval_space.clone(), archive);
    let mut hv_history: Vec<(usize, f64)> = Vec::with_capacity(cli.iterations);
    let mut total_evaluated: usize = 0;

    println!("Running {} SMCO iterations...", cli.iterations);

    while !optimizer.is_done() {
        let eval_iteration = optimizer.iteration();
        let candidates = optimizer.generate_candidates();

        let results: Vec<(ParamVec, [f64; 3], Vec<([f64; 3], OptimizationContext)>)> =
            candidates
                .into_par_iter()
                .map(|params| {
                    let detailed = evaluator.evaluate_detailed(&eval_space, &params);
                    let aggregate = aggregate_scores(&evaluator.scenarios, &detailed);
                    let policy = decode_heterogeneous_policy(&params);
                    let surrogate_scores = surrogate.score(&build_graph_snapshot(&policy));
                    let objectives = blend_scores(
                        aggregate,
                        surrogate_scores,
                        heterogeneity_bonus(&policy),
                        graph_weight,
                    );
                    let contextual = evaluator
                        .scenarios
                        .iter()
                        .zip(detailed.iter())
                        .map(|(scenario, scores)| {
                            (*scores, context_for_scenario(doctrine, scenario.name.as_str()))
                        })
                        .collect();
                    (params, objectives, contextual)
                })
                .collect();

        for (params, _objectives, contextual_entries) in &results {
            for (scenario_scores, context) in contextual_entries {
                let _ = contextual_archive.insert(
                    context.clone(),
                    ParetoSolution::new(
                        params.clone(),
                        *scenario_scores,
                        vec![*scenario_scores],
                        eval_iteration,
                    ),
                );
            }
        }

        total_evaluated += results.len();
        optimizer.report_results(
            results
                .iter()
                .map(|(params, objectives, _)| (params.clone(), *objectives))
                .collect(),
        );

        let iter = optimizer.iteration();
        let removed = contextual_archive.forget_stale(iter);
        if iter.is_multiple_of(20) {
            migrate_contextual_elites(&mut contextual_archive);
        }

        let hv = optimizer.hypervolume(HV_REFERENCE);
        hv_history.push((iter, hv));

        if iter.is_multiple_of(10) || iter == 1 {
            println!(
                "  iter {:>4}/{}: archive={} hv={:.6} step={:.4} evaluated={} contexts={} trimmed={}",
                iter,
                cli.iterations,
                optimizer.archive().len(),
                hv,
                optimizer.current_step(),
                total_evaluated,
                contextual_archive.contexts().count(),
                removed
            );
        }
    }

    let elapsed = t0.elapsed().as_secs_f64();
    let report = OptimizationReport::from_archive(
        optimizer.archive(),
        doctrine,
        evaluator.objective_labels(),
        hv_history,
        total_evaluated,
        elapsed,
    )
    .attach_contextual_fronts(&contextual_archive);

    report.print_summary();

    if !cli.output.is_empty() {
        if let Err(e) = report.to_json(Path::new(&cli.output)) {
            eprintln!("Failed to write output: {}", e);
        } else {
            println!("Report written to: {}", cli.output);
        }
    }
}

fn aggregate_scores(scenarios: &[ScenarioSpec], scores: &[[f64; 3]]) -> [f64; 3] {
    let total_weight = scenarios
        .iter()
        .map(|scenario| scenario.weight)
        .sum::<f64>()
        .max(1e-9);
    let mut aggregate = [0.0; 3];
    for (scenario, score) in scenarios.iter().zip(scores.iter()) {
        let weight = scenario.weight / total_weight;
        aggregate[0] += score[0] * weight;
        aggregate[1] += score[1] * weight;
        aggregate[2] += score[2] * weight;
    }
    aggregate
}

fn blend_scores(
    base: [f64; 3],
    surrogate: [f64; 3],
    heterogeneity: [f64; 3],
    graph_weight: f64,
) -> [f64; 3] {
    let graph_weight = graph_weight.clamp(0.0, 0.5);
    [
        clamp01(base[0] * (1.0 - graph_weight) + surrogate[0] * graph_weight + heterogeneity[0]),
        clamp01(base[1] * (1.0 - graph_weight) + surrogate[1] * graph_weight + heterogeneity[1]),
        clamp01(base[2] * (1.0 - graph_weight) + surrogate[2] * graph_weight + heterogeneity[2]),
    ]
}

fn heterogeneity_bonus(policy: &HeterogeneousPolicy) -> [f64; 3] {
    let exploration_span = span(policy.gains.iter().map(|gain| gain.exploration_gain));
    let coordination_span = span(policy.gains.iter().map(|gain| gain.coordination_gain));
    let relay_platoon = policy.gain_for(Role::Relay, Echelon::Platoon).unwrap();
    let strike_squad = policy.gain_for(Role::Strike, Echelon::Squad).unwrap();
    let decoy_pair = policy.gain_for(Role::Decoy, Echelon::Pair).unwrap();

    [
        clamp_bonus((relay_platoon.relay_weight - 1.0) * 0.05 + coordination_span * 0.03),
        clamp_bonus(exploration_span * 0.05 + coordination_span * 0.03),
        clamp_bonus(
            (strike_squad.strike_weight - 1.0) * 0.05
                + (decoy_pair.deception_weight - 1.0) * 0.02,
        ),
    ]
}

fn build_graph_snapshot(policy: &HeterogeneousPolicy) -> GraphSnapshot {
    let scout_pair = policy.gain_for(Role::Scout, Echelon::Pair).unwrap();
    let relay_squad = policy.gain_for(Role::Relay, Echelon::Squad).unwrap();
    let relay_platoon = policy.gain_for(Role::Relay, Echelon::Platoon).unwrap();
    let strike_squad = policy.gain_for(Role::Strike, Echelon::Squad).unwrap();
    let decoy_pair = policy.gain_for(Role::Decoy, Echelon::Pair).unwrap();

    GraphSnapshot {
        nodes: vec![
            GraphNode {
                id: 1,
                role: Role::Scout.as_str().to_string(),
            },
            GraphNode {
                id: 2,
                role: Role::Relay.as_str().to_string(),
            },
            GraphNode {
                id: 3,
                role: Role::Strike.as_str().to_string(),
            },
            GraphNode {
                id: 4,
                role: Role::Decoy.as_str().to_string(),
            },
        ],
        edges: vec![
            GraphEdge {
                src: 1,
                dst: 2,
                weight: scout_pair.coordination_gain,
                latency: latency_from_gain(relay_platoon.relay_weight),
            },
            GraphEdge {
                src: 2,
                dst: 3,
                weight: relay_squad.coordination_gain,
                latency: latency_from_gain(strike_squad.strike_weight),
            },
            GraphEdge {
                src: 1,
                dst: 4,
                weight: scout_pair.exploration_gain,
                latency: latency_from_gain(decoy_pair.deception_weight),
            },
            GraphEdge {
                src: 2,
                dst: 4,
                weight: relay_platoon.relay_weight,
                latency: latency_from_gain(decoy_pair.coordination_gain),
            },
            GraphEdge {
                src: 4,
                dst: 3,
                weight: decoy_pair.deception_weight,
                latency: latency_from_gain(strike_squad.strike_weight),
            },
        ],
        threat_couplings: vec![
            ThreatCoupling {
                src: 1,
                dst: 3,
                pressure: clamp01(
                    (strike_squad.strike_weight - relay_platoon.relay_weight).abs() * 0.4,
                ),
            },
            ThreatCoupling {
                src: 4,
                dst: 3,
                pressure: clamp01(decoy_pair.deception_weight * 0.25),
            },
        ],
    }
}

fn context_for_scenario(doctrine: DoctrineProfile, scenario_name: &str) -> OptimizationContext {
    let (environment, regime) = match scenario_name {
        "ambush" => ("contested", "engage"),
        "gps" | "gps_denied" => ("degraded_comms", "patrol"),
        "attrition" => ("attrition", "evade"),
        "stress" | "stress_test" => ("stress", "mixed"),
        _ => ("default", "patrol"),
    };

    OptimizationContext {
        doctrine: doctrine.as_str().to_string(),
        scenario_family: scenario_name.to_string(),
        environment: environment.to_string(),
        regime: regime.to_string(),
    }
}

fn migrate_contextual_elites(archive: &mut ContextualArchive) {
    let mut ranked: Vec<(OptimizationContext, usize)> = archive
        .contexts()
        .map(|(context, front)| (context.clone(), front.len()))
        .collect();
    ranked.sort_by(|left, right| {
        right
            .1
            .cmp(&left.1)
            .then_with(|| left.0.key().cmp(&right.0.key()))
    });

    if let Some((source, _)) = ranked.first().cloned() {
        for (target, _) in ranked.iter().skip(1).take(3) {
            let _ = archive.migrate_elites(&source, target);
        }
    }
}

fn span<I>(values: I) -> f64
where
    I: Iterator<Item = f64>,
{
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for value in values {
        min = min.min(value);
        max = max.max(value);
    }
    if min.is_finite() && max.is_finite() {
        (max - min).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn latency_from_gain(gain: f64) -> f64 {
    (1.6 - gain.clamp(0.5, 1.8)).max(0.05)
}

fn clamp01(value: f64) -> f64 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn clamp_bonus(value: f64) -> f64 {
    if value.is_finite() {
        value.clamp(-0.08, 0.08)
    } else {
        0.0
    }
}