//! Optimization result export — Pareto front summary + hypervolume history.

use std::io;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::contextual_archive::ContextualArchive;
use crate::evaluator::DoctrineProfile;
use crate::pareto::{ParetoArchive, ParetoSolution};

/// Reference point for hypervolume computation (worst-case objectives).
pub const HV_REFERENCE: [f64; 3] = [0.0, 0.0, 0.0];

fn finite_or_zero(value: f64) -> f64 {
    if value.is_finite() {
        value
    } else {
        0.0
    }
}

fn compare_f64_slices(a: &[f64], b: &[f64]) -> std::cmp::Ordering {
    a.len().cmp(&b.len()).then_with(|| {
        a.iter()
            .zip(b.iter())
            .find_map(|(left, right)| {
                let ordering = left.total_cmp(right);
                (ordering != std::cmp::Ordering::Equal).then_some(ordering)
            })
            .unwrap_or(std::cmp::Ordering::Equal)
    })
}

fn compare_score_slices(a: &[[f64; 3]], b: &[[f64; 3]]) -> std::cmp::Ordering {
    a.len().cmp(&b.len()).then_with(|| {
        a.iter()
            .zip(b.iter())
            .find_map(|(left, right)| {
                let ordering = left
                    .iter()
                    .zip(right.iter())
                    .find_map(|(l, r)| {
                        let ordering = l.total_cmp(r);
                        (ordering != std::cmp::Ordering::Equal).then_some(ordering)
                    })
                    .unwrap_or(std::cmp::Ordering::Equal);
                (ordering != std::cmp::Ordering::Equal).then_some(ordering)
            })
            .unwrap_or(std::cmp::Ordering::Equal)
    })
}

fn sanitize_solution(solution: &ParetoSolution) -> ParetoSolution {
    ParetoSolution {
        params: solution
            .params
            .iter()
            .copied()
            .map(finite_or_zero)
            .collect(),
        objectives: solution.objectives.map(finite_or_zero),
        scenario_scores: solution
            .scenario_scores
            .iter()
            .map(|scores| (*scores).map(finite_or_zero))
            .collect(),
        iteration: solution.iteration,
        crowding_distance: finite_or_zero(solution.crowding_distance),
    }
}

fn compare_solution(a: &ParetoSolution, b: &ParetoSolution) -> std::cmp::Ordering {
    a.iteration
        .cmp(&b.iteration)
        .then_with(|| {
            a.objectives[0]
                .total_cmp(&b.objectives[0])
                .then_with(|| a.objectives[1].total_cmp(&b.objectives[1]))
                .then_with(|| a.objectives[2].total_cmp(&b.objectives[2]))
        })
        .then_with(|| compare_f64_slices(&a.params, &b.params))
        .then_with(|| compare_score_slices(&a.scenario_scores, &b.scenario_scores))
}

fn compare_by_objective(
    objective: usize,
    a: &ParetoSolution,
    b: &ParetoSolution,
) -> std::cmp::Ordering {
    a.objectives[objective]
        .total_cmp(&b.objectives[objective])
        .then_with(|| compare_solution(a, b))
}

/// Full optimization result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextualFrontSummary {
    pub context: String,
    pub size: usize,
    pub hypervolume: f64,
    pub best_objectives: [f64; 3],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationReport {
    /// Doctrine profile used when producing this report.
    pub doctrine_profile: String,
    /// Human-readable objective labels for F0/F1/F2.
    pub objective_labels: [String; 3],
    /// Final Pareto front as a flat list of solutions.
    pub pareto_front: Vec<ParetoSolution>,
    /// Optional per-context front summaries.
    pub contextual_fronts: Vec<ContextualFrontSummary>,
    /// Best solution maximising F0.
    pub best_survival: Option<ParetoSolution>,
    /// Best solution maximising F1.
    pub best_stability: Option<ParetoSolution>,
    /// Best solution maximising F2.
    pub best_efficiency: Option<ParetoSolution>,
    /// `(iteration, hypervolume)` recorded after each optimizer step.
    pub hypervolume_history: Vec<(usize, f64)>,
    /// Total number of candidate evaluations (including rejected).
    pub total_evaluations: usize,
    /// Wall-clock seconds for the full run.
    pub elapsed_secs: f64,
}

impl OptimizationReport {
    /// Build from a finalised Pareto archive and history bookkeeping.
    pub fn from_archive(
        archive: &ParetoArchive,
        doctrine: DoctrineProfile,
        objective_labels: [&str; 3],
        hypervolume_history: Vec<(usize, f64)>,
        total_evaluations: usize,
        elapsed_secs: f64,
    ) -> Self {
        let mut solutions: Vec<ParetoSolution> =
            archive.solutions.iter().map(sanitize_solution).collect();
        solutions.sort_by(compare_solution);

        let hypervolume_history = hypervolume_history
            .into_iter()
            .map(|(iteration, hv)| (iteration, finite_or_zero(hv).max(0.0)))
            .collect();
        let elapsed_secs = finite_or_zero(elapsed_secs).max(0.0);

        let best_survival = solutions
            .iter()
            .filter(|solution| solution.objectives[0].is_finite())
            .max_by(|a, b| compare_by_objective(0, a, b))
            .cloned();
        let best_stability = solutions
            .iter()
            .filter(|solution| solution.objectives[1].is_finite())
            .max_by(|a, b| compare_by_objective(1, a, b))
            .cloned();
        let best_efficiency = solutions
            .iter()
            .filter(|solution| solution.objectives[2].is_finite())
            .max_by(|a, b| compare_by_objective(2, a, b))
            .cloned();

        Self {
            doctrine_profile: doctrine.as_str().to_string(),
            objective_labels: objective_labels.map(str::to_string),
            pareto_front: solutions,
            contextual_fronts: Vec::new(),
            best_survival,
            best_stability,
            best_efficiency,
            hypervolume_history,
            total_evaluations,
            elapsed_secs,
        }
    }

    /// Attach archive-of-archives summaries to this report.
    pub fn attach_contextual_fronts(mut self, archive: &ContextualArchive) -> Self {
        self.contextual_fronts = archive
            .contexts()
            .map(|(context, front)| {
                let best_objectives = front
                    .solutions
                    .iter()
                    .max_by(|left, right| {
                        let left_sum = left.objectives.iter().sum::<f64>();
                        let right_sum = right.objectives.iter().sum::<f64>();
                        left_sum.total_cmp(&right_sum)
                    })
                    .map(|solution| solution.objectives)
                    .unwrap_or([0.0, 0.0, 0.0]);
                ContextualFrontSummary {
                    context: context.key(),
                    size: front.len(),
                    hypervolume: front.hypervolume(HV_REFERENCE),
                    best_objectives,
                }
            })
            .collect();
        self
    }
    /// Serialize to a pretty-printed JSON file at `path`.
    pub fn to_json(&self, path: &Path) -> io::Result<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        std::fs::write(path, json)
    }

    /// Print a human-readable summary to stdout.
    pub fn print_summary(&self) {
        println!("=== STRIX Optimizer Results ===");
        println!("  Doctrine:           {}", self.doctrine_profile);
        println!(
            "  Objectives:         [{}, {}, {}]",
            self.objective_labels[0], self.objective_labels[1], self.objective_labels[2]
        );
        println!("  Pareto front size:  {}", self.pareto_front.len());
        println!("  Total evaluations:  {}", self.total_evaluations);
        println!("  Elapsed:            {:.1}s", self.elapsed_secs);

        if let Some((_, hv)) = self.hypervolume_history.last() {
            println!("  Final hypervolume:  {:.6}", hv);
        }

        println!();
        println!("  Best by objective:");

        if let Some(s) = &self.best_survival {
            println!(
                "    {}: {:.4}  ({}={:.4}, {}={:.4})",
                self.objective_labels[0],
                s.objectives[0],
                self.objective_labels[1],
                s.objectives[1],
                self.objective_labels[2],
                s.objectives[2]
            );
        }
        if let Some(s) = &self.best_stability {
            println!(
                "    {}: {:.4}  ({}={:.4}, {}={:.4})",
                self.objective_labels[1],
                s.objectives[1],
                self.objective_labels[0],
                s.objectives[0],
                self.objective_labels[2],
                s.objectives[2]
            );
        }
        if let Some(s) = &self.best_efficiency {
            println!(
                "    {}: {:.4}  ({}={:.4}, {}={:.4})",
                self.objective_labels[2],
                s.objectives[2],
                self.objective_labels[0],
                s.objectives[0],
                self.objective_labels[1],
                s.objectives[1]
            );
        }

        println!();
        let hv_pts = &self.hypervolume_history;
        println!("  Hypervolume history ({} pts):", hv_pts.len());
        let stride = (hv_pts.len() / 10).max(1);
        for (iter, hv) in hv_pts.iter().step_by(stride) {
            println!("    iter {:>4}: {:.6}", iter, hv);
        }
        if let Some((iter, hv)) = hv_pts.last() {
            let last_idx = hv_pts.len().saturating_sub(1);
            if !last_idx.is_multiple_of(stride) {
                println!("    iter {:>4}: {:.6}", iter, hv);
            }
        }

        println!();
        println!("=== END ===");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pareto::{ParetoArchive, ParetoSolution};

    #[test]
    fn from_archive_sanitizes_non_finite_export_fields() {
        let mut archive = ParetoArchive::new(2);
        archive.insert(ParetoSolution::new(vec![0.0], [0.2, 0.1, 0.3], vec![], 1));
        archive.insert(ParetoSolution::new(vec![1.0], [0.4, 0.5, 0.6], vec![], 2));
        archive.insert(ParetoSolution::new(vec![2.0], [0.7, 0.8, 0.9], vec![], 3));
        archive.solutions[0].crowding_distance = f64::INFINITY;

        let report = OptimizationReport::from_archive(
            &archive,
            DoctrineProfile::Balanced,
            ["f0", "f1", "f2"],
            vec![(1, f64::NAN), (2, 0.5)],
            7,
            f64::INFINITY,
        );

        let json = serde_json::to_string(&report).unwrap();
        assert!(!json.contains("Infinity"));
        assert!(!json.contains("NaN"));
        assert!(report
            .pareto_front
            .iter()
            .all(|solution| solution.crowding_distance.is_finite()));
        assert_eq!(report.hypervolume_history[0], (1, 0.0));
        assert_eq!(report.elapsed_secs, 0.0);
    }
}
