//! Optimization result export — Pareto front summary + hypervolume history.

use std::io;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::evaluator::DoctrineProfile;
use crate::pareto::{ParetoArchive, ParetoSolution};

/// Reference point for hypervolume computation (worst-case objectives).
pub const HV_REFERENCE: [f64; 3] = [0.0, 0.0, 0.0];

/// Full optimization result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationReport {
    /// Doctrine profile used when producing this report.
    pub doctrine_profile: String,
    /// Human-readable objective labels for F0/F1/F2.
    pub objective_labels: [String; 3],
    /// Final Pareto front as a flat list of solutions.
    pub pareto_front: Vec<ParetoSolution>,
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
        let solutions = archive.solutions.clone();

        let best_survival = solutions
            .iter()
            .max_by(|a, b| a.objectives[0].partial_cmp(&b.objectives[0]).unwrap())
            .cloned();
        let best_stability = solutions
            .iter()
            .max_by(|a, b| a.objectives[1].partial_cmp(&b.objectives[1]).unwrap())
            .cloned();
        let best_efficiency = solutions
            .iter()
            .max_by(|a, b| a.objectives[2].partial_cmp(&b.objectives[2]).unwrap())
            .cloned();

        Self {
            doctrine_profile: doctrine.as_str().to_string(),
            objective_labels: objective_labels.map(str::to_string),
            pareto_front: solutions,
            best_survival,
            best_stability,
            best_efficiency,
            hypervolume_history,
            total_evaluations,
            elapsed_secs,
        }
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
            if last_idx % stride != 0 {
                println!("    iter {:>4}: {:.6}", iter, hv);
            }
        }

        println!();
        println!("=== END ===");
    }
}
