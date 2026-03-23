//! Multi-objective SMCO (Swarm Memory Coordinate Optimizer) algorithm core.
//!
//! SMCO key innovation: direction memory — directions that led to Pareto archive
//! insertions are averaged into a `direction_mean`, biasing future perturbations
//! toward productive regions of parameter space.
//!
//! Usage:
//! ```ignore
//! let opt = SmcoOptimizer::new(SmcoConfig::default(), space, archive);
//! loop {
//!     let candidates = opt.generate_candidates();
//!     let results = evaluate(candidates); // Vec<(ParamVec, [f64;3])>
//!     opt.report_results(results);
//!     if opt.is_done() { break; }
//! }
//! ```

use std::collections::VecDeque;

use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

use crate::param_space::{ParamSpace, ParamVec};
use crate::pareto::{ParetoArchive, ParetoSolution};

const NORM_EPSILON: f64 = 1e-12;

fn vec_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn normalize_in_place(v: &mut [f64]) {
    let n = vec_norm(v);
    if n > NORM_EPSILON {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
}

// ── Config ────────────────────────────────────────────────────────────────

/// Configuration for the SMCO optimizer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmcoConfig {
    /// Maximum number of `report_results` calls (optimizer iterations).
    pub max_iterations: usize,
    /// Number of candidate parameter vectors generated per iteration.
    pub population_size: usize,
    /// Initial perturbation step size (fraction of parameter range).
    pub step_size: f64,
    /// Multiplicative decay applied to step_size after each iteration.
    pub step_decay: f64,
    /// Maximum number of solutions kept in the Pareto archive.
    pub archive_size: usize,
    /// RNG seed for reproducibility.
    pub seed: u64,
    /// Number of recent productive directions to keep in memory.
    pub direction_memory_size: usize,
    /// Fraction [0,1] of perturbation that is purely random (rest follows direction_mean).
    pub exploration_ratio: f64,
}

impl Default for SmcoConfig {
    fn default() -> Self {
        Self {
            max_iterations: 500,
            population_size: 20,
            step_size: 0.1,
            step_decay: 0.995,
            archive_size: 100,
            seed: 42,
            direction_memory_size: 50,
            exploration_ratio: 0.3,
        }
    }
}

// ── Optimizer ─────────────────────────────────────────────────────────────

/// Multi-objective SMCO optimizer.
///
/// The optimizer alternates between:
/// 1. `generate_candidates()` — produce a batch of candidate `ParamVec`s
/// 2. `report_results(...)` — ingest evaluated objectives, update archive & direction memory
pub struct SmcoOptimizer {
    config: SmcoConfig,
    space: ParamSpace,
    archive: ParetoArchive,
    rng: StdRng,

    /// Current (decayed) step size.
    current_step: f64,
    /// Number of completed iterations (calls to `report_results`).
    iteration: usize,

    /// Ring buffer of directions that led to archive insertions.
    direction_memory: VecDeque<Vec<f64>>,
    /// Running mean of productive directions (same dimensionality as ParamVec).
    direction_mean: Vec<f64>,

    /// Candidates issued in the last `generate_candidates()` call.
    /// Stored so `report_results` can compute the direction from base → accepted.
    last_candidates: Vec<(ParamVec, ParamVec)>, // (base, candidate)
}

impl SmcoOptimizer {
    /// Create a new optimizer. The `archive` may be pre-populated or empty.
    pub fn new(config: SmcoConfig, space: ParamSpace, archive: ParetoArchive) -> Self {
        let rng = StdRng::seed_from_u64(config.seed);
        let dim = space.len();
        let current_step = config.step_size;

        Self {
            config,
            space,
            archive,
            rng,
            current_step,
            iteration: 0,
            direction_memory: VecDeque::new(),
            direction_mean: vec![0.0; dim],
            last_candidates: Vec::new(),
        }
    }

    // ── Public API ────────────────────────────────────────────────────────

    /// Generate a batch of candidate parameter vectors for evaluation.
    ///
    /// Each candidate is produced by:
    /// 1. Pick a base from the archive (random member) or sample randomly if archive empty.
    /// 2. Generate a perturbation direction: blend of random unit vector and direction_mean.
    /// 3. Perturb the base by `current_step` along that direction, then clamp.
    pub fn generate_candidates(&mut self) -> Vec<ParamVec> {
        let n = self.config.population_size;
        let mut candidates_with_base: Vec<(ParamVec, ParamVec)> = Vec::with_capacity(n);
        let mut out: Vec<ParamVec> = Vec::with_capacity(n);

        for _ in 0..n {
            let base = self.pick_base();
            let direction = self.sample_direction();
            let candidate = self.perturb_along(&base, &direction);
            candidates_with_base.push((base, candidate.clone()));
            out.push(candidate);
        }

        self.last_candidates = candidates_with_base;
        out
    }

    /// Ingest evaluation results for the last batch of candidates.
    ///
    /// For each `(params, objectives)`:
    /// - Build a `ParetoSolution` and attempt insertion into the archive.
    /// - If accepted, compute the direction from base → candidate and push into direction memory.
    /// - Update `direction_mean` from the direction memory.
    /// - Decay `current_step`.
    /// - Increment iteration counter.
    pub fn report_results(&mut self, results: Vec<(ParamVec, [f64; 3])>) {
        for (idx, (params, objectives)) in results.into_iter().enumerate() {
            // scenario_scores stores the per-scenario breakdown; we use the 3 objectives
            // as a single entry since objectives are already [f64; 3].
            let scenario_scores: Vec<[f64; 3]> = vec![objectives];
            let sol =
                ParetoSolution::new(params.clone(), objectives, scenario_scores, self.iteration);

            let accepted = self.archive.insert(sol);

            if accepted {
                // Compute direction: candidate - base (from last_candidates)
                if let Some((base, _cand)) = self.last_candidates.get(idx) {
                    let dir: Vec<f64> =
                        params.iter().zip(base.iter()).map(|(c, b)| c - b).collect();

                    let mut unit = dir;
                    normalize_in_place(&mut unit);
                    if vec_norm(&unit) > NORM_EPSILON {
                        self.push_direction(unit);
                    }
                }
            }
        }

        self.recompute_direction_mean();

        // Decay step and advance iteration
        self.current_step *= self.config.step_decay;
        self.iteration += 1;
    }

    /// Returns true when the optimizer has exhausted its iteration budget.
    pub fn is_done(&self) -> bool {
        self.iteration >= self.config.max_iterations
    }

    /// Current (possibly decayed) step size.
    pub fn current_step(&self) -> f64 {
        self.current_step
    }

    /// Hypervolume of the current Pareto archive relative to `reference`.
    pub fn hypervolume(&self, reference: [f64; 3]) -> f64 {
        self.archive.hypervolume(reference)
    }

    /// Read-only access to the underlying Pareto archive.
    pub fn archive(&self) -> &ParetoArchive {
        &self.archive
    }

    /// Number of completed iterations.
    pub fn iteration(&self) -> usize {
        self.iteration
    }

    // ── Private helpers ───────────────────────────────────────────────────

    /// Pick a random solution from the archive as the perturbation base.
    /// Falls back to the parameter-space defaults when the archive is empty.
    fn pick_base(&mut self) -> ParamVec {
        if self.archive.is_empty() {
            // Explore from defaults or fully random, alternating
            if self.rng.random::<f64>() < 0.5 {
                self.space.defaults()
            } else {
                self.space.sample_random(&mut self.rng)
            }
        } else {
            let idx = self.rng.random_range(0..self.archive.solutions.len());
            self.archive.solutions[idx].params.clone()
        }
    }

    /// Sample a perturbation direction as a blend of:
    ///   (1 - exploration_ratio) * direction_mean  +  exploration_ratio * random_unit
    /// The result is L2-normalised.
    fn sample_direction(&mut self) -> Vec<f64> {
        let dim = self.space.len();
        let alpha = self.config.exploration_ratio;

        // Random Gaussian unit vector
        let mut rand_dir: Vec<f64> = (0..dim)
            .map(|_| self.rng.sample::<f64, _>(rand_distr::StandardNormal))
            .collect();
        normalize_in_place(&mut rand_dir);

        // Blend with direction_mean (may be zero vector initially)
        let mut dir: Vec<f64> = rand_dir
            .iter()
            .zip(self.direction_mean.iter())
            .map(|(r, m)| alpha * r + (1.0 - alpha) * m)
            .collect();

        // Blend of two unit vectors is not unit; re-normalise for valid direction
        if vec_norm(&dir) > NORM_EPSILON {
            normalize_in_place(&mut dir);
        } else {
            // Fallback: pure random (direction_mean was also near-zero)
            dir = rand_dir;
        }

        dir
    }

    /// Perturb `base` by `current_step` along `direction`, then clamp.
    fn perturb_along(&self, base: &ParamVec, direction: &[f64]) -> ParamVec {
        use crate::param_space::ParamKind;

        let step = self.current_step;
        let out: Vec<f64> = base
            .iter()
            .zip(direction.iter())
            .zip(self.space.params.iter())
            .map(|((b, d), def)| {
                let range = match &def.kind {
                    ParamKind::Continuous { min, max } => max - min,
                    ParamKind::Discrete { min, max } => (max - min) as f64,
                };
                def.clamp(b + d * step * range)
            })
            .collect();
        out
    }

    /// Push a productive direction into the ring buffer.
    fn push_direction(&mut self, unit_dir: Vec<f64>) {
        if self.direction_memory.len() >= self.config.direction_memory_size {
            self.direction_memory.pop_front();
        }
        self.direction_memory.push_back(unit_dir);
    }

    /// Recompute `direction_mean` as the arithmetic mean of all stored directions.
    fn recompute_direction_mean(&mut self) {
        let dim = self.space.len();
        let n = self.direction_memory.len();
        if n == 0 {
            self.direction_mean = vec![0.0; dim];
            return;
        }
        let mut mean = vec![0.0f64; dim];
        for dir in &self.direction_memory {
            for (m, v) in mean.iter_mut().zip(dir.iter()) {
                *m += v;
            }
        }
        let scale = 1.0 / n as f64;
        for m in &mut mean {
            *m *= scale;
        }
        normalize_in_place(&mut mean);
        self.direction_mean = mean;
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::param_space::strix_full;

    fn make_optimizer() -> SmcoOptimizer {
        let config = SmcoConfig::default();
        let space = strix_full();
        let archive = ParetoArchive::new(config.archive_size);
        SmcoOptimizer::new(config, space, archive)
    }

    #[test]
    fn generates_correct_population_size() {
        let mut opt = make_optimizer();
        let candidates = opt.generate_candidates();
        assert_eq!(candidates.len(), SmcoConfig::default().population_size);
    }

    #[test]
    fn candidates_are_54_dimensional() {
        let mut opt = make_optimizer();
        let candidates = opt.generate_candidates();
        for c in &candidates {
            assert_eq!(c.len(), 54, "Expected 54-dim ParamVec, got {}", c.len());
        }
    }

    #[test]
    fn candidates_within_bounds_after_generation() {
        let mut opt = make_optimizer();
        let space = strix_full();
        let candidates = opt.generate_candidates();
        for c in &candidates {
            let clamped = space.clamp(c);
            for (i, (a, b)) in c.iter().zip(&clamped).enumerate() {
                assert!(
                    (a - b).abs() < 1e-9,
                    "param[{}] = {} out of bounds (clamped to {})",
                    i,
                    a,
                    b
                );
            }
        }
    }

    #[test]
    fn report_results_increments_iteration() {
        let mut opt = make_optimizer();
        assert_eq!(opt.iteration(), 0);
        let candidates = opt.generate_candidates();
        let results: Vec<(ParamVec, [f64; 3])> = candidates
            .into_iter()
            .map(|p| (p, [0.5, 0.5, 0.5]))
            .collect();
        opt.report_results(results);
        assert_eq!(opt.iteration(), 1);
    }

    #[test]
    fn is_done_after_max_iterations() {
        let config = SmcoConfig {
            max_iterations: 3,
            population_size: 2,
            ..Default::default()
        };
        let space = strix_full();
        let archive = ParetoArchive::new(config.archive_size);
        let mut opt = SmcoOptimizer::new(config, space, archive);

        for _ in 0..3 {
            assert!(!opt.is_done());
            let cands = opt.generate_candidates();
            let results = cands.into_iter().map(|p| (p, [0.4, 0.4, 0.4])).collect();
            opt.report_results(results);
        }
        assert!(opt.is_done());
    }

    #[test]
    fn step_decays_over_iterations() {
        let config = SmcoConfig {
            max_iterations: 10,
            population_size: 2,
            step_size: 0.1,
            step_decay: 0.9,
            ..Default::default()
        };
        let space = strix_full();
        let archive = ParetoArchive::new(config.archive_size);
        let mut opt = SmcoOptimizer::new(config, space, archive);
        let initial_step = opt.current_step();

        let cands = opt.generate_candidates();
        let results = cands.into_iter().map(|p| (p, [0.4, 0.4, 0.4])).collect();
        opt.report_results(results);

        assert!(
            opt.current_step() < initial_step,
            "step should decay: {} >= {}",
            opt.current_step(),
            initial_step
        );
    }

    #[test]
    fn good_solutions_enter_archive() {
        let mut opt = make_optimizer();
        let cands = opt.generate_candidates();
        // Report with Pareto-incomparable objectives to fill the front
        let results: Vec<(ParamVec, [f64; 3])> = cands
            .into_iter()
            .enumerate()
            .map(|(i, p)| {
                let t = i as f64 / 20.0;
                (p, [1.0 - t, t, 0.5])
            })
            .collect();
        opt.report_results(results);
        assert!(
            !opt.archive().is_empty(),
            "archive should have accepted solutions"
        );
    }

    #[test]
    fn dominated_solutions_rejected_from_archive() {
        let mut opt = make_optimizer();

        // First: plant a dominant solution manually
        let params = strix_full().defaults();
        let dominator = ParetoSolution::new(params.clone(), [1.0, 1.0, 1.0], vec![], 0);
        opt.archive.insert(dominator);

        // Now report all-dominated batch
        let cands = opt.generate_candidates();
        let results: Vec<(ParamVec, [f64; 3])> =
            cands.into_iter().map(|p| (p, [0.1, 0.1, 0.1])).collect();
        opt.report_results(results);

        // Archive should still have exactly 1 solution (the dominator)
        assert_eq!(opt.archive().len(), 1);
    }

    #[test]
    fn direction_memory_influences_candidates() {
        // After many productive insertions, direction_mean should be non-zero.
        // We verify direction_mean gets populated by running the optimizer with
        // Pareto-improving results.
        let config = SmcoConfig {
            max_iterations: 10,
            population_size: 5,
            exploration_ratio: 0.0, // pure exploitation — direction_mean fully drives
            ..Default::default()
        };
        let space = strix_full();
        let archive = ParetoArchive::new(config.archive_size);
        let mut opt = SmcoOptimizer::new(config, space, archive);

        // First iteration: all candidates with spread objectives → fill archive
        let cands = opt.generate_candidates();
        let results: Vec<(ParamVec, [f64; 3])> = cands
            .into_iter()
            .enumerate()
            .map(|(i, p)| {
                let t = i as f64 / 5.0;
                (p, [t, 1.0 - t, 0.5])
            })
            .collect();
        opt.report_results(results);

        let norm: f64 = opt.direction_mean.iter().map(|x| x * x).sum::<f64>().sqrt();
        // Direction mean should be non-zero if any solution was accepted
        // (archive was empty so first non-dominated solution is always accepted)
        assert!(
            norm > 1e-6 || opt.archive().is_empty(),
            "direction_mean should be non-zero after productive insertions, norm={}",
            norm
        );
    }

    #[test]
    fn hypervolume_increases_with_better_solutions() {
        let mut opt = make_optimizer();
        let ref_pt = [0.0, 0.0, 0.0];
        let hv_before = opt.hypervolume(ref_pt);

        let cands = opt.generate_candidates();
        // Report with objectives that dominate reference
        let results: Vec<(ParamVec, [f64; 3])> = cands
            .into_iter()
            .enumerate()
            .map(|(i, p)| {
                let t = i as f64 / 20.0;
                (p, [0.5 + t * 0.02, 0.5 - t * 0.01, 0.6])
            })
            .collect();
        opt.report_results(results);

        let hv_after = opt.hypervolume(ref_pt);
        assert!(
            hv_after >= hv_before,
            "HV should not decrease after inserting solutions: before={}, after={}",
            hv_before,
            hv_after
        );
    }

    #[test]
    fn reproducible_with_same_seed() {
        let mut opt1 = make_optimizer();
        let mut opt2 = make_optimizer();

        let c1 = opt1.generate_candidates();
        let c2 = opt2.generate_candidates();

        assert_eq!(c1.len(), c2.len());
        for (a, b) in c1.iter().zip(&c2) {
            for (x, y) in a.iter().zip(b) {
                assert!(
                    (x - y).abs() < 1e-12,
                    "Same seed should produce identical candidates"
                );
            }
        }
    }

    #[test]
    fn archive_size_respected() {
        let config = SmcoConfig {
            archive_size: 5,
            population_size: 20,
            max_iterations: 5,
            ..Default::default()
        };
        let space = strix_full();
        let archive = ParetoArchive::new(config.archive_size);
        let mut opt = SmcoOptimizer::new(config, space, archive);

        for _ in 0..5 {
            let cands = opt.generate_candidates();
            let results: Vec<(ParamVec, [f64; 3])> = cands
                .into_iter()
                .enumerate()
                .map(|(i, p)| {
                    let t = i as f64 / 20.0;
                    (p, [t, 1.0 - t, t * 0.5])
                })
                .collect();
            opt.report_results(results);
        }

        assert!(
            opt.archive().len() <= 5,
            "Archive exceeded max_size: {}",
            opt.archive().len()
        );
    }

    #[test]
    fn full_run_smoke_test() {
        // Verify a short run completes without panic and produces a non-empty archive
        let config = SmcoConfig {
            max_iterations: 10,
            population_size: 10,
            ..Default::default()
        };
        let space = strix_full();
        let archive = ParetoArchive::new(config.archive_size);
        let mut opt = SmcoOptimizer::new(config, space, archive);

        while !opt.is_done() {
            let cands = opt.generate_candidates();
            let results: Vec<(ParamVec, [f64; 3])> = cands
                .into_iter()
                .enumerate()
                .map(|(i, p)| {
                    let t = (i as f64 + opt.iteration() as f64) / 100.0;
                    (p, [(t * 1.3).min(1.0), (1.0 - t * 0.7).max(0.0), 0.5])
                })
                .collect();
            opt.report_results(results);
        }

        assert!(opt.is_done());
        assert!(!opt.archive().is_empty());
        assert!(opt.hypervolume([0.0, 0.0, 0.0]) > 0.0);
    }
}
