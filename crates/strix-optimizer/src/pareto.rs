// Pareto archive with crowding distance (NSGA-II style)
//
// Objectives are maximized: solution A dominates B iff
//   A[i] >= B[i] for all i  AND  A[j] > B[j] for at least one j.
//
// Hypervolume is computed via 3-D WFG sweep against a reference point.

use rand::Rng;
use serde::{Deserialize, Serialize};

/// A single solution in the Pareto archive.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ParetoSolution {
    /// Raw parameter vector (optimizer coordinates).
    pub params: Vec<f64>,
    /// Three-objective vector: [survival, stability, efficiency].
    pub objectives: [f64; 3],
    /// Per-scenario breakdown scores (one [f64; 3] per scenario).
    pub scenario_scores: Vec<[f64; 3]>,
    /// Optimizer iteration at which this solution was found.
    pub iteration: usize,
    /// Crowding distance assigned during last prune (f64::INFINITY = boundary).
    #[serde(default)]
    pub crowding_distance: f64,
}

impl ParetoSolution {
    pub fn new(
        params: Vec<f64>,
        objectives: [f64; 3],
        scenario_scores: Vec<[f64; 3]>,
        iteration: usize,
    ) -> Self {
        Self {
            params,
            objectives,
            scenario_scores,
            iteration,
            crowding_distance: 0.0,
        }
    }

    /// Returns true if `self` dominates `other` (maximisation semantics).
    pub fn dominates(&self, other: &ParetoSolution) -> bool {
        dominates(&self.objectives, &other.objectives)
    }
}

/// Free-function dominance check: `a` dominates `b` iff
/// a[i] >= b[i] for all i and a[j] > b[j] for at least one j (maximisation).
pub fn dominates(a: &[f64; 3], b: &[f64; 3]) -> bool {
    a.iter().zip(b.iter()).all(|(x, y)| x >= y) && a.iter().zip(b.iter()).any(|(x, y)| x > y)
}

/// NSGA-II crowding distance for a slice of solutions.
/// Returns a Vec<f64> of length `solutions.len()`.
/// Boundary solutions (extreme values per objective) get f64::INFINITY.
pub fn crowding_distance(solutions: &[ParetoSolution]) -> Vec<f64> {
    let n = solutions.len();
    let mut dist = vec![0.0f64; n];

    if n <= 2 {
        // All boundary
        for d in dist.iter_mut() {
            *d = f64::INFINITY;
        }
        return dist;
    }

    for obj_idx in 0..3usize {
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| {
            solutions[a].objectives[obj_idx]
                .partial_cmp(&solutions[b].objectives[obj_idx])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let f_min = solutions[order[0]].objectives[obj_idx];
        let f_max = solutions[order[n - 1]].objectives[obj_idx];
        let range = f_max - f_min;

        dist[order[0]] = f64::INFINITY;
        dist[order[n - 1]] = f64::INFINITY;

        if range < f64::EPSILON {
            continue;
        }

        for i in 1..n - 1 {
            let prev = solutions[order[i - 1]].objectives[obj_idx];
            let next = solutions[order[i + 1]].objectives[obj_idx];
            dist[order[i]] += (next - prev) / range;
        }
    }

    dist
}

/// Bounded Pareto archive with automatic dominated-solution pruning and
/// crowding-distance–based size enforcement.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ParetoArchive {
    pub solutions: Vec<ParetoSolution>,
    pub max_size: usize,
}

impl ParetoArchive {
    pub fn new(max_size: usize) -> Self {
        Self {
            solutions: Vec::new(),
            max_size: max_size.max(1),
        }
    }

    /// Attempt to insert `candidate`.
    ///
    /// 1. If dominated by any existing solution → reject (return false).
    /// 2. Remove all existing solutions dominated by `candidate`.
    /// 3. Add `candidate`.
    /// 4. If archive exceeds `max_size`, call `prune()`.
    ///
    /// Returns `true` if the candidate was accepted.
    pub fn insert(&mut self, candidate: ParetoSolution) -> bool {
        // Single-pass: check if candidate is dominated AND collect indices it dominates
        let mut dominated_by_existing = false;
        let mut keep = Vec::with_capacity(self.solutions.len());
        for s in &self.solutions {
            if s.dominates(&candidate) {
                dominated_by_existing = true;
                break;
            }
            if !candidate.dominates(s) {
                keep.push(true);
            } else {
                keep.push(false);
            }
        }
        if dominated_by_existing {
            return false;
        }
        // Remove dominated solutions
        let mut i = 0;
        self.solutions.retain(|_| {
            let k = keep[i];
            i += 1;
            k
        });
        self.solutions.push(candidate);

        if self.solutions.len() > self.max_size {
            self.prune();
        }

        true
    }

    /// Prune the archive to `max_size` by iteratively removing the solution
    /// with the smallest crowding distance (NSGA-II style).
    /// Tie-break: prefer to keep solutions from more recent iterations.
    pub fn prune(&mut self) {
        while self.solutions.len() > self.max_size {
            let dist = crowding_distance(&self.solutions);

            let worst_idx = dist
                .iter()
                .enumerate()
                .min_by(|(ia, da), (ib, db)| {
                    da.partial_cmp(db)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| {
                            self.solutions[*ia]
                                .iteration
                                .cmp(&self.solutions[*ib].iteration)
                        })
                })
                .map(|(i, _)| i)
                .unwrap();

            // Store the computed distances back into the solutions before removing.
            for (s, d) in self.solutions.iter_mut().zip(dist.iter()) {
                s.crowding_distance = *d;
            }

            self.solutions.swap_remove(worst_idx);
        }
    }

    /// Number of non-dominated solutions in the archive.
    pub fn len(&self) -> usize {
        self.solutions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.solutions.is_empty()
    }

    /// Return the solution with the highest value on the given objective index (0, 1, or 2).
    /// Returns `None` if the archive is empty.
    pub fn best_for(&self, objective: usize) -> Option<&ParetoSolution> {
        self.solutions.iter().max_by(|a, b| {
            a.objectives[objective]
                .partial_cmp(&b.objectives[objective])
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Return a uniformly random member of the archive, or `None` if empty.
    pub fn random_member<R: Rng>(&self, rng: &mut R) -> Option<&ParetoSolution> {
        if self.solutions.is_empty() {
            return None;
        }
        let idx = rng.random_range(0..self.solutions.len());
        Some(&self.solutions[idx])
    }

    /// 3-D hypervolume dominated by the archive relative to `ref_point`.
    ///
    /// Uses the WFG sweep algorithm (sort by obj[0] descending, accumulate
    /// 2-D staircase slabs). Solutions that do not strictly dominate
    /// `ref_point` on all objectives are excluded.
    pub fn hypervolume(&self, ref_point: [f64; 3]) -> f64 {
        let mut pts: Vec<[f64; 3]> = self
            .solutions
            .iter()
            .filter(|s| {
                s.objectives[0] > ref_point[0]
                    && s.objectives[1] > ref_point[1]
                    && s.objectives[2] > ref_point[2]
            })
            .map(|s| s.objectives)
            .collect();

        if pts.is_empty() {
            return 0.0;
        }

        hypervolume_3d(&mut pts, ref_point)
    }
}

// ---------------------------------------------------------------------------
// 3-D Hypervolume (WFG sweep)
// ---------------------------------------------------------------------------

/// Compute 3-D hypervolume using the sweep-line algorithm.
/// Points must already be filtered to strictly dominate the reference.
/// Mutates `pts` (sorts in place by obj[0] descending).
fn hypervolume_3d(pts: &mut [[f64; 3]], reference: [f64; 3]) -> f64 {
    pts.sort_by(|a, b| b[0].partial_cmp(&a[0]).unwrap_or(std::cmp::Ordering::Equal));

    let mut volume = 0.0;
    // Sweep plane starts at reference[0]; points are at higher values.
    let mut prev_x = reference[0];
    // 2-D non-dominated front for (obj[1], obj[2]) projection.
    let mut front_2d: Vec<[f64; 2]> = Vec::new();

    for pt in pts.iter() {
        let x = pt[0];
        // Slab between prev_x (lower, already swept) and x (new, higher):
        // Note: sorted descending so x <= prev_x after first point is processed.
        // First iteration: area of empty front = 0, slab = ref - x (negative) → 0 contribution.
        let area = hypervolume_2d(&front_2d, [reference[1], reference[2]]);
        let slab = prev_x - x; // prev_x >= x since we sorted descending and update prev_x = x each iter
        volume += area * slab;

        insert_2d(&mut front_2d, [pt[1], pt[2]]);
        prev_x = x;
    }

    // Final slab from reference[0] up to last processed x.
    let area = hypervolume_2d(&front_2d, [reference[1], reference[2]]);
    volume += area * (prev_x - reference[0]);

    volume
}

/// Insert a 2-D point into a non-dominated front (maximisation),
/// removing dominated points.
fn insert_2d(front: &mut Vec<[f64; 2]>, pt: [f64; 2]) {
    front.retain(|p| !(pt[0] >= p[0] && pt[1] >= p[1] && (pt[0] > p[0] || pt[1] > p[1])));

    let dominated = front
        .iter()
        .any(|p| p[0] >= pt[0] && p[1] >= pt[1] && (p[0] > pt[0] || p[1] > pt[1]));

    if !dominated {
        front.push(pt);
    }
}

/// 2-D hypervolume of a set of points relative to `reference` (maximisation).
/// Staircase algorithm: sort by obj[0] descending, sweep accumulating rectangles.
fn hypervolume_2d(front: &[[f64; 2]], reference: [f64; 2]) -> f64 {
    if front.is_empty() {
        return 0.0;
    }

    let mut pts: Vec<[f64; 2]> = front.to_vec();
    pts.sort_by(|a, b| b[0].partial_cmp(&a[0]).unwrap_or(std::cmp::Ordering::Equal));

    let mut area = 0.0;
    let mut prev_y = reference[1];

    for pt in &pts {
        if pt[1] > prev_y {
            area += (pt[0] - reference[0]) * (pt[1] - prev_y);
            prev_y = pt[1];
        }
    }

    area
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn sol(objectives: [f64; 3], iteration: usize) -> ParetoSolution {
        ParetoSolution::new(vec![0.0; 3], objectives, vec![], iteration)
    }

    // --- Free-function dominance (spec-required tests) ---

    #[test]
    fn test_dominates() {
        // a=(1,2,3) dominates b=(1,1,3)
        let a = [1.0f64, 2.0, 3.0];
        let b = [1.0f64, 1.0, 3.0];
        assert!(dominates(&a, &b), "a should dominate b");

        // a does NOT dominate c=(2,1,3) — c is better on obj0
        let c = [2.0f64, 1.0, 3.0];
        assert!(!dominates(&a, &c), "a should not dominate c");
        assert!(!dominates(&c, &a), "c should not dominate a (tradeoff)");
    }

    #[test]
    fn test_dominates_equal_not_dominate() {
        let a = [0.8, 0.8, 0.8];
        assert!(!dominates(&a, &a));
    }

    #[test]
    fn test_dominates_strict() {
        let a = [0.9, 0.8, 0.7];
        let b = [0.8, 0.7, 0.6];
        assert!(dominates(&a, &b));
        assert!(!dominates(&b, &a));
    }

    // --- Archive insertion (spec-required tests) ---

    #[test]
    fn test_insert_non_dominated() {
        let mut archive = ParetoArchive::new(100);
        // Insert 3 non-dominated solutions
        archive.insert(sol([1.0, 0.0, 0.0], 0));
        archive.insert(sol([0.0, 1.0, 0.0], 1));
        archive.insert(sol([0.0, 0.0, 1.0], 2));
        assert_eq!(archive.len(), 3);
    }

    #[test]
    fn test_insert_removes_dominated() {
        let mut archive = ParetoArchive::new(100);
        archive.insert(sol([0.5, 0.5, 0.5], 0));
        archive.insert(sol([0.3, 0.4, 0.4], 1));
        // Insert a dominator — both previous should be removed
        archive.insert(sol([0.9, 0.9, 0.9], 2));
        assert_eq!(archive.len(), 1);
        assert_eq!(archive.solutions[0].objectives, [0.9, 0.9, 0.9]);
    }

    #[test]
    fn test_insert_dominated_rejected() {
        let mut archive = ParetoArchive::new(100);
        archive.insert(sol([0.9, 0.9, 0.9], 0));
        let accepted = archive.insert(sol([0.5, 0.5, 0.5], 1));
        assert!(!accepted);
        assert_eq!(archive.len(), 1);
    }

    #[test]
    fn test_insert_incomparable_both_kept() {
        let mut archive = ParetoArchive::new(100);
        archive.insert(sol([1.0, 0.0, 0.5], 0));
        archive.insert(sol([0.0, 1.0, 0.5], 1));
        assert_eq!(archive.len(), 2);
    }

    // --- Pruning (spec-required test) ---

    #[test]
    fn test_prune_crowding() {
        // Archive of 10 non-dominated solutions, pruned to max_size=5.
        // After prune the kept solutions should be diverse (boundary solutions preserved).
        let mut archive = ParetoArchive::new(5);
        for i in 0..10usize {
            let t = i as f64 / 9.0 * std::f64::consts::PI / 2.0;
            archive.insert(sol([t.cos(), t.sin(), 0.5], i));
        }
        assert!(
            archive.len() <= 5,
            "expected pruned to 5, got {}",
            archive.len()
        );

        // All remaining solutions must be mutually non-dominated
        for i in 0..archive.len() {
            for j in 0..archive.len() {
                if i != j {
                    assert!(
                        !archive.solutions[i].dominates(&archive.solutions[j]),
                        "solution {} dominates {} after prune",
                        i,
                        j
                    );
                }
            }
        }
    }

    #[test]
    fn test_prune_explicit() {
        let mut archive = ParetoArchive::new(3);
        for i in 0..5usize {
            let t = i as f64 / 4.0 * std::f64::consts::PI / 2.0;
            archive.insert(sol([t.cos(), t.sin(), 0.5 + i as f64 * 0.05], i));
        }
        assert!(archive.len() <= 3, "expected <=3, got {}", archive.len());
    }

    // --- Hypervolume (spec-required test) ---

    #[test]
    fn test_hypervolume() {
        // Known geometry: two points A=[2,1,1] B=[1,2,1] with ref=[0,0,0]
        // Trace of WFG sweep gives HV=3.0 (verified analytically).
        let mut archive = ParetoArchive::new(10);
        archive.insert(sol([2.0, 1.0, 1.0], 0));
        archive.insert(sol([1.0, 2.0, 1.0], 1));
        let hv = archive.hypervolume([0.0, 0.0, 0.0]);
        assert!((hv - 3.0).abs() < 1e-9, "expected HV=3.0, got {hv}");
    }

    #[test]
    fn test_hypervolume_single_point() {
        let mut archive = ParetoArchive::new(10);
        archive.insert(sol([1.0, 1.0, 1.0], 0));
        let hv = archive.hypervolume([0.0, 0.0, 0.0]);
        assert!((hv - 1.0).abs() < 1e-9, "expected 1.0, got {hv}");
    }

    #[test]
    fn test_hypervolume_monotone_with_more_solutions() {
        let mut small = ParetoArchive::new(100);
        small.insert(sol([1.0, 0.5, 0.5], 0));

        let mut large = ParetoArchive::new(100);
        large.insert(sol([1.0, 0.5, 0.5], 0));
        large.insert(sol([0.5, 1.0, 0.5], 1));
        large.insert(sol([0.5, 0.5, 1.0], 2));

        assert!(large.hypervolume([0.0, 0.0, 0.0]) >= small.hypervolume([0.0, 0.0, 0.0]));
    }

    // --- best_for (spec-required test) ---

    #[test]
    fn test_best_for() {
        let mut archive = ParetoArchive::new(100);
        archive.insert(sol([0.9, 0.1, 0.5], 0)); // best on obj0
        archive.insert(sol([0.1, 0.9, 0.5], 1)); // best on obj1
        archive.insert(sol([0.5, 0.5, 0.9], 2)); // best on obj2

        let b0 = archive.best_for(0).unwrap();
        assert!((b0.objectives[0] - 0.9).abs() < 1e-9, "wrong best for obj0");

        let b1 = archive.best_for(1).unwrap();
        assert!((b1.objectives[1] - 0.9).abs() < 1e-9, "wrong best for obj1");

        let b2 = archive.best_for(2).unwrap();
        assert!((b2.objectives[2] - 0.9).abs() < 1e-9, "wrong best for obj2");
    }

    // --- Empty archive (spec-required test) ---

    #[test]
    fn test_empty_archive() {
        let mut archive = ParetoArchive::new(10);
        assert!(archive.is_empty());
        assert_eq!(archive.len(), 0);
        assert_eq!(archive.hypervolume([0.0, 0.0, 0.0]), 0.0);
        assert!(archive.best_for(0).is_none());
        assert!(archive.best_for(1).is_none());
        assert!(archive.best_for(2).is_none());

        let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
        assert!(archive.random_member(&mut rng).is_none());

        // prune on empty should not panic
        archive.prune();
        assert!(archive.is_empty());
    }

    // --- random_member ---

    #[test]
    fn test_random_member_returns_archive_element() {
        let mut archive = ParetoArchive::new(100);
        archive.insert(sol([1.0, 0.0, 0.0], 0));
        archive.insert(sol([0.0, 1.0, 0.0], 1));
        archive.insert(sol([0.0, 0.0, 1.0], 2));

        let mut rng = rand::rngs::SmallRng::seed_from_u64(7);
        for _ in 0..20 {
            let m = archive.random_member(&mut rng).unwrap();
            assert!(
                archive.solutions.iter().any(|s| std::ptr::eq(s, m)),
                "random_member returned a pointer not in the archive"
            );
        }
    }

    // --- Crowding distance ---

    #[test]
    fn test_crowding_distance_boundaries_infinite() {
        let solutions = vec![
            sol([0.0, 0.0, 0.0], 0),
            sol([0.5, 0.5, 0.5], 1),
            sol([1.0, 1.0, 1.0], 2),
        ];
        let dist = crowding_distance(&solutions);
        // Extremes of obj[0] are indices 0 and 2
        assert_eq!(dist[0], f64::INFINITY);
        assert_eq!(dist[2], f64::INFINITY);
        assert!(dist[1].is_finite());
    }

    #[test]
    fn test_crowding_distance_two_solutions() {
        let solutions = vec![sol([0.0, 0.0, 0.0], 0), sol([1.0, 1.0, 1.0], 1)];
        let dist = crowding_distance(&solutions);
        for d in &dist {
            assert_eq!(*d, f64::INFINITY);
        }
    }

    // --- Serialization ---

    #[test]
    fn test_serde_round_trip() {
        let mut archive = ParetoArchive::new(10);
        archive.insert(ParetoSolution::new(
            vec![1.0, 2.0],
            [0.9, 0.8, 0.7],
            vec![[0.1, 0.2, 0.3]],
            0,
        ));
        archive.insert(ParetoSolution::new(
            vec![3.0, 4.0],
            [0.7, 0.9, 0.8],
            vec![[0.4, 0.5, 0.6]],
            1,
        ));

        let json = serde_json::to_string(&archive).unwrap();
        let restored: ParetoArchive = serde_json::from_str(&json).unwrap();

        assert_eq!(restored.len(), archive.len());
        assert_eq!(restored.max_size, archive.max_size);
        for (orig, rest) in archive.solutions.iter().zip(restored.solutions.iter()) {
            assert_eq!(orig.objectives, rest.objectives);
            assert_eq!(orig.iteration, rest.iteration);
            assert_eq!(orig.scenario_scores, rest.scenario_scores);
        }
    }

    // --- scenario_scores type check ---

    #[test]
    fn test_scenario_scores_type() {
        let scores: Vec<[f64; 3]> = vec![[0.8, 0.7, 0.9], [0.6, 0.5, 0.7]];
        let s = ParetoSolution::new(vec![], [0.8, 0.7, 0.6], scores.clone(), 5);
        assert_eq!(s.scenario_scores, scores);
    }
}
