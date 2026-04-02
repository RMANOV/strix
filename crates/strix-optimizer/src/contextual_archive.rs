use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::pareto::{ParetoArchive, ParetoSolution};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct OptimizationContext {
    pub doctrine: String,
    pub scenario_family: String,
    pub environment: String,
    pub regime: String,
}

impl OptimizationContext {
    pub fn key(&self) -> String {
        format!(
            "{}:{}:{}:{}",
            self.doctrine, self.scenario_family, self.environment, self.regime
        )
    }
}

impl Default for OptimizationContext {
    fn default() -> Self {
        Self {
            doctrine: "balanced".to_string(),
            scenario_family: "mixed".to_string(),
            environment: "default".to_string(),
            regime: "patrol".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextualArchiveConfig {
    pub per_context_max_size: usize,
    pub global_max_size: usize,
    pub forgetting_window: usize,
    pub migration_elites: usize,
}

impl Default for ContextualArchiveConfig {
    fn default() -> Self {
        Self {
            per_context_max_size: 48,
            global_max_size: 128,
            forgetting_window: 200,
            migration_elites: 3,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextualArchive {
    pub config: ContextualArchiveConfig,
    pub global: ParetoArchive,
    pub contexts: BTreeMap<OptimizationContext, ParetoArchive>,
    pub last_seen_iteration: BTreeMap<OptimizationContext, usize>,
}

impl ContextualArchive {
    pub fn new(config: ContextualArchiveConfig) -> Self {
        Self {
            global: ParetoArchive::new(config.global_max_size),
            contexts: BTreeMap::new(),
            last_seen_iteration: BTreeMap::new(),
            config,
        }
    }

    pub fn insert(&mut self, context: OptimizationContext, candidate: ParetoSolution) -> bool {
        let iteration = candidate.iteration;
        let accepted_global = self.global.insert(candidate.clone());
        let accepted_local = self
            .contexts
            .entry(context.clone())
            .or_insert_with(|| ParetoArchive::new(self.config.per_context_max_size))
            .insert(candidate);
        self.last_seen_iteration.insert(context, iteration);
        accepted_global || accepted_local
    }

    pub fn archive_for(&self, context: &OptimizationContext) -> Option<&ParetoArchive> {
        self.contexts.get(context)
    }

    pub fn contexts(&self) -> impl Iterator<Item = (&OptimizationContext, &ParetoArchive)> {
        self.contexts.iter()
    }

    pub fn forget_stale(&mut self, current_iteration: usize) -> usize {
        let stale_contexts: Vec<OptimizationContext> = self
            .last_seen_iteration
            .iter()
            .filter_map(|(context, last_seen)| {
                (current_iteration.saturating_sub(*last_seen) > self.config.forgetting_window)
                    .then_some(context.clone())
            })
            .collect();

        let mut removed = 0;
        for context in stale_contexts {
            let retain_window = (self.config.forgetting_window / 2).max(1);
            let became_empty = if let Some(archive) = self.contexts.get_mut(&context) {
                let before = archive.solutions.len();
                archive
                    .solutions
                    .retain(|solution| current_iteration.saturating_sub(solution.iteration) <= retain_window);
                removed += before.saturating_sub(archive.solutions.len());
                archive.solutions.is_empty()
            } else {
                false
            };

            if became_empty {
                self.contexts.remove(&context);
                self.last_seen_iteration.remove(&context);
            }
        }

        removed
    }

    pub fn migrate_elites(
        &mut self,
        from: &OptimizationContext,
        to: &OptimizationContext,
    ) -> usize {
        let Some(source) = self.contexts.get(from) else {
            return 0;
        };
        let elites = elite_solutions(source, self.config.migration_elites);
        if elites.is_empty() {
            return 0;
        }

        let destination = self
            .contexts
            .entry(to.clone())
            .or_insert_with(|| ParetoArchive::new(self.config.per_context_max_size));
        let mut migrated = 0;
        for solution in elites {
            if destination.insert(solution) {
                migrated += 1;
            }
        }
        migrated
    }
}

fn elite_solutions(archive: &ParetoArchive, count: usize) -> Vec<ParetoSolution> {
    let mut ranked = archive.solutions.clone();
    ranked.sort_by(|left, right| {
        objective_sum(&right.objectives)
            .total_cmp(&objective_sum(&left.objectives))
            .then_with(|| right.iteration.cmp(&left.iteration))
    });
    ranked.truncate(count.max(1));
    ranked
}

fn objective_sum(objectives: &[f64; 3]) -> f64 {
    objectives
        .iter()
        .filter(|value| value.is_finite())
        .copied()
        .sum::<f64>()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solution(iteration: usize, objectives: [f64; 3]) -> ParetoSolution {
        ParetoSolution::new(vec![iteration as f64], objectives, vec![objectives], iteration)
    }

    #[test]
    fn separates_contexts() {
        let mut archive = ContextualArchive::new(ContextualArchiveConfig::default());
        let patrol = OptimizationContext {
            regime: "patrol".into(),
            ..OptimizationContext::default()
        };
        let evade = OptimizationContext {
            regime: "evade".into(),
            ..OptimizationContext::default()
        };

        archive.insert(patrol.clone(), solution(1, [0.8, 0.3, 0.4]));
        archive.insert(evade.clone(), solution(2, [0.3, 0.9, 0.7]));

        assert_eq!(archive.archive_for(&patrol).unwrap().len(), 1);
        assert_eq!(archive.archive_for(&evade).unwrap().len(), 1);
        assert_eq!(archive.global.len(), 2);
    }

    #[test]
    fn migrates_elites_across_contexts() {
        let mut archive = ContextualArchive::new(ContextualArchiveConfig {
            migration_elites: 2,
            ..ContextualArchiveConfig::default()
        });
        let patrol = OptimizationContext::default();
        let engage = OptimizationContext {
            regime: "engage".into(),
            ..OptimizationContext::default()
        };

        archive.insert(patrol.clone(), solution(1, [0.9, 0.4, 0.3]));
        archive.insert(patrol.clone(), solution(2, [0.7, 0.7, 0.7]));

        let migrated = archive.migrate_elites(&patrol, &engage);
        assert!(migrated >= 1);
        assert!(!archive.archive_for(&engage).unwrap().is_empty());
    }

    #[test]
    fn forgets_stale_contexts() {
        let mut archive = ContextualArchive::new(ContextualArchiveConfig {
            forgetting_window: 4,
            ..ContextualArchiveConfig::default()
        });
        let patrol = OptimizationContext::default();
        archive.insert(patrol.clone(), solution(1, [0.6, 0.6, 0.6]));
        archive.insert(patrol.clone(), solution(2, [0.7, 0.5, 0.5]));

        let removed = archive.forget_stale(10);
        assert!(removed >= 1);
    }
}
