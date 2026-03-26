//! Optimizer configuration (CLI-overridable, serde-compatible).

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::evaluator::DoctrineProfile;
use crate::smco::SmcoConfig;

/// Top-level configuration for the STRIX SMCO optimizer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerConfig {
    /// Core SMCO algorithm parameters.
    pub smco: SmcoConfig,
    /// Scenario names to include in evaluation (empty = all defaults).
    pub scenarios: Vec<String>,
    /// Doctrine profile used to shape scenario weighting and objectives.
    pub doctrine_profile: DoctrineProfile,
    /// Output JSON path.
    pub output_path: PathBuf,
    /// Number of rayon threads (0 = auto).
    pub threads: usize,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            smco: SmcoConfig::default(),
            scenarios: Vec::new(),
            doctrine_profile: DoctrineProfile::Balanced,
            output_path: PathBuf::from("optimization_results.json"),
            threads: 0,
        }
    }
}
