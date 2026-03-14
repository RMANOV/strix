//! # Digital Pheromone Fields (Stigmergy)
//!
//! Bio-inspired indirect communication for emergent swarm behaviour.
//! Each pheromone deposit is ~20 bytes, enabling ultra-low-bandwidth
//! coordination of:
//!
//! - **Area coverage** (`Explored` pheromone — avoid redundancy)
//! - **Threat avoidance** (`Threat` pheromone — repels drones)
//! - **Target investigation** (`Target` pheromone — attracts drones)
//! - **Regrouping** (`Rally` pheromone — congregation)
//! - **Path optimization** (`Corridor` pheromone — safe transit)
//!
//! The pheromone field is stored on a 3-D spatial grid with configurable
//! resolution. All operations are O(1) per cell.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::{NodeId, Position3D};

// ---------------------------------------------------------------------------
// Pheromone types
// ---------------------------------------------------------------------------

/// Category of pheromone — determines behavioural response.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PheromoneType {
    /// "I've been here" — repels further exploration to avoid redundant coverage.
    Explored,
    /// "Danger here" — repels drones from hazardous areas.
    Threat,
    /// "Interesting target" — attracts investigation by nearby drones.
    Target,
    /// "Regroup here" — congregation / rally point.
    Rally,
    /// "Safe path" — attracts transit along established corridors.
    Corridor,
}

// ---------------------------------------------------------------------------
// Pheromone deposit
// ---------------------------------------------------------------------------

/// A single pheromone deposit event (~20 bytes on the wire).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pheromone {
    /// World position of the deposit.
    pub position: Position3D,
    /// What kind of pheromone.
    pub ptype: PheromoneType,
    /// Intensity (arbitrary units, decays over time).
    pub intensity: f64,
    /// Simulation timestamp of the deposit.
    pub timestamp: f64,
    /// Which drone deposited it.
    pub depositor: NodeId,
}

// ---------------------------------------------------------------------------
// Grid cell key
// ---------------------------------------------------------------------------

/// Discretised 3-D cell index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
struct CellKey {
    x: i64,
    y: i64,
    z: i64,
}

// ---------------------------------------------------------------------------
// Cell contents
// ---------------------------------------------------------------------------

/// Pheromone concentration stored per cell per type.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CellData {
    /// Concentration per pheromone type.
    concentrations: HashMap<PheromoneType, f64>,
    /// Last-update timestamp per type (for lazy evaporation).
    last_update: HashMap<PheromoneType, f64>,
}

impl CellData {
    fn new() -> Self {
        Self {
            concentrations: HashMap::new(),
            last_update: HashMap::new(),
        }
    }

    /// Get concentration after applying lazy evaporation.
    fn get(&self, ptype: PheromoneType, now: f64, decay_rate: f64) -> f64 {
        let raw = self.concentrations.get(&ptype).copied().unwrap_or(0.0);
        let last = self.last_update.get(&ptype).copied().unwrap_or(now);
        let dt = (now - last).max(0.0);
        raw * (-decay_rate * dt).exp()
    }

    /// Deposit and return new concentration.
    fn deposit(&mut self, ptype: PheromoneType, amount: f64, now: f64, decay_rate: f64) -> f64 {
        // First evaporate existing.
        let current = self.get(ptype, now, decay_rate);
        let new_val = current + amount;
        self.concentrations.insert(ptype, new_val);
        self.last_update.insert(ptype, now);
        new_val
    }

    /// Eagerly evaporate all types to `now`.
    fn evaporate_all(&mut self, now: f64, decay_rate: f64) {
        let types: Vec<PheromoneType> = self.concentrations.keys().copied().collect();
        for ptype in types {
            let val = self.get(ptype, now, decay_rate);
            if val < 1e-6 {
                self.concentrations.remove(&ptype);
                self.last_update.remove(&ptype);
            } else {
                self.concentrations.insert(ptype, val);
                self.last_update.insert(ptype, now);
            }
        }
    }

    /// True when all concentrations are essentially zero.
    fn is_empty(&self) -> bool {
        self.concentrations.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Pheromone field
// ---------------------------------------------------------------------------

/// 3-D spatial grid storing pheromone concentrations.
///
/// Resolution is configurable (default 10 m cells). All lookups are O(1)
/// via `HashMap<CellKey, CellData>`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PheromoneField {
    /// Cell resolution in metres.
    resolution: f64,
    /// Exponential decay rate (per second).
    decay_rate: f64,
    /// Grid storage (sparse — only cells with non-zero pheromone exist).
    cells: HashMap<CellKey, CellData>,
}

impl PheromoneField {
    /// Create a new empty field.
    ///
    /// - `resolution`: cell edge length in metres (default 10.0).
    /// - `decay_rate`: exponential decay constant per second (default 0.05).
    pub fn new(resolution: f64, decay_rate: f64) -> Self {
        Self {
            resolution: resolution.max(0.1),
            decay_rate: decay_rate.max(0.0),
            cells: HashMap::new(),
        }
    }

    /// Create a field with default parameters (10 m cells, 0.05/s decay).
    pub fn default_field() -> Self {
        Self::new(10.0, 0.05)
    }

    /// Number of cells currently holding pheromone.
    pub fn active_cells(&self) -> usize {
        self.cells.len()
    }

    // --- Core operations (all O(1) per cell) ---

    /// Deposit pheromone at a position.
    pub fn deposit(&mut self, pheromone: &Pheromone) {
        let key = self.pos_to_key(&pheromone.position);
        let cell = self.cells.entry(key).or_insert_with(CellData::new);
        cell.deposit(
            pheromone.ptype,
            pheromone.intensity,
            pheromone.timestamp,
            self.decay_rate,
        );
    }

    /// Read pheromone level at a position (with lazy evaporation).
    /// Returns the concentration of the requested type.
    pub fn sense(&self, position: &Position3D, ptype: PheromoneType, now: f64) -> f64 {
        let key = self.pos_to_key(position);
        self.cells
            .get(&key)
            .map_or(0.0, |cell| cell.get(ptype, now, self.decay_rate))
    }

    /// Read all pheromone levels at a position.
    pub fn sense_all(&self, position: &Position3D, now: f64) -> HashMap<PheromoneType, f64> {
        let key = self.pos_to_key(position);
        let mut result = HashMap::new();
        if let Some(cell) = self.cells.get(&key) {
            for &ptype in &[
                PheromoneType::Explored,
                PheromoneType::Threat,
                PheromoneType::Target,
                PheromoneType::Rally,
                PheromoneType::Corridor,
            ] {
                let val = cell.get(ptype, now, self.decay_rate);
                if val > 1e-6 {
                    result.insert(ptype, val);
                }
            }
        }
        result
    }

    /// Eagerly evaporate all cells to `now`. Removes dead cells.
    pub fn evaporate(&mut self, now: f64) {
        let decay = self.decay_rate;
        self.cells.retain(|_key, cell| {
            cell.evaporate_all(now, decay);
            !cell.is_empty()
        });
    }

    /// Compute the pheromone gradient at a position for a given type.
    ///
    /// Returns a 3-D vector `[dx, dy, dz]` pointing in the direction of
    /// increasing concentration (for attractive pheromones) or you can
    /// negate it for repulsive navigation.
    ///
    /// Uses central-difference across neighbouring cells.
    pub fn gradient(&self, position: &Position3D, ptype: PheromoneType, now: f64) -> [f64; 3] {
        let r = self.resolution;
        let p = position.0;

        let sample = |dx: f64, dy: f64, dz: f64| -> f64 {
            let pos = Position3D([p[0] + dx, p[1] + dy, p[2] + dz]);
            self.sense(&pos, ptype, now)
        };

        let gx = (sample(r, 0.0, 0.0) - sample(-r, 0.0, 0.0)) / (2.0 * r);
        let gy = (sample(0.0, r, 0.0) - sample(0.0, -r, 0.0)) / (2.0 * r);
        let gz = (sample(0.0, 0.0, r) - sample(0.0, 0.0, -r)) / (2.0 * r);

        [gx, gy, gz]
    }

    // --- Helpers ---

    /// Convert a world position to a discrete cell key.
    fn pos_to_key(&self, pos: &Position3D) -> CellKey {
        let inv = 1.0 / self.resolution;
        CellKey {
            x: (pos.0[0] * inv).floor() as i64,
            y: (pos.0[1] * inv).floor() as i64,
            z: (pos.0[2] * inv).floor() as i64,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn deposit_at(
        field: &mut PheromoneField,
        x: f64,
        y: f64,
        z: f64,
        ptype: PheromoneType,
        intensity: f64,
        t: f64,
    ) {
        field.deposit(&Pheromone {
            position: Position3D([x, y, z]),
            ptype,
            intensity,
            timestamp: t,
            depositor: NodeId(0),
        });
    }

    #[test]
    fn deposit_and_sense() {
        let mut field = PheromoneField::new(10.0, 0.0); // no decay
        deposit_at(&mut field, 5.0, 5.0, 5.0, PheromoneType::Explored, 1.0, 0.0);

        let val = field.sense(&Position3D([5.0, 5.0, 5.0]), PheromoneType::Explored, 0.0);
        assert!((val - 1.0).abs() < 1e-10);
    }

    #[test]
    fn deposit_accumulates() {
        let mut field = PheromoneField::new(10.0, 0.0);
        deposit_at(&mut field, 5.0, 5.0, 5.0, PheromoneType::Threat, 1.0, 0.0);
        deposit_at(&mut field, 5.0, 5.0, 5.0, PheromoneType::Threat, 2.0, 0.0);

        let val = field.sense(&Position3D([5.0, 5.0, 5.0]), PheromoneType::Threat, 0.0);
        assert!((val - 3.0).abs() < 1e-10);
    }

    #[test]
    fn evaporation_decays() {
        let decay = 0.1;
        let mut field = PheromoneField::new(10.0, decay);
        deposit_at(&mut field, 5.0, 5.0, 5.0, PheromoneType::Target, 10.0, 0.0);

        // At t=10, concentration = 10 * exp(-0.1 * 10) = 10 * e^-1 ≈ 3.679
        let val = field.sense(&Position3D([5.0, 5.0, 5.0]), PheromoneType::Target, 10.0);
        let expected = 10.0 * (-1.0_f64).exp();
        assert!(
            (val - expected).abs() < 0.01,
            "got {val}, expected {expected}"
        );
    }

    #[test]
    fn evaporate_removes_dead_cells() {
        let mut field = PheromoneField::new(10.0, 10.0); // very fast decay
        deposit_at(&mut field, 5.0, 5.0, 5.0, PheromoneType::Explored, 1.0, 0.0);
        assert_eq!(field.active_cells(), 1);

        field.evaporate(100.0); // way past decay
        assert_eq!(field.active_cells(), 0);
    }

    #[test]
    fn gradient_points_toward_concentration() {
        let mut field = PheromoneField::new(10.0, 0.0);
        // Deposit a strong source at (100, 0, 0).
        deposit_at(
            &mut field,
            100.0,
            0.0,
            0.0,
            PheromoneType::Rally,
            100.0,
            0.0,
        );

        // Gradient at (95, 0, 0) — central difference probes (105, 0, 0)
        // which falls in the same cell as the deposit, and (85, 0, 0) which
        // is empty. This gives a positive x gradient.
        let grad = field.gradient(&Position3D([95.0, 0.0, 0.0]), PheromoneType::Rally, 0.0);
        assert!(
            grad[0] > 0.0,
            "gradient x should be positive toward source, got {}",
            grad[0]
        );
    }

    #[test]
    fn sense_all_returns_multiple_types() {
        let mut field = PheromoneField::new(10.0, 0.0);
        deposit_at(&mut field, 5.0, 5.0, 5.0, PheromoneType::Explored, 1.0, 0.0);
        deposit_at(&mut field, 5.0, 5.0, 5.0, PheromoneType::Threat, 2.0, 0.0);

        let all = field.sense_all(&Position3D([5.0, 5.0, 5.0]), 0.0);
        assert_eq!(all.len(), 2);
        assert!((all[&PheromoneType::Explored] - 1.0).abs() < 1e-10);
        assert!((all[&PheromoneType::Threat] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn different_cells_are_independent() {
        let mut field = PheromoneField::new(10.0, 0.0);
        deposit_at(&mut field, 5.0, 5.0, 5.0, PheromoneType::Corridor, 3.0, 0.0);

        // Different cell.
        let val = field.sense(
            &Position3D([50.0, 50.0, 50.0]),
            PheromoneType::Corridor,
            0.0,
        );
        assert!((val).abs() < 1e-10);
    }

    #[test]
    fn default_field_params() {
        let field = PheromoneField::default_field();
        assert!((field.resolution - 10.0).abs() < 1e-10);
        assert!((field.decay_rate - 0.05).abs() < 1e-10);
    }
}
