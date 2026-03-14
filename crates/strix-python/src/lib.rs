//! **strix-python** — PyO3 bindings for the STRIX drone swarm system.
//!
//! This crate is the cdylib target for maturin. It depends on both
//! `strix-core` and `strix-auction`, exposing their types to Python
//! without creating a cyclic dependency.

use pyo3::prelude::*;

pub mod auction_wrappers;
pub mod core_wrappers;

/// Python module entry point for `strix._strix_core`.
#[pymodule]
fn _strix_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    // Core types
    m.add_class::<core_wrappers::PyRegime>()?;
    m.add_class::<core_wrappers::PySensorConfig>()?;
    m.add_class::<core_wrappers::PyDroneState>()?;
    m.add_class::<core_wrappers::PyParticleNavFilter>()?;

    // Core functions
    m.add_function(wrap_pyfunction!(core_wrappers::py_detect_jamming, m)?)?;
    m.add_function(wrap_pyfunction!(core_wrappers::py_detect_regime, m)?)?;

    // Auction types
    m.add_class::<auction_wrappers::PyPosition>()?;
    m.add_class::<auction_wrappers::PyCapabilities>()?;
    m.add_class::<auction_wrappers::PyAuctionDroneState>()?;
    m.add_class::<auction_wrappers::PyTask>()?;
    m.add_class::<auction_wrappers::PyThreatType>()?;
    m.add_class::<auction_wrappers::PyThreatState>()?;
    m.add_class::<auction_wrappers::PyAssignment>()?;
    m.add_class::<auction_wrappers::PyAuctionResult>()?;
    m.add_class::<auction_wrappers::PyAuctioneer>()?;
    m.add_class::<auction_wrappers::PyLossAnalyzer>()?;

    Ok(())
}
