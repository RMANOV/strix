//! # strix-xai — Glass Box Explainability Engine
//!
//! Every decision the STRIX swarm orchestrator makes is recorded as a
//! [`DecisionTrace`](trace::DecisionTrace) — a structured log entry that
//! captures inputs, reasoning steps, alternatives considered, and the
//! final output.
//!
//! The three pillars of the explainability engine:
//!
//! - **[`trace`]** — Record decisions with full causal chains.
//! - **[`narrator`]** — Convert machine traces into human-readable text.
//! - **[`replay`]** — After-action review, what-if analysis, and timeline export.

pub mod correlation;
pub mod narrator;
pub mod reason_codes;
pub mod replay;
pub mod trace;
