//! Command lifecycle types for the STRIX adapter layer.
//!
//! These types provide a structured way to track commands from submission
//! through completion, enabling the orchestrator to monitor in-flight
//! commands and react to failures or delays.

use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};

// ---------------------------------------------------------------------------
// CommandId
// ---------------------------------------------------------------------------

/// Unique command identifier assigned at submission time.
///
/// IDs are process-local, monotonically increasing, and never zero.
/// Two commands on the same drone will never share an ID within a
/// process lifetime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CommandId(pub u64);

static NEXT_COMMAND_ID: AtomicU64 = AtomicU64::new(1);

impl CommandId {
    /// Generate a new unique command ID.
    ///
    /// Thread-safe and wait-free (uses `Relaxed` ordering — IDs only need
    /// to be unique, not ordered relative to other memory operations).
    pub fn next() -> Self {
        Self(NEXT_COMMAND_ID.fetch_add(1, Ordering::Relaxed))
    }
}

impl std::fmt::Display for CommandId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Cmd#{}", self.0)
    }
}

// ---------------------------------------------------------------------------
// CommandAcceptance
// ---------------------------------------------------------------------------

/// Whether a command was accepted by the platform.
///
/// Returned immediately from [`CommandSink::submit_waypoint`] and
/// [`CommandSink::submit_action`]. Acceptance does NOT mean completion —
/// use [`CommandSink::command_status`] to track lifecycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CommandAcceptance {
    /// Command accepted and assigned this ID.
    ///
    /// Use the [`CommandId`] to poll [`CommandSink::command_status`].
    Accepted(CommandId),
    /// Command rejected with a human-readable reason.
    Rejected(String),
}

impl CommandAcceptance {
    /// Return the `CommandId` if accepted, `None` if rejected.
    pub fn id(&self) -> Option<CommandId> {
        match self {
            Self::Accepted(id) => Some(*id),
            Self::Rejected(_) => None,
        }
    }

    /// `true` if the command was accepted.
    pub fn is_accepted(&self) -> bool {
        matches!(self, Self::Accepted(_))
    }

    /// `true` if the command was rejected.
    pub fn is_rejected(&self) -> bool {
        matches!(self, Self::Rejected(_))
    }
}

// ---------------------------------------------------------------------------
// CommandOutcome
// ---------------------------------------------------------------------------

/// Outcome of a previously accepted command.
///
/// Returned by [`CommandSink::command_status`]. Adapters that do not support
/// fine-grained status tracking may return `Completed` immediately after
/// acceptance or `Unknown` for expired IDs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CommandOutcome {
    /// Command is queued but not yet executing.
    Pending,
    /// Command is actively being executed.
    InProgress,
    /// Command completed successfully.
    Completed,
    /// Command failed with a reason.
    Failed(String),
    /// Command timed out before completion.
    TimedOut,
    /// Command ID not found (expired or never issued).
    Unknown,
}

impl CommandOutcome {
    /// `true` if this is a terminal state (no further transitions expected).
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            Self::Completed | Self::Failed(_) | Self::TimedOut | Self::Unknown
        )
    }

    /// `true` if the command succeeded.
    pub fn is_success(&self) -> bool {
        matches!(self, Self::Completed)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn command_id_uniqueness() {
        let ids: HashSet<u64> = (0..100).map(|_| CommandId::next().0).collect();
        assert_eq!(ids.len(), 100, "all 100 IDs must be unique");
    }

    #[test]
    fn command_id_nonzero() {
        // IDs start at 1, never 0.
        for _ in 0..20 {
            assert_ne!(CommandId::next().0, 0);
        }
    }

    #[test]
    fn command_acceptance_helpers() {
        let id = CommandId::next();
        let accepted = CommandAcceptance::Accepted(id);
        assert!(accepted.is_accepted());
        assert!(!accepted.is_rejected());
        assert_eq!(accepted.id(), Some(id));

        let rejected = CommandAcceptance::Rejected("not armed".into());
        assert!(!rejected.is_accepted());
        assert!(rejected.is_rejected());
        assert_eq!(rejected.id(), None);
    }

    #[test]
    fn command_outcome_terminal() {
        assert!(!CommandOutcome::Pending.is_terminal());
        assert!(!CommandOutcome::InProgress.is_terminal());
        assert!(CommandOutcome::Completed.is_terminal());
        assert!(CommandOutcome::Failed("x".into()).is_terminal());
        assert!(CommandOutcome::TimedOut.is_terminal());
        assert!(CommandOutcome::Unknown.is_terminal());
    }

    #[test]
    fn command_outcome_success() {
        assert!(CommandOutcome::Completed.is_success());
        assert!(!CommandOutcome::Failed("x".into()).is_success());
        assert!(!CommandOutcome::TimedOut.is_success());
    }
}
