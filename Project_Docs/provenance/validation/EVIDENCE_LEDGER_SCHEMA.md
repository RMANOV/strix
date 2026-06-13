# STRIX Evidence-Ledger Artifact Schema (Validation-Phase Design)

> **Status: validation-phase design — no integration code exists today.**
> This document specifies the *intended future* handoff from existing STRIX
> outputs to a governed mission-decision record ("evidence ledger"). It is
> **documentation only**: there is no ledger implementation, no persistence
> layer, and **no STRIX integration into any external memory system (e.g.
> sqlite-memory) — that seam is a future validation target, not a shipped or
> in-progress capability.** Nothing here changes the safety story: classical
> CBF remains the only safety guarantee, and all source evidence is
> software-simulation evidence only.

Governing claim map: [`Project_Docs/CAPABILITY_BOUNDARY.md`](../../CAPABILITY_BOUNDARY.md).

## 1. Purpose

STRIX already emits structured, replayable decision artifacts (DecisionTrace,
BattleReport, replay JSON). What does **not** exist is a single governed record
format that binds each consequential mission decision to its provenance
(scenario, seed, config hash, commit), its integrity hash, and its review
status. This schema defines that target record so the validation phase can
implement the handoff without re-deriving it from code.

Every per-source section below answers three questions about **real, current**
code: what the type is (exact name + path), when it is emitted, and what a
governed mission-decision record would capture from it.

## 2. Source artifact inventory (current, real types)

> Validation-phase design — no integration code exists today. The types below
> are the *existing* in-tree producers the future ledger would consume.

### 2.1 BattleReport — full-run simulation output

- **Type:** `BattleReport` — `crates/strix-playground/src/report.rs` (serde
  Serialize/Deserialize).
- **Fields:** `scenario_name: String`, `duration: f64`, `n_drones_initial:
  usize`, `n_threats_initial: usize`, `timeline: Vec<TimelineEntry>`,
  `aggregates: Aggregates`, `per_drone: HashMap<u32, DroneSummary>`,
  `tick_data: Option<Vec<TickSnapshot>>`.
- **Supporting types (same file):** `TimelineEntry { time: f64, event_type:
  TimelineEventType }`; `TimelineEventType` variants include `RegimeChange`,
  `HysteresisBlock`, `DroneLost`, `GpsJammed`, `GpsRestored`, `CusumFired`,
  `CbfCorrection`, `AuctionRound`, `ThreatSpawned`, `ForcedEvade`,
  `WindChanged`, `NfzAdded`; `Aggregates` carries the CBF counters
  (`cbf_activation_ticks`, `cbf_constraints_total`, `cbf_constraints_peak`,
  `cbf_burden_mean`), auction/coordination/gossip/formation statistics,
  `drones_lost`, `drones_survived`, `kill_zones_created`,
  `forced_evade_count`, battery stats; `DroneSummary`; `TickSnapshot`.
- **When emitted:** once per simulation run, built by `build_report()`
  (`crates/strix-playground/src/engine.rs`, line 538) at the end of
  `Engine::run()`.
- **Governed record would capture:** the full report as the run-level evidence
  payload, bound to scenario id, seed, config hash, and commit (none of which
  `BattleReport` itself carries today — `scenario_name` is a free-form string).

### 2.2 DecisionTrace — per-decision audit record

- **Type:** `DecisionTrace` — `crates/strix-xai/src/trace.rs` (serde).
- **Fields:** `id: u64`, `timestamp: f64`, `decision_type: DecisionType`,
  `inputs: TraceInputs`, `reasoning: Vec<ReasoningStep>`, `output:
  TraceOutput`, `confidence: f64`, `alternatives_considered:
  Vec<Alternative>`.
- **Supporting types (same file):** `TraceInputs { drone_ids: Vec<u32>,
  regime: String, metrics: serde_json::Value, context: serde_json::Value,
  fear_level: Option<f64>, courage_level: Option<f64>, tension: Option<f64>,
  calibration_quality: Option<f64> }`; `ReasoningStep { step: u32,
  description: String, data: serde_json::Value }`; `TraceOutput { action:
  String, details: serde_json::Value }`; `Alternative { description: String,
  score: f64, rejection_reason: String }`.
- **When emitted** (all in `crates/strix-swarm/src/tick.rs`):
  `DecisionType::ReAuction` (line 608), `RegimeChange` (lines 1132, 1278),
  `TaskAssignment` (line 1401), `CriticalityAdjustment` (line 2057),
  `SafetyClamp` (line 2233), `ThreatResponse` (line 2506). The variants
  `EpistemicEscalation`, `EpistemicConflict`, `EpistemicVacuum`,
  `FormationChange`, and `LeaderElection` are defined in the enum but are
  **not currently emitted** anywhere in the workspace.
- **When emitted (storage):** recorded into the in-memory `TraceRecorder` log
  (`crates/strix-xai/src/trace.rs`), with JSON export/import. **Bounded, not
  unbounded append-only:** `record()` keeps at most 10,000 traces and evicts the
  oldest at the cap (`self.traces.remove(0)`, FIFO ring buffer), so long runs or
  large imports silently drop the earliest decisions — a future ledger consumer
  must not treat the recorder as a complete append-only source.
- **Governed record would capture:** the trace verbatim as payload, plus the
  provenance envelope and an integrity hash; the ledger adds durable,
  tamper-evident storage that the in-memory recorder does not provide.

### 2.3 ROE denial / escalation outcomes

- **Type:** `EngagementAuth` — `crates/strix-core/src/roe.rs`, line 56
  (serde). Variants: `Authorized { conditions: Vec<String> }`, `Denied {
  reason: String }`, `EscalationRequired { reason: String, urgency:
  EscalationUrgency }` with `EscalationUrgency::{Routine, Priority,
  Immediate}`.
- **Context types (same file):** `RoeEngine`, `EngagementContext`,
  `WeaponsPosture::{WeaponsHold, WeaponsTight, WeaponsFree}`,
  `ThreatClassification::{ConfirmedHostile, SuspectedHostile, Unknown,
  Friendly, Civilian}`.
- **When emitted:** returned inline by `RoeEngine::authorize_engagement()` at
  gate-decision time. Denial / escalation outcomes **are** persisted per-event
  today: the swarm tick loop calls `record_roe_trace()`
  (`crates/strix-swarm/src/tick.rs`, line 2498; invoked on the denial /
  escalation paths at lines 1197 / 1250 / 1258), which records a
  `DecisionType::ThreatResponse` `DecisionTrace` (regime `"ROE"`) into the
  bounded (10,000-entry FIFO) in-memory `TraceRecorder` carrying the task id, threat distance, action,
  and a formatted reason string. **Gap (honest):** that trace is *lossy* — it
  does not carry the structured `EngagementAuth` variant payload (typed
  `reason` / `urgency`), the `EngagementContext` snapshot, or the deciding
  `WeaponsPosture`; the `EngagementAuth` enum is not serialized verbatim, and
  `Authorized` outcomes are not traced at all (the helper covers only the
  denial / escalation paths).
- **Governed record would capture:** one record per `Denied` /
  `EscalationRequired` outcome — the structured variant payload (typed reason,
  urgency), the `EngagementContext` snapshot that drove it, and the deciding
  posture — replacing the current free-text `ThreatResponse` trace with a
  typed, context-complete ROE artifact.

### 2.4 CBF safety interventions

- **Types:** `CbfResult` — `crates/strix-core/src/cbf.rs` (fields:
  `safe_velocity: Vector3<f64>`, `correction: Vector3<f64>`, `any_active:
  bool`, `active_count: u32`; **no serde derives** — transient by design).
  Aggregated counters live in `Aggregates` (section 2.1). A
  `TimelineEventType::CbfCorrection { drone_id, magnitude }` variant exists
  in `crates/strix-playground/src/report.rs` but is **not currently emitted**.
- **When emitted:** computed every tick by `cbf_filter()` /
  `cbf_filter_with_neighbor_states()`; when any constraint fires, the tick
  loop emits a `DecisionType::SafetyClamp` trace
  (`crates/strix-swarm/src/tick.rs`, line 2233) whose `inputs.metrics`
  carries `active_constraints`, `deadlock_escapes`, `neural_agents`,
  `classical_agents`, and whose `output` is `"safety_clamp"` with
  `{constraints, escapes}`.
- **Governed record would capture:** each SafetyClamp trace as a
  safety-intervention record (see Sample A), preserving the
  classical-vs-neural agent split — which documents in evidence that the
  active safety mechanism is classical CBF (`classical_agents`), consistent
  with `fallback_to_classical: true` being the default
  (`crates/strix-core/src/gcbf/config.rs`, line 37). A future refinement may
  also persist per-drone correction magnitudes (today only aggregate counts
  survive the tick).

### 2.5 Authority / fallback events

- **Current reality (honest):** there is **no discrete authority-transition
  or fallback event type** in the codebase today. Fallback behavior is
  visible only indirectly:
  - `EwResponse::GossipFallback { reduced_fanout, priority_only }` and
    `EwResponse::InertialFallback` (variants of `EwResponse`, used inside
    `EwResponsePlan.actions`) (`crates/strix-core/src/ew_response.rs`,
    lines 77/93);
  - the `classical_agents` / `neural_agents` counts inside SafetyClamp trace
    metrics (section 2.4), which record GCBF+→classical fallback in effect.
- **Governed record would capture:** an explicit authority/fallback record
  (which mechanism held safety authority for which agents, and why) derived
  from those existing signals. This is a **future validation-phase
  derivation**, not an existing event stream.

### 2.6 Loss and re-auction events

- **Types:** `LossRecord` — `crates/strix-auction/src/antifragile.rs`, line
  73 (serde). Fields: `drone_id: u32`, `position: Position`, `altitude: f64`,
  `heading: f64`, `velocity: [f64; 3]`, `threat_bearing: Option<f64>`,
  `regime_at_loss: Regime`, `classification: LossClassification`,
  `orphaned_tasks: Vec<u32>`, `timestamp: f64`.
  `LossClassification::{Sam, SmallArms, Collision, ElectronicWarfare,
  Unknown}` (line 24). `KillZone { center, radius, penalty, classification,
  loss_count }` (line 98).
- **When emitted:** `LossAnalyzer::record_loss()` accepts a caller-built
  `LossRecord` (the tick loop constructs it), stores it, and returns its
  orphaned task IDs; the tick loop then emits a
  `DecisionType::ReAuction` trace (`crates/strix-swarm/src/tick.rs`, line
  608) and the playground timeline logs `TimelineEventType::DroneLost`
  (`crates/strix-playground/src/engine.rs`).
- **Governed record would capture:** the loss record plus its consequence
  chain — orphaned tasks, the ReAuction trace id, resulting kill zones — as
  one linked attrition/recovery record (see Sample B).

### 2.7 Degraded-mode / EW events

- **Types:** `EwEvent` — `crates/strix-core/src/ew_response.rs`, line 54
  (serde). Fields: `threat: EwThreat`, `severity: EwSeverity`,
  `source_bearing: Option<f64>`, `source_range: Option<f64>`, `confidence:
  f64`, `timestamp: f64`. `EwThreat::{GpsDenial, GpsSpoofing, CommsJamming,
  RadarLock, DirectedEnergy}` (line 26); `EwSeverity::{Detected, Degraded,
  Severe, Denied}` (line 41); `EwResponsePlan { event, actions, summary }`
  (line 100) with `EwResponse` actions (line 73).
- **When emitted:** `EwEngine::respond()` produces an `EwResponsePlan` per
  detected event and tracks active events; the playground timeline records
  `GpsJammed` / `GpsRestored` markers.
- **Governed record would capture:** the event + chosen response plan as a
  degraded-mode evidence record — documenting that responses are defensive
  (noise expansion, gossip/inertial fallback, evasion, terrain masking, zone
  marking, monitoring).

## 3. Governed record envelope (future schema)

> Validation-phase design — no integration code exists today.

Each ledger record wraps one source payload (sections 2.1–2.7) in a common
envelope. Provenance fields deliberately reuse what the existing replay
harness already computes (`scripts/strix_sim_replay.py`: `scenario_hash()`
line 175 — SHA-256 of scenario YAML, first 16 hex chars — and the `repo`
block with `commit`, `branch`, `working_tree_clean`):

| Envelope field | Type | Source |
|---|---|---|
| `record_id` | string (ULID/UUID) | ledger-assigned (future) |
| `schema_version` | integer | this document, v1 draft |
| `kind` | string enum | `safety_intervention` \| `roe_outcome` \| `loss_reauction` \| `ew_degraded` \| `authority_fallback` \| `run_report` |
| `mission.scenario_id`, `mission.seed` | string, u64 | scenario YAML (`sim/scenarios/*.yaml`). **For Rust-produced payloads (`safety_intervention`, `loss_reauction`), `mission.seed` is provenance metadata only — not a determinism key**: the Rust orchestrator is not wired to the scenario seed (it uses unseeded `ParticleNavFilter::new(...)`; scenario-seed→Rust wiring is the GAP in `SCENARIO_REPLAY_CONTRACT.md` row 2), so such a record is not bit-reproducible from `mission.seed` alone. The seed determines only the Python replay |
| `mission.config_hash` | string | `scenario_hash()` (existing) |
| `repo.commit` / `repo.branch` / `repo.working_tree_clean` | string/string/bool | existing replay `repo` block |
| `payload_type` | string | exact Rust type name (e.g. `strix_xai::trace::DecisionTrace`) |
| `payload` | object | serde JSON of the source artifact, verbatim |
| `integrity.sha256` | string | hash of canonicalized `payload` (future) |
| `classification` | string | fixed: `"software-simulation evidence only"` |
| `review` | object | reviewer/disposition fields (future, validation phase) |

## 4. Sample JSON mappings (built from real struct fields)

> Validation-phase design — no integration code exists today. Payload field
> names below are exactly the serde field names of the cited types; envelope
> fields are the future schema of section 3. Values are illustrative.

### Sample A — `SafetyClamp` DecisionTrace → safety-intervention record

Payload fields mirror `DecisionTrace`/`TraceInputs`/`TraceOutput`
(`crates/strix-xai/src/trace.rs`) as populated at
`crates/strix-swarm/src/tick.rs` line 2233:

```json
{
  "record_id": "01J0FUTURE0000000000000000",
  "schema_version": 1,
  "kind": "safety_intervention",
  "mission": { "scenario_id": "gps_denied_recon", "seed": 42001, "config_hash": "3f6c2b9a1d4e8f07" },
  "repo": { "commit": "7ce303cb8b9cd05654870c7922bd3c467804c230", "branch": "main", "working_tree_clean": true },
  "payload_type": "strix_xai::trace::DecisionTrace",
  "payload": {
    "id": 1042,
    "timestamp": 184.2,
    "decision_type": "SafetyClamp",
    "inputs": {
      "drone_ids": [1, 2, 3, 4],
      "regime": "CBF",
      "metrics": {
        "active_constraints": 3,
        "deadlock_escapes": 0,
        "neural_agents": 0,
        "classical_agents": 4
      },
      "context": null,
      "fear_level": 0.31,
      "courage_level": null,
      "tension": null,
      "calibration_quality": null
    },
    "reasoning": [],
    "output": { "action": "safety_clamp", "details": { "constraints": 3, "escapes": 0 } },
    "confidence": 0.95,
    "alternatives_considered": []
  },
  "integrity": { "sha256": "<sha256-of-canonical-payload>" },
  "classification": "software-simulation evidence only"
}
```

Note `classical_agents: 4`, `neural_agents: 0`: with default configuration
(`fallback_to_classical: true`, no trained weights shipped) every agent is
safeguarded by classical CBF — the ledger record preserves that fact.

### Sample B — `LossRecord` → loss/re-auction record

Payload fields mirror `LossRecord`
(`crates/strix-auction/src/antifragile.rs`, line 73); the linked trace id
refers to the `ReAuction` `DecisionTrace` emitted at
`crates/strix-swarm/src/tick.rs` line 608:

```json
{
  "record_id": "01J0FUTURE0000000000000001",
  "schema_version": 1,
  "kind": "loss_reauction",
  "mission": { "scenario_id": "mass_attrition", "seed": 42002, "config_hash": "9b1e44d020c7aa5f" },
  "repo": { "commit": "7ce303cb8b9cd05654870c7922bd3c467804c230", "branch": "main", "working_tree_clean": true },
  "payload_type": "strix_auction::antifragile::LossRecord",
  "payload": {
    "drone_id": 7,
    "position": { "x": 412.5, "y": -88.0, "z": -120.0 },
    "altitude": -120.0,
    "heading": 1.5707963,
    "velocity": [12.0, 0.5, -0.2],
    "threat_bearing": 0.7853982,
    "regime_at_loss": "Evade",
    "classification": "Sam",
    "orphaned_tasks": [3, 9],
    "timestamp": 642.8
  },
  "links": { "reauction_trace_id": 2210, "timeline_event": "DroneLost" },
  "integrity": { "sha256": "<sha256-of-canonical-payload>" },
  "classification": "software-simulation evidence only"
}
```

## 5. Explicit non-goals and boundaries

> Validation-phase design — no integration code exists today.

- **No implementation exists.** No ledger writer, store, or reader is in the
  tree; this document is the design artifact, not a feature.
- **No external memory-system integration exists.** In particular, there is
  **no STRIX↔sqlite-memory integration**; that seam is a future validation
  target only, and per the claim freeze it must never be described as shipped.
- **No change to the safety story.** Classical CBF + ROE + traces + simulator
  remains the entire shipped safety claim; the ledger records evidence, it
  does not gate or alter behavior.
- **Software-simulation evidence only.** Every record carries that
  classification verbatim; nothing in the ledger can upgrade simulation
  evidence into hardware, RF, sensor-fidelity, or field-readiness evidence.
- **Defensive scope only.** Source artifacts document deny-first ROE gating,
  safety clamping, loss recovery, and degraded-mode defensive responses.
