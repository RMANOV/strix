# STRIX Scenario / Replay Contract

**Scope: algorithmic, software-only pre-field gate — explicitly NOT field
validation.** This contract describes the Python player's reproducible
kinematic output and its acceptance checks. A pass supports only the quantities
actually computed by that player; it demonstrates nothing about uninstrumented
navigation / formation / radio performance, hardware, RF, sensor fidelity, or
field readiness, and there
have been zero hardware flights. Governing claim map:
[`Project_Docs/CAPABILITY_BOUNDARY.md`](../../CAPABILITY_BOUNDARY.md).

**Metric correction, 2026-10-14:** the former GPS-denied PASS is withdrawn.
Coverage was an active-agent ratio; RMS and coherence were constants. Fixed
avoidance / recovery / relay / response-time / coherence outputs in the other
scenario families are likewise withdrawn. Scenario acceptance bounds have not
been lowered. The corrected player marks unsupported metrics `not_observed`
and fails these scenario envelopes. See [the corrected capture](../../demo/replay_output.txt).

This document describes the existing harness after that correction. Where a
contract field is not yet produced by the harness, it is explicitly marked
**GAP (future)**.

## 1. Existing capabilities this contract is grounded in

| Capability | Where it lives today |
|---|---|
| Scenario definitions (YAML) | `sim/scenarios/gps_denied_recon.yaml`, `sim/scenarios/mass_attrition.yaml`, `sim/scenarios/multi_domain.yaml` (+ `sim/scenarios/README.md`) |
| Scenario contract validation | `scripts/strix_scenario_contract.py` — requires `scenario_id`, `seed`, `name`, `description`, `duration_seconds`, `environment`, `mission`, `metrics`, `pass_envelope` (lines 95–105) |
| Deterministic replay harness | `scripts/strix_sim_replay.py` — `build_replay()`, `replay_metrics()`, `evaluate_envelope()`, `scenario_hash()` |
| Replay artifacts | JSON replay (default `target/strix-replays/latest.json`) + self-contained HTML canvas viewer |
| Public evidence harness | `scripts/strix_test_matrix.py` driven by `Project_Docs/testing/public_test_matrix.json` (checks include `scenario_contract`, `software_replay_gps_denied`, `scenario_schema_contract`); see `Project_Docs/testing/EVIDENCE_HARNESS.md` |
| Rust simulation runs | `crates/strix-playground/src/playground.rs` builder + `presets.rs` (`ambush()`, `gps_denied()`, `attrition()`, `stress_test()`, `temporal()`), producing `BattleReport` (`crates/strix-playground/src/report.rs`) with timeline, aggregates, and optional `tick_data` snapshots |
| Deterministic seeding (Rust) | `ParticleNavFilter::new_seeded()` — `crates/strix-core/src/particle_nav.rs`, line 774 (`ChaCha8Rng::seed_from_u64(seed)`) |
| Deterministic seeding (replay) | `random.Random(seed)` in `strix_sim_replay.py`, seed taken from the scenario YAML |

Canonical invocation (already documented in
`Project_Docs/testing/EVIDENCE_HARNESS.md`):

```bash
python scripts/strix_sim_replay.py \
  --scenario sim/scenarios/gps_denied_recon.yaml \
  --output target/strix-replays/gps_denied_recon.json \
  --html target/strix-replays/gps_denied_recon.html
```

## 2. The canonical contract

A conforming validation run is the tuple below. "Carrier (today)" cites the
exact existing field or file that already carries the item.

| # | Contract field | Carrier (today) | Status |
|---|---|---|---|
| 1 | **Scenario id** | `scenario_id` in scenario YAML; echoed as `scenario.id` in replay JSON; format enforced by `strix_scenario_contract.py` | EXISTS |
| 2 | **Seed** | `seed` (u64) in scenario YAML (e.g. 42001/42002/42003); echoed as `scenario.seed`; drives `random.Random(seed)` in the **Python replay only**. The Rust orchestrator is **not** wired to the scenario seed — `SwarmOrchestrator::new` / `register_drone` use `ParticleNavFilter::new(...)` (unseeded); the seeded `new_seeded()` / `ChaCha8Rng::seed_from_u64` path exists but is exercised only by Rust tests, not driven by scenario YAML | PARTIAL — **GAP (future):** wire the scenario seed into the Rust orchestrator, or keep Rust determinism a separate lane |
| 3 | **Config hash** | `scenario.config_hash` in replay JSON — SHA-256 of the scenario YAML bytes, first 16 hex chars (`scenario_hash()`, `strix_sim_replay.py` line 175) | EXISTS |
| 4 | **Git commit** | `repo.commit`, `repo.branch`, `repo.working_tree_clean` in replay JSON (`build_replay()`) | EXISTS |
| 5 | **Build / runtime info** | Partially: `report_version: 1`, `kind: "software_replay"`, `simulator`, `fidelity: "deterministic_kinematic_public_replay"`, `tick_s` in replay JSON. Toolchain version, build profile, and OS are **not** recorded | PARTIAL — **GAP (future):** add toolchain/profile/platform stamp |
| 6 | **Pass envelope and metric provenance** | `pass_envelope` per metric (`min`/`max` bounds) in scenario YAML; evaluated by `evaluate_envelope()` into `envelope.status` (`passed`/`failed`) + per-metric `envelope.checks[]`. `metric_evidence` distinguishes computed kinematic diagnostics from unavailable measurements and gives reasons. Missing metrics surface as `not_observed` and fail the run; targets or placeholders never count as observations | EXISTS — public scenario acceptance metrics remain uninstrumented |
| 7 | **Event timeline** | Replay JSON `frames[]` (`t_s`, per-agent `x/y/z`, `energy`, `status`, `mode`) with `frames[].events` (e.g. `gps_loss`, `wind_gust` from the scenario `events:` schedule). Rust path: `BattleReport.timeline` (`TimelineEntry`/`TimelineEventType`) and optional `tick_data` | EXISTS |
| 8 | **Safety interventions** | Replay JSON (today, indirect only): `metrics.min_constraint_clearance_m` only when a constraint distance was actually recorded, plus per-frame `mode` flags (e.g. `avoid_constraint`). No constraints means clearance is `not_observed`, not zero. Rust path: `Aggregates` CBF counters and `DecisionType::SafetyClamp` traces. The replay artifact carries **no** per-intervention records today. | PARTIAL — **GAP (future):** fold per-intervention records into the replay artifact (see `EVIDENCE_LEDGER_SCHEMA.md`, validation-phase design) |
| 9 | **Replay artifact** | Deterministic JSON replay + self-contained HTML viewer written by `strix_sim_replay.py`; outputs default under `target/` so evidence stays out of source control unless deliberately promoted | EXISTS |
| 10 | **Failure incident pack** | Today: a failing run yields `envelope.status: "failed"` plus the failing `checks[]` inside the replay JSON, and a non-passing entry in the test-matrix report (JSON + Markdown under `target/strix-test-reports/`) | PARTIAL — **GAP (future):** normalize into one bundle = scenario YAML + seed + config hash + commit + replay JSON/HTML + failing checks, archived per incident |

## 3. Gate semantics

A scenario **passes the pre-field gate** when all of the following hold:

The current GPS-denied, attrition, and multi-domain Python replays do **not**
meet this gate: they lack measurements for required acceptance metrics.

1. `strix_scenario_contract.py` validates the scenario file (required fields
   present, `seed` non-negative integer, each `pass_envelope` metric provides a
   numeric `min` and/or `max` — at least one bound — with `min <= max` enforced
   only when both bounds are present).
2. The replay is generated from a recorded commit, and `repo.working_tree_clean`
   is `true` (a dirty tree makes the commit stamp non-reproducible).
3. `envelope.status == "passed"` — every declared metric is observed and
   inside its bounds; `not_observed` counts as failure. A successful regression
   test of this failure behavior is not a successful scenario acceptance run.
4. The corresponding public-test-matrix checks (`scenario_contract`,
   `software_replay_gps_denied`, `scenario_schema_contract`) pass on the same
   commit.

Reproducibility claim: same scenario YAML (same `config_hash`) + same `seed`
+ same `tick_s` + same commit ⇒ same frames, metrics, and envelope verdict.
**`tick_s` must be pinned**: it is the `--tick-s` invocation step (default
`10.0`, echoed into the replay JSON), **not** part of `config_hash` (which
hashes only the scenario YAML), yet it drives frame count, timestamps,
movement distance, GPS-noise scaling (`sqrt(tick_s)`), energy burn, and event
windows — so a different `--tick-s` yields different frames/metrics/envelope
from the same YAML + seed + commit. This is what "deterministic kinematic
public replay" (`fidelity` field) means — and all it means.

## 4. What this gate is — and is not

**Is:** a software-only, deterministic, seeded regression and behavior-review
gate over the **Python public replay** — a kinematic scenario model
(`strix_sim_replay.py`: `orbit_target` / `apply_constraint_avoidance` /
`move_toward`) that emits reproducible frames, metrics, and a pass-envelope
verdict, suitable as a *pre-field* screen and for after-action visual
inspection. **It does not exercise the Rust orchestrator** —
`strix_sim_replay.py` invokes no `SwarmOrchestrator`, `RoeEngine`, `Auctioneer`,
or Rust CBF code. The Rust OODA / ROE / classical-CBF behavior is validated in a
**separate evidence lane** (the Rust test suite and `BattleReport`; see
`EVIDENCE_PACKET.md`), not by this replay.

**Is not** (per the claim freeze, `Project_Docs/CAPABILITY_BOUNDARY.md` and
`Project_Docs/testing/EVIDENCE_HARNESS.md`):

- field validation, flight testing, or any hardware evidence — **zero
  hardware flights**; ROS2/MAVLink adapters remain stubs
  (`crates/strix-adapters/src/ros2.rs`, `crates/strix-adapters/src/mavlink.rs`);
- RF, sensor-fidelity, or environment-fidelity evidence — the replay is
  kinematic by declaration;
- a neural-safety demonstration — the safety behavior exercised in the Python
  replay is **kinematic constraint avoidance** (`apply_constraint_avoidance`),
  not a neural or classical-CBF mechanism. The classical-CBF safety guarantee
  lives in the **separate Rust evidence lane** (`fallback_to_classical: true`
  by default; GCBF+ ships no trained weights);
- scale evidence beyond what a scenario actually ran — public scenarios use
  single-digit-to-tens of agents; the ~400–500 single-node ceiling is an
  estimate and 2000+ a roadmap target, neither demonstrated by this gate;
- evidence of any external memory-system integration (none exists; future
  validation target only).

## 5. Forward extensions (future, not implemented)

Documented next steps already noted in
`Project_Docs/testing/EVIDENCE_HARNESS.md`: scenario-family batch replay
(every scenario through `evaluate_envelope`), Monte Carlo seed sweeps, richer
trace exports, and — per `Project_Docs/provenance/validation/EVIDENCE_LEDGER_SCHEMA.md` — binding replay
artifacts into governed evidence records. All of these remain software-only
pre-field instruments; none of them convert simulation evidence into field
evidence.
