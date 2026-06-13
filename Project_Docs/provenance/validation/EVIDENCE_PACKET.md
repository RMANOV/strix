# STRIX Evidence Packet

**Audience:** external reviewers (DIANA pre-submission).
**Scope:** documentation only — this packet asserts nothing beyond the frozen
claim-set in [`Project_Docs/CAPABILITY_BOUNDARY.md`](../../CAPABILITY_BOUNDARY.md),
which governs wherever wording differs.

**Posture (one paragraph).** STRIX is a simulator-first, software-only autonomy
research stack. Classical control-barrier-function (CBF) gating is the only
safety guarantee. GCBF+ ships no trained weights and defaults to classical
fallback. There have been **zero hardware flights**; ROS2/MAVLink adapters are
stubs. The practical single-node scale estimate is ~400–500 agents; 2000+ is a
forward roadmap target, not a demonstrated capability.

---

## Live test evidence (this packet's only freshly measured fact)

| Field | Value |
|---|---|
| Command | `cargo test --workspace` |
| Commit | `7ce303cb8b9cd05654870c7922bd3c467804c230` (`7ce303c`) |
| Branch | `docs/diana-evidence-pack` (docs-only branch off `main` at the same commit) |
| Date of run | 2026-06-12 |
| Result | **843 passed, 0 failed, 3 ignored** across 27 test binaries/suites |

Ignored tests are 3 doc-tests (`strix_core` gcbf module doc, `strix_optimizer`
smco doc, `strix_playground` playground doc) explicitly marked `ignore`.

Per-suite breakdown (passed counts, all 0 failed):

| Crate | Suite | Passed |
|---|---|---|
| strix-adapters | unit (`src/lib.rs`) | 43 |
| strix-adapters | `tests/phase1_integration.rs` | 5 |
| strix-auction | unit | 87 |
| strix-core | unit | 298 |
| strix-core | `tests/proptest_particle.rs` | 6 |
| strix-mesh | unit | 191 |
| strix-optimizer | unit | 57 |
| strix-playground | unit | 14 |
| strix-playground | `tests/presets_smoke.rs` | 5 |
| strix-python | unit | 0 (Rust shell; Python tests run separately) |
| strix-swarm | unit | 48 |
| strix-swarm | `tests/fault_injection.rs` | 5 |
| strix-swarm | `tests/feedback_stability.rs` | 6 |
| strix-swarm | `tests/island_module_integration.rs` | 15 |
| strix-swarm | `tests/phi_sim_integration.rs` | 0 (feature-gated; runs only with the `phi-sim` feature) |
| strix-swarm | `tests/regression.rs` | 3 |
| strix-swarm | `tests/twenty_drone_integration.rs` | 12 |
| strix-xai | unit | 47 |
| (workspace) | doc-tests | 1 (+3 ignored) |

Per the claim freeze, test counts are **point-in-time software facts** that
drift with development; they must be re-run on the exact submission commit and
never quoted as a frozen final number. This 843/0/3 was measured at `7ce303c`,
**before** the pyo3 0.29 security-dependency update (commit `5bd998d`) was
merged; the current tree includes that update, so the suite must be re-run on
the exact submission commit before any external use.

---

## Claim → evidence → caveat → source matrix

| # | Claim (as allowed by claim freeze) | Evidence | Caveat | Source |
|---|---|---|---|---|
| 1 | Workspace test suite passes: 843 passed / 0 failed / 3 ignored at commit `7ce303c`, 2026-06-12. | Live `cargo test --workspace` run recorded above, with per-suite breakdown. | Point-in-time count; re-run on the exact submission commit. `phi_sim_integration` contributes 0 tests unless the `phi-sim` feature is enabled. | This document, section above; suites listed per crate (e.g. `crates/strix-core/src/`, `crates/strix-swarm/tests/twenty_drone_integration.rs`). |
| 2 | Full swarm tick measured at ~1.15 ms for 20 drones (prior measured software result). | Criterion benchmark group `bench_swarm_tick` covering estimation, regime updates, assignment, coordination, safety clamps, and trace capture. | **Prior measured result, not re-measured for this packet.** Bench source last modified 2026-03-23; README snapshot line last updated 2026-04-13 (commit `748a707`). Single core, unoptimized test profile. **Sub-millisecond figures refer to subcomponents only** (particle-filter step 42–226 µs, auction round 2.7–465 µs, 5/10-drone ticks 298/580 µs) — the full 20-drone tick is 1.15 ms, i.e. above one millisecond. Must be re-run on the submission commit before being quoted as current. | Bench: `crates/strix-swarm/benches/tick.rs` (`bench_swarm_tick`). Numbers: `README.md`, "Performance Snapshot" table. |
| 3 | Classical CBF gating is the **only** safety guarantee STRIX ships. | Implementation: `cbf_filter()`, `cbf_filter_with_neighbor_states()`, `is_position_safe()`, `detect_deadlock()`, `generate_escape_maneuvers()` with `CbfConfig`/`CbfResult`. 18 in-file unit tests, e.g. `close_neighbor_gets_corrected`, `altitude_floor_enforced`, `nfz_avoidance`, `correction_clamped_to_max`, `fast_ingress_to_nfz_triggers_predictive_margin`, `escape_maneuvers_separate_drones`. | Safety properties are demonstrated in software simulation only — no formal-methods certificate, no field validation, no hardware-in-the-loop evidence. | `crates/strix-core/src/cbf.rs` (types, functions, and all listed tests). |
| 4 | GCBF+ is experimental: **no trained weights ship**, and the runtime default falls back to classical CBF. | Config default: `fallback_to_classical: true` in `impl Default for GcbfConfig` (`config.rs` line 37). Weight infrastructure exists, but `default_weights()` generates Xavier-like **random-initialized** weights from a fixed PRNG seed (12345); no `.onnx`/`.pt`/`.safetensors` or other model files exist anywhere in the repository. | GCBF+ must never be presented as a shipped or default neural-safety guarantee. The safety story remains classical-CBF + ROE + traces + simulator (claim freeze, "Experimental surfaces"). | `crates/strix-core/src/gcbf/config.rs` (field line 21, default line 37); `crates/strix-core/src/gcbf/weights.rs` (`default_weights()`); repo-wide absence of weight files. |
| 5 | All evidence is simulation-only; **zero hardware flights** have occurred. | The entire evidence base is software: the playground engine (`Engine::run()` → `BattleReport`), the deterministic kinematic replay harness, YAML scenarios, and the public test matrix. No flight logs, no hardware telemetry, no field artifacts exist in the repository. | No RF, sensor-fidelity, or field-readiness conclusions may be inferred from software replay alone (claim freeze, "Forbidden public claims"). | `crates/strix-playground/src/engine.rs`; `scripts/strix_sim_replay.py`; `sim/scenarios/*.yaml`; `Project_Docs/testing/EVIDENCE_HARNESS.md`. |
| 6 | ROS2/MAVLink adapters are **stubs** (adapter-boundary, validation-ahead). | `Ros2Adapter` doc comment: "**STUB** — no actual ROS2 middleware is running. Methods return dummy data."; `connect()` logs a stub-mode warning and only flips a flag. `MavlinkAdapter` in default build: "**Stub** — returns deterministic dummy data; zero external deps; safe for simulation"; waypoint/action methods are `TODO` no-ops. | Not delivered hardware integration; must not be claimed as fielded or on-hardware deployment. The optional `mavlink-hw` feature flag exists but is not a validated or claimed path. | `crates/strix-adapters/src/ros2.rs`; `crates/strix-adapters/src/mavlink.rs`. |
| 7 | Practical single-node scale: **estimate ~400–500 agents** at 10 Hz; **2000+ is a roadmap target only**. | The frozen claim map classifies the ~400–500 ceiling and the 2000+ target as **estimate / roadmap only "unless a concrete benchmark backs them"** — and no such benchmark exists: the largest benchmarked configuration is 100 drones in `bench_swarm_tick`, so neither figure is benchmark-backed. | Estimate / roadmap only — **not demonstrated**. No benchmark at these scales exists in the repository; the largest benchmarked configuration is 100 drones in `bench_swarm_tick`. Never state the ceiling or the 2000+ target as fact (claim freeze, "Forbidden public claims"). | `Project_Docs/CAPABILITY_BOUNDARY.md` ("Forbidden public claims" — the 400–500 ceiling / 2000+ target line); `crates/strix-swarm/benches/tick.rs`. |
| 8 | No STRIX integration into any external memory system (e.g. sqlite-memory) exists. | No such integration code exists in the repository. The evidence-ledger handoff is documented as a **future validation-phase design** only. | The seam is a future validation target, not a shipped or in-progress integration (claim freeze, "Forbidden public claims"). | `Project_Docs/provenance/validation/EVIDENCE_LEDGER_SCHEMA.md` (design doc, documentation only). |

---

## Reproduction

```bash
git checkout 7ce303cb8b9cd05654870c7922bd3c467804c230
cargo test --workspace                      # live test evidence (item 1)
cargo bench -p strix-swarm                  # re-measure tick timings (item 2)
python scripts/strix_test_matrix.py --list  # public software-only evidence harness
python scripts/strix_sim_replay.py \
  --scenario sim/scenarios/gps_denied_recon.yaml  # deterministic replay artifact
```

All commands are software-only. None of them produce, or substitute for,
hardware, RF, sensor-fidelity, or field-readiness evidence.
