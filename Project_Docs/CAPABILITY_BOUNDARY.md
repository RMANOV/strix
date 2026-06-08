# STRIX Capability Boundary — Canonical Public Claim Map

This document is the canonical, authoritative claim map for STRIX. It freezes
which surfaces are load-bearing for public claims, which are experimental or
optional, and which public claims STRIX explicitly does **not** make.

**Posture.** STRIX is civilian-dogfooded, single-operator, test-backed, and
simulator-first. The public repository exposes a reusable autonomy core
(coordination, safety, simulation, explainability) plus platform-agnostic
adapter boundaries.

**Important: these are claim-posture labels, not a product cut.** Labelling a
surface "experimental", "optional", "adapter-boundary", or "offline-tool"
describes how STRIX talks about that surface in public material. It does **not**
mean the surface is removed, deprecated, disabled, unsupported, or scheduled for
removal. GCBF+, ROS2/MAVLink adapters, and the Python LLM / edge-inference path
all remain present and functional in the source tree. This is claim hygiene, not
a feature reduction.

## Core / load-bearing surfaces (the claimable autonomy story)

These are the surfaces STRIX stands behind as the public autonomy claim:

- **Rust-centered OODA / tick path.** The orchestration loop and per-tick
  decision path are Rust-centered; this is the load-bearing runtime surface.
- **Classical safety constraints.** Classical control-barrier-function (CBF)
  gating is the default safety mechanism, with `fallback_to_classical = true`.
  The safety story is classical-CBF + ROE + traces + simulator.
- **ROE / policy gates.** Friendly-and-civilian deny-first rules-of-engagement
  and conservative task gating.
- **DecisionTrace / BattleReport / replay.** Structured, replayable, auditable
  decision artifacts for after-action inspection.
- **Degraded-mode / EW behavior.** Documented behavior under GPS loss, comms
  degradation, sensor noise, and electronic-warfare conditions.
- **Simulator-first evidence.** Deterministic software-only scenario replay and
  the public test matrix as the evidence base.

## Experimental surfaces (in-tree, not the public claim center)

- **GCBF+ (graph control-barrier-function, trained path).** Experimental /
  training path. STRIX does not present it as a shipped neural-safety guarantee,
  trained-weights default, or default-runtime safety mechanism. The shipped
  safety claim rests on classical-CBF + ROE + traces + simulator.

## Optional surfaces (in-tree, advanced / supporting)

- **Python LLM / edge inference.** Optional / facade / degraded-mode support.
  It is not a required runtime and not the core autonomy path.
- **Optimizer.** An offline tool, not a live autonomy core.

## Adapter-boundary surfaces (validation-ahead, not delivered integration)

- **ROS2 / MAVLink adapters.** Adapter-boundary and validation-ahead. They mark
  where platform integration would attach. They are **not** delivered hardware
  integration and must not be claimed as fielded or on-hardware deployment.

## D2-only changes (out of scope for docs/labels)

The following are explicitly **not** D1 (docs/labels) work. They require a D2
inventory / RFC and explicit operator approval before any change:

- Editing `Cargo.toml` crate descriptions, features, or workspace metadata for
  `strix-core`, `strix-adapters`, `strix-optimizer`, `strix-swarm`, or any
  crate — package metadata and feature flags affect the runtime/build surface.
- Adding labels, docstrings, or changed exports inside code under
  `python/strix/gcbf_training/`, `python/strix/llm/`, `python/strix/brain.py`,
  `python/strix/digital_twin/`, or any `crates/strix-*/` source.
- Changing `public_test_matrix.json`, tests, scripts, benchmark claims, or
  generated-report rules.
- Moving MissionBrain (Python) authority into SwarmOrchestrator (Rust), or
  changing the PyO3 surface — blocked until a call-graph / API inventory,
  parity tests, a deprecation plan, and broad-suite proof exist.
- Removing ROS2/MAVLink stubs, GCBF+ modules, LLM/edge modules, the optimizer
  crate, or any public export. There are no deletions in D1.

## Forbidden public claims (claims STRIX does NOT make)

The following must never appear in DIANA-facing or other public material. They
are listed as the do-not-claim set, not as assertions:

- Fielded or on-hardware drone deployment; edge-drone deployment.
- Delivered ROS2 / MAVLink hardware integration.
- Defence validation, accreditation, or certification.
- A default-runtime or trained-neural GCBF+ safety guarantee.
- Edge-LLM autonomous decision authority as core autonomy.
- A shipped STRIX integration into an external memory system (e.g.
  sqlite-memory) in production.
- Sensor / RF / field readiness inferred from software replay alone.
- Unbounded "production" claims — only production-quality, single-operator, with
  no external customers or deployment.
- A 400–500 drone ceiling or 2000+ drone target stated as fact; these are
  estimate / roadmap only unless a concrete benchmark backs them.
- Frozen final tick-timing or pass-count numbers (for example, a fixed
  per-tick millisecond figure or an "all tests pass" count) presented as final
  live facts. Treat such figures as prior measured software-replay results to be
  re-run on the exact submission commit.

## Allowed public claims (frozen claim-set)

STRIX may publicly claim:

- A public-safe research / prototype autonomy stack.
- A Rust-centered tick / orchestration path.
- State estimation, task allocation, a coordination mesh, classical safety
  constraints, ROE / policy gating, and degraded-mode / EW behavior.
- DecisionTrace / BattleReport / replay and auditability.
- Simulator-first software-replay evidence.
- Platform-agnostic adapter boundaries.

Tick-timing and scale figures are stated as prior measured software-replay
results to be re-run on the exact submission commit, never as final live facts.

---

This file is the claim-freeze reference for STRIX. The README "Capability
status / public claim boundary" section is a summary pointer to this document;
where they differ, this document governs.
