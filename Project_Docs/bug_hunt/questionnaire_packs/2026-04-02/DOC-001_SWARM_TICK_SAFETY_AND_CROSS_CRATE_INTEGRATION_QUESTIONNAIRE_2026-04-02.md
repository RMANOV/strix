# DOC-001 Swarm Tick, Safety and Cross-Crate Integration Questionnaire

Pack: `2026-04-02`

Primary surfaces:
- `crates/strix-swarm/src/tick.rs`
- `crates/strix-swarm/src/criticality.rs`
- `crates/strix-core` safety modules: regime, ROE, EW, CBF, temporal
- `crates/strix-xai`

Mission risk focus:
- T0 safety or loss-of-control regressions
- T1 wrong tasking caused by broken orchestration order
- T2 hidden integration drift between crates that still compiles but changes behavior

## 1. Tick Entry and Data Sanitation

- [ ] DOC-001/1.1/1 — `dt` sanitation prevents NaN, negative or absurdly large values from poisoning `sim_time` or downstream state.
- [ ] DOC-001/1.1/2 — non-finite telemetry values are sanitized before they reach navigation, gossip, CBF and trace layers.
- [ ] DOC-001/1.1/3 — missing or partial telemetry for one drone degrades only that drone path, not the whole swarm tick.
- [ ] DOC-001/1.1/4 — adding a new field to `SwarmDecision` cannot compile unless the constructor path populates it everywhere.

## 2. Causal Order of the Main Loop

- [ ] DOC-001/2.1/1 — particle/filter predict-update steps happen before regime evaluation, not after final assignment.
- [ ] DOC-001/2.1/2 — criticality modulation is applied before auction fanout / pheromone decay / bid aggression are used.
- [ ] DOC-001/2.1/3 — regime decisions are finalized before auction and mesh propagation consume them.
- [ ] DOC-001/2.1/4 — mesh-side emergent directives cannot bypass the explicit safety pass later in the tick.
- [ ] DOC-001/2.1/5 — output `SwarmDecision` reflects the final post-safety state, not a pre-clamp intermediate snapshot.

## 3. Safety Chain Integrity

- [ ] DOC-001/3.1/1 — ROE denial or escalation cannot be bypassed by auction re-run, dark-pool scoping or threat-intent spikes.
- [ ] DOC-001/3.1/2 — CBF corrections still apply when formation control is disabled or when all drones enter EVADE.
- [ ] DOC-001/3.1/3 — EW response plans cannot silently clear themselves before they are consumed by the orchestration step.
- [ ] DOC-001/3.1/4 — temporal constraints can veto unsafe maneuvers without corrupting tactical state for the next tick.
- [ ] DOC-001/3.1/5 — threat-tracker update order is stable relative to fear state, criticality and auction side effects.

## 4. Decision Payload Integrity

- [ ] DOC-001/4.1/1 — every emitted assignment has valid `drone_id`, `task_id`, confidence and position semantics.
- [ ] DOC-001/4.1/2 — `fear_level`, `criticality`, `exploration_noise`, `pheromone_decay_multiplier` and `bid_aggression` stay internally coherent for the same tick.
- [ ] DOC-001/4.1/3 — any field exposed in `SwarmDecision` is either consumed, traced or intentionally documented as informational only.
- [ ] DOC-001/4.1/4 — changes in `tick.rs` that affect mission semantics must have at least one directly relevant test, not just generic workspace green.

## 5. Traceability and Explainability

- [ ] DOC-001/5.1/1 — ROE denials, escalations and forced regime transitions produce trace evidence, not only behavior.
- [ ] DOC-001/5.1/2 — trace payloads are based on sanitized inputs, so NaN or invalid telemetry never pollutes audit output.
- [ ] DOC-001/5.1/3 — when a new branch of logic is added to the tick, either XAI coverage expands or the omission is explicitly accepted.

## 6. Regression Traps Worth Hunting Actively

- [ ] DOC-001/6.1/1 — constructor drift: struct fields added but not written in final decision assembly.
- [ ] DOC-001/6.1/2 — patch residue or dead code text accidentally committed into runtime files.
- [ ] DOC-001/6.1/3 — benchmark-only or nightly-only failures that signal unstable integration even when stable tests pass.
- [ ] DOC-001/6.1/4 — compile-clean changes that alter swarm semantics because tick ordering moved by only a few lines.

## 7. Evidence Discipline

За всяка checked точка запиши поне едно от следните:
- exact file path and relevant function
- failing/passing test name
- benchmark or scenario name
- commit or PR that introduced / fixed the behavior

## 8. Exit Criteria

Не маркирай `DOC-001` като complete, ако липсва някое от следните:
- поне една safety-oriented проверка
- поне една causality-order проверка
- evidence за final `SwarmDecision` integrity
- изрично statement дали има или няма T0/T1 findings
