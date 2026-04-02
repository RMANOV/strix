# DOC-003 Mesh, Auction and Higher-Order Coordination Questionnaire

Pack: `2026-04-02`

Primary surfaces:
- `crates/strix-mesh`
- `crates/strix-auction`

Mission risk focus:
- T1 wrong allocation or coordination collapse under degraded comms
- T2 contagion, gossip or quorum logic that overreacts or underreacts
- T2 compartment leaks between dark-pool or bundle scopes

## 1. Gossip and State Propagation

- [ ] DOC-003/1.1/1 — gossip fanout changes do not break convergence guarantees or deterministic peer selection assumptions.
- [ ] DOC-003/1.1/2 — stale or non-finite remote state is rejected or replaced cleanly, not merged into valid mesh state.
- [ ] DOC-003/1.1/3 — threat union semantics remain monotonic under partition and later rejoin.
- [ ] DOC-003/1.1/4 — mesh message priority order still matches operational intent: threat and directive traffic outrank background chatter.

## 2. Contagion Semantics

- [ ] DOC-003/2.1/1 — simple contagion forwards fresh threat or pheromone signals without reinforcement requirements.
- [ ] DOC-003/2.1/2 — complex contagion truly requires multi-source reinforcement for directives, not mere repeated replay from one sender.
- [ ] DOC-003/2.1/3 — damped affect signals cool down over time instead of re-amplifying forever from identical repeats.
- [ ] DOC-003/2.1/4 — new message kinds are explicitly mapped to contagion mode; defaulting silently is not acceptable.

## 3. Higher-Order Coordination

- [ ] DOC-003/3.1/1 — hypergraph or bundle quorum logic activates only where there is a real group effect, not for every pairwise event.
- [ ] DOC-003/3.1/2 — anti-deception checks reject divergent votes without suppressing legitimate multi-agent consensus.
- [ ] DOC-003/3.1/3 — coordination directives with focus positions preserve spatial identity across serialization and propagation.
- [ ] DOC-003/3.1/4 — failure of one higher-order path degrades to pairwise behavior instead of crashing the coordination layer.

## 4. Auction Floor Integrity

- [ ] DOC-003/4.1/1 — bid compression preserves bundle semantics and does not silently drop required bundle mates.
- [ ] DOC-003/4.1/2 — dark-pool filtering enforces compartmentalization without hiding tasks from their intended eligible sub-swarms.
- [ ] DOC-003/4.1/3 — greedy fallback and optimal assignment produce compatible invariants when bid volume crosses threshold.
- [ ] DOC-003/4.1/4 — bundle validation, quorum thresholds and reauction triggers cannot clear contradictory assignments into the same round.

## 5. Cross-Layer Side Effects

- [ ] DOC-003/5.1/1 — mesh-side directive or affect propagation can trigger reauction, but never bypasses ROE or safety gates downstream.
- [ ] DOC-003/5.1/2 — pheromone deposits created from emergent directives stay consistent with actual directive intent.
- [ ] DOC-003/5.1/3 — auction state remains valid when comms quality degrades and gossip fanout is modulated by criticality.

## 6. Regression Classes to Hunt Actively

- [ ] DOC-003/6.1/1 — one-line semantic bugs in contagion or quorum code that simultaneously break tests, FFI builds and benches.
- [ ] DOC-003/6.1/2 — code that compiles and passes unit tests but creates coordination storms under repeated replay.
- [ ] DOC-003/6.1/3 — new coordination fields added to messages but not included in signatures, serialization or priority routing.

## 7. Evidence Discipline

Запиши поне едно от следните:
- relevant `strix-mesh` or `strix-auction` unit test name
- exact function path for quorum / contagion / bid compression logic
- if scenario-based, which simulated partition or swarm preset was used

## 8. Exit Criteria

`DOC-003` не е complete ако липсва:
- поне една contagion check
- поне една quorum / higher-order coordination check
- поне една auction bundle or dark-pool integrity check
- statement дали има compartment leak or coordination storm risk
