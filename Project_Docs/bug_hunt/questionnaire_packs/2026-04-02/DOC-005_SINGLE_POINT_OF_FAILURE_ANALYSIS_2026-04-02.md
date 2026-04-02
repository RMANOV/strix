# DOC-005 Single Point of Failure Analysis

Pack: `2026-04-02`

Цел: да идентифицира онези surfaces в `strix`, които формално не са single-threaded или single-process, но practically концентрират твърде много control, truth, safety или release confidence в една точка.

## 1. How to Read This Document

`SPOF` тук означава не само "ако този файл се счупи, всичко пада", а и:
- едно място, където semantic regression променя мисийното поведение на много слоеве наведнъж
- една граница, където detection lag е висок, а mission impact е голям
- една зависимост, за която няма достатъчно независима валидация

Използвай следната бърза скала:
- `Impact`: T0, T1, T2, T3
- `Detectability`: `low`, `medium`, `high`
- `Tracking metric`: какво да следим във времето

## 2. SPOF Register

| SPOF ID | Surface | Why It Is a SPOF | Impact | Detectability | Tracking Metric |
|---|---|---|---|---|---|
| SPOF-01 | `crates/strix-swarm/src/tick.rs` | Central orchestration choke point: ordering bugs here can change fear, criticality, auction, mesh and safety in one diff. | T0/T1 | medium | count of tick-order touching PRs, integration-test coverage, swarm preset failures |
| SPOF-02 | `python/strix/brain.py` | Single owner of active-plan semantics, intent routing, fallback logic and mission-level confidence shaping. | T1/T2 | medium | FFI parity failures, mission-plan smoke regressions, active-plan mutation defects |
| SPOF-03 | `crates/strix-python` and PyO3 boundary | Cross-language seam where build success does not guarantee semantic parity. | T1/T2 | medium | wheel/smoke pass rate, Python-vs-Rust path delta, maturin build stability |
| SPOF-04 | `crates/strix-mesh/src/gossip.rs` plus contagion routing | Shared dissemination path for swarm knowledge; stale or wrong semantics here distort coordination globally. | T1/T2 | low | convergence metrics, partition recovery tests, contagion-specific failures |
| SPOF-05 | Safety gate chain: ROE + CBF + EW response | Safety is distributed in code but mission-trust depends on the chain acting coherently under stress. | T0 | low | ROE denial traces, CBF-trigger coverage, EW scenario regressions |
| SPOF-06 | `crates/strix-auction/src/auctioneer.rs` | Assignment engine is the practical resource allocator; wrong semantics here mis-task the whole swarm. | T1 | medium | bundle/dark-pool test results, assignment stability under permutation, greedy-vs-optimal divergence |
| SPOF-07 | `crates/strix-optimizer` reports and exported tuning assumptions | Design-time policy drift can become runtime dogma if optimizer outputs are treated as truth without contextual caveats. | T2/T3 | low | contextual-front retention, report integrity, seed reproducibility |
| SPOF-08 | CI and scenario evidence layer | If release confidence depends on one workflow shape or too few scenarios, regressions escape despite green badges. | T1/T3 | medium | stable/nightly duration, failed-job clustering, scenario diversity over time |

## 3. Detailed Notes

### DOC-005/SPOF-01 — `tick.rs`

Failure mode:
- silent reordering of update, regime, auction, mesh or safety phases
- field added to `SwarmDecision` but not propagated to final assembly
- patch residue or dead code text committed into a hot path

Mitigation direction:
- keep high-value integration tests around orchestration order
- require docs touch or explicit rationale on cross-cutting tick changes
- treat benchmark-only or nightly-only tick failures as real signals, not noise

### DOC-005/SPOF-02 — `brain.py`

Failure mode:
- async/sync wrapper drift
- fallback semantics diverge from Rust-backed path
- `_active_plan` becomes partially updated after loss or reauction

Mitigation direction:
- require parity-oriented smoke paths through the extension module
- keep plan ownership and mutation rules explicit
- do not accept mission explanation strings as evidence of correctness by themselves

### DOC-005/SPOF-03 — PyO3 / `strix-python`

Failure mode:
- build and install succeed, but symbol shape or runtime semantics diverge
- keyword mismatch or constructor mismatch between Python caller and Rust-backed helper path

Mitigation direction:
- keep dedicated FFI smoke tests mandatory
- track failures here separately from generic Python tests
- require at least one real wheel/install path in CI evidence

### DOC-005/SPOF-04 — Mesh dissemination path

Failure mode:
- contagion mode mismatch for new message types
- stale state accepted as live truth
- repeated single-source replay mimics reinforcement incorrectly

Mitigation direction:
- insist on explicit message-to-mode mapping
- preserve partition / recovery tests
- watch for changes that make coordination storms easier to trigger than to detect

### DOC-005/SPOF-05 — Safety gate chain

Failure mode:
- ROE says "deny" but downstream path still emits dangerous tasking
- CBF or EW path is technically present but bypassed by orchestration order
- tactical feasibility and strategic intent drift apart with no veto path

Mitigation direction:
- maintain scenario evidence where safety modules disagree under stress
- treat missing trace evidence as a defect when safety decisions happen
- prefer chain-level checks over isolated module checks

### DOC-005/SPOF-06 — Auctioneer

Failure mode:
- bundle semantics collapse under bid compression
- dark-pool compartment boundaries leak or over-filter
- greedy fallback introduces mission-level allocation drift under load

Mitigation direction:
- keep adversarial permutation tests
- track greedy vs optimal invariants explicitly
- tie assignment correctness back to mission intent, not only bid score

### DOC-005/SPOF-07 — Optimizer truth inflation

Failure mode:
- contextual archive data gets flattened into a fake universal optimum
- surrogate score or heterogeneity bonus dominates actual mission objectives
- reports sanitize too little or explain too little

Mitigation direction:
- keep contextual fronts first-class in reports
- track where runtime assumptions are derived from optimizer outputs
- separate design-time fitness from field-truth claims

### DOC-005/SPOF-08 — Evidence monoculture

Failure mode:
- too much trust in one CI path, one scenario family or one benchmark profile
- release confidence collapses if stable/nightly coverage is not complementary

Mitigation direction:
- preserve both stable and nightly signals
- track long-running scenario presets explicitly
- use the run registry to see repeated failure clusters over time

## 4. Review Cadence

- Every architecture-heavy PR: re-read `SPOF-01`, `SPOF-02`, `SPOF-05`
- Every mesh or auction PR: re-read `SPOF-04`, `SPOF-06`
- Every optimizer or tuning PR: re-read `SPOF-07`
- Every release candidate: review all SPOFs and log what is accepted vs mitigated

## 5. Exit Criteria

`DOC-005` е complete за конкретен run само ако:
- поне един SPOF е reviewed with evidence
- ако има accepted residual risk, той е named explicitly
- ако няма new SPOFs, това е stated explicitly, not implied
