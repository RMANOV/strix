# DOC-004 Optimizer, Contextual Evolution and Offline Graph Surrogate Questionnaire

Pack: `2026-04-02`

Primary surfaces:
- `crates/strix-optimizer`

Mission risk focus:
- T2 design-time drift that later distorts runtime behavior
- T2 contextual optimization collapse into a fake global optimum
- T3 reporting, parameter export and reproducibility gaps

## 1. Contextual Archive Integrity

- [ ] DOC-004/1.1/1 — per-context Pareto fronts remain separated by doctrine, scenario family, environment and regime.
- [ ] DOC-004/1.1/2 — forgetting logic removes stale local solutions without silently corrupting the global archive.
- [ ] DOC-004/1.1/3 — migration of elites between contexts is explicit, bounded and reviewable.
- [ ] DOC-004/1.1/4 — context keys are stable enough to compare runs across time.

## 2. No Single Global Optimum Assumption

- [ ] DOC-004/2.1/1 — optimizer output does not implicitly assume one universal best policy across incompatible scenarios.
- [ ] DOC-004/2.1/2 — aggregate score blending does not hide catastrophic local failures behind strong average performance.
- [ ] DOC-004/2.1/3 — contextual entries are retained with sufficient provenance to explain why a candidate was locally useful.

## 3. Heterogeneity and Policy Decode

- [ ] DOC-004/3.1/1 — heterogeneity policy decode covers every expected role/echelon pair deterministically.
- [ ] DOC-004/3.1/2 — parameter-count or ordering changes are reflected in tests, reports and downstream config consumers.
- [ ] DOC-004/3.1/3 — heterogeneity bonus cannot dominate the real mission objectives to the point of gaming the search.

## 4. Offline Graph Surrogate Scope

- [ ] DOC-004/4.1/1 — graph surrogate is used only as offline evaluator input, not silently introduced into real-time orchestration paths.
- [ ] DOC-004/4.1/2 — global encodings or master-node signals are explainable in reports and not treated as magic scalar improvements.
- [ ] DOC-004/4.1/3 — surrogate blending weight is bounded and visible in CLI output, reports or config.

## 5. Reporting and Reproducibility

- [ ] DOC-004/5.1/1 — `OptimizationReport` faithfully represents archive contents, contextual fronts and run metadata.
- [ ] DOC-004/5.1/2 — non-finite values are sanitized before export so report consumers are not poisoned.
- [ ] DOC-004/5.1/3 — same seed and same scenario set give reproducible search trajectories within expected tolerance.
- [ ] DOC-004/5.1/4 — docs and release notes do not present optimizer output as field-validated runtime truth.

## 6. Regression Classes to Hunt Actively

- [ ] DOC-004/6.1/1 — `type_complexity` or other lint-driven refactors that change meaning while trying to satisfy CI.
- [ ] DOC-004/6.1/2 — archive / report code that still compiles but drops contextual information silently.
- [ ] DOC-004/6.1/3 — parameter-space changes that desync optimizer, swarm config and report interpretation.

## 7. Exit Criteria

`DOC-004` не е complete ако липсва:
- поне една contextual archive check
- поне една heterogeneity decode check
- поне една surrogate-scope check
- поне една report / reproducibility check
