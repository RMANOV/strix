# DOC-002 Mission Brain, Intent, FFI and Stateful Adaptation Questionnaire

Pack: `2026-04-02`

Primary surfaces:
- `python/strix/brain.py`
- `crates/strix-python`
- Python fallback logic vs Rust-backed execution paths

Mission risk focus:
- T1 wrong mission planning or stale plan reuse
- T2 mismatch between Python behavior and Rust-backed runtime
- T2 degraded adaptation due to hidden state ownership errors

## 1. Intent to Plan Conversion

- [ ] DOC-002/1.1/1 — `MissionIntent` to `MissionPlan` uses live fleet state, not stale snapshots cached before the current request.
- [ ] DOC-002/1.1/2 — requested drone count is clamped deterministically and never allocates more than alive drones.
- [ ] DOC-002/1.1/3 — fallback behavior for missing mission area is explicit and traceable, not silent planner degradation.
- [ ] DOC-002/1.1/4 — task IDs are stable across plan build, auction pass and subsequent tick execution.

## 2. Stateful Adaptation and World Model Drift

- [ ] DOC-002/2.1/1 — active plan ownership is single-source-of-truth; plan mutation after loss or reauction does not leave shadow state behind.
- [ ] DOC-002/2.1/2 — threat updates can bias regime and trigger reauction without corrupting current mission plan structure.
- [ ] DOC-002/2.1/3 — comms freshness and stale-age logic affects planning confidence consistently across sync and async entry points.
- [ ] DOC-002/2.1/4 — any future explicit orientation layer must inherit these checks: source trust, doctrine bias, novelty, self-model mismatch and broken-assumption memory.

## 3. FFI Boundary and Parity

- [ ] DOC-002/3.1/1 — behavior remains sane when Rust FFI is unavailable and Python fallback takes over.
- [ ] DOC-002/3.1/2 — keyword names, field names and constructor signatures match between async wrappers, sync paths and helper methods.
- [ ] DOC-002/3.1/3 — `maturin` / wheel path exercises the same semantics that local Python tests assume.
- [ ] DOC-002/3.1/4 — build-only green is not treated as parity evidence; at least one smoke path must execute through the extension module.

## 4. Runtime Communications as Planning Input

- [ ] DOC-002/4.1/1 — packet success rate and state age influence planning validation but do not create impossible oscillation between ENGAGE and EVADE.
- [ ] DOC-002/4.1/2 — per-drone link state and fleet-wide link state are combined in a predictable way, with no silent overwrite.
- [ ] DOC-002/4.1/3 — link-quality fallback tracker cannot grow stale indefinitely without visible effect on planning confidence.

## 5. Loss Handling and Reauction

- [ ] DOC-002/5.1/1 — drone loss marks kill-zone or threat memory and removes the lost drone from active assignments.
- [ ] DOC-002/5.1/2 — attrition-triggered regime changes do not leave orphan assignments on dead drones.
- [ ] DOC-002/5.1/3 — reauction trigger is idempotent and cannot create runaway auction churn under repeated loss reports.

## 6. Known Regression Classes to Hunt

- [ ] DOC-002/6.1/1 — sync/async wrapper drift where one path passes the wrong keyword or omits a new state update.
- [ ] DOC-002/6.1/2 — Python fallback logic diverges from Rust path because only one side got updated during a refactor.
- [ ] DOC-002/6.1/3 — plan explanation strings stay green while actual confidence or assignment content is wrong.
- [ ] DOC-002/6.1/4 — `_active_plan` is mutated in-place by one subsystem while another assumes immutable semantics.

## 7. Evidence Discipline

За checked point приложи поне едно от следните:
- Python unit / smoke test name
- FFI smoke evidence
- exact method path in `brain.py`
- commit or PR proving the regression or fix

## 8. Exit Criteria

`DOC-002` не е complete ако липсва:
- поне една fallback-vs-FFI проверка
- поне една active-plan ownership проверка
- поне една loss/reauction проверка
- statement дали има stateful adaptation gap или не
