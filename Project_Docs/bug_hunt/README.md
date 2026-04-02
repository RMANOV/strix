# STRIX Bug Hunt Tracking Hub

Цел: едно място за дълбок, repo-specific bug hunt, architectural risk review, single point of failure анализ и run traceability за `RMANOV/strix`.

STRIX не е единичен бинарен артефакт. Това е смесена Rust/Python система с няколко критични граници:
- cross-crate orchestration между `strix-core`, `strix-swarm`, `strix-mesh`, `strix-auction`, `strix-optimizer`, `strix-xai`
- Python mission brain и planning слой
- Rust/Python FFI чрез `strix-python`
- safety-sensitive logic: ROE, CBF, EW response, regime transitions
- симулационна и benchmark evidence layer, която може да маскира runtime regressions, ако не е структурирана

Този hub съществува, за да направи bug hunt-а:
1. Дълбок: проверява не само функции, а причинно-следствени вериги и mission-level behavior.
2. Прецизен: всяка точка има стабилен reference и ясна evidence discipline.
3. Стратегически: фокусира се върху surfaces, които могат да счупят swarm behavior, safety или trustworthiness.
4. Проследим във времето: questionnaire packs, per-run folders и append-only run registry.

## Какво се следи

Тук се следят три различни класа риск:

| Tier | Meaning | Typical STRIX Example |
|---|---|---|
| T0 | Safety / loss-of-control | ROE bypass, broken CBF guarantees, corrupt tick output |
| T1 | Mission failure / wrong tasking | wrong auction assignment, stale plan execution, mesh coordination collapse |
| T2 | Adaptation / resilience degradation | contextual archive drift, contagion over-propagation, FFI fallback mismatch |
| T3 | Observability / release hygiene | missing traces, perf budget drift, incomplete docs or run evidence |

## Operating Model

1. `questionnaire_packs/<date>/` съдържа versioned snapshot на bug-hunt documents.
2. `MANIFEST.json` е machine-readable index на pack-а.
3. `INDEX.md` е human-readable входна точка.
4. `runs/_TEMPLATE_RUN_FOLDER/` е canonical skeleton за всеки нов run.
5. `RUN_REGISTRY.jsonl` е append-only ledger на всички изпълнени run-ове.
6. Всеки реален run записва:
   - кои docs са ползвани
   - срещу кой branch / commit е run-ът
   - кои surfaces са проверени
   - какви findings и SPOFs са отворени, затворени или отложени

## Структура

```text
Project_Docs/bug_hunt/
├─ README.md
├─ questionnaire_packs/
│  └─ 2026-04-02/
│     ├─ MANIFEST.json
│     ├─ INDEX.md
│     ├─ DOC-001_...
│     ├─ DOC-002_...
│     ├─ DOC-003_...
│     ├─ DOC-004_...
│     └─ DOC-005_...
└─ runs/
   ├─ RUN_REGISTRY.jsonl
   └─ _TEMPLATE_RUN_FOLDER/
      ├─ run_result.json
      ├─ checkpoints.jsonl
      └─ summary.md
```

## Препоръчани run типове

- `STATIC_AUDIT`: code-path review без execution
- `POST_PR_REGRESSION`: run след конкретен PR или branch
- `FFI_PARITY`: Rust/Python parity, wheel/build/smoke focus
- `SIM_PRESET_SWEEP`: playground presets и scenario smoke behavior
- `SAFETY_GATE_REVIEW`: ROE, CBF, EW, regime transitions
- `SPOF_REVIEW`: single point of failure reassessment
- `PERF_BUDGET_REVIEW`: benchmarks, test duration, orchestration budget

## Reference Format

Използвай стабилни references:

- `DOC-001/3.2/4` за checklist item
- `DOC-005/SPOF-03` за single point of failure record

Където:
- `DOC-001` е документът
- `3.2` е секцията
- `4` е конкретният item в секцията

## Минимални полета за `run_result.json`

- `run_id`
- `pack_id`
- `repo`
- `branch`
- `commit_sha`
- `run_type`
- `focus_surfaces`
- `docs_touched`
- `checkpoints_total`
- `checkpoints_passed`
- `findings_open`
- `findings_fixed`
- `findings_deferred`
- `spofs_reviewed`
- `highest_severity`
- `status`
- `summary`
- `linked_prs`
- `linked_commits`
- `next_actions`

## Минимални полета за `checkpoints.jsonl`

По един JSON object на ред:
- `timestamp`
- `doc_ref`
- `surface`
- `status`
- `severity`
- `finding_kind`
- `summary`
- `evidence`
- `linked_issue`
- `linked_pr`
- `fixed`

## Какво се счита за завършен run

Run е завършен само ако има:
- попълнен `summary.md`
- попълнен `run_result.json`
- поне един смислен checkpoint или изрично `no findings` обяснение
- append към `RUN_REGISTRY.jsonl`
- ясно stated next action: `close`, `fix`, `defer`, `retest` или `needs_manual`

## STRIX-specific правило

Не маркирай точка като проверена само защото unit test минава. За STRIX това не е достатъчно. Приоритет са cross-layer invariants:
- `brain -> swarm -> mesh -> safety -> xai`
- `auction -> assignment -> decision trace`
- `optimizer -> exported params -> runtime behavior`
- `Python fallback -> Rust FFI path`

Точно там се появяват най-скъпите regressions.
