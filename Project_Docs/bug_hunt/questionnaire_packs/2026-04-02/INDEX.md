# STRIX Bug Hunt Pack 2026-04-02

Pack ID: `2026-04-02`

Този pack е първият strategic bug-hunt baseline за `RMANOV/strix`. Той е структуриран по fault lines, не по crate alphabetically.

## Документи

1. `DOC-001` — Swarm tick, safety и cross-crate integration
   Focus: `strix-swarm`, `strix-core`, `strix-xai`
   Използвай при промени в orchestration order, decision payload, criticality, ROE, CBF, EW или telemetry sanitation.

2. `DOC-002` — Mission brain, intent pipeline, FFI и stateful adaptation
   Focus: `python/strix/brain.py`, `strix-python`, fallback vs Rust paths
   Използвай при промени в planning, intent routing, loss handling, plan ownership, Python/Rust parity.

3. `DOC-003` — Mesh, auction и higher-order coordination
   Focus: `strix-mesh`, `strix-auction`
   Използвай при промени в gossip, contagion, hypergraph coordination, dark pools, bundle validation.

4. `DOC-004` — Optimizer, contextual evolution и offline graph surrogate
   Focus: `strix-optimizer`
   Използвай при промени в contextual archives, heterogeneity, surrogate blending, reports и param mapping.

5. `DOC-005` — Single Point of Failure analysis
   Focus: repo-wide critical choke points
   Използвай като периодичен strategic review и задължително преди release hardening.

## Препоръчана последователност

1. Започни с `DOC-005`, ако промените са архитектурни или cross-cutting.
2. Мини през `DOC-001` при всяка промяна в `tick.rs`, safety, regime или final decision output.
3. Добави `DOC-002`, ако има Python / FFI / planning / mission-level work.
4. Добави `DOC-003`, ако има mesh, contagion, auction или coordination semantics.
5. Добави `DOC-004`, ако има optimizer, tuning или config export implications.

## Minimum Coverage by Change Type

| Change Type | Minimum Docs |
|---|---|
| Tick orchestration / safety | DOC-001 + DOC-005 |
| Brain / planning / FFI | DOC-002 + DOC-005 |
| Mesh / auction / dark-pool | DOC-003 + DOC-005 |
| Optimizer / surrogate / param export | DOC-004 + DOC-005 |
| Release candidate | DOC-001 + DOC-002 + DOC-003 + DOC-004 + DOC-005 |

## Notes

- Ако run-ът е purely static, маркирай това изрично в `run_result.json`.
- Ако finding е simulation-only, не го маркирай като closed без code-path evidence.
- Ако SPOF е приет временно, това трябва да се вижда както в summary, така и в registry entry.
