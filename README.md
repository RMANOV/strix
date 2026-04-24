# STRIX -- Swarm Coordination, Safety, and Explainable Autonomy

## Technical deep-dives

- **Medium:** [Sixty Seconds of Fear: How a Drone Swarm Learned to Survive](https://medium.com/@r.manov/sixty-seconds-of-fear-how-a-drone-swarm-learned-to-survive-bec79d34de0b)
- **Medium:** [I Taught a Drone Swarm to Be Afraid — and It Made Better Decisions](https://medium.com/@r.manov/i-taught-a-drone-swarm-to-be-afraid-and-it-made-better-decisions-a32bdea5fb93)
- **Dev.to:** [The Day the Swarm Got Scared — And Saved Everyone](https://dev.to/ruslan_manov/the-day-the-swarm-got-scared-and-saved-everyone-3gl2)
- **Dev.to:** [Building a Drone Swarm Orchestrator That Gets Scared — 20 Subsystems in 35K Lines of Rust](https://dev.to/ruslan_manov/building-a-drone-swarm-orchestrator-that-gets-scared-20-subsystems-in-35k-lines-of-rust-4bp4)

[![CI](https://github.com/RMANOV/strix/actions/workflows/ci.yml/badge.svg)](https://github.com/RMANOV/strix/actions/workflows/ci.yml)

STRIX is a Rust + Python research platform for coordinating heterogeneous autonomous systems in degraded environments. The public repository focuses on state estimation, task allocation, resilient mesh coordination, safety constraints, simulation, and explainable decision traces.

The core design draws from quantitative finance, control theory, and distributed systems: particle filters for hidden-state estimation, auction-based assignment for constrained resource allocation, regime detection for degraded-mode adaptation, and replayable traces for auditability. The result is a platform-agnostic autonomy stack aimed at research, evaluation, and prototype integration.

The public tree is intentionally conservative. It exposes the reusable autonomy core: coordination, safety, simulation, explainability, and platform-agnostic adapter boundaries. Evaluator collateral, internal review packs, and program-specific material are not maintained as part of the public repository.

## Official Project Identity

The official public upstream is `https://github.com/RMANOV/strix`.

Forks are allowed under Apache-2.0, but the STRIX name, project identity, and official release channel are separate from the source-code license. See [NOTICE](NOTICE), [TRADEMARKS.md](TRADEMARKS.md), and [Project_Docs/provenance/OFFICIAL_RELEASES.md](Project_Docs/provenance/OFFICIAL_RELEASES.md).

Public releases should be traceable to the official upstream and maintainer release authority. Private keys, local machine release state, customer-specific material, and private companion modules do not belong in the public repository.

## Focus Areas

- **State estimation and prediction**: particle filters, regime detection, anomaly handling, and degraded-mode reasoning for uncertain environments.
- **Task allocation and coordination**: combinatorial auctions, stigmergic coordination, fractal scaling, and bandwidth-aware mesh behavior.
- **Safety and policy gates**: barrier functions, conservative task gating, and resilience to GPS loss, comms degradation, and sensor noise.
- **Explainability and replay**: structured decision traces, narration hooks, and after-action inspection.
- **Simulator-first integration**: reusable adapters and a simulation playground before platform-specific deployment work.

## Architecture

```text
+-----------------------------------------------------------+
|              LAYER 0: HUMAN / API INTERFACE               |
|  intents, constraints, confirmation, execution requests   |
+-----------------------------------------------------------+
|          LAYER 1: PLANNING AND STATE ESTIMATION           |
|  particle filters, regime detection, anomaly handling     |
+-----------------------------------------------------------+
|             LAYER 2: TASK ALLOCATION ENGINE               |
|  auction scoring, assignment, energy/risk tradeoffs       |
+-----------------------------------------------------------+
|              LAYER 3: COORDINATION AND MESH               |
|  stigmergy, gossip, hierarchy, distributed convergence    |
+-----------------------------------------------------------+
|           LAYER 4: SAFETY AND PLATFORM ADAPTERS           |
|  safety constraints, simulator-first adapters, I/O edge   |
+-----------------------------------------------------------+
|           LAYER 5: EXPLAINABILITY AND REPLAY              |
|  decision traces, narration hooks, audit and playback     |
+-----------------------------------------------------------+
```

## Performance Snapshot

All measurements below are from Criterion benchmarks on a single core in an unoptimized test profile. Release builds are materially faster.

| Benchmark | Configuration | Time |
|-----------|--------------|------|
| Particle filter step | 50 particles | 42 us |
| Particle filter step | 200 particles (default) | 75 us |
| Particle filter step | 1000 particles | 226 us |
| Combinatorial auction | 5 drones, 3 tasks | 2.7 us |
| Combinatorial auction | 20 drones, 10 tasks | 47 us |
| Combinatorial auction | 50 drones, 20 tasks | 465 us |
| Full swarm tick | 5 drones | 298 us |
| Full swarm tick | 10 drones | 580 us |
| Full swarm tick | 20 drones | 1.15 ms |

The full tick benchmark covers estimation, regime updates, assignment, coordination, safety clamps, and trace capture. At 1.15 ms per tick for 20 drones, the system comfortably fits inside a 10 Hz orchestration loop with significant headroom for sensor processing and platform I/O.

## Quick Start

```bash
git clone https://github.com/RMANOV/strix.git
cd strix

cargo test --workspace
cargo build --release
pip install -e .
```

**Requirements**: Rust 1.75+, Python 3.11+, maturin 1.11+

## Project Structure

```text
strix/
├── crates/
│   ├── strix-core/        state estimation, resilience modules, safety constraints
│   ├── strix-auction/     task allocation and portfolio-style optimization
│   ├── strix-mesh/        coordination mesh, gossip, stigmergy, hierarchy
│   ├── strix-adapters/    simulator-first adapter boundary and platform stubs
│   ├── strix-xai/         explainability engine and decision traces
│   ├── strix-swarm/       integration tick loop across the Rust crates
│   ├── strix-python/      PyO3 bindings for Rust/Python interop
│   └── strix-playground/  simulation playground and preset execution
├── python/strix/
│   ├── brain.py           orchestration loop and planning shell
│   ├── adversarial.py     prediction and hidden-state modeling helpers
│   ├── nlp/               intent parsing and confirmation flow
│   ├── temporal/          multi-horizon planning logic
│   ├── digital_twin/      world model, rehearsal, visualization
│   └── llm/               optional narration and generic provider hooks
├── sim/scenarios/         public simulation scenarios and placeholders
├── demo/                  public demo placeholders and lightweight examples
├── Project_Docs/          sanitized public notes
└── paper/                 paper source and generated PDF
```

## Licensing

This public repository is licensed under [Apache License 2.0](LICENSE-APACHE).

## Disclaimer

STRIX is a research prototype under active development. It is intended for research, experimentation, evaluation, and technology demonstration. Users are responsible for validating fitness for their own domain, integration path, and regulatory context.
