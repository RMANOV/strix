# STRIX -- Swarm Tactical Reasoning and Intelligence eXchange

[![CI](https://github.com/RMANOV/strix/actions/workflows/ci.yml/badge.svg)](https://github.com/RMANOV/strix/actions/workflows/ci.yml)

*From the genus* Strix *(owls) -- silent predators with exceptional sensor fusion.*

> **The battlefield is a market. Drones are traders. Missions are positions. The enemy is a counterparty.**

Quantitative trading firms have spent decades solving problems structurally identical to drone swarm orchestration: decentralized decision-making under uncertainty, adversarial prediction, resource allocation with constraints, and graceful degradation under stress. STRIX applies these battle-tested algorithms -- particle filters, regime-switching models, combinatorial auctions, portfolio optimization -- directly to autonomous multi-vehicle coordination in contested, GPS-denied, comms-degraded environments.

The mapping is not a metaphor. It is a structural isomorphism. The mathematics that tracks hidden asset prices from noisy market data is the same mathematics that tracks drone position from noisy IMU/barometer/magnetometer readings. The auction that allocates capital across a portfolio is the same auction that allocates drones across missions. The counterparty model that predicts institutional flow is the same model that predicts enemy maneuvers.

---

## Key Innovations

### 1. Adversarial Particle Filter ("Enemy as Market")

A **dual particle filter** architecture. The first filter tracks friendly drone state in 6D `[x, y, z, vx, vy, vz]` for GPS-denied navigation (configurable 50–2000 particles, default 200 at 10 Hz). The second filter tracks enemy entities, encoding each adversary as a kinematic state plus an intent hypothesis: DEFENDING, ATTACKING, or RETREATING. This directly mirrors counterparty prediction in quantitative trading -- model the opponent's likely actions before they complete them. The informational advantage is measured in seconds, which in combat is the difference between initiative and reaction.

### 2. Anti-Fragile Loss Recovery (Taleb)

The swarm gets **stronger** after losses, not weaker. When a drone is destroyed:
- The loss location is marked as a kill zone with elevated risk scoring (spatial memory).
- Future auction bids automatically penalize tasks near kill zones (risk repricing).
- Attrition analysis classifies the failure mode (cause identification).
- Surviving drones adjust approach vectors to avoid the learned threat pattern (behavioral adaptation).

Each loss provides information that improves all future decisions. This is the military application of Taleb's anti-fragility -- a convex response to adversity where volatility becomes a source of strength.

### 3. Multi-Horizon Temporal Reasoning

Three parallel planning horizons with cascade constraints:

| Horizon | Update Rate | Lookahead | Purpose |
|---------|-------------|-----------|---------|
| H1 Tactical | 0.1 s | 10 s | Obstacle avoidance, collision prevention |
| H2 Operational | 5 s | 5 min | Formation, coordination, sensor coverage |
| H3 Strategic | 60 s | 1 hr | Mission phasing, resource allocation |

**Top-down**: strategic decisions constrain operational plans, which constrain tactical maneuvers. **Bottom-up**: a tactical impossibility vetoes the operational plan. An operational infeasibility flags the strategic plan for re-evaluation. No planning horizon operates in isolation.

### 4. Stigmergy + LLM Hybrid

Bio-inspired coordination fused with explainable AI. Drones deposit digital pheromones (`explored`, `danger`, `interest`, `relay`) onto a shared spatial grid. Other drones read the field to avoid redundant coverage and steer toward high-value areas. Pheromone payloads are ~20 bytes -- bandwidth-efficient enough for contested RF environments.

On top of this, an edge-deployable LLM (3B parameters, Phi-3 / Llama-3.2 class) converts raw decision traces into natural-language explanations. The commander sees not just what the swarm decided, but *why*. When the LLM is unavailable (comms-denied, resource-constrained), a rule-based keyword parser provides baseline NLP capability -- the system never depends on a single inference path.

### 5. Integrated Safety and Engagement

Five operational modules are wired directly into the orchestration loop:

- **Formation Control** — 7 tactical formations (Vee, Line, Wedge, Column, Echelon L/R, Spread) with proportional correction and deadband control
- **Rules of Engagement (ROE)** — go/no-go gate with WeaponsHold/Tight/Free postures, self-defense override, collateral risk cap, and human-in-the-loop escalation
- **Electronic Warfare Response** — GPS denial/spoofing, comms jamming, radar lock, and directed energy detection with automated multi-action response plans
- **Control Barrier Functions (CBF)** — provably safe collision avoidance (inter-drone separation, altitude bounds, no-fly zones) with formal dh/dt + αh ≥ 0 guarantees
- **Multi-Horizon Temporal Manager** — three parallel particle filters (H1 tactical 0.1s, H2 operational 5s, H3 strategic 60s) with top-down cascade constraints

### 6. Fractal Self-Similarity

The same organizational algorithms operate at every echelon:

```
Pair (2) --> Squad (5-8) --> Platoon (20-30) --> Company (60-100)
```

A squad of 5–8 drones has the same command structure, consensus mechanism, and coordination protocol as a platoon of 20–30 or a company of 60+. Leadership is emergent via consensus, not assigned. Losing a leader degrades performance, not capability. There is no "head" to cut off.

### 7. "Dark Pool" Mission Compartmentalization

Sensitive tasks (strike missions, electronic warfare operations) are restricted to specific sub-swarms via `dark_pool` identifiers. Reconnaissance drones in sub-swarm A cannot see or bid on strike tasks assigned to sub-swarm B. This enforces need-to-know compartmentalization at the algorithmic level -- the military equivalent of dark pool order routing in financial markets, where large orders execute without revealing intent to the broader market.

---

## Architecture

```
+-----------------------------------------------------------+
|               LAYER 0: HUMAN INTERFACE                    |
|  Voice/NLP -> Intent -> Confirmation -> Execution         |
|  "Recon north ridge, avoid AA" -> MissionIntent           |
+-----------------------------------------------------------+
|               LAYER 1: MARKET BRAIN                       |
|  predict -> update -> regime_check -> auction -> assign   |
|  Particle filters (6D) + Regime-switching Markov model    |
+-----------------------------------------------------------+
|               LAYER 2: AUCTION FLOOR                      |
|  Sealed-bid scoring + Hungarian assignment + Dark pools   |
|  score = urgency*10 + capability*3 + proximity*5          |
|         + energy*2 - risk*4                               |
+-----------------------------------------------------------+
|               LAYER 3: MESH COORDINATION                  |
|  Fractal hierarchy + Stigmergy + Gossip protocol          |
|  O(log N) convergence, bandwidth-aware prioritization     |
+-----------------------------------------------------------+
|               LAYER 4: PUPPET MASTER                      |
|  PlatformAdapter trait -> MAVLink / ROS2 / Simulator      |
|  Unified command interface across all drone platforms      |
+-----------------------------------------------------------+
|               LAYER 5: GLASS BOX                          |
|  Decision traces + Narration + After-action replay        |
|  Every decision is explainable and auditable              |
+-----------------------------------------------------------+
```

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Core Engine | Rust (rayon, ndarray, nalgebra) | Particle filter, auction, mesh -- parallel, zero-copy |
| Orchestration | Python 3.12+ (asyncio) | Mission brain, NLP, planning, digital twin |
| FFI Bridge | PyO3 + maturin | Rust-to-Python interop with minimal overhead |
| Math | ndarray, nalgebra | Linear algebra, matrix ops, state estimation |
| Autopilot | MAVLink v2 | PX4 / ArduPilot integration |
| Robotics | ROS2 (DDS) | Ground vehicle / naval craft integration |
| Edge LLM | llama.cpp / ONNX | On-device NLP for comms-denied operations |
| Serialization | serde + JSON | All state is serializable for replay and audit |

---

## Performance (Criterion Benchmarks)

All measurements on a single core, unoptimized test profile. Release builds are 2–5× faster.

| Benchmark | Configuration | Time |
|-----------|--------------|------|
| Particle filter step | 50 particles | 42 µs |
| Particle filter step | 200 particles (default) | 75 µs |
| Particle filter step | 1000 particles | 226 µs |
| Combinatorial auction | 5 drones, 3 tasks | 2.7 µs |
| Combinatorial auction | 20 drones, 10 tasks | 47 µs |
| Combinatorial auction | 50 drones, 20 tasks | 465 µs |
| **Full swarm tick** | **5 drones** | **298 µs** |
| **Full swarm tick** | **10 drones** | **580 µs** |
| **Full swarm tick** | **20 drones** | **1.15 ms** |

The full tick includes: particle filter update, regime detection (CUSUM + Hurst + intent), formation correction, ROE authorization, combinatorial auction, gossip propagation, pheromone update, CBF safety clamp, and XAI trace recording. All modules integrated, all safety checks active.

At 1.15 ms per tick for 20 drones, the system supports **870 Hz orchestration** — well within the 10 Hz real-time target with 99% headroom for sensor processing, communication, and platform I/O.

---

## Quick Start

```bash
git clone https://github.com/RMANOV/strix.git
cd strix

# Build and test (315+ tests, all passing)
cargo test --workspace

# Build optimized
cargo build --release

# Install Python orchestration layer
pip install -e .
```

**Requirements**: Rust 1.75+ (2021 edition), Python 3.11+, maturin 1.11+

---

## Project Structure

```
strix/
├── crates/
│   ├── strix-core/          6D particle filter, regime model, CUSUM, formation, ROE, EW, CBF, temporal
│   ├── strix-auction/       Combinatorial auction, portfolio optimization, anti-fragile
│   ├── strix-mesh/          Mesh coordination, gossip protocol, stigmergy, fractal hierarchy
│   ├── strix-adapters/      Platform adapters (MAVLink, ROS2, simulator)
│   ├── strix-xai/           Explainability engine, decision traces, narration
│   ├── strix-swarm/         Integration orchestrator: tick loop chains all crates
│   ├── strix-python/        PyO3 Python bindings (cdylib)
│   └── strix-playground/    Simulation engine with 4 presets + BattleReport
├── python/strix/
│   ├── brain.py             Mission planner (Market Brain, 10 Hz main loop)
│   ├── adversarial.py       Adversarial prediction engine (dual particle filter)
│   ├── nlp/                 Intent parsing, acknowledgment loop
│   ├── temporal/            Multi-horizon planner (H1/H2/H3 cascade)
│   ├── digital_twin/        3D world model, rehearsal, visualization
│   └── llm/                 Military LLM, edge inference, narration
├── sim/scenarios/           YAML simulation scenarios
├── demo/                    Dashboard and demo scripts
└── docs/                    Architecture, trading mapping, ITAR analysis
```

---

## Documentation

- [Architecture](docs/architecture.md) -- 6-layer system design with data flow diagrams
- [Trading-to-Warfare Mapping](docs/trading_mapping.md) -- complete algorithm mapping (14 transformations)
- [Competitor Comparison](docs/competitor_comparison.md) -- differentiation vs. OpenAI, xAI/SpaceX, Shield AI, Anduril, Auterion
- [ITAR Analysis](docs/itar_analysis.md) -- open-source legal strategy and export control precedents

---

## License

Apache 2.0 -- see [LICENSE-APACHE](LICENSE-APACHE).

The open-source core contains published mathematics and general-purpose algorithms. It is ITAR-exempt under **EAR Section 734.7 (Published Information Exemption)**: algorithms derived from published academic literature and publicly available quantitative finance research do not constitute controlled technical data. Military-specific integrations (weapon adapters, classified platform interfaces, classified operational scenarios) are maintained in a separate, access-controlled repository with appropriate export controls. See [ITAR Analysis](docs/itar_analysis.md) for the full legal framework.

---

## Disclaimer

STRIX is a **research prototype** under active development. It has not been validated for operational deployment and carries no warranty regarding fitness for any particular use case, including defense applications. The system is intended for research, experimentation, and technology demonstration. Users are responsible for compliance with all applicable laws, regulations, and export controls.

---

*STRIX -- because the most dangerous swarm is the one that thinks like a trading firm.*
