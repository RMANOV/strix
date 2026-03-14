# STRIX

**Swarm Tactical Reasoning and Intelligence eXchange**

*From the genus Strix (owls) -- silent predators with exceptional sensor fusion.*

> The battlefield is a market. Drones are traders. Missions are positions. The enemy is a counterparty.

---

A drone swarm orchestrator built on quantitative trading mathematics. STRIX applies algorithms proven in the most adversarial competitive environment on Earth -- financial markets -- to the problem of autonomous multi-vehicle coordination in contested environments.

## Key Innovations

1. **GPS-Denied Navigation via Particle Filter** -- the same 6D particle filter that tracks hidden asset prices now tracks drone position from IMU, barometer, magnetometer, and visual odometry. 1000 particles, 6 dimensions, 10 Hz.

2. **Adversarial Prediction Engine** -- a dual particle filter that models enemy intent as a first-class concept. Each enemy particle is a hypothesis about whether the adversary is defending, attacking, or retreating. Predict their maneuver before they complete it.

3. **Anti-Fragile Swarm** -- drone losses do not simply degrade the swarm. They improve it. Loss locations become kill zones with elevated risk scores. Surviving drones automatically avoid learned threats. Mission effectiveness per drone increases after attrition.

4. **Combinatorial Task Auction** -- drones bid on tasks like traders bid on assets. Sealed-bid scoring, modified Hungarian assignment, dark pool compartmentalization. The same mathematics that allocates capital across a portfolio now allocates drones across missions.

5. **Multi-Horizon Planning (t-CoT)** -- three parallel planning horizons cascade top-down (strategic 60s, operational 5s, tactical 0.1s) with bottom-up veto capability. A tactical impossibility vetoes the operational plan that requested it.

6. **Glass Box Explainability** -- every decision produces a human-readable explanation. No black boxes. The commander sees not just what the system decided, but why, in real time.

## Architecture

```
Layer 0  Human Interface     Voice/NLP -> Intent -> Confirm -> Execute
Layer 1  Market Brain        Particle filters + Regime-switching + Adversarial
Layer 2  Auction Floor       Sealed-bid scoring + Hungarian + Dark pools
Layer 3  Mesh Coordination   Fractal hierarchy + Stigmergy + Gossip
Layer 4  Puppet Master       PlatformAdapter -> MAVLink / ROS2 / Simulator
Layer 5  Glass Box           Decision traces + Narration + After-action replay
```

## Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| Core Engine | Rust + rayon | Particle filter, auction, mesh -- parallel, zero-copy |
| Orchestration | Python 3.12 + asyncio | Mission brain, NLP, planning, digital twin |
| FFI Bridge | PyO3 + maturin | Rust-Python interop with minimal overhead |
| Autopilot | MAVLink v2 | PX4 / ArduPilot integration |
| Robotics | ROS2 (DDS) | Ground vehicle / naval craft integration |
| Visualization | rerun.io | 3D digital twin with uncertainty clouds |
| Edge LLM | llama.cpp / ONNX | On-device NLP for comms-denied ops |
| Serialization | serde + JSON | All state is serializable for replay |

## Quick Start

```bash
# Clone
git clone https://github.com/RMANOV/strix.git
cd strix

# Build and test (189 tests)
cargo test --workspace

# Build optimized
cargo build --release

# Install Python layer
pip install -e .
```

### Requirements

- Rust 1.75+ (2021 edition)
- Python 3.11+
- NumPy 1.26+
- maturin 1.11+

## Project Structure

```
strix/
├── crates/
│   ├── strix-core/          # 6D particle filter, state types, regime model
│   ├── strix-auction/       # Combinatorial auction, portfolio, risk, anti-fragile
│   ├── strix-mesh/          # Mesh coordination, gossip, stigmergy, fractal
│   ├── strix-adapters/      # Platform adapters (MAVLink, ROS2, simulator)
│   └── strix-xai/           # Explainability engine, decision traces
├── python/strix/
│   ├── brain.py             # Mission planner (Market Brain)
│   ├── adversarial.py       # Adversarial prediction engine
│   ├── nlp/                 # Intent parsing, acknowledgment loop
│   ├── temporal/            # Multi-horizon planner (t-CoT)
│   ├── digital_twin/        # 3D world model, rehearsal, visualization
│   └── llm/                 # Military LLM, edge inference
├── sim/scenarios/           # YAML simulation scenarios
├── docs/                    # Architecture, trading mapping, ITAR analysis
└── demo/                    # Demo scripts and dashboard
```

## Documentation

- [Architecture](docs/architecture.md) -- 6-layer system design with data flow
- [Trading Mapping](docs/trading_mapping.md) -- complete algorithm-to-warfare mapping (14 rows)
- [ITAR Analysis](docs/itar_analysis.md) -- open-source legal strategy and precedents
- [Competitor Comparison](docs/competitor_comparison.md) -- differentiation vs. 5 AVO competitors

## Simulation Scenarios

| Scenario | Drones | Duration | Key Test |
|---|---|---|---|
| [GPS-Denied Recon](sim/scenarios/gps_denied_recon.yaml) | 4 | 10 min | Particle filter navigation without GPS |
| [Contested Strike](sim/scenarios/contested_strike.yaml) | 6 | 8 min | SAM avoidance, regime switching, dark pools |
| [Mass Attrition](sim/scenarios/mass_attrition.yaml) | 12 | 15 min | 50% loss survivability, anti-fragile adaptation |
| [Multi-Domain](sim/scenarios/multi_domain.yaml) | 3 UAV + 2 UGV | 12 min | Cross-platform coordination, air-ground escort |

## License

Apache 2.0 -- see [LICENSE-APACHE](LICENSE-APACHE).

The open-source core contains published mathematics and general-purpose algorithms. Military-specific integrations (weapon adapters, classified scenarios, classified platform adapters) are maintained in a separate proprietary repository. See [ITAR Analysis](docs/itar_analysis.md) for the legal framework.

## Roadmap

| Phase | Timeline | Deliverable |
|---|---|---|
| Phase 1 | Q1 2026 | Particle filter + auction core in Rust, Python orchestration |
| Phase 2 | Q2 2026 | Simulator integration, 4-drone GPS-denied demo |
| Phase 3 | Q3 2026 | MAVLink adapter, real hardware flight test |
| Phase 4 | Q4 2026 | Adversarial prediction, multi-horizon planning |
| Phase 5 | Q1 2027 | Edge LLM integration, voice command interface |
| Phase 6 | Q2 2027 | Multi-domain (UAV + UGV), production hardening |

---

**Note**: STRIX is a research prototype. It has not been tested with live hardware or in operational environments.

*STRIX -- because the most dangerous swarm is the one that thinks like a trading firm.*
