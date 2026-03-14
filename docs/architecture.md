# STRIX Architecture

## The Core Thesis

**The battlefield is a market.**

Quantitative trading firms have spent decades solving problems that are structurally identical to drone swarm coordination: decentralized decision-making under uncertainty, adversarial prediction, resource allocation with constraints, and graceful degradation under stress. STRIX applies these solutions directly.

| Trading Concept | STRIX Application |
|---|---|
| Particle filter | GPS-denied 6D navigation |
| Regime-switching | PATROL / ENGAGE / EVADE |
| Combinatorial auction | Task allocation |
| Portfolio optimization | Fleet diversification |
| Counterparty modeling | Adversarial prediction |
| Risk management | Attrition protection |

---

## 6-Layer Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  LAYER 0: HUMAN INTERFACE               │
│  Voice/NLP → Intent → Confirmation → Execution          │
│  "Recon north ridge, avoid AA"  →  MissionIntent        │
├─────────────────────────────────────────────────────────┤
│                  LAYER 1: MARKET BRAIN                  │
│  predict → update → regime_check → auction → assign     │
│  Particle filters (6D) + Regime-switching Markov model  │
├─────────────────────────────────────────────────────────┤
│                  LAYER 2: AUCTION FLOOR                 │
│  Sealed-bid scoring + Hungarian assignment + Dark pools  │
│  score = urgency×10 + capability×3 + proximity×5        │
│         + energy×2 - risk×4                             │
├─────────────────────────────────────────────────────────┤
│                  LAYER 3: MESH COORDINATION             │
│  Fractal hierarchy + Stigmergy + Gossip protocol        │
│  O(log N) convergence, bandwidth-aware prioritization   │
├─────────────────────────────────────────────────────────┤
│                  LAYER 4: PUPPET MASTER                 │
│  PlatformAdapter trait → MAVLink / ROS2 / Simulator     │
│  Unified command interface across all drone platforms    │
├─────────────────────────────────────────────────────────┤
│                  LAYER 5: GLASS BOX                     │
│  Decision traces + Narration + After-action replay       │
│  Every decision is explainable and auditable             │
└─────────────────────────────────────────────────────────┘
```

---

## Layer 0: Human Interface

**Module**: `python/strix/nlp/`

The commander speaks; STRIX listens, confirms, and executes.

### Intent Parsing

Natural language commands are converted to structured `MissionIntent` objects through a two-tier parser:

1. **Keyword Parser** (`intent_parser.py`): rule-based extraction of mission type, area, constraints, timeline, and drone count. Always available, even without compute for LLM inference.

2. **Military LLM** (`llm/military_llm.py`): finetuned 3B-parameter model (Phi-3 or Llama-3.2) for richer understanding. Handles implicit constraints, doctrinal conventions, and ambiguity resolution.

### Acknowledgment Loop

STRIX never acts on ambiguous orders. Every parsed intent generates a structured confirmation:

```
Commander: "Recon north ridge with 4 drones, avoid SAM corridor"
STRIX:     "Understood: reconnaissance at designated area (radius 300m).
            Allocating 4 drones. Constraints: Avoid SAM engagement zone.
            Estimated time: 8 minutes. Current regime: PATROL. Confirm?"
Commander: "Confirmed"
```

### Design Rationale

The two-tier architecture ensures degraded-mode operation. When the LLM is unavailable (edge devices, comms-denied, resource-constrained), the keyword parser provides baseline capability. This mirrors how quantitative trading systems maintain simple rule-based fallbacks when their ML models fail.

---

## Layer 1: Market Brain

**Module**: `python/strix/brain.py`, `crates/strix-core/`

The central orchestrator. Runs a 10 Hz main loop that mirrors a quantitative trading engine:

```
predict → update → regime_check → auction → assign
```

### Particle Filter Navigation (GPS-Denied)

The core innovation borrowed from quantitative finance. The original 2D trading filter `[log_price, velocity]` is extended to 6D `[x, y, z, vx, vy, vz]` with regime-specific dynamics.

**State space**: Each drone is tracked by N particles (default: 1000), where each particle is a hypothesis about the drone's true 6D kinematic state.

**Prediction step**: particles are propagated through regime-specific dynamics:
- **PATROL**: mean-reverting velocity (alpha=0.5), low process noise
- **ENGAGE**: velocity tracks threat bearing (beta=0.3), medium noise
- **EVADE**: high-noise random walk, rapid direction changes

**Update step**: sensor observations (IMU, barometer, magnetometer, visual odometry, radio bearing) update particle weights via Gaussian likelihood.

**Resampling**: systematic resampling when ESS drops below threshold (default: 50% of N).

### Regime-Switching Markov Model

Three operating regimes with a 3x3 Markov transition matrix:

```
       PATROL  ENGAGE  EVADE
PATROL  0.90    0.07    0.03
ENGAGE  0.10    0.80    0.10
EVADE   0.15    0.10    0.75
```

Regime transitions are diagonal-dominant (regimes persist) but shift based on sensor evidence:
- Threat proximity triggers PATROL → ENGAGE
- Active fire or high attrition triggers ENGAGE → EVADE
- Threat clearance triggers EVADE → PATROL

### Adversarial Prediction

**Module**: `python/strix/adversarial.py`

The key innovation: a **dual particle filter** that maintains separate hypothesis clouds for enemy entities. Each enemy particle encodes a kinematic state plus an intent hypothesis:

- **DEFENDING**: mean-reverting velocity, stationary clustering
- **ATTACKING**: velocity vector toward friendly centroid
- **RETREATING**: velocity vector away from engagement area

This directly mirrors counterparty prediction in quantitative trading, where you model the likely actions of other market participants to gain informational advantage.

### Multi-Horizon Planning

**Module**: `python/strix/temporal/multi_horizon.py`

Three parallel planning horizons with cascade constraints:

| Horizon | dt | Particles | Lookahead | Purpose |
|---|---|---|---|---|
| H1 Tactical | 0.1s | 100 | 10s | Obstacle avoidance, collision prevention |
| H2 Operational | 5s | 500 | 5min | Formation, coordination, sensor coverage |
| H3 Strategic | 60s | 2000 | 1hr | Mission phasing, resource allocation |

**Top-down**: strategic decisions constrain operational plans, which constrain tactical manoeuvres.

**Bottom-up**: a tactical impossibility (terrain collision) vetoes the operational plan. An operational infeasibility (insufficient drones) flags the strategic plan for re-evaluation.

---

## Layer 2: Auction Floor

**Module**: `crates/strix-auction/`

Task allocation is a combinatorial auction where drones bid on tasks like traders bid on assets.

### Bidding Function

Each drone independently evaluates its fitness for each task:

```
score = urgency × 10 + capability × 3 + proximity × 5 + energy × 2 - risk × 4
```

Components:
- **Proximity**: inverse distance to task (1/d)
- **Capability**: fraction of required capabilities met [0, 1]
- **Energy**: remaining battery/fuel [0, 1]
- **Risk**: aggregated threat exposure at task location [0, 1]
- **Urgency**: task urgency × priority

### Sealed-Bid Protocol

Drones never see each other's bids. The Auctioneer resolves assignments using a modified Hungarian algorithm for optimal global allocation.

### Dark Pools

Sensitive tasks can be restricted to specific sub-swarms via `dark_pool` IDs. A strike task assigned to the attack sub-swarm is invisible to reconnaissance drones. This prevents information leakage and enables need-to-know compartmentalization.

### Kill Zone Adaptation

After a drone is lost, the loss location becomes a "kill zone" with elevated risk scoring. Subsequent bids automatically penalize tasks near kill zones. **The swarm learns from its losses.**

### Portfolio Optimization

The fleet is treated as a portfolio where each drone is an asset. The optimizer ensures:
- **Diversification**: capabilities are spread across the fleet, not concentrated
- **Correlation management**: drones assigned to adjacent tasks should have uncorrelated failure modes
- **Risk budgets**: no single task consumes more than a configurable fraction of fleet capability

---

## Layer 3: Mesh Coordination

**Module**: `crates/strix-mesh/`

Decentralized coordination without a central authority.

### Fractal Hierarchy

Self-similar command structure at every scale. A squad of 4 drones has the same organizational pattern as a platoon of 16 or a company of 64. Leadership is emergent via consensus, not assigned.

### Stigmergy (Digital Pheromones)

Bio-inspired coordination using digital pheromone fields:

| Pheromone Type | Meaning | Decay Rate |
|---|---|---|
| `explored` | Area has been surveyed | Fast (5%/s) |
| `danger` | Threat detected here | Slow (1%/s) |
| `interest` | Point of interest found | Medium (3%/s) |
| `relay` | Comm relay coverage | Fast (5%/s) |

Drones deposit pheromones as they operate. Other drones read the field to avoid redundant coverage and steer toward unexplored areas. Pheromone payload is ~20 bytes, making it bandwidth-efficient.

### Gossip Protocol

State synchronization via epidemic gossip with O(log N) convergence. Each gossip round, a drone shares its state with `fanout` randomly selected peers (default: 3).

### Bandwidth-Aware Prioritization

Messages are priority-queued:

| Priority | Message Type |
|---|---|
| 0 (highest) | ThreatAlert |
| 1 | TaskAssignment |
| 2 | StateUpdate |
| 3 | PheromoneDeposit |
| 4 (lowest) | Heartbeat |

When bandwidth is constrained, low-priority messages are dropped first.

---

## Layer 4: Puppet Master

**Module**: `crates/strix-adapters/`

Unified platform abstraction via the `PlatformAdapter` trait:

```rust
pub trait PlatformAdapter: Send + Sync {
    fn id(&self) -> u32;
    fn send_waypoint(&self, wp: &Waypoint) -> Result<(), AdapterError>;
    fn get_telemetry(&self) -> Result<Telemetry, AdapterError>;
    fn execute_action(&self, action: &Action) -> Result<(), AdapterError>;
    fn capabilities(&self) -> &Capabilities;
    fn is_connected(&self) -> bool;
    fn health_check(&self) -> Result<HealthStatus, AdapterError>;
}
```

Implementations:
- **MAVLink**: PX4 / ArduPilot autopilots via MAVLink v2
- **ROS2**: ground vehicles and naval craft via DDS
- **Simulator**: built-in physics sim for testing

The trait-based design means adding support for a new platform (e.g., a proprietary military autopilot) requires implementing one trait, with no changes to the rest of the system.

---

## Layer 5: Glass Box

**Module**: `crates/strix-xai/`

Every decision is explainable and auditable.

### Decision Traces

Each brain tick produces a `DecisionTrace` containing:
- Input state (fleet positions, threats, regime)
- Particle filter statistics (ESS, regime probabilities)
- Auction bids and scores
- Selected assignment and rationale
- Confidence score

### Narration

The military LLM converts decision traces into natural language:

> "Drone 7 was assigned to the northern waypoint because it had the highest energy (82%) and closest proximity (43m), despite moderate threat exposure (0.3). The kill zone penalty from the loss of Drone 2 reduced the attractiveness of the eastern approach by 40%."

### After-Action Replay

WorldSnapshot history enables full replay of any mission. The visualization layer can scrub through time, showing how decisions evolved and where things went right or wrong.

---

## Data Flow

```
Voice Command
    │
    ▼
┌──────────────┐     ┌──────────────┐
│ Intent Parser │────▶│  Ack Loop    │──── Commander confirms
└──────────────┘     └──────────────┘
    │                       │
    ▼                       ▼
┌──────────────────────────────────────┐
│           MISSION BRAIN              │
│  ┌────────┐  ┌────────┐  ┌────────┐ │
│  │Particle│  │ Regime │  │Adversar│ │
│  │ Filter │  │ Model  │  │  ial   │ │
│  └────────┘  └────────┘  └────────┘ │
│        │          │          │       │
│        ▼          ▼          ▼       │
│  ┌─────────────────────────────────┐ │
│  │      Multi-Horizon Planner     │ │
│  │   H3:strat → H2:ops → H1:tac  │ │
│  └─────────────────────────────────┘ │
└──────────────────────────────────────┘
    │
    ▼
┌──────────────┐     ┌──────────────┐
│   Auction    │────▶│    Mesh      │
│   Floor      │     │ Coordination │
└──────────────┘     └──────────────┘
    │                       │
    ▼                       ▼
┌──────────────────────────────────────┐
│          PUPPET MASTER               │
│  MAVLink / ROS2 / Simulator         │
└──────────────────────────────────────┘
    │
    ▼
Physical Drones / Digital Twin / Glass Box
```

---

## Design Principles

1. **Separation of concerns**: each layer has a single responsibility and communicates through well-defined interfaces.

2. **Graceful degradation**: every capability has a fallback. LLM unavailable? Keyword parser. GPS lost? Particle filter. Drones destroyed? Anti-fragile reallocation.

3. **No single point of failure**: the mesh is leaderless. Any drone can assume command. The fractal hierarchy means losing a leader degrades performance, not capability.

4. **Explainability by default**: every decision produces a trace. The Glass Box layer ensures that no autonomous action is opaque.

5. **Platform independence**: the Puppet Master layer isolates all platform-specific code behind a single trait. The brain never knows what hardware it's controlling.

6. **Performance where it matters**: the particle filter and auction engine run in Rust with `rayon` parallelism. The orchestration and NLP layers run in Python for flexibility. The FFI boundary (via PyO3) is designed for minimal overhead.
