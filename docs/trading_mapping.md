# Trading-to-Warfare Algorithm Mapping

## The Core Insight

Every algorithm in STRIX was originally developed for quantitative finance. The mapping is not a metaphor -- it is a structural isomorphism. The mathematical foundations are identical; only the semantics change.

This document provides the complete mapping table with technical detail on each transformation.

---

## Complete Mapping Table

| # | Trading Algorithm | STRIX Application | State Space | Key Transformation |
|---|---|---|---|---|
| 1 | Particle Filter (2D) | GPS-Denied Navigation (6D) | `[log_price, velocity]` → `[x,y,z,vx,vy,vz]` | Extend from 2 to 6 dimensions; add multi-sensor fusion |
| 2 | Regime-Switching HMM | Tactical Regime Detection | `{RANGE, TREND, PANIC}` → `{PATROL, ENGAGE, EVADE}` | Transition matrix governs operating mode shifts |
| 3 | Combinatorial Auction | Task Allocation | `bid = f(urgency, alpha, risk)` → `bid = f(urgency, capability, proximity, energy, risk)` | Replace financial alpha with military scoring components |
| 4 | Dark Pool Execution | Covert Sub-Swarm Ops | Hidden order routing → dark_pool sub-swarm filtering | Need-to-know compartmentalization of sensitive tasks |
| 5 | Portfolio Optimization | Fleet Diversification | Markowitz mean-variance → capability/failure-mode decorrelation | Maximize expected mission value while minimizing correlated failure |
| 6 | Value at Risk (VaR) | Attrition Threshold | Portfolio drawdown limit → maximum acceptable drone losses | Trigger regime shift when VaR threshold is breached |
| 7 | Anti-Fragile Strategy | Kill Zone Adaptation | Convex response to volatility → learn from losses | Each loss makes the swarm stronger through spatial memory |
| 8 | Order Book Imbalance | Threat-Bearing Signal | `bid_vol - ask_vol` → threat bearing vector | Enemy position delta drives velocity adjustment in ENGAGE regime |
| 9 | Market Microstructure | Pheromone Stigmergy | Limit order book → digital pheromone grid | Spatial coordination via decaying chemical traces |
| 10 | Counterparty Prediction | Adversarial Particle Filter | Model other traders → model enemy intent | Dual particle filter predicts enemy behaviour before completion |
| 11 | HFT Latency Optimization | Mesh Gossip Protocol | Co-location, FPGA → O(log N) gossip convergence | Bandwidth-aware prioritization replaces speed-of-light racing |
| 12 | Multi-Strategy Portfolio | Multi-Horizon Planning | Intraday + swing + position → tactical + operational + strategic | Three parallel planning horizons with cascade constraints |
| 13 | Risk Parity | Capability Balancing | Equal risk contribution per asset → equal failure-mode coverage | No single drone type dominates the risk budget |
| 14 | Execution Algorithm (TWAP/VWAP) | Formation Maneuver | Time-weighted execution → smooth coordinated movement | Minimize signature while completing formation transitions |

---

## Detailed Explanations

### 1. Particle Filter: Price Discovery → Position Discovery

**Trading**: A particle filter tracks the hidden "true price" of an asset from noisy market data (bid/ask, trades, order flow). Each particle is a hypothesis about `[log_price, velocity]`. The filter predicts price movement, incorporates new observations, and resamples.

**STRIX**: The same filter tracks the hidden "true position" of a drone from noisy sensor data (IMU, barometer, magnetometer, visual odometry, radio bearing). Each particle is a hypothesis about `[x, y, z, vx, vy, vz]`. The mathematics are identical: predict, update weights via Gaussian likelihood, resample when ESS drops.

**What changes**: dimensionality (2 → 6), observation model (market data → sensor data), process noise profile (volatility regimes → movement regimes).

### 2. Regime-Switching: Market Regimes → Tactical Regimes

**Trading**: Markets exhibit distinct regimes -- ranging (mean-reverting), trending (momentum), and panic (high volatility, correlation breakdown). A Hidden Markov Model detects transitions using a Markov transition matrix.

**STRIX**: The battlespace exhibits identical regime structure:
- **PATROL** (= RANGE): mean-reverting velocity, hold position, low noise
- **ENGAGE** (= TREND): velocity tracks threat bearing, directional bias
- **EVADE** (= PANIC): high noise, rapid direction changes, survival mode

The transition matrix is structurally identical. Diagonal dominance ensures regime persistence. Off-diagonal elements model the probability of tactical transitions.

### 3. Combinatorial Auction: Asset Allocation → Task Allocation

**Trading**: In a call auction, market participants submit sealed bids for assets. The exchange runs a combinatorial optimization to find the allocation that maximizes total value.

**STRIX**: Drones submit sealed bids for tasks. Each bid scores the drone's fitness: `urgency×10 + capability×3 + proximity×5 + energy×2 - risk×4`. The Auctioneer runs a modified Hungarian algorithm for optimal global assignment. Drones never see each other's bids -- just as traders never see each other's limit orders in a dark pool.

### 4. Dark Pool: Hidden Orders → Covert Operations

**Trading**: Dark pools are private exchanges where large orders are executed without revealing intent to the broader market. This prevents information leakage and front-running.

**STRIX**: Sensitive tasks (strike missions, EW operations) can be restricted to specific sub-swarms via `dark_pool` IDs. Reconnaissance drones in sub-swarm A cannot see or bid on strike tasks assigned to sub-swarm B. This enforces need-to-know at the algorithmic level.

### 5. Portfolio Optimization: Diversification → Fleet Resilience

**Trading**: Markowitz mean-variance optimization selects asset weights to maximize expected return while minimizing portfolio variance. The key insight: correlation between assets matters as much as individual risk.

**STRIX**: The fleet is a portfolio of capabilities. The optimizer ensures that sensor, weapon, EW, and relay capabilities are distributed across the fleet rather than concentrated. Drones assigned to adjacent tasks should have uncorrelated failure modes (different altitudes, different approach vectors). Losing one drone should not eliminate an entire capability class.

### 6. Value at Risk: Drawdown Protection → Attrition Protection

**Trading**: VaR sets a maximum acceptable loss threshold. When portfolio drawdown exceeds VaR, risk management triggers position reduction or liquidation.

**STRIX**: An AttritionMonitor tracks cumulative drone losses. When attrition exceeds the maximum drawdown threshold (default: 30%), the system forces a regime shift to EVADE and begins managed withdrawal. This prevents catastrophic total-loss scenarios.

### 7. Anti-Fragile Strategy: Volatility Harvesting → Loss Learning

**Trading**: Taleb's anti-fragile strategies are designed to benefit from volatility. A convex payoff profile means that large market moves generate outsized returns.

**STRIX**: After losing a drone, the swarm does not simply degrade. It adapts:
1. The loss location is marked as a kill zone (spatial memory)
2. Future bids penalize tasks near kill zones (risk repricing)
3. The attrition analysis identifies the failure mode (cause classification)
4. Surviving drones adjust their approach vectors (behavioral adaptation)

**The swarm gets stronger after losses**, because each loss provides information that improves future decisions.

### 8. Order Book Imbalance: Price Pressure → Threat Pressure

**Trading**: Order book imbalance (excess bid volume vs. ask volume) predicts short-term price movement. An imbalance toward bids signals upward pressure.

**STRIX**: The threat bearing vector is the drone equivalent of order book imbalance. When a threat is detected at bearing 045, the `threat_bearing` signal drives ENGAGE-regime particles to adjust velocity toward that direction, exactly as `imbalance > 0` drives price particles upward.

### 9. Market Microstructure: Order Book → Pheromone Grid

**Trading**: The limit order book is a spatial data structure where agents deposit orders (buy/sell at price X). Orders interact through matching rules and decay through cancellation.

**STRIX**: The pheromone grid is a spatial data structure where drones deposit digital pheromones. Pheromones interact through superposition and decay exponentially over time. Just as traders read the order book to decide where to place their next order, drones read the pheromone field to decide where to fly next.

### 10. Counterparty Prediction: Model Traders → Model Enemy

**Trading**: Sophisticated trading firms model the behavior of other market participants (institutional flow, retail sentiment, market maker positioning) to predict future price movements.

**STRIX**: The adversarial particle filter models enemy intent using the same mathematics. Each enemy particle is a hypothesis about `[position, velocity, intent_regime]`. The regime model classifies enemy behavior as DEFENDING, ATTACKING, or RETREATING. Time-to-contact prediction enables preemptive tactical decisions.

### 11. HFT Latency: Speed → Convergence

**Trading**: High-frequency trading optimizes for latency -- co-located servers, FPGA accelerators, microwave links. The goal is to process information faster than competitors.

**STRIX**: The mesh gossip protocol optimizes for convergence speed under bandwidth constraints. O(log N) convergence with fanout=3 means that state changes propagate to the entire swarm in approximately log2(N) gossip rounds. Priority queuing ensures that threat alerts propagate faster than heartbeats.

### 12. Multi-Strategy: Time Horizons → Planning Horizons

**Trading**: Quantitative firms run multiple strategies simultaneously across different time horizons: intraday (seconds-minutes), swing (hours-days), position (weeks-months). Higher horizons set risk budgets for lower horizons.

**STRIX**: Three planning horizons cascade top-down:
- Strategic (60s dt): sets mission phases and resource budgets
- Operational (5s dt): manages formation and coordination within strategic constraints
- Tactical (0.1s dt): avoids obstacles within operational constraints

Bottom-up feedback: a tactical impossibility vetoes the operational plan.

### 13. Risk Parity: Equal Risk → Equal Capability

**Trading**: Risk parity allocates capital so that each asset contributes equally to portfolio risk, rather than equally to portfolio weight.

**STRIX**: Capability balancing ensures that no single drone type dominates the risk budget. If sensor drones are overrepresented and weapon drones are scarce, the optimizer flags the imbalance and adjusts task allocation to protect the scarce capability.

### 14. Execution Algorithms: TWAP/VWAP → Formation Maneuver

**Trading**: Time-weighted average price (TWAP) and volume-weighted average price (VWAP) algorithms execute large orders gradually to minimize market impact.

**STRIX**: Formation transitions are executed gradually to minimize electromagnetic signature and acoustic signature. Rather than all drones snapping to new positions simultaneously (high signature), they transition smoothly over a configurable time window -- the military equivalent of "don't move the market."

---

## Why This Mapping Works

The structural isomorphism exists because both domains share fundamental properties:

1. **Uncertainty**: both operate under incomplete information
2. **Adversarial dynamics**: both face intelligent opponents
3. **Resource constraints**: both manage finite assets with opportunity costs
4. **Time pressure**: both require decisions faster than the opponent
5. **Catastrophic risk**: both face scenarios where total loss is possible
6. **Decentralized execution**: both distribute decision-making across many agents

The mathematics that solves these problems does not care whether the agent is a trading algorithm or a drone. STRIX exploits this insight.
