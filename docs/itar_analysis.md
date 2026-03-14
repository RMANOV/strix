# ITAR / EAR Legal Strategy

## Executive Summary

STRIX is designed from the ground up with a clean separation between **published mathematics** (ITAR-exempt, open-source) and **military-specific integration** (controlled, proprietary). This document analyzes the regulatory framework and establishes the legal basis for the open-source core.

---

## Regulatory Framework

### ITAR (International Traffic in Arms Regulations)

- Administered by the Directorate of Defense Trade Controls (DDTC)
- Governs the United States Munitions List (USML)
- Category VIII: Aircraft and Related Articles
- Category XI: Military Electronics

### EAR (Export Administration Regulations)

- Administered by the Bureau of Industry and Security (BIS)
- Governs the Commerce Control List (CCL)
- Category 7: Navigation and Avionics
- Category 9: Propulsion Systems

---

## The Published Information Exemption

### EAR section 734.7(a) -- Published Information

> "Published" means information that has been made available to the public without restrictions upon its further dissemination.

Software and technical data that are **published** (i.e., made freely available with no access restrictions) are excluded from the EAR. This is the legal basis for open-source defense-adjacent software.

### ITAR section 120.34 -- Public Domain

> The term "public domain" means information which is published and which is generally accessible or available to the public [...] through unlimited distribution at a conference, meeting, seminar, trade show, or exhibition, generally accessible to the interested public.

Technical data in the public domain is not subject to ITAR controls.

---

## What Is Apache 2.0 (Open Source)

The following components are published under Apache 2.0 and constitute published information / public domain:

### 1. Particle Filter Mathematics

- 6D state prediction: `predict_particles_6d()`
- Gaussian likelihood measurement update: `update_weights_6d()`
- Systematic resampling: `systematic_resample_6d()`
- Effective sample size computation

**Legal basis**: particle filters are a well-known statistical technique published in thousands of academic papers. The 6D extension is a straightforward generalization. No classified or controlled information is embedded.

### 2. Regime-Switching Hidden Markov Model

- Markov transition matrices
- Regime classification from observation evidence
- Regime-specific dynamics parameters

**Legal basis**: HMMs are standard probabilistic models taught in every graduate statistics program.

### 3. Combinatorial Auction Algorithm

- Sealed-bid scoring function
- Hungarian algorithm for assignment
- Portfolio optimization (Markowitz)

**Legal basis**: auction theory and portfolio optimization are published economics and finance research.

### 4. Mesh Networking Protocols

- Gossip protocol for state synchronization
- Pheromone-based stigmergy
- Fractal hierarchy organization
- Bandwidth-aware message prioritization

**Legal basis**: gossip protocols, stigmergy, and mesh networking are published computer science research with extensive civilian applications.

### 5. Platform Adapter Trait

- Generic `PlatformAdapter` interface
- Waypoint, Telemetry, Action types
- Health monitoring abstractions

**Legal basis**: abstract interfaces define no controlled technology. They are structural contracts, not implementations.

### 6. NLP / Intent Parsing

- Keyword-based command parser
- Acknowledgment loop logic

**Legal basis**: natural language processing is a civilian technology with no controlled applications in this context.

### 7. Explainability Engine

- Decision trace recording
- After-action replay
- Narrative generation

**Legal basis**: explainability and audit logging are standard software engineering practices.

---

## What Is Proprietary (Controlled)

The following are NOT included in the open-source release and would require export licensing:

### 1. Weapon Adapter Implementations

- Specific weapon system integration code
- Targeting algorithms tied to specific munitions
- Weapon release authorization logic
- Kill chain automation

**Control basis**: USML Category IV (Launch Vehicles, Guided Missiles, Ballistic Missiles, Rockets, Torpedoes, Bombs, and Mines) or Category VIII.

### 2. Classified Scenario Configurations

- Real-world terrain data above NGA classification thresholds
- Threat databases with classified performance parameters
- Electronic warfare vulnerability databases
- Specific adversary force modeling parameters

**Control basis**: classification by originating agency.

### 3. Specific Military Platform Adapters

- Adapters for classified autopilot systems
- Integration with classified communication systems (Link 16, JREAP, etc.)
- Adapters for specific military drone platforms not in the public domain

**Control basis**: USML Category XI or CCL Category 7.

### 4. Military LLM Training Data

- Finetuning datasets derived from classified doctrine
- After-action reports with classified content
- Performance parameters of classified systems

**Control basis**: classification by originating agency.

### 5. Performance Benchmarks

- Benchmarks that reveal the performance capabilities of the system in classified scenarios
- Timing data that could inform adversary countermeasures

**Control basis**: derivative classification from scenario data.

---

## Precedents

Several prominent open-source projects operate successfully under the same legal framework:

### PX4 Autopilot (BSD 3-Clause)

- Full autopilot software stack for drones
- Used by both civilian and military operators
- Published as open source without ITAR restriction
- Military-specific integrations remain proprietary

### ArduPilot (GPL v3)

- Complete flight controller firmware
- Powers military drones in multiple NATO countries
- Published as open source
- No ITAR restrictions on the published code

### ROS2 (Apache 2.0)

- Robot Operating System 2
- Used extensively in military robotics programs
- Open-source middleware and DDS communication
- Military extensions are proprietary

### MAVLink (MIT)

- Lightweight messaging protocol for drones
- Standardized by multiple defense organizations
- Published as open source
- The protocol itself is not controlled; specific implementations may be

### OpenCV (Apache 2.0)

- Computer vision library
- Used in military targeting and surveillance systems
- Published as open source without restriction
- Specific military applications are proprietary

### Common Thread

In every precedent, the **published mathematical algorithms and software infrastructure** are open source. The **military-specific integration** (specific weapon systems, classified data, classified platforms) is proprietary and controlled.

STRIX follows this exact pattern.

---

## Architectural Enforcement

The codebase architecture physically separates controlled from uncontrolled:

```
strix/                          # Apache 2.0 -- all published
├── crates/
│   ├── strix-core/             # Particle filter, regime model
│   ├── strix-auction/          # Auction, portfolio, risk
│   ├── strix-mesh/             # Mesh coordination
│   ├── strix-adapters/         # Adapter trait + simulator
│   └── strix-xai/              # Explainability
├── python/strix/               # Orchestration, NLP, planning
└── sim/scenarios/              # Unclassified test scenarios

strix-military/                 # PROPRIETARY -- not published
├── adapters/                   # Classified platform adapters
├── weapons/                    # Weapon integration
├── scenarios/                  # Classified scenarios
└── training/                   # Classified LLM training data
```

The `strix-military` repository is a separate, private repository that depends on the published `strix` core. It can never contaminate the open-source release because it exists in a completely separate repository with independent access controls.

---

## Recommendations

1. **Publish early, publish often**: the earlier the core algorithms are published, the stronger the public domain claim.

2. **No classified data in open-source repo**: enforce this with CI checks that scan for known classified marking patterns (FOUO, SECRET, etc.).

3. **Clean API boundary**: the `PlatformAdapter` trait is the moat. Everything above the trait is uncontrolled. Everything below it (specific military implementations) is controlled.

4. **Seek a Commodity Jurisdiction (CJ) determination**: file a CJ request with DDTC to confirm that the published core is not ITAR-controlled. This provides a formal written determination.

5. **EAR classification request**: file a CCATS (Commerce Classification Automated Tracking System) request with BIS to get a formal ECN (Export Control Classification Number) for the published core. Target: EAR99 (no restrictions).

6. **Legal counsel**: engage ITAR-specialized counsel before any engagement with DoD or foreign military customers.
