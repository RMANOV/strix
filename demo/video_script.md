# STRIX Demo Video Script

## Duration: 4 minutes

## Target Audience
Military R&D evaluators, defense program managers, venture capital investors with defense portfolios.

---

## 0:00 - 0:15 | Opening Hook

**Visual**: Black screen. Sound of wind. A single drone propeller spins up.

**Text on screen** (white on black, typewriter effect):

> *"The most dangerous opponent is the one who can predict your next move."*

**Cut to**: 3D visualization of a particle cloud coalescing into a drone swarm formation. The particles shift from red (uncertainty) to green (converged).

**Voiceover**: "Every competitor in autonomous drone operations solves the same problem the same way. STRIX solves it the way Wall Street does."

---

## 0:15 - 0:45 | The Thesis

**Visual**: Split screen. Left: stock market trading floor with order books flashing. Right: drone swarm in formation over terrain.

**Voiceover**: "Quantitative trading firms have spent decades solving exactly the problems drone swarms face: decentralized decision-making under uncertainty, adversarial prediction, resource allocation with constraints, and graceful degradation under stress."

**Visual**: Trading algorithms morph into their drone equivalents:
- Particle filter price tracking → GPS-denied navigation
- Portfolio optimization → Fleet diversification
- Counterparty modeling → Enemy prediction

**Text overlay**: "The battlefield is a market. Drones are traders. The enemy is a counterparty."

---

## 0:45 - 1:30 | GPS-Denied Navigation

**Visual**: 3D view of 4 drones in formation. GPS indicator shows "CONNECTED" in green.

**Voiceover**: "Scenario one: GPS-denied reconnaissance."

**Visual**: GPS indicator turns red: "DENIED". Standard drones would drift and fail. STRIX drones show particle clouds forming around each drone -- thousands of translucent dots representing position hypotheses.

**Voiceover**: "When GPS is jammed, STRIX does not stop. Each drone runs a 6-dimensional particle filter -- the same mathematics that quantitative trading firms use to track hidden asset prices from noisy market data. One thousand particles per drone. Six dimensions. Ten hertz."

**Visual**: Particle clouds tighten as sensor fusion incorporates IMU, barometer, magnetometer, and visual odometry. The drones continue their reconnaissance pattern. Coverage percentage ticks upward: 20%... 40%... 60%... 80%.

**Visual**: Position error graph shows sub-10-meter accuracy without GPS. A comparison bar shows competitor drift at 50m+.

**Text overlay**: "80% area coverage. Sub-10m accuracy. Zero GPS."

---

## 1:30 - 2:15 | Anti-Fragile Response

**Visual**: 12-drone patrol formation. Dashboard shows all drones green.

**Voiceover**: "Scenario two: mass attrition."

**Visual**: Drone 2 is hit. Its icon turns red. A kill zone circle appears at the loss location. The auction dashboard shows bids being recalculated -- scores near the kill zone drop.

**Voiceover**: "Drone 2 is destroyed by a SAM. Every competitor's swarm degrades. STRIX's swarm gets stronger."

**Visual**: Drone 7 approaches the same area. Its planned path shows a yellow curve rerouting AWAY from the kill zone.

**Voiceover**: "The loss location becomes a kill zone. Future bids automatically penalize tasks near kill zones. Drone 7 never makes the same mistake."

**Visual**: Attrition counter: 1 lost... 2 lost... 3 lost. At 30%, the regime indicator shifts from green (PATROL) to red (EVADE). Surviving drones scatter into evasive patterns.

**Visual**: Per-drone effectiveness graph. Despite losing half the fleet, the line goes UP. Each surviving drone covers more area, more efficiently, with better threat awareness.

**Text overlay**: "Anti-fragile. The swarm learns from every loss."

---

## 2:15 - 3:00 | Adversarial Prediction

**Visual**: Enemy vehicle appears on the 3D display. A cloud of red particles forms around it -- the adversarial particle filter.

**Voiceover**: "Scenario three: adversarial prediction. The innovation no competitor has."

**Visual**: The red particle cloud splits into three colored clusters:
- Blue: DEFENDING hypothesis (stationary particles)
- Red: ATTACKING hypothesis (particles moving toward friendly fleet)
- Yellow: RETREATING hypothesis (particles moving away)

**Voiceover**: "STRIX maintains a dual particle filter. One for friendly drones. One for the enemy. Each enemy particle is a hypothesis about their intent. Is the enemy defending? Attacking? Retreating?"

**Visual**: The ATTACKING cluster grows dominant. A countdown timer appears: "Time to contact: 47 seconds."

**Voiceover**: "Before the enemy completes their maneuver, STRIX has already classified their intent and calculated time to contact. This is not reaction. This is prediction."

**Visual**: The swarm preemptively repositions into an engagement formation 15 seconds before the enemy reaches engagement range. The commander's display shows: "Predicted: ATTACKING. Confidence: 87%. Repositioning."

**Text overlay**: "Predict. Preempt. Prevail."

---

## 3:00 - 3:30 | Glass Box Explainability

**Visual**: Commander's tablet interface. A natural language explanation appears:

> "Drone 7 assigned to northern overwatch. Reason: highest energy (82%), closest proximity (43m), acceptable threat exposure (0.3). Kill zone from Drone 2 loss reduced eastern approach score by 40%."

**Voiceover**: "Every decision STRIX makes is explainable. Not after the fact. In real time. The commander sees not just what the system decided, but why. No black boxes. No trust-me AI."

**Visual**: After-action replay scrubber. The commander drags a timeline slider, watching the entire mission unfold with decision annotations at every point.

**Text overlay**: "Glass Box Autonomy. Every decision, explained."

---

## 3:30 - 3:50 | Technical Summary

**Visual**: Architecture diagram animating layer by layer:

```
Human Interface → Market Brain → Auction Floor → Mesh → Puppet Master → Glass Box
```

**Voiceover**: "Six layers. Rust core for performance. Python orchestration for flexibility. Open-source mathematics. Platform agnostic. Edge deployable."

**Visual**: Tech badges fly in: Rust, Python, PyO3, MAVLink, ROS2, rerun.io

**Text overlay**: "Apache 2.0 open core. ITAR-clean published mathematics."

---

## 3:50 - 4:00 | Closing

**Visual**: The STRIX owl logo materializes from a particle cloud.

**Text on screen**:

> **STRIX**
>
> *The Battlefield is a Market.*

**Voiceover**: "STRIX. Because the most dangerous swarm is the one that thinks like a trading firm."

**Visual**: Contact information. Website URL. QR code.

---

## Production Notes

- All visuals rendered from rerun.io / WebGL 3D engine using actual simulation data
- Particle clouds use real particle filter output, not artist animations
- Kill zone adaptation uses real auction bid scores
- Adversarial prediction uses real dual particle filter output
- After-action replay uses real WorldSnapshot history
- No mock-ups, no concept art -- everything shown is running code
- Target resolution: 4K, 60fps for particle cloud smoothness
- Audio: subtle ambient drone hum, no music (military audience prefers clean audio)
- Narration: professional male voice, measured pace, no hype language
