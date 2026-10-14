# STRIX Reviewer Demo — `gps_denied_recon` (Corrected Software Replay)

> Simulation-only. Zero hardware flights. No RF/sensor fidelity. No field validation. Safety claims are limited to classical-CBF + ROE + traces + simulator; no trained-neural safety guarantee is claimed.

This is a **static, reviewer-facing** walkthrough of one deterministic software-only
scenario replay. It shows exactly how to reproduce a single worked scenario and what
the harness emits, so a reviewer can re-run it and obtain byte-identical output.

- **Correction date:** 2026-10-14. Replay metric production and its regression tests
  were corrected along with this walkthrough.
- **Evidence kind:** `software_replay`, `fidelity = deterministic_kinematic_public_replay`.
- **Withdrawn result:** the capture at `3a5d46de4ed69f4321c058ae55d84047e2ccad49`
  reported 100% area coverage, 4.5 m position RMS, 0.78 formation coherence, and
  PASS. Coverage was an active-agent ratio; RMS and coherence were fixed
  constants selected by scenario id. These are not measurements of those
  quantities and must not be cited as achieved results.
- **Corrected result:** the original three acceptance criteria are
  `not_observed`; the envelope is `failed` and the process exits `1`.
  The computed counters and energy below describe the Python kinematic model
  only. See [`../CAPABILITY_BOUNDARY.md`](../CAPABILITY_BOUNDARY.md).

The replay harness is a lightweight **kinematic** scenario player. It is **not** a
field-physics, RF, or hardware-fidelity simulator. Its purpose is repeatable,
inspectable public scenario behavior before any integration testing.

---

## 1. The worked scenario

`gps_denied_recon` — four aerial agents fly a reconnaissance pattern and transition to
the scripted `degraded_nav` mode when the GPS-loss event occurs at `t = 30 s`.
This is a mode flag and noisy kinematic movement, not a run of the Rust particle
filter or a measurement of GPS-denied navigation accuracy. The scenario's seed is
fixed at **42001** inside the scenario file itself
(`sim/scenarios/gps_denied_recon.yaml`), so the replay is deterministic by construction.

```
Scenario:  gps_denied_recon
Seed:      42001            (defined in the scenario YAML; the harness reads it)
Duration:  600 s           Tick: 10 s            Frames: 61            Agents: 4
```

---

## 2. Exact replay command

The harness takes the scenario file as input and reads `seed: 42001` from it (there is no
`--seed` flag — the seed is a scenario property). Run from the repository root:

```bash
python3 scripts/strix_sim_replay.py \
  --scenario sim/scenarios/gps_denied_recon.yaml \
  --output /tmp/a1_run.json \
  --no-html
```

`--no-html` writes only the JSON evidence. This scenario currently exits `1`
after writing the artifact because its required performance metrics are
unmeasured. Exit `0` requires every acceptance metric to be observed and within
the original bounds.

---

## 3. Determinism proof (double run, byte-identical)

Run the command twice into separate files, preserving both expected exit codes:

```bash
python3 scripts/strix_sim_replay.py --scenario sim/scenarios/gps_denied_recon.yaml --output /tmp/a1_run1.json --no-html
python3 scripts/strix_sim_replay.py --scenario sim/scenarios/gps_denied_recon.yaml --output /tmp/a1_run2.json --no-html
cmp -s /tmp/a1_run1.json /tmp/a1_run2.json && echo "byte-identical"
```

Within a single checkout, the two runs are **byte-for-byte identical to each other** —
`cmp -s` reports no difference. Both replay commands still exit `1`. Determinism
does not make an unmeasured acceptance criterion pass.

> The whole-file `sha256sum` is **intentionally not pinned** in this doc. The replay
> JSON embeds a `repo` block (`commit`, `branch`, `working_tree_clean`) that reflects
> *your* git context, not the simulation, so the same seeded run yields a *different*
> whole-file hash per checkout — any concrete value here would be a checkout-specific
> reviewer trap.
>
> What **does** reproduce deterministically on any checkout is the seed-driven
> simulation content — the `scenario`, `metrics`, and `envelope` blocks (Section 4).
> Verify reproducibility by comparing **those blocks**, not a whole-file SHA.

---

## 4. Captured output excerpt

A captured excerpt of the harness output for this run is stored alongside this doc at
[`replay_output.txt`](replay_output.txt). The load-bearing parts are reproduced below.

### 4.1 Scenario header — `replay["scenario"]`

```json
{
  "config_hash": "829cc17c46b98d72",
  "duration_s": 600.0,
  "id": "gps_denied_recon",
  "name": "GPS-Denied Reconnaissance",
  "path": "sim/scenarios/gps_denied_recon.yaml",
  "seed": 42001,
  "tick_s": 10.0
}
```

### 4.2 Metrics — `replay["metrics"]`

```json
{
  "active_agents": 4,
  "frame_count": 61,
  "mean_energy_remaining_pct": 66.111,
  "offline_agents": 0
}
```

These are computed replay diagnostics for seed 42001 and tick 10 s. Active
agents do not measure spatial coverage. The energy value is the model's
endurance-based accounting, not battery telemetry. Unsupported metrics are
omitted from `metrics`; `metric_evidence` records why each cannot be observed.

### 4.3 Pass-envelope evaluation — `replay["envelope"]`

| Required metric | Unchanged acceptance bounds | Corrected result | Missing measurement |
|---|---|---|---|
| `area_coverage_pct` | 80–100% | `not_observed` | Sensor footprint / spatial coverage accumulation |
| `position_error_rms_m` | 0–10 m | `not_observed` | Estimated and reference positions to compare |
| `formation_coherence` | 0.6–1.0 | `not_observed` | Defined, instrumented formation-quality calculation |

Overall `envelope.status = "failed"`; CLI exit code **1**. The full checks and
their reasons are captured in [`replay_output.txt`](replay_output.txt). This
means the replay cannot determine whether those performance targets were met;
it does not supply measured values below the target. The matrix entry still
requires exit `0` and therefore also fails.

### 4.4 Timeline excerpt — degraded-nav transition at GPS loss

The per-agent `mode` field flips from `nominal` to `degraded_nav` exactly at the
scheduled `t = 30 s` `gps_loss` event, and scenario events surface at their scheduled
times. This establishes scheduled event playback and the Python mode transition.

```
  t_s  agent_1.mode      agent_1 (x, y, z)               events
-----  ----------------  ------------------------------  ------------------------
    0  nominal           (    0.00,   200.00,  -50.00)   -
   10  nominal           (  -51.89,   218.94,  -50.00)   -
   20  nominal           ( -100.98,   201.07,  -50.00)   -
   30  degraded_nav      ( -144.39,   171.75,  -50.00)   30.0s:gps_loss
   40  degraded_nav      ( -180.79,   134.60,  -50.00)   -
  120  degraded_nav      (  -76.23,  -212.21,  -50.00)   120.0s:wind_gust
  300  degraded_nav      ( -144.40,   172.08,  -50.00)   300.0s:sensor_degradation
  600  degraded_nav      ( -222.22,    39.13,  -50.00)   -
```

This scenario defines no threat constraints, so `min_constraint_clearance_m`
is **not observed** and is omitted, rather than reported as zero. This Python
player does not invoke the Rust orchestrator, classical CBF, ROE, or particle
filter. Those need separate Rust evidence; this capture establishes none of
their performance or safety properties.

---

## 5. Diagrams (labels only)

### 5.1 Replay data flow

```mermaid
flowchart LR
    Y["gps_denied_recon.yaml<br/>seed=42001"] --> R["strix_sim_replay.py<br/>kinematic replay"]
    R --> J["replay JSON<br/>scenario / metrics / envelope / frames"]
    J --> E["pass-envelope check"]
    J --> T["timeline / modes"]
```

### 5.2 Navigation-mode state (this scenario)

```
   t < 30s              t >= 30s (gps_loss)
 +----------+          +---------------+
 | nominal  | -------> | degraded_nav  |
 +----------+          +---------------+
```

---

## 6. Reproduce checklist

1. Check out the commit containing this correction, with a clean working tree;
   record its hash. The withdrawn historical commit reproduces the defect.
2. From the repo root, run the command in Section 2.
3. Run it a second time to a different output path.
4. `cmp -s` the two outputs — within your checkout they are byte-identical to each
   other. (The whole-file `sha256sum` is checkout-specific and intentionally not
   pinned — see Section 3.)
5. Confirm exit `1`, three `not_observed` acceptance checks, and overall `failed`.
   Compare the `scenario` / `metrics` / `metric_evidence` / `envelope` blocks to
   the corrected capture. A Python regression-suite PASS verifies this honest
   failure behavior; it is not a GPS-denied scenario PASS.

---

## 7. Boundary reminder

- Software replay only. No hardware flights, no RF/sensor fidelity, no field validation.
- The historical fixed / proxy KPI values and their PASS are withdrawn.
- Current counters / energy are kinematic diagnostics; the three performance
  acceptance metrics remain unmeasured. Re-run on the exact submission commit.
- No fielded deployment, no delivered ROS2/MAVLink hardware integration, no defence
  validation/accreditation, no trained-neural safety guarantee is claimed.
- Authoritative claim map: [`../CAPABILITY_BOUNDARY.md`](../CAPABILITY_BOUNDARY.md).
