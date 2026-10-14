# Public Evidence Harness

STRIX public validation should produce repeatable evidence, not one-off terminal
claims. The public test matrix is a small software-only harness for running
approved checks and recording the exact commit, branch, selected tags, command
results, and report timestamps.

This is an upstream-maintainer advantage: downstream forks can still run or
modify the code, but they do not inherit official release provenance, maintained
test matrices, or the project-specific validation discipline unless they keep
that evidence layer current themselves.

## Scope

The public matrix is intentionally conservative. It covers:

- public surface hygiene;
- release manifest generation;
- harness self-tests;
- scenario contract validation;
- software-only scenario replay generation;
- public scenario envelope checks;
- Python regression tests;
- targeted Rust contract tests.

Program-specific scenarios, customer data, internal review ledgers, and
non-public benchmark data do not belong in this public matrix.

The evidence produced here is **software-only** scenario, replay, and test
evidence for the auditable OODA path. It is explicitly **not** evidence of:

- hardware integration or on-hardware / edge-drone deployment;
- RF or sensor fidelity, or field readiness;
- a default-runtime or trained-neural GCBF+ safety guarantee (the shipped safety
  story is classical-CBF + ROE + traces + simulator);
- delivered ROS2 / MAVLink deployment (these remain adapter-boundary,
  validation-ahead surfaces); or
- edge-LLM autonomy (the Python LLM / edge-inference path is optional /
  degraded-mode support, not the core autonomy runtime).

For the full claim map, see
[../CAPABILITY_BOUNDARY.md](../CAPABILITY_BOUNDARY.md).

## Usage

List the available checks:

```bash
python scripts/strix_test_matrix.py --list
```

Run the fast smoke layer:

```bash
python scripts/strix_test_matrix.py --select smoke
```

Run all non-manual checks and write an explicit report path:

```bash
python scripts/strix_test_matrix.py \
  --matrix Project_Docs/testing/public_test_matrix.json \
  --output target/strix-test-reports/public.json
```

Each run writes both JSON and Markdown reports. The default output directory is
under `target/`, so generated evidence stays out of source control unless a
maintainer intentionally promotes a report into release notes.

Generate a deterministic scenario replay plus a browser-viewable HTML canvas:

```bash
python scripts/strix_sim_replay.py \
  --scenario sim/scenarios/gps_denied_recon.yaml \
  --output target/strix-replays/gps_denied_recon.json \
  --html target/strix-replays/gps_denied_recon.html
```

The replay harness is intentionally described as a deterministic kinematic
public replay. It is useful for visual inspection, regression evidence, seeded
event playback, and pre-field behavior review. It is not a hardware, RF,
sensor-fidelity, or field-readiness simulator.

### Replay metric correction — 2026-10-14

The former `gps_denied_recon` PASS is withdrawn. Its position-error RMS and
formation coherence were fixed constants; its “area coverage” was the fraction
of active agents. None measured the named quantity. Analogous fixed avoidance,
recovery, relay, response-time, and formation metrics in the other two scenario
families are also withdrawn.

The corrected player reports only computed replay diagnostics. Missing
acceptance metrics produce `not_observed`, `envelope.status = "failed"`, and
process exit `1`, after writing JSON / HTML. Scenario targets are unchanged.
The matrix entry `software_replay_gps_denied` still requires exit `0`, so it
correctly reports **failed** while these requirements are unmeasured. A passing
Python regression test that verifies this refusal is not a passing scenario.
See [the corrected capture](../demo/replay_output.txt).

Corrected runs of all three public scenarios, with their declared seeds and
`--tick-s 10` (2026-10-14), produced the following **model diagnostics**:

| Scenario | Active / offline agents | Frames | Mean model energy remaining | Minimum sampled constraint clearance | Acceptance result |
|---|---|---|---|---|---|
| `gps_denied_recon` | 4 / 0 | 61 | 66.111% | Not observed — no constraints | Failed; all 3 required metrics `not_observed`; exit 1 |
| `mass_attrition` | 6 / 6 | 91 | 46.1% | −290.985 m | Failed; all 3 required metrics `not_observed`; exit 1 |
| `multi_domain` | 5 / 0 | 73 | 62.626% | −88.155 m | Failed; all 3 required metrics `not_observed`; exit 1 |

Loss counts follow scripted events. Negative sampled clearances are preserved
as reported diagnostics, not converted to an avoidance-success claim. No
physical survival, avoidance-success rate, formation-recovery time, relay
availability, or threat-response latency is established by these runs.

Verification of this correction used `/usr/bin/python3` 3.14.7, pytest 9.0.2,
NumPy 2.3.5, and PyYAML 6.0.3. The complete Python suite changed from
**153 passed** at parent commit `b883500e8be932bd226960278f15e7aaa62a2e90`
to **175 passed**, including **30 replay tests**. New regressions failed
against the old producer before the fix. The selected `replay` matrix check
returned exit `1` / **failed**, as required while the three metrics are missing.
These are dated local Python verification results, not a new Rust, native FFI,
hardware, or scenario-performance acceptance receipt.

## Next Capabilities

The next useful expansion is scenario-family batch replay: every public scenario
already declares a seed, metric set, and `pass_envelope`; the next step is to
run every scenario through replay and compare observed metrics against that
envelope. After that, add statistical Monte Carlo sweeps, richer trace exports,
and integration checks for criticality, contagion, and quorum-style
confirmation loops.
