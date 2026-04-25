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

## Next Capabilities

The next useful expansion is scenario-family batch replay: every public scenario
already declares a seed, metric set, and `pass_envelope`; the next step is to
run every scenario through replay and compare observed metrics against that
envelope. After that, add statistical Monte Carlo sweeps, richer trace exports,
and integration checks for criticality, contagion, and quorum-style
confirmation loops.
