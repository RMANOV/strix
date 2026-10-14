# Public Scenario Set

This directory contains public simulation presets used to exercise coordination, resilience, and safety behavior in generic settings.

Program-specific, evaluator-facing, or operationally framed scenarios are not maintained as part of the public repository. Public scenarios should remain suitable for research, testing, and documentation.

Each public scenario can be rendered as deterministic software-only replay
evidence:

```bash
python scripts/strix_sim_replay.py --scenario sim/scenarios/gps_denied_recon.yaml
```

The replay output is a JSON timeline plus an optional self-contained HTML canvas
for visual inspection. It reproduces the Python player's seeded movement and
scheduled events. Acceptance targets are not observations: a metric without
an implemented measurement is `not_observed` and fails the envelope. The
GPS-denied, attrition, and multi-domain scenarios currently have unsupported
acceptance metrics; their replay output must not be presented as scenario PASS.
The Python player does not exercise the Rust orchestrator.

These scenarios are public software-only replay **inputs**. They are not sensor,
RF, or hardware validation artifacts, and they do not by themselves demonstrate
field readiness. See
[../../Project_Docs/CAPABILITY_BOUNDARY.md](../../Project_Docs/CAPABILITY_BOUNDARY.md)
for the full public claim boundary.
