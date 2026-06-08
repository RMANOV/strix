# Public Scenario Set

This directory contains public simulation presets used to exercise coordination, resilience, and safety behavior in generic settings.

Program-specific, evaluator-facing, or operationally framed scenarios are not maintained as part of the public repository. Public scenarios should remain suitable for research, testing, and documentation.

Each public scenario can be rendered as deterministic software-only replay
evidence:

```bash
python scripts/strix_sim_replay.py --scenario sim/scenarios/gps_denied_recon.yaml
```

The replay output is a JSON timeline plus an optional self-contained HTML canvas
for visual inspection. This validates seeded orchestration behavior and public
scenario envelopes; it does not replace hardware, RF, or field validation.

These scenarios are public software-only replay **inputs**. They are not sensor,
RF, or hardware validation artifacts, and they do not by themselves demonstrate
field readiness. See
[../../Project_Docs/CAPABILITY_BOUNDARY.md](../../Project_Docs/CAPABILITY_BOUNDARY.md)
for the full public claim boundary.
