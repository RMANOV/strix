# Public Demo Material

The public `demo/` tree contains only lightweight examples and placeholders.

Evaluator-facing collateral, narrated demo scripts, and richer presentation assets are not maintained as part of the public repository. Public examples should stay focused on generic orchestration, simulation, and developer-facing integration.

For a public-safe visual replay, generate a self-contained HTML view from one of
the public scenarios:

```bash
python scripts/strix_sim_replay.py --scenario sim/scenarios/gps_denied_recon.yaml --output target/strix-replays/gps_denied_recon.json --html target/strix-replays/gps_denied_recon.html
```

Open the generated HTML file locally to inspect agent movement, event timing,
constraint avoidance, energy, and replay metrics. Generated replay assets live
under `target/` by default and are not committed.

Demo output is public-safe software replay intended for visual inspection of
seeded behavior. It is **not** hardware, RF, sensor, or field validation. See
[../Project_Docs/CAPABILITY_BOUNDARY.md](../Project_Docs/CAPABILITY_BOUNDARY.md)
for the full public claim boundary.
