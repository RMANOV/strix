# Contributing to STRIX

STRIX accepts contributions to the public Apache-2.0 open core. The public
repository is intentionally limited to reusable autonomy, safety, simulation,
explainability, and platform-agnostic adapter work.

## Contribution Rules

- Keep contributions compatible with Apache-2.0.
- Do not submit secrets, credentials, private keys, customer data, operational
  scenarios, evaluator collateral, internal review packs, or private companion
  modules.
- Do not add dependencies on private repositories or local filesystem paths.
- Keep public examples generic and simulator-first.
- Preserve the public/private boundary described in `Project_Docs/README.md`.

## Certificate of Origin

By contributing, you certify that you have the right to submit the work under
Apache-2.0 and that it can be included in the public STRIX open core.

Use a Signed-off-by trailer in commits where possible:

```text
Signed-off-by: Your Name <you@example.com>
```

The trailer is provenance hygiene. It is not a copyright assignment.

## Pull Request Expectations

Before opening a pull request:

- run the relevant tests for the touched area;
- run `python scripts/verify_public_surface.py`;
- keep terminology and examples aligned with the neutral public README;
- include a short explanation of any public API or schema change.

Private companion work, customer-specific material, and release authority
operations belong outside this public repository.
