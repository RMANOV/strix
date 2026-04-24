# Official Upstream Branch Protection

This document describes the recommended GitHub settings for the official STRIX
upstream.

The goal is to protect the official project identity and release channel. It is
not intended to stop Apache-2.0 forks.

## Required Settings

- Require pull requests before merging into `main`.
- Require review from code owners.
- Require status checks before merging.
- Require branches to be up to date before merging when practical.
- Restrict direct pushes to `main` to the maintainer account.
- Block force pushes on protected branches.
- Block branch deletion on protected branches.
- Require conversation resolution before merging.

## Required Checks

At minimum, the official upstream should require:

- Public Surface
- Format
- Clippy
- Python Tests
- Python FFI Smoke Tests
- Python Type Check
- Test (stable)
- Security Audit

If the workflow keeps nightly tests or benchmark compilation as required gates,
keep them required as well. If they are advisory, document that explicitly in
the repository settings and PR description.

## Local Maintainer Guardrails

Install local hooks in maintainer checkouts:

```bash
python scripts/install_local_git_hooks.py --force
```

The hooks run the public-surface guard before commits and pushes. The pre-push
hook also blocks direct pushes to `main` unless the maintainer deliberately sets
`STRIX_ALLOW_MAIN_PUSH=1` for an exceptional release operation.

These hooks are a local safety layer only. GitHub branch protection remains the
authoritative upstream gate.

## Review Discipline

- Treat official identity files as high-risk review surfaces.
- Treat release provenance files as high-risk review surfaces.
- Keep private signing keys, local release-authority state, and customer-specific
  material out of the public repository.
- Prefer small PRs with one security or governance purpose per PR.
