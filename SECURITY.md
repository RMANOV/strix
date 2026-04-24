# Security Policy

## Reporting

Do not open public issues for vulnerabilities, leaked credentials, sensitive
scenario material, or private companion boundary failures.

Report privately to the maintainer using a direct private channel. If no
private channel has already been established, request one without including
exploit details or sensitive material in the public issue tracker.

## Public Repository Boundary

The public repository must not contain:

- secrets, tokens, private keys, entitlement keys, or signing keys;
- customer data or deployment-specific configuration;
- internal review ledgers or evaluator collateral;
- private companion modules or customer-specific policy;
- release authority state from `.git/` or local machine keyrings.

## Release Provenance

Official release artifacts should be traceable to the upstream repository and
to a maintainer release authority. Public verification material may be
published, but private keys and local machine authority state must never be
committed.

See `Project_Docs/provenance/OFFICIAL_RELEASES.md`.
