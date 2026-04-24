# Official Releases and Provenance

This document defines the public STRIX release authority model.

The goal is not to prevent Apache-2.0 forks. Forks are allowed by the license.
The goal is to make the official project, official releases, and supported
distribution channel easy to verify and hard to impersonate.

## Official Source

```text
https://github.com/RMANOV/strix
```

Only artifacts built from this upstream and released by the maintainer release
authority should be treated as official STRIX artifacts.

## Maintainer Release Authority

The release authority is local to the maintainer environment and must not be
committed to Git. The local state lives under `.git/` or the maintainer's local
keyring. Public releases may publish only non-secret verification material such
as:

- release manifest JSON;
- commit SHA;
- source repository URL;
- build workflow reference;
- public signing key fingerprint;
- artifact digest;
- tag name.

The local release authority can be initialized with:

```bash
python scripts/strix_release_manifest.py init-local-authority
```

This creates `.git/strix-release-authority.json`. That file is local-only.

## What Machine Binding Means Here

Machine binding is used for maintainer-side release provenance, not for locking
the public open core to one workstation.

Correct:

- the maintainer machine holds local release authority state;
- official release manifests include an authority ID and public fingerprint;
- private signing keys stay out of the repository;
- forks can build the code, but cannot claim the official release channel.

Incorrect:

- committing private keys or machine IDs;
- making normal development depend on one workstation;
- claiming that public Apache-2.0 code cannot be forked;
- embedding private companion logic in the public repository.

## Recommended Release Flow

1. Confirm the working tree is clean.
2. Run public validation.
3. Build artifacts from the official upstream commit.
4. Generate a release manifest.
5. Sign artifacts and manifest with the maintainer release key.
6. Publish public verification material with the release.

Example:

```bash
python scripts/verify_public_surface.py
python scripts/strix_release_manifest.py manifest --output release-manifest.json
```

The manifest is useful provenance by itself, but it becomes authoritative only
when paired with a published signature or another verifiable release process.
