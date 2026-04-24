<!-- SPDX-License-Identifier: Apache-2.0 -->

## Scope

- [ ] This PR keeps the public repository aligned with the Apache-2.0 open core.
- [ ] This PR does not introduce customer-specific material, internal ledgers, or evaluator collateral.
- [ ] This PR does not add alternate public licensing metadata.
- [ ] This PR does not add local release-authority state or private signing material.
- [ ] This PR does not make the public README claim capabilities or artifacts that are not present in the tree.

## Boundary Checks

- [ ] `python scripts/verify_public_surface.py`
- [ ] `pytest -q`
- [ ] Relevant Rust checks or tests were run, if Rust code changed.

## Release Provenance

- [ ] Official-source, trademark, or release-process changes were reviewed by the maintainer.
- [ ] If this PR affects release artifacts, provenance docs and manifests were updated.
