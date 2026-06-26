# external/ — third-party engine (gitignored, not committed)

## VLMEvalKit

`VLMEvalKit/` is the evaluation engine. It is **cloned by `setup.sh` at a pinned
commit** and is **gitignored** — not committed to this repo. The local clone also
serves as a read-only reference while writing/aligning our model wrappers.

- **Upstream:** https://github.com/open-compass/VLMEvalKit
- **Pinned commit:** `2cf2a36c6e79b51faf676a7011b3a0f5b579814d` (HEAD of `main` @ 2026-06-26, just ahead of tag `v0.3rc1`)
- **Never modified** — our custom wrappers live in our own code and register via VLMEvalKit's plugin path.

Recreate it any time with `bash setup.sh` (clones + installs editable), or clone
manually and `git -C external/VLMEvalKit checkout 2cf2a36c6e79b51faf676a7011b3a0f5b579814d`.

> Note: we tried *vendoring* (committing the source) but VLMEvalKit ships its own
> `.gitignore` that excludes ~35 force-tracked source files (e.g. `api/bailingmm.py`),
> producing a broken committed copy. Cloning gets the complete upstream tree.
