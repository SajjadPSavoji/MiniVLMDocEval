# Vendored dependencies

## VLMEvalKit

`VLMEvalKit/` is **vendored** (committed into this repo) rather than cloned at
setup time — for reproducibility and local development. It is pinned to an exact
upstream commit and is **never modified**: our custom model wrappers live in our
own code and are registered via VLMEvalKit's plugin path.

- **Upstream:** https://github.com/open-compass/VLMEvalKit
- **Pinned commit:** `2cf2a36c6e79b51faf676a7011b3a0f5b579814d` (HEAD of `main` @ 2026-06-26, just ahead of tag `v0.3rc1`)
- **Vendored on:** 2026-06-26 (nested `.git` removed; source frozen in this repo's history)

### Why vendored (not a clone or submodule)

- **Always present** after a plain `git clone` — no submodule init, no network at setup.
- **Frozen in git history** — the exact engine source is pinned alongside our code.
- **Dev-friendly** — the source is readable locally when writing/aligning wrappers.

### How it's installed

`setup.sh` installs it **editable** from this path (`pip install --no-deps -e external/VLMEvalKit`).
Editable is required because a built wheel drops subpackages lacking `__init__.py`
(e.g. `megabench/parsing`). No cloning happens at setup.

### Updating the pin (deliberate, rare)

Re-vendor a new commit explicitly:

```bash
rm -rf external/VLMEvalKit
git clone https://github.com/open-compass/VLMEvalKit.git external/VLMEvalKit
git -C external/VLMEvalKit checkout <new-commit>
rm -rf external/VLMEvalKit/.git
# update the commit hash above and in setup.sh / requirements.txt
```
