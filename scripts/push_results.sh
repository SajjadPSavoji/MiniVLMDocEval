#!/usr/bin/env bash
# Push the light results/ table to GitHub from Colab.
#
# Colab has no stored git credentials, so provide a GitHub token (a fine-grained
# PAT with repo write). Easiest: store it in Colab Secrets as GH_TOKEN, then in a
# cell: import os; os.environ['GH_TOKEN'] = userdata.get('GH_TOKEN')
#
# Only results/ is pushed (small score tables) — heavy predictions stay in the
# Drive work-dir, out of git. Pull --rebase first so this coexists with code
# pushes from the local machine.
set -euo pipefail

: "${GH_TOKEN:?Set GH_TOKEN (GitHub PAT with repo write) — e.g. from Colab Secrets}"
REMOTE="https://x-access-token:${GH_TOKEN}@github.com/SajjadPSavoji/MiniVLMDocEval.git"

git config user.email "${GIT_EMAIL:-colab-runner@users.noreply.github.com}"
git config user.name  "${GIT_NAME:-colab-runner}"

git add results/
if git diff --cached --quiet; then
  echo "No results changes to push."
  exit 0
fi
git commit -m "results: update comparison table (Colab run)"
git pull --rebase "$REMOTE" main
git push "$REMOTE" main
echo "Pushed results/ to GitHub."
