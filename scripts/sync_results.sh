#!/usr/bin/env bash
# Pull eval results from Google Drive to a local (gitignored) folder so they can
# be inspected on the dev machine. Uses rclone (purpose-built for cloud<->local).
#
# ONE-TIME SETUP (you, on the Mac):
#   brew install rclone
#   rclone config
#     -> n (new remote)
#     -> name it:  gdrive
#     -> storage:  drive   (Google Drive)
#     -> follow the browser OAuth; pick the account that stores the results
#     -> "Configure as team drive?" -> n ; accept defaults
#   Verify:  rclone lsd gdrive:MiniVLMDocEval/outputs
#
# USAGE:
#   bash scripts/sync_results.sh          # light: summary/ + logs/ only
#   bash scripts/sync_results.sh --full   # also predictions/ (HEAVY: base64 images)
#
# Override via env: RCLONE_REMOTE, DRIVE_DIR, LOCAL_DIR
set -euo pipefail

REMOTE="${RCLONE_REMOTE:-gdrive}"
DRIVE_DIR="${DRIVE_DIR:-MiniVLMDocEval/outputs}"
LOCAL_DIR="${LOCAL_DIR:-drive_sync}"

if ! command -v rclone >/dev/null; then
  echo "rclone not found. Install + configure once:" >&2
  echo "  brew install rclone && rclone config   # create a 'gdrive' Google Drive remote" >&2
  exit 1
fi

# Deep sync of all result metadata: summary/, logs/, and predictions/ (per-pair
# score.json + acc.csv). The heavy prediction .xlsx (base64 images) are excluded
# unless --full is passed.
EXCLUDE=(--exclude "*.xlsx")
[ "${1:-}" = "--full" ] && EXCLUDE=()

echo "sync $REMOTE:$DRIVE_DIR -> $LOCAL_DIR/ (summary + logs + predictions metadata${1:+ $1})"
rclone sync "$REMOTE:$DRIVE_DIR/summary"      "$LOCAL_DIR/summary"      -P
rclone sync "$REMOTE:$DRIVE_DIR/logs"         "$LOCAL_DIR/logs"         -P
rclone sync "$REMOTE:$DRIVE_DIR/predictions"  "$LOCAL_DIR/predictions"  "${EXCLUDE[@]}" -P

echo "done -> $LOCAL_DIR/"
