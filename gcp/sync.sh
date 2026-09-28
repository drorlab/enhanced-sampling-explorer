#!/usr/bin/env bash
# push: copy sims/ analysis/ to the VM.  pull: copy results/ and dashboard/data back.
set -euo pipefail
cd "$(dirname "$0")/.."
NAME=${NAME:-esdemo}; ZONE=$(cat gcp/.zone)
case "${1:-}" in
  push) tar -czf - --exclude=dashboard/data --exclude=__pycache__ sims analysis environment.yml dashboard | gcloud compute ssh "$NAME" --zone "$ZONE" -- \
          'mkdir -p ~/esdemo && tar -xzf - -C ~/esdemo' ;;
  pull) gcloud compute ssh "$NAME" --zone "$ZONE" -- 'cd ~/esdemo && tar -czf - dashboard/data 2>/dev/null' \
          | tar -xzf - ;;
  *) echo "usage: sync.sh push|pull" >&2; exit 2 ;;
esac
