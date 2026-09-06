#!/usr/bin/env bash
# Compatibility name retained for old automation. package-line.sh requires the complete inference
# payload plus a hash-bound build manifest and therefore refuses stale/mislabeled legacy outputs.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo ">> package-legacy.sh is retired; delegating to provenance-checked package-line.sh" >&2
exec "$REPO/hiveos/package-line.sh" dist-legacy legacy "$@"
