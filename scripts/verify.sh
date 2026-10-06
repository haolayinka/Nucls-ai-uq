#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
sha256sum -c config/expected_hashes.sha256
echo "PUBLIC_ARTIFACT_VERIFICATION_PASS"
