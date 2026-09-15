#!/usr/bin/env bash
# Build the training image.
set -euo pipefail

IMAGE="${IMAGE:-asv-rl:3.0}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

docker build "$HERE" -f "$HERE/Dockerfile" -t "$IMAGE"
echo "Built $IMAGE"
