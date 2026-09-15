#!/usr/bin/env bash
# Run the training image with the repository bind-mounted and X11 forwarded.
#
#   ./docker/run.sh                      # interactive shell
#   ./docker/run.sh python scripts/train.py configs/ddpg.yaml
#
# GPUs are requested only if the host exposes them, so this also works on a
# CPU-only machine.
set -euo pipefail

IMAGE="${IMAGE:-asv-rl:3.0}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MOUNT=/home/docker/compare_rl

GPU_FLAG=()
if docker info 2>/dev/null | grep -qi nvidia; then
    GPU_FLAG=(--gpus all)
fi

X11_FLAGS=()
if [ -d /tmp/.X11-unix ]; then
    X11_FLAGS=(--env "DISPLAY=${DISPLAY:-}" --volume /tmp/.X11-unix:/tmp/.X11-unix:rw)
fi

exec docker run -it --rm \
    "${GPU_FLAG[@]}" \
    --name asv_rl \
    --net=host \
    "${X11_FLAGS[@]}" \
    --workdir "$MOUNT" \
    --volume "$REPO:$MOUNT" \
    "$IMAGE" "${@:-bash}"
