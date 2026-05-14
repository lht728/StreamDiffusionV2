#!/usr/bin/env bash
# Convenience wrapper to run the StreamDiffusionV2 image with the right
# GPU + volume + port flags. The image itself is slim; checkpoints and
# outputs MUST be mounted from the host.
#
# Usage:
#   ./docker-run.sh                       # interactive bash
#   ./docker-run.sh demo                  # launch the web demo (port 7860)
#   ./docker-run.sh single                # run offline single-GPU inference
#   ./docker-run.sh pipe                  # run offline multi-GPU pipeline
#   ./docker-run.sh download              # run _download_ckpts.sh into the mounted ckpts/
#
# Env overrides:
#   IMAGE          full image ref (default: mirrors.tencent.com/gputest/streamdiffusionv2:latest)
#   GPUS           --gpus value (default: all)
#   HOST_CKPTS     host path mounted to /workspace/StreamDiffusionV2/ckpts
#   HOST_WAN       host path mounted to /workspace/StreamDiffusionV2/wan_models
#   HOST_OUTPUTS   host path mounted to /workspace/StreamDiffusionV2/outputs
#   PORT           host port forwarded to container 7860 (demo mode only, default 7860)
#   GPU_IDS        passed through to demo/run.sh (default: 0)

set -euo pipefail

IMAGE="${IMAGE:-mirrors.tencent.com/gputest/streamdiffusionv2:latest}"
GPUS="${GPUS:-all}"
HOST_CKPTS="${HOST_CKPTS:-$(pwd)/ckpts}"
HOST_WAN="${HOST_WAN:-$(pwd)/wan_models}"
HOST_OUTPUTS="${HOST_OUTPUTS:-$(pwd)/outputs}"
PORT="${PORT:-7860}"
GPU_IDS="${GPU_IDS:-0}"

mkdir -p "$HOST_CKPTS" "$HOST_WAN" "$HOST_OUTPUTS"

COMMON_ARGS=(
  --gpus "$GPUS"
  --rm
  --ipc=host
  --ulimit memlock=-1
  --ulimit stack=67108864
  -v "$HOST_CKPTS":/workspace/StreamDiffusionV2/ckpts
  -v "$HOST_WAN":/workspace/StreamDiffusionV2/wan_models
  -v "$HOST_OUTPUTS":/workspace/StreamDiffusionV2/outputs
)

MODE="${1:-shell}"
shift || true

case "$MODE" in
  shell|bash|"")
    exec docker run -it "${COMMON_ARGS[@]}" -p "${PORT}:7860" "$IMAGE" /bin/bash
    ;;

  demo)
    # Run the web UI in foreground; demo/run.sh -> demo/start.sh expects the
    # frontend already built (we baked it in stage 1) and SKIP_FRONTEND_BUILD=1
    # is set in the image env.
    exec docker run -it "${COMMON_ARGS[@]}" \
      -p "${PORT}:7860" \
      -e GPU_IDS="$GPU_IDS" \
      "$IMAGE" \
      bash -lc "cd demo && ./start.sh"
    ;;

  single|single-wo|pipe)
    exec docker run -it "${COMMON_ARGS[@]}" \
      -e CUDA_VISIBLE_DEVICES="$GPU_IDS" \
      "$IMAGE" \
      bash -lc "./run_v2v.sh $MODE $*"
    ;;

  download)
    exec docker run -it "${COMMON_ARGS[@]}" \
      "$IMAGE" \
      bash -lc "export HF_HUB_ENABLE_HF_TRANSFER=1 && \
                hf download Wan-AI/Wan2.1-T2V-1.3B --local-dir wan_models/Wan2.1-T2V-1.3B && \
                hf download jerryfeng/StreamDiffusionV2 --local-dir ./ckpts --include 'wan_causal_dmd_v2v/*' && \
                curl -L https://github.com/madebyollin/taehv/raw/main/taew2_1.pth -o ckpts/taew2_1.pth"
    ;;

  *)
    echo "Unknown mode: $MODE" >&2
    echo "Usage: $0 {shell|demo|single|single-wo|pipe|download} [extra args...]" >&2
    exit 2
    ;;
esac
