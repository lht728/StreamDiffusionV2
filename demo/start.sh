#!/bin/bash
set -eu

# Build the Svelte frontend, then launch the Python demo backend.
# Override HOST, PORT, GPU_IDS, STEP, and MODEL_TYPE via environment variables.
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname "$0")" && pwd)"
FRONTEND_DIR="$SCRIPT_DIR/frontend"

PORT="${PORT:-7860}"
HOST="${HOST:-0.0.0.0}"
GPU_IDS="${GPU_IDS:-0}"
STEP="${STEP:-1}"
MODEL_TYPE="${MODEL_TYPE:-T2V-1.3B}"
# Always-on fast path: TAEHV decoder + TensorRT acceleration.
# `--fast` is syntactic sugar for `--use_taehv --use_tensorrt` and also picks
# the `_fast.yaml` model config; see streamv2v.inference_common.
# To run the slow reference path for A/B, set FAST=0 (and USE_TAEHV=0,
# USE_TENSORRT=0) explicitly in the environment.
USE_TAEHV="${USE_TAEHV:-1}"
USE_TENSORRT="${USE_TENSORRT:-1}"
FAST="${FAST:-1}"
# Path to the python interpreter (defaults to the project's venv if it exists,
# otherwise falls back to whatever `python` is on PATH).
if [ -z "${PYTHON_BIN:-}" ]; then
  if [ -x "$SCRIPT_DIR/../.venv/bin/python" ]; then
    PYTHON_BIN="$SCRIPT_DIR/../.venv/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi
# Skip the npm install + npm run build step (set to 1/true/yes/on when frontend
# was already built, e.g. for systemd / nohup restarts of the backend only).
SKIP_FRONTEND_BUILD="${SKIP_FRONTEND_BUILD:-0}"
# Disable flash-attn at runtime by default (FA2/FA3 are no faster than SDPA on H20
# for step=1 + KV-cache + short sequences; FA3 was observed to be slower).
# Set STREAMDIFF_DISABLE_FLASH=0 to re-enable flash-attn for A/B comparison.
STREAMDIFF_DISABLE_FLASH="${STREAMDIFF_DISABLE_FLASH:-1}"
export STREAMDIFF_DISABLE_FLASH

IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
LOCAL_GPU_IDS="$(seq 0 $((${#GPU_ARRAY[@]} - 1)) | paste -sd, -)"

case "$(printf '%s' "$SKIP_FRONTEND_BUILD" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|on)
    echo "skip frontend build"
    ;;
  *)
    cd "$FRONTEND_DIR"
    npm install
    npm run build
    echo "frontend build success"
    ;;
esac

cd "$SCRIPT_DIR"
TAEHV_FLAG=""
case "$(printf '%s' "$USE_TAEHV" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|on)
    TAEHV_FLAG="--use_taehv"
    ;;
esac

TENSORRT_FLAG=""
case "$(printf '%s' "$USE_TENSORRT" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|on)
    TENSORRT_FLAG="--use_tensorrt"
    ;;
esac

FAST_FLAG=""
case "$(printf '%s' "$FAST" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|on)
    FAST_FLAG="--fast"
    ;;
esac

# Optional metrics / latency target controls (main.py CLI flags).
TARGET_LATENCY_FLAG=""
if [ -n "${TARGET_LATENCY:-}" ]; then
  TARGET_LATENCY_FLAG="--target-latency $TARGET_LATENCY"
fi

ENABLE_METRICS_FLAG=""
case "$(printf '%s' "${ENABLE_METRICS:-0}" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|on)
    ENABLE_METRICS_FLAG="--enable-metrics"
    ;;
esac

# Extra raw CLI args appended verbatim (advanced).
EXTRA_ARGS="${EXTRA_ARGS:-}"

CUDA_VISIBLE_DEVICES="$GPU_IDS" \
STREAMDIFF_DISABLE_FLASH="$STREAMDIFF_DISABLE_FLASH" \
"$PYTHON_BIN" main.py \
  --port "$PORT" \
  --host "$HOST" \
  --num_gpus "$(printf '%s' "$GPU_IDS" | awk -F',' '{print NF}')" \
  --gpu_ids "$LOCAL_GPU_IDS" \
  --step "$STEP" \
  --model_type "$MODEL_TYPE" \
  $TAEHV_FLAG \
  $TENSORRT_FLAG \
  $FAST_FLAG \
  $TARGET_LATENCY_FLAG \
  $ENABLE_METRICS_FLAG \
  $EXTRA_ARGS
