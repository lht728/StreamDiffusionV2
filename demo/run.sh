#!/bin/bash
# Container-friendly process manager for the StreamDiffusionV2 demo backend.
# Usage:
#   demo/run.sh start | stop | restart | status | logs
# Honors the same env vars as start.sh (PORT, GPU_IDS, ...).  Most importantly:
#   STREAMDIFF_DISABLE_FLASH=1 (default) -> SDPA fallback (recommended on H20)
#   STREAMDIFF_DISABLE_FLASH=0           -> use installed flash-attn
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname "$0")" && pwd)"
LOG_DIR="$SCRIPT_DIR/../logs"
mkdir -p "$LOG_DIR"
PID_FILE="$LOG_DIR/demo.pid"
LOG_FILE="$LOG_DIR/server.log"

# Defaults that match the production-tuned nohup invocation.
export PORT="${PORT:-7860}"
export HOST="${HOST:-0.0.0.0}"
export GPU_IDS="${GPU_IDS:-0,1}"
export STEP="${STEP:-1}"
export MODEL_TYPE="${MODEL_TYPE:-T2V-1.3B}"
# Fast path is the supported production config (TAEHV decoder + TensorRT engines,
# ~2x VAE speedup, negligible quality loss). Both run.sh and start.sh default
# these to 1; demo/config.py also defaults --fast / --use_taehv / --use_tensorrt
# to True. To run the slow reference path for A/B, set FAST=0 USE_TAEHV=0
# USE_TENSORRT=0 in the environment.
export USE_TAEHV="${USE_TAEHV:-0}"
export USE_TENSORRT="${USE_TENSORRT:-0}"
export FAST="${FAST:-0}"
export TARGET_LATENCY="${TARGET_LATENCY:-1}"
export ENABLE_METRICS="${ENABLE_METRICS:-1}"
export SKIP_FRONTEND_BUILD="${SKIP_FRONTEND_BUILD:-1}"
export STREAMDIFF_DISABLE_FLASH="${STREAMDIFF_DISABLE_FLASH:-1}"
export DEMO_DECODE_OVERLAP="${DEMO_DECODE_OVERLAP:-1}"

is_running() {
  [ -s "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null
}

# Path of the venv python; used as a safety filter when sweeping orphans so
# we never touch python procs from outside this repo. We must use the
# logical absolute path (no "..", no symlink resolution): the kernel
# records argv[0] verbatim as the launcher passed it, which for our
# wrapper -> start.sh -> .venv/bin/python chain is the *logical* path
# below. Resolving with realpath would yield /usr/bin/python3.11 (venv
# python is a symlink) and would risk matching unrelated system pythons.
REPO_ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)"
VENV_PY="$REPO_ROOT/.venv/bin/python"

cmd_start() {
  if is_running; then
    echo "already running, pid=$(cat "$PID_FILE")"
    exit 0
  fi
  # rotate previous log if any
  if [ -f "$LOG_FILE" ] && [ ! -L "$LOG_FILE" ]; then
    mv "$LOG_FILE" "$LOG_FILE.$(date +%Y%m%d_%H%M%S)"
  fi
  cd "$SCRIPT_DIR"
  # Run the wrapper in its own session so wrapper + main.py + every
  # mp.spawn rank worker + every torch._inductor.compile_worker share one
  # process group (pgid == wrapper pid). cmd_stop relies on this to reap
  # the whole tree with a single signal-to-pgid.
  setsid nohup ./start.sh >"$LOG_FILE" 2>&1 < /dev/null &
  echo $! > "$PID_FILE"
  sleep 1
  if is_running; then
    echo "started, pid=$(cat "$PID_FILE")  STREAMDIFF_DISABLE_FLASH=$STREAMDIFF_DISABLE_FLASH"
    echo "log: $LOG_FILE"
  else
    echo "failed to start, see $LOG_FILE"
    exit 1
  fi
}

# Sweep any leftover python procs from this repo's venv. Safe because we
# match on the absolute path to .venv/bin/python which is unique to this
# checkout. Used as belt-and-suspenders in cmd_stop.
sweep_repo_python() {
  local sig="$1"
  # -f matches against full cmdline; pattern is anchored to the absolute
  # interpreter path so we never hit unrelated python procs on the host.
  pkill "-${sig}" -f "^${VENV_PY}( |$)" 2>/dev/null || true
}

cmd_stop() {
  if ! is_running; then
    echo "not running"
    rm -f "$PID_FILE"
    # Even when the wrapper is gone, mp.spawn / inductor compile workers
    # may have outlived it on previous unclean exits; sweep them too.
    sweep_repo_python TERM
    sleep 2
    sweep_repo_python KILL
    fuser -k "${PORT}/tcp" 2>/dev/null || true
    fuser -k 29500/tcp 2>/dev/null || true
    return 0
  fi
  PID=$(cat "$PID_FILE")
  # With setsid in cmd_start, pgid == PID. Signalling -PID hits every
  # descendant in the group at once (rank workers, compile workers, ...).
  echo "stopping pid=$PID (process group) ..."
  kill -TERM -- "-$PID" 2>/dev/null || kill -TERM "$PID" 2>/dev/null || true
  for i in $(seq 1 15); do
    sleep 1
    is_running || break
  done
  if is_running; then
    echo "TERM did not work, sending KILL to group"
    kill -KILL -- "-$PID" 2>/dev/null || true
    kill -KILL "$PID" 2>/dev/null || true
  fi
  rm -f "$PID_FILE"
  # Belt-and-suspenders: any straggler from this repo's venv that somehow
  # detached from the group (rare, but observed historically), plus port
  # cleanup for the listener and the torch.distributed rendezvous port.
  sweep_repo_python TERM
  sleep 2
  sweep_repo_python KILL
  fuser -k "${PORT}/tcp" 2>/dev/null || true
  fuser -k 29500/tcp 2>/dev/null || true
  # Final sanity check: report if any /dev/nvidia* holders remain that
  # belong to this repo, so the operator notices before the next start.
  # Match by reading argv[0] from /proc/<pid>/cmdline (kernel-recorded
  # logical path) instead of /proc/<pid>/exe (which would resolve the
  # venv python symlink to /usr/bin/python3.11 and over-match).
  if command -v fuser >/dev/null 2>&1; then
    leftover=$(fuser /dev/nvidia* 2>/dev/null | tr ' ' '\n' \
      | grep -E '^[0-9]+$' | sort -u \
      | while read -r p; do
          [ -d "/proc/$p" ] || continue
          argv0=$(tr '\0' '\n' <"/proc/$p/cmdline" 2>/dev/null | head -n1)
          [ "$argv0" = "$VENV_PY" ] && echo "$p"
        done)
    if [ -n "$leftover" ]; then
      echo "warning: repo procs still holding /dev/nvidia*: $(echo "$leftover" | tr '\n' ' ')"
    fi
  fi
  echo "stopped"
}

cmd_status() {
  if is_running; then
    PID=$(cat "$PID_FILE")
    echo "running, pid=$PID"
    ps -p "$PID" -o pid,etime,stat,pcpu,pmem,cmd | sed 1d
    echo "STREAMDIFF_DISABLE_FLASH=$(tr '\0' '\n' </proc/$PID/environ 2>/dev/null | grep '^STREAMDIFF_DISABLE_FLASH=' || echo 'unset')"
    ss -lntp 2>/dev/null | grep ":${PORT}\b" || true
  else
    echo "not running"
    return 1
  fi
}

cmd_logs() {
  exec tail -F "$LOG_FILE"
}

case "${1:-}" in
  start)   cmd_start ;;
  stop)    cmd_stop ;;
  restart) cmd_stop; cmd_start ;;
  status)  cmd_status ;;
  logs)    cmd_logs ;;
  *)
    echo "Usage: $0 {start|stop|restart|status|logs}" >&2
    echo "Tunables (env vars):"
    echo "  STREAMDIFF_DISABLE_FLASH=1|0  (default 1: use SDPA, 0: use flash-attn)"
    echo "  PORT GPU_IDS STEP MODEL_TYPE USE_TAEHV USE_TENSORRT FAST"
    echo "  TARGET_LATENCY ENABLE_METRICS SKIP_FRONTEND_BUILD PYTHON_BIN"
    exit 2
    ;;
esac
