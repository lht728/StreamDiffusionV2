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
export USE_TAEHV="${USE_TAEHV:-1}"
export USE_TENSORRT="${USE_TENSORRT:-1}"
export FAST="${FAST:-1}"
export TARGET_LATENCY="${TARGET_LATENCY:-0.4}"
export ENABLE_METRICS="${ENABLE_METRICS:-1}"
export SKIP_FRONTEND_BUILD="${SKIP_FRONTEND_BUILD:-1}"
export STREAMDIFF_DISABLE_FLASH="${STREAMDIFF_DISABLE_FLASH:-1}"

is_running() {
  [ -s "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null
}

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
  nohup ./start.sh >"$LOG_FILE" 2>&1 &
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

cmd_stop() {
  if ! is_running; then
    echo "not running"
    rm -f "$PID_FILE"
    return 0
  fi
  PID=$(cat "$PID_FILE")
  echo "stopping pid=$PID ..."
  kill -TERM "$PID" 2>/dev/null || true
  for i in $(seq 1 15); do
    sleep 1
    is_running || break
  done
  if is_running; then
    echo "TERM did not work, sending KILL"
    kill -KILL "$PID" 2>/dev/null || true
    pkill -9 -f 'demo/main.py' 2>/dev/null || true
    pkill -9 -f 'StreamDiffusionV2' 2>/dev/null || true
  fi
  rm -f "$PID_FILE"
  # belt & suspenders: free 7860 / 29500 if leaked
  fuser -k "${PORT}/tcp" 2>/dev/null || true
  fuser -k 29500/tcp 2>/dev/null || true
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
