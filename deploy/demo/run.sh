#!/usr/bin/env bash
# Control the public mepd demo.
#
#   deploy/demo/run.sh start      # start (or restart) the demo container; prints the password
#   deploy/demo/run.sh share      # open a public https://....trycloudflare.com tunnel to it
#   deploy/demo/run.sh status     # what is running, and the public URL
#   deploy/demo/run.sh unshare    # close the tunnel (demo keeps running, private again)
#   deploy/demo/run.sh stop       # close the tunnel and stop the demo
#   deploy/demo/run.sh logs       # follow the server log
#   deploy/demo/run.sh password NEW   # set the shared password (restarts the demo if running)
#
# Settings (environment):
#   DEMO_DATA   host dir for visitor workspaces + demo profiles  (default ~/mepd_demo)
#   DEMO_PORT   loopback port the container publishes            (default 8790)
#   DEMO_CPUS   CPU cap for the whole demo                       (default 12)
#   DEMO_MEM    memory cap                                       (default 32g)
#   MEPD_DEMO_PASSWORD  shared password for `start` (saved to $DEMO_DATA/root/.demo_password,
#                       the single source of truth; default: generated once)
set -euo pipefail

NAME=mepd-demo
IMAGE=mepd-demo-runtime
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
DEMO_DATA="${DEMO_DATA:-$HOME/mepd_demo}"
DEMO_PORT="${DEMO_PORT:-8790}"
DEMO_CPUS="${DEMO_CPUS:-12}"
DEMO_MEM="${DEMO_MEM:-32g}"

PW_FILE="$DEMO_DATA/root/.demo_password"

save_password() {
  mkdir -p "$DEMO_DATA/root"
  (umask 077; printf '%s\n' "$1" >"$PW_FILE")
}

TUNNEL_PID="$DEMO_DATA/tunnel.pid"
TUNNEL_LOG="$DEMO_DATA/tunnel.log"
TUNNEL_URL="$DEMO_DATA/tunnel.url"

tunnel_running() { [ -f "$TUNNEL_PID" ] && kill -0 "$(cat "$TUNNEL_PID")" 2>/dev/null; }

unshare() {
  if tunnel_running; then
    kill "$(cat "$TUNNEL_PID")" && echo "tunnel closed: the demo is no longer public"
  else
    echo "no tunnel running"
  fi
  rm -f "$TUNNEL_PID" "$TUNNEL_URL"
}

share() {
  if ! curl -fs -o /dev/null "http://127.0.0.1:$DEMO_PORT/login"; then
    echo "the demo is not running; start it first: $0 start"; exit 1
  fi
  if tunnel_running; then
    echo "already public: $(cat "$TUNNEL_URL" 2>/dev/null)"; return
  fi
  command -v cloudflared >/dev/null || { echo "cloudflared not found on PATH"; exit 1; }
  # A Cloudflare quick tunnel: no account, random hostname, HTTPS terminated
  # by Cloudflare, forwarding only to the demo's loopback port.
  nohup cloudflared tunnel --no-autoupdate --url "http://127.0.0.1:$DEMO_PORT" >"$TUNNEL_LOG" 2>&1 &
  echo $! >"$TUNNEL_PID"
  for _ in $(seq 1 45); do
    url="$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' "$TUNNEL_LOG" | head -1 || true)"
    [ -n "$url" ] && break
    tunnel_running || { echo "cloudflared exited:"; tail -5 "$TUNNEL_LOG"; exit 1; }
    sleep 1
  done
  [ -n "${url:-}" ] || { echo "no URL yet; see $TUNNEL_LOG"; exit 1; }
  echo "$url" >"$TUNNEL_URL"
  echo "PUBLIC: $url"
  echo "password: $(cat "$DEMO_DATA/root/.demo_password" 2>/dev/null || echo '(set via MEPD_DEMO_PASSWORD)')"
  echo "close it with: $0 unshare"
}

status() {
  if docker ps --filter "name=^$NAME\$" --format '{{.Status}}' | grep -q .; then
    echo "demo:   running ($(docker ps --filter "name=^$NAME\$" --format '{{.Status}}')) on http://127.0.0.1:$DEMO_PORT"
  else
    echo "demo:   stopped"
  fi
  if tunnel_running; then echo "public: $(cat "$TUNNEL_URL" 2>/dev/null)"; else echo "public: no (private)"; fi
}

case "${1:-start}" in
  stop) unshare; docker rm -f "$NAME" >/dev/null 2>&1 && echo "stopped $NAME" || echo "$NAME not running"; exit 0 ;;
  logs) exec docker logs -f "$NAME" ;;
  share) share; exit 0 ;;
  unshare) unshare; exit 0 ;;
  status) status; exit 0 ;;
  password)
    [ -n "${2:-}" ] || { echo "usage: $0 password NEW_PASSWORD"; exit 2; }
    save_password "$2"
    if docker ps --filter "name=^$NAME\$" --format '{{.Names}}' | grep -q .; then
      docker restart "$NAME" >/dev/null   # the server reads the password at startup
      for _ in $(seq 1 60); do curl -fs -o /dev/null "http://127.0.0.1:$DEMO_PORT/login" && break; sleep 1; done
      echo "password changed and demo restarted (logged-in visitors stay logged in)"
    else
      echo "password saved; it applies when you run: $0 start"
    fi
    exit 0 ;;
  start) ;;
  *) echo "usage: $0 [start|share|status|unshare|stop|logs|password NEW]"; exit 2 ;;
esac

PY_HOME="$(readlink -f "$REPO/.venv/bin/python")"          # uv-managed interpreter
UV_PYTHONS="$(dirname "$(dirname "$(dirname "$PY_HOME")")")"   # ~/.local/share/uv/python
OPT="$HOME/.local/opt"                                         # gxtb, gsm, gsm-blas
CREST="$(readlink -f "$(command -v crest)")"
GXTB="$(readlink -f "${GXTB_EXECUTABLE:-$OPT/gxtb-2.0.1/bin/xtb}")"
GSM="$(readlink -f "${GSM_EXECUTABLE:-$OPT/gsm/bin/gsm}")"

mkdir -p "$DEMO_DATA/root" "$DEMO_DATA/home" "$DEMO_DATA/state"
[ -n "${MEPD_DEMO_PASSWORD:-}" ] && save_password "$MEPD_DEMO_PASSWORD"
docker build -q -t "$IMAGE" "$REPO/deploy/demo" >/dev/null
docker rm -f "$NAME" >/dev/null 2>&1 || true

# (--mount, not -v src:dst:ro: the short form silently skipped some of these
# binds on this Docker install.)
# Only the package and its venv are mounted from the checkout -- not the rest
# of the repo (workspaces, .git, ...), which visitors have no business seeing.
# /tmp must allow exec: GSM runs a ./grad.py helper from its temp workdir
# (Docker mounts --tmpfs noexec by default, which silently broke GSM).
# Hardening: non-root (your uid), read-only root fs and code mounts, no
# capabilities, no privilege escalation, capped CPU/memory/processes, and the
# port published on loopback only (Tailscale Funnel proxies to it).
# TORCHDYNAMO_DISABLE: the image has no C++ compiler, which torch.compile
# (used inside AIMNet2) needs on CPU; eager mode gives the same numbers.
docker run -d --name "$NAME" --restart unless-stopped \
  --user "$(id -u):$(id -g)" \
  --read-only --tmpfs /tmp:rw,exec,size=4g \
  --cap-drop ALL --security-opt no-new-privileges \
  --cpus "$DEMO_CPUS" --memory "$DEMO_MEM" --pids-limit 2048 \
  -p "127.0.0.1:$DEMO_PORT:$DEMO_PORT" \
  --mount "type=bind,src=$REPO/mepd,dst=$REPO/mepd,readonly" \
  --mount "type=bind,src=$REPO/.venv,dst=$REPO/.venv,readonly" \
  --mount "type=bind,src=$UV_PYTHONS,dst=$UV_PYTHONS,readonly" \
  --mount "type=bind,src=$OPT,dst=$OPT,readonly" \
  --mount "type=bind,src=$CREST,dst=/opt/bin/crest,readonly" \
  --mount "type=bind,src=$DEMO_DATA,dst=/data" \
  -e HOME=/data/home -e MEPD_WEB_STATE_DIR=/data/state \
  -e PATH=/opt/bin:/usr/local/bin:/usr/bin:/bin \
  -e GXTB_EXECUTABLE="$GXTB" -e GSM_EXECUTABLE="$GSM" \
  -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
  -e TORCHDYNAMO_DISABLE=1 \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -w /data \
  "$IMAGE" \
  "$REPO/.venv/bin/python" -m mepd.cli web /data/root --demo --host 0.0.0.0 --port "$DEMO_PORT" --no-open --max-jobs 2 \
  >/dev/null

echo "started $NAME on http://127.0.0.1:$DEMO_PORT (data: $DEMO_DATA)"
for _ in $(seq 1 60); do
  curl -fs -o /dev/null "http://127.0.0.1:$DEMO_PORT/login" && break
  sleep 1
done
docker logs "$NAME" 2>&1 | grep -E "password|profiles|limits" || docker logs --tail 20 "$NAME"
echo "private for now; make it public with: $0 share"
