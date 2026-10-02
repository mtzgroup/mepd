#!/usr/bin/env bash
# Your own remote mepd web: the full app, unrestricted (unlike deploy/demo/),
# behind your password, reachable over a Cloudflare quick tunnel. For you
# only: whoever logs in can read your files and run programs as you.
#
#   deploy/remote/run.sh start        # start (or restart) it on the latest develop
#   deploy/remote/run.sh share        # open the https://....trycloudflare.com tunnel
#   deploy/remote/run.sh status       # running? public? at which URL
#   deploy/remote/run.sh unshare      # close the tunnel (it keeps running, local only)
#   deploy/remote/run.sh stop         # close the tunnel and stop it
#   deploy/remote/run.sh logs         # follow the server log
#   deploy/remote/run.sh password NEW # set the password (restarts it if running)
#
# Settings (environment):
#   REMOTE_DATA  its workspace, state and password      (default ~/mepd_remote)
#   REMOTE_PORT  loopback port                           (default 8791)
#   REMOTE_SRC   the code it runs: a clean git worktree of REMOTE_REF, updated on
#                every `start` (default ~/mepd_remote_src), never this checkout's
#                working tree (other work in progress must not break it)
#   REMOTE_REF   what REMOTE_SRC checks out              (default origin/develop)
#   REMOTE_JOBS  calculations at once                    (default 4)
set -euo pipefail

UNIT=mepd-remote
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
REMOTE_DATA="${REMOTE_DATA:-$HOME/mepd_remote}"
REMOTE_PORT="${REMOTE_PORT:-8791}"
REMOTE_SRC="${REMOTE_SRC:-$HOME/mepd_remote_src}"
REMOTE_REF="${REMOTE_REF:-origin/develop}"
REMOTE_JOBS="${REMOTE_JOBS:-4}"
PW_FILE="$REMOTE_DATA/.password"
TUNNEL_PID="$REMOTE_DATA/tunnel.pid"
TUNNEL_LOG="$REMOTE_DATA/tunnel.log"
TUNNEL_URL="$REMOTE_DATA/tunnel.url"

save_password() { mkdir -p "$REMOTE_DATA"; (umask 077; printf '%s\n' "$1" >"$PW_FILE"); }
running() { systemctl --user is-active --quiet "$UNIT"; }
tunnel_running() { [ -f "$TUNNEL_PID" ] && kill -0 "$(cat "$TUNNEL_PID")" 2>/dev/null; }

unshare() {
  if tunnel_running; then kill "$(cat "$TUNNEL_PID")" && echo "tunnel closed: local only again"; else echo "no tunnel running"; fi
  rm -f "$TUNNEL_PID" "$TUNNEL_URL"
}

share() {
  curl -fs -o /dev/null "http://127.0.0.1:$REMOTE_PORT/login" || { echo "not running; start it first: $0 start"; exit 1; }
  if tunnel_running; then echo "already public: $(cat "$TUNNEL_URL" 2>/dev/null)"; return; fi
  command -v cloudflared >/dev/null || { echo "cloudflared not found on PATH"; exit 1; }
  nohup cloudflared tunnel --no-autoupdate --url "http://127.0.0.1:$REMOTE_PORT" >"$TUNNEL_LOG" 2>&1 &
  echo $! >"$TUNNEL_PID"
  for _ in $(seq 1 45); do
    url="$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' "$TUNNEL_LOG" | head -1 || true)"
    [ -n "$url" ] && break
    tunnel_running || { echo "cloudflared exited:"; tail -5 "$TUNNEL_LOG"; exit 1; }
    sleep 1
  done
  [ -n "${url:-}" ] || { echo "no URL yet; see $TUNNEL_LOG"; exit 1; }
  echo "$url" >"$TUNNEL_URL"
  echo "PUBLIC: $url  (log in with your password)"
  echo "close it with: $0 unshare"
}

status() {
  if running; then echo "remote: running on http://127.0.0.1:$REMOTE_PORT ($(git -C "$REMOTE_SRC" log -1 --format='%h %s' 2>/dev/null))"
  else echo "remote: stopped"; fi
  if tunnel_running; then echo "public: $(cat "$TUNNEL_URL" 2>/dev/null)"; else echo "public: no (local only)"; fi
}

case "${1:-start}" in
  stop) unshare; systemctl --user stop "$UNIT" 2>/dev/null && echo "stopped" || echo "not running"; exit 0 ;;
  logs) exec journalctl --user -u "$UNIT" -f ;;
  share) share; exit 0 ;;
  unshare) unshare; exit 0 ;;
  status) status; exit 0 ;;
  password)
    [ -n "${2:-}" ] || { echo "usage: $0 password NEW_PASSWORD"; exit 2; }
    save_password "$2"
    if running; then systemctl --user restart "$UNIT"; echo "password changed and restarted (log in again)"; else echo "password saved"; fi
    exit 0 ;;
  start) ;;
  *) echo "usage: $0 [start|share|status|unshare|stop|logs|password NEW]"; exit 2 ;;
esac

[ -s "$PW_FILE" ] || { echo "set a password first: $0 password NEW_PASSWORD"; exit 1; }
# The code: REMOTE_REF as committed, in its own worktree. PYTHONPATH puts it
# ahead of the venv's editable install (jobs inherit it, so they run it too).
git -C "$REPO" fetch -q origin
if [ -e "$REMOTE_SRC/.git" ]; then
  git -C "$REMOTE_SRC" checkout -q --detach "$REMOTE_REF"
  git -C "$REMOTE_SRC" reset -q --hard "$REMOTE_REF"
  git -C "$REMOTE_SRC" clean -qfdx
else
  git -C "$REPO" worktree add -q --detach "$REMOTE_SRC" "$REMOTE_REF"
fi
mkdir -p "$REMOTE_DATA/workspace" "$REMOTE_DATA/state"
systemctl --user stop "$UNIT" 2>/dev/null || true
systemctl --user reset-failed "$UNIT" 2>/dev/null || true
# A transient user service: survives this shell, restarts if it dies. The
# password reaches it through a file (not the command line, which `ps` shows).
systemd-run --user --unit "$UNIT" --description "mepd web (remote, $REMOTE_REF)" \
  --property Restart=on-failure --property WorkingDirectory="$REMOTE_DATA" \
  --setenv PYTHONPATH="$REMOTE_SRC" --setenv MEPD_WEB_STATE_DIR="$REMOTE_DATA/state" \
  --setenv OMP_NUM_THREADS=1 --setenv PATH="$PATH" --setenv HOME="$HOME" \
  bash -c "MEPD_WEB_PASSWORD=\"\$(cat '$PW_FILE')\" exec '$REPO/.venv/bin/python' -m mepd.cli web '$REMOTE_DATA/workspace' --host 127.0.0.1 --port $REMOTE_PORT --no-open --max-jobs $REMOTE_JOBS" >/dev/null
for _ in $(seq 1 60); do curl -fs -o /dev/null "http://127.0.0.1:$REMOTE_PORT/login" && break; sleep 1; done
curl -fs -o /dev/null "http://127.0.0.1:$REMOTE_PORT/login" || { echo "it did not come up:"; journalctl --user -u "$UNIT" -n 20 --no-pager; exit 1; }
echo "started on http://127.0.0.1:$REMOTE_PORT, code $(git -C "$REMOTE_SRC" log -1 --format='%h %s')"
tunnel_running && echo "public: $(cat "$TUNNEL_URL")" || echo "local only; make it reachable with: $0 share"
