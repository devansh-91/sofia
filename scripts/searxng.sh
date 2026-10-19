#!/usr/bin/env bash
# searxng.sh — a private web search for Sofia: SearXNG in Docker, reachable from this laptop only.
#
# Sofia's web_search (sofia/tools.py) asks it first. Without it she falls back to Brave Search's
# web page, which turns scripts away after a few searches in a row, and then to news headlines.
#
#   ./scripts/searxng.sh          start it (the first time pulls the image, a few hundred MB)
#   ./scripts/searxng.sh stop     stop and remove it
set -euo pipefail

NAME=sofia-searxng
PORT="${SOFIA_SEARXNG_PORT:-8888}"
CONFIG="$HOME/.local/share/sofia/searxng"
IMAGE=docker.io/searxng/searxng:latest

if [ "${1:-}" = stop ]; then
  docker rm -f "$NAME"
  exit
fi

mkdir -p "$CONFIG"
if [ ! -f "$CONFIG/settings.yml" ]; then
  # JSON output is what Sofia reads; the limiter guards public instances, and this one is private.
  cat >"$CONFIG/settings.yml" <<EOF
use_default_settings: true
server:
  secret_key: "$(head -c 32 /dev/urandom | base64 | tr -dc 'A-Za-z0-9')"
  limiter: false
  image_proxy: false
search:
  formats:
    - html
    - json
EOF
fi

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --restart unless-stopped -p "127.0.0.1:$PORT:8080" \
  -v "$CONFIG:/etc/searxng" "$IMAGE" >/dev/null

for _ in $(seq 60); do
  if curl -fsS -o /dev/null "http://127.0.0.1:$PORT/search?q=test&format=json" 2>/dev/null; then
    echo "SearXNG is up on 127.0.0.1:$PORT — Sofia searches through it now."
    exit 0
  fi
  sleep 1
done
echo "SearXNG did not answer; see: docker logs $NAME" >&2
exit 1
