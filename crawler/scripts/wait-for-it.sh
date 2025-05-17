#!/usr/bin/env bash
#   Use this script to test if a given TCP host/port are available
#   Usage: wait-for-it.sh host:port -- command args

set -e

HOSTPORT="$1"
shift

HOST="${HOSTPORT%%:*}"
PORT="${HOSTPORT##*:}"

TIMEOUT=60
START_TIME=$(date +%s)

while :; do
  if nc -z "$HOST" "$PORT"; then
    echo "Service $HOST:$PORT is up!"
    break
  fi
  NOW=$(date +%s)
  if [ $((NOW - START_TIME)) -ge $TIMEOUT ]; then
    echo "Timeout waiting for $HOST:$PORT" >&2
    exit 1
  fi
  echo "Waiting for $HOST:$PORT..."
  sleep 2
done

exec "$@" 