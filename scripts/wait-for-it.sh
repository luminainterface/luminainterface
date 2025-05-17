#!/bin/bash
# wait-for-it.sh script to wait for a host:port to be available

set -e

hostport="$1"
shift
cmd="$@"

host="${hostport%%:*}"
port="${hostport##*:}"

until nc -z "$host" "$port"; do
  >&2 echo "Waiting for $host:$port to be available..."
  sleep 1
done

>&2 echo "$host:$port is up - executing command"
exec $cmd 