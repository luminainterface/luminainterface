#!/usr/bin/env bash
set -euo pipefail

url="http://crawler:8400/metrics"
echo "🔍 hitting $url"
body=$(curl -sf "$url")

grep -q "crawler_pages_total"   <<< "$body" || { echo "Missing pages_total";   exit 1; }
grep -q "crawler_errors_total"  <<< "$body" || { echo "Missing errors_total";  exit 1; }
grep -q "crawler_fetch_seconds" <<< "$body" || { echo "Missing histogram";     exit 1; }
grep -q "crawler_queue_depth"   <<< "$body" || { echo "Missing queue_depth";   exit 1; }
grep -q "crawler_response_bytes" <<< "$body" || { echo "Missing bytes_pulled"; exit 1; }
grep -q "crawler_http_status"   <<< "$body" || { echo "Missing http_status";   exit 1; }

echo "✅ crawler metrics present" 