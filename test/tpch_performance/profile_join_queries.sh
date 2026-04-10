#!/usr/bin/env bash
# Targeted Nsight Systems capture for TPC-H queries that are typically join-heavy.
#
# Usage (from repo root):
#   export SIRIUS_CONFIG_FILE=$(pwd)/test/cpp/integration/integration.cfg
#   ./test/tpch_performance/profile_join_queries.sh <scale_factor> [query_numbers...]
#
# Default query set: 3 5 9 21 (override by listing numbers after scale factor).
# Join-heavy extras used by extract_join_data recipes: 7 8 10 17 (e.g. "$0 10 3 5 7 8 9 10 17 21").
# Forwards to profile_tpch_nsys.sh (same env vars: PARQUET_DIR, OUTPUT_DIR, etc.).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <scale_factor> [query_numbers...]" >&2
    echo "  Example: $0 100" >&2
    echo "  Example: $0 100 3 5 9 21" >&2
    exit 1
fi

SF="$1"
shift
if [ $# -eq 0 ]; then
    set -- 3 5 9 21
fi

exec bash "$SCRIPT_DIR/profile_tpch_nsys.sh" "$SF" "$@"
