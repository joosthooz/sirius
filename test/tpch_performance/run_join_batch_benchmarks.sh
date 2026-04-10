#!/usr/bin/env bash
# Run join_batch_benchmark for every recipe in extract_join_data.py against extracted parquets.
#
# Usage (repo root, after extract + build):
#   python3 test/tpch_performance/extract_join_data.py 10 --python
#   cmake --build build/release --target join_batch_benchmark
#   ./test/tpch_performance/run_join_batch_benchmarks.sh 10
#
# Optional: second arg = output CSV path (default: test/tpch_performance/join_batch_benchmark_sf<SF>.csv)
#
# Batch sizes default to 64MiB … 4GiB (powers of two). Override with env BATCH_SIZES
# (space-separated, e.g. BATCH_SIZES="64M 128M" ./run_join_batch_benchmarks.sh 10).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

SF="${1:?usage: $0 <scale_factor> [out.csv]}"
OUT_CSV="${2:-$ROOT/test/tpch_performance/join_batch_benchmark_sf${SF}.csv}"
DATA="$ROOT/test_datasets/join_extracts_sf${SF}"
BIN="$ROOT/build/release/extension/sirius/test/cpp/join_batch_benchmark"

# 64 MiB through 4 GiB, doubling each step (binary suffixes per join_batch_benchmark).
DEFAULT_BATCH_SIZES=(64M 128M 256M 512M 1G 2G 4G)
if [[ -n "${BATCH_SIZES:-}" ]]; then
  read -r -a BATCH_ARGS <<< "${BATCH_SIZES}"
else
  BATCH_ARGS=("${DEFAULT_BATCH_SIZES[@]}")
fi
BATCH_FLAGS=()
for b in "${BATCH_ARGS[@]}"; do
  BATCH_FLAGS+=(--batch-bytes "$b")
done

if [[ ! -x "$BIN" ]]; then
  echo "missing binary: $BIN (build join_batch_benchmark)" >&2
  exit 1
fi
if [[ ! -d "$DATA" ]]; then
  echo "missing extract dir: $DATA (run extract_join_data.py)" >&2
  exit 1
fi

rm -f "$OUT_CSV"
RECIPES="$(python3 "$ROOT/test/tpch_performance/extract_join_data.py" --list-names)"
if [[ -z "$RECIPES" ]]; then
  echo "no recipes from extract_join_data.py --list-names" >&2
  exit 1
fi

for name in $RECIPES; do
  meta="$DATA/${name}_meta.json"
  build="$DATA/${name}_build.parquet"
  probe="$DATA/${name}_probe.parquet"
  if [[ ! -f "$meta" || ! -f "$build" || ! -f "$probe" ]]; then
    echo "skip $name (missing parquet under $DATA)" >&2
    continue
  fi
  key_cols="$(python3 -c "import json; print(json.load(open('''$meta'''))['num_key_columns'])")"
  echo "=== $name key_cols=$key_cols batches=${BATCH_ARGS[*]} ==="
  "$BIN" \
    --join-name "$name" \
    --build "$build" \
    --probe "$probe" \
    --key-cols "$key_cols" \
    --iterations 2 \
    --warmup 1 \
    "${BATCH_FLAGS[@]}" \
    --csv "$OUT_CSV"
done

echo "Wrote $OUT_CSV"
