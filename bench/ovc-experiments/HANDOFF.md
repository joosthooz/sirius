# Handoff: offset value codes and sorted-input fast paths

State as of 2026-10-27, measured on an RTX 3500 Ada laptop GPU (sm_89, 12 GB). Work continues on
GB300.

## The question

Can sortedness, and offset value codes (OVC) on top of it, make cuDF's joins and group-bys
faster on TPC-H-shaped data? Lineitem stored sorted by `l_orderkey` is the motivating case
(Q3, Q18, and joins of lineitem with orders).

The codes: for rows sorted on keys `(k0, k1, ...)`, a row's code is the index of the first key
column where it differs from the previous row (0 = a new `k0` run). Run heads, group boundaries
and "can this lookup be skipped" all fall out of the code.

## Where everything is

| What | Where |
|---|---|
| Sirius branch | `joosthooz/sirius` `perf/ovc-sorted-joins-cudf2612` |
| Patched cuDF | `joosthooz/cudf` `sirius/ovc-experiments` (commit `e2d8469573`, on top of `01a7871610`) |
| Benchmark | `bench/ovc-experiments/ovc_bench.cu`, built by `build.sh` |
| cuDF join benchmark (Catch2, `[!benchmark]`) | `test/cpp/operator/bench_sorted_join.cpp` |

The branch is on cuDF nightly `26.12.00a568`, whose build string `01a78716` is the commit the
cuDF patches sit on. The Sirius commits under the benchmark already use sortedness: natural-order
parquet pinning, cuDF's sorted-keys group-by for sorted batches, order-preserving partitions and
merges, and pin-time proof of unique join keys.

The raw result logs from the laptop are gone; the numbers below are from reruns on 2026-10-27.

## The benchmark

`ovc_bench` generates its own data: 16.7M orders, and 60.8M lineitem rows (about 3.6 per order),
sorted by `(l_orderkey, l_linenumber)`. It needs no dataset. Every variant validates against a
reference (row counts, key and value checksums) and prints `ok`.

- `[join]`: cuDF `distinct_hash_join`, `hash_join` and `sort_merge_join` against custom sorted
  kernels (binary search per probe row, a merge walk with 8 rows per thread, and searching only
  run heads, found from codes or by comparing keys). Scenarios: full orders build, a 50%-filtered
  build, and a shuffled probe.
- `[agg]`: Q18-shaped (`GROUP BY l_orderkey`, `SUM(l_quantity)`) and Q3-shaped (three keys) group-bys.
  These compare cuDF hash against cuDF sorted with and without an `is_sorted` check, OVC labels
  fed to cuDF, a thread-per-run sum over OVC runs, and CUB `reduce_by_key`.
- `[ovc]`: what computing the codes costs, and recomputing them against carrying them through a
  50% filter.

## The cuDF patches

Each fast path is enabled by a bit of `CUDF_EXP_SORTED_FASTPATH`. With the variable unset, cuDF
behaves as stock. Use `31` for all of them.

| Bit | Change |
|---|---|
| 1 | `sort_merge_join`: typed binary-search probe for one non-null integer or timestamp key |
| 2 | sort group-by: pass "keys presorted" through to the aggregation, which skips the identity gather. A plain bug fix, upstreamable as is |
| 4 | sort group-by: typed group-boundary detection for presorted fixed-width keys |
| 8 | sort group-by: SUM straight over group offsets (falls back for nulls, non-numeric types, huge groups) |
| 16 | `sort_merge_join`: direct inner/left output when build keys are unique |

cuDF's own tests pass with all bits on: the 4429 group-by tests, and the join suite, which runs
`sort_merge_join` for every inner and left case. Two join tests try to allocate 20 GB and fail on
a 12 GB GPU either way.

## Results (laptop, ms)

| Operation | cuDF nightly | patched cuDF | best custom kernel |
|---|---:|---:|---:|
| Sort-merge inner join, full build | 89.4 | 15.4 | 5.2 (merge walk) |
| Sort-merge left join, full build | 89.3 | 12.3 | — |
| Sort-merge inner join, filtered build | 82.7 | 11.3 | 5.5 |
| Sort-merge join, shuffled probe | 286 | 134 | `distinct_hash_join` 57–60 |
| Q18 hash group-by | 34.4 | 33.9 | — |
| Q18 sorted group-by, no check | 18.1 | 5.7 | 4.7 (per-run sum) |
| Q18 sorted group-by with `is_sorted` | 20.6 | 8.5 | — |
| Q3 hash group-by | 43.5 | 42.3 | — |
| Q3 sorted group-by, no check | 24.2 | 11.1 | 7.4 (per-run sum) |
| Q3 sorted group-by with `is_sorted` | 31.6 | 19.3 | — |

For comparison, the hash joins on the full build: `distinct_hash_join` 25.2 ms, `hash_join` 40.6 ms.

OVC costs on 60.8M rows: computing codes on two keys takes 2.4 ms. After a 50% filter,
recomputing them takes 5.0 ms and carrying them through takes 1.6 ms.

## What we learned

- **cuDF is slow on sorted input because of generic machinery, not the algorithm.**
  - Sort-merge join spent 91% of its time in a binary search through the generic row comparator.
  - Sorted group-by still gathered through an identity order, and found boundaries with that
    same comparator.
  - Typed fast paths make the patched `sort_merge_join` the fastest cuDF join for sorted
    primitive keys.
- **For joins, sortedness is enough; codes add nothing.**
  - Searching only run heads (8.3–8.9 ms) loses to the plain merge walk (5.2 ms), and a probe
    that gallops from the previous match was slower than plain typed search, so it was dropped.
  - The patched join emits rows in probe order, so sortedness survives a chain of joins.
- **Shuffled probes favour hash joins.** Use the sort-merge join only when the probe is known to
  be sorted.
- **The `is_sorted` check is now the bottleneck**, about 3 of 9 ms for Q18 and 8 of 19 ms for
  Q3 after patching. Knowing sortedness from metadata (pin time, carried through operators) is
  worth more than any further kernel work.
- **Where codes still pay off:**
  - Free group boundaries on multi-column keys. On stock cuDF, OVC labels cut Q3's sorted
    group-by with check from 31.6 to 24.1 ms, but once bit 4 is in, the labels gain nothing over
    the typed boundary detection.
  - Expensive keys, multi-column or string, where comparing adjacent rows costs the most.
  - Merging sorted runs.
  - Carrying codes through a filter is about 3x cheaper than recomputing them.
- **Headroom left:** the custom kernels beat patched cuDF by 1.2–3x (join 5.2 vs 15.4 ms; Q18
  4.7 vs 5.7 ms; Q3 7.4 vs 11.1 ms).

## Getting it running on GB300 (aarch64, Grace)

1. Get the Sirius branch and install its environment:

   ```bash
   git fetch joost && git checkout perf/ovc-sorted-joins-cudf2612
   git submodule update --init --recursive
   pixi install
   ```

   The lockfile covers `linux-aarch64`. If `pixi install` 404s, the nightlies have rotated out:
   bump `libcudf`/`libcuvs` in `pixi.toml` and run `pixi update` on the `rapidsai-nightly`
   packages (`librmm`, `libkvikio`, `libraft-headers*`, `libucxx`, `ucxx`, `rapids-logger`,
   `libcuvs-headers`). A failed `pixi run` can leave the environment without `libcudf`, so check
   with `pixi install` before you `pixi run` anything.
2. Build and run the benchmark against the nightly. `build.sh` picks the GPU architecture from
   `nvidia-smi` and the aarch64 paths from `uname -m`:

   ```bash
   pixi run bench/ovc-experiments/build.sh
   pixi run build/ovc-experiments/ovc_bench
   ```

   The data is sized for a 12 GB GPU. On GB300, scale the row counts in `ovc_bench.cu` up to SF100
   or larger shapes, so the timings aren't dominated by launch overhead.
3. Rebuild the patched cuDF for GB300. The laptop build was for sm_89 on x86_64.

   ```bash
   git clone git@github.com:joosthooz/cudf.git && cd cudf && git checkout sirius/ovc-experiments
   pixi run --manifest-path <sirius>/pixi.toml bash -c '
     cmake -S cpp -B cpp/build-sirius -G Ninja -DCMAKE_BUILD_TYPE=Release \
       -DCMAKE_CUDA_ARCHITECTURES=NATIVE -DBUILD_TESTS=OFF -DBUILD_BENCHMARKS=OFF \
       -DCUDF_BUILD_STREAMS_TEST_UTIL=OFF -DCUDF_BUILD_TESTUTIL=OFF \
       -DCMAKE_PREFIX_PATH=$CONDA_PREFIX \
       -DCMAKE_CUDA_HOST_COMPILER=$CONDA_PREFIX/bin/aarch64-conda-linux-gnu-c++ \
       "-DCMAKE_CXX_FLAGS=-isystem <nvjitlink>/include -DCCCL_IGNORE_DEPRECATED_STREAM_REF_HEADER" \
       "-DCMAKE_CUDA_FLAGS=-isystem <nvjitlink>/include -DCCCL_IGNORE_DEPRECATED_STREAM_REF_HEADER" &&
     ninja -C cpp/build-sirius cudf'
   ```

   The Sirius environment lacks the nvJitLink headers that cuDF's build needs. On the laptop,
   the `libnvjitlink-dev` conda package matching the environment's CUDA version was unpacked
   into `~/workspace/cuda/cudf-side-deps/pkg`, outside the environment, so `pixi.toml` stays
   untouched. On aarch64, take the `linux-aarch64` package; `<nvjitlink>` is then
   `.../pkg/targets/sbsa-linux`.
4. Run against the patched library. `build.sh` links with RUNPATH, so `LD_LIBRARY_PATH` takes
   precedence:

   ```bash
   CUDF_EXP_SORTED_FASTPATH=31 LD_LIBRARY_PATH=<cudf>/cpp/build-sirius \
     pixi run build/ovc-experiments/ovc_bench
   ```

   To see what each change contributes, also run with single bits (1, 2, 4, 8, 16) and with the
   variable unset.

## Next steps

1. **Re-measure on GB300.** Check whether the 2–7x gains of the patched paths over the nightly
   hold, and whether the custom-kernel headroom grows or shrinks with HBM bandwidth.
2. **Upstream bit 2 to cuDF as a bug fix.** It's small and independent.
3. **Turn bits 1, 4, 8 and 16 into an upstream cuDF PR.** None of them needs a new API.
   - Fix the shuffled-probe case first, or document it: the patched sort-merge join still loses
     to `distinct_hash_join` there.
   - Close the gap to the custom kernels before proposing.
4. **In Sirius, take sortedness from metadata instead of `is_sorted`.**
   - Use the patched `sort_merge_join` when the build keys are unique and sorted and the probe is
     sorted; the pin-time uniqueness proof is already on this branch.
   - There is a local-only experiment that carries sortedness and value ranges on batches through
     a cuCascade properties slot (`feat/batch-data-properties`, on the laptop, not pushed). It
     gave a 21% faster Q18-style group-by on pinned SF10 lineitem.
   - The limit is joins: cuDF's hash joins don't keep probe order, so orderings die at every hash
     join. The patched sort-merge join does keep it.
5. **Prototype offset value codes where they can still win** (requested, not started):
   - Pass precomputed group boundaries into cuDF's sorted group-by. This needs a new cuDF API, or
     a Sirius kernel like the thread-per-run sum.
   - Multi-column and string keys, where boundary detection is expensive. Add a string-key
     scenario to `ovc_bench`.
   - A merge of sorted runs (spilled sorted batches, out-of-core sort) that compares codes instead
     of keys.
