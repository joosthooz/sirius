#!/usr/bin/env python3
# =============================================================================
# Copyright 2025, Sirius Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License.
# =============================================================================
"""Rank TPC-H queries for join-focused benchmarking using run timings + Sirius logs + nsys.

Combines:
  * Wall-clock timings from ``runs/<...>/timings.csv`` or per-query
    ``runs/.../sirius/q<N>/timings.csv`` produced by benchmark_and_validate /
    run_tpch_parquet.
  * **Hash-join execution time from Sirius trace logs** (``SIRIUS_LOG_LEVEL=trace``):
    sums ``execution time: … ms`` on ``operator HASH_JOIN (id=…) produced`` lines in
    ``runs/.../sirius/q<N>/sirius.log`` (every task completion, e.g. cold+warm).
    ``hash_join_log_pct_of_wall`` can exceed 100% when joins run in parallel across
    pipelines (sum of task times > single-thread wall clock).
  * Optional Nsight Systems SQLite exports (``q<N>.sqlite``), aggregating NVTX time in
    ``sirius_physical_hash_join::execute``.

Typical workflow
----------------
1. Collect timings from a recent benchmark run directory under ``runs/``.
2. Use the same run directory so per-query ``sirius/qN/sirius.log`` files are parsed for
   join GPU task execution times (trace logging required).
3. Optionally profile join-heavy queries::

     export SIRIUS_CONFIG_FILE=$(pwd)/test/cpp/integration/integration.cfg
     ./test/tpch_performance/profile_join_queries.sh 100

4. Point this script at the run dir and optionally nsys::

     python3 test/tpch_performance/rank_tpch_joins.py \\
       --runs-dir runs/2026-03-05_13-28-45_sf10_4iter \\
       --nsys-dir nsys_profiles/sf100

``--runs-dir`` is also used automatically to find ``sirius/q*/sirius.log`` when present.
Use ``--sirius-run-dir`` only to override the log root.

Writes ``test/tpch_performance/join_query_ranking.csv`` by default.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# gpu_pipeline_task trace: ``produced`` line with ``size: … execution time: … ms``
_PRODUCED_RE = re.compile(
    r"Pipeline (\d+): operator (\w+) \(id=(\d+)\) produced (\d+) batches, num rows?: "
    r"([\d\s]+?)\s*, size:.*?execution time: ([\d.]+) ms"
)


@dataclass
class HashJoinLogStats:
    """Aggregated HASH_JOIN ``produced`` execution times from one query log."""

    total_exec_ms: float = 0.0
    num_tasks: int = 0
    max_single_task_ms: float = 0.0


def parse_hash_join_exec_from_log(log_text: str) -> HashJoinLogStats:
    """Sum / count / max of ``execution time`` on HASH_JOIN produced lines."""
    st = HashJoinLogStats()
    for mo in _PRODUCED_RE.finditer(log_text):
        if mo.group(2) != "HASH_JOIN":
            continue
        ms = float(mo.group(6))
        st.total_exec_ms += ms
        st.num_tasks += 1
        st.max_single_task_ms = max(st.max_single_task_ms, ms)
    return st


def collect_hash_join_log_stats_run_dir(run_dir: Path) -> Dict[int, HashJoinLogStats]:
    """Load ``sirius/qN/sirius.log`` under a benchmark run directory."""
    out: Dict[int, HashJoinLogStats] = {}
    sirius_root = run_dir / "sirius"
    if not sirius_root.is_dir():
        return out
    for qn in range(1, 23):
        log_path = sirius_root / f"q{qn}" / "sirius.log"
        if not log_path.is_file():
            continue
        try:
            text = log_path.read_text(errors="replace")
        except OSError:
            continue
        st = parse_hash_join_exec_from_log(text)
        if st.num_tasks > 0:
            out[qn] = st
    return out


# High-impact joins from TPC-H structure (used when filling recommendations).
KNOWN_CRITICAL_JOINS = [
    (
        "lineitem × orders (l_orderkey = o_orderkey)",
        "Q3, Q5, Q10, Q12, Q18, Q21",
    ),
    (
        "lineitem × part (l_partkey = p_partkey)",
        "Q14, Q17, Q19",
    ),
    (
        "lineitem × supplier (l_suppkey = s_suppkey)",
        "Q5, Q7, Q9, Q21",
    ),
    (
        "partsupp × lineitem (composite keys)",
        "Q9",
    ),
    (
        "orders × customer (o_custkey = c_custkey)",
        "Q3, Q5, Q10, Q13, Q18",
    ),
]


@dataclass
class QueryTimings:
    query: int
    sirius_warm_s: Optional[float] = None
    sirius_cold_s: Optional[float] = None
    run_label: str = ""


def _parse_query_from_path(path: str) -> Optional[int]:
    m = re.search(r"/q(\d+)/", path.replace("\\", "/"))
    if m:
        return int(m.group(1))
    m = re.search(r"_q(\d+)\.csv$", path)
    if m:
        return int(m.group(1))
    return None


def _warm_from_step_csv(path: Path) -> Tuple[Optional[float], Optional[float]]:
    """Return (cold iter_1, best warm) from a timings.csv with step,runtime_s."""
    cold = None
    warm_vals: List[float] = []
    try:
        with path.open(newline="") as f:
            r = csv.DictReader(f)
            if not r.fieldnames:
                return None, None
            # Accept step,name or single column names from different writers
            step_key = "step" if "step" in r.fieldnames else r.fieldnames[0]
            time_key = (
                "runtime_s"
                if "runtime_s" in r.fieldnames
                else r.fieldnames[1] if len(r.fieldnames) > 1 else None
            )
            if time_key is None:
                return None, None
            for row in r:
                step = row.get(step_key, "")
                try:
                    t = float(row[time_key])
                except (TypeError, ValueError):
                    continue
                if step == "iter_1":
                    cold = t
                elif step.startswith("iter_") and step != "iter_1":
                    warm_vals.append(t)
    except OSError:
        return None, None
    best_warm = min(warm_vals) if warm_vals else None
    return cold, best_warm


def collect_sirius_timings_runs_dir(runs_dir: Path) -> Dict[int, QueryTimings]:
    """Load Sirius warm times from a single benchmark run directory."""
    out: Dict[int, QueryTimings] = {}
    label = runs_dir.name

    combined = runs_dir / "timings.csv"
    if combined.is_file():
        by_q: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        with combined.open(newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                if row.get("engine", "").lower() != "sirius":
                    continue
                qraw = row.get("query", "")
                m = re.match(r"Q?(\d+)", qraw, re.I)
                if not m:
                    continue
                qn = int(m.group(1))
                try:
                    it = int(row["iteration"])
                    t = float(row["runtime_s"])
                except (KeyError, ValueError):
                    continue
                by_q[qn].append((it, t))
        for qn, pairs in by_q.items():
            cold_vals = [t for it, t in pairs if it == 1]
            warm_vals = [t for it, t in pairs if it > 1]
            out[qn] = QueryTimings(
                query=qn,
                sirius_cold_s=cold_vals[0] if cold_vals else None,
                sirius_warm_s=min(warm_vals) if warm_vals else None,
                run_label=label,
            )
        return out

    # Per-query sirius/qN/timings.csv
    pattern = str(runs_dir / "sirius" / "q*" / "timings.csv")
    for p in glob.glob(pattern):
        qn = _parse_query_from_path(p)
        if qn is None:
            continue
        cold, warm = _warm_from_step_csv(Path(p))
        out[qn] = QueryTimings(
            query=qn,
            sirius_warm_s=warm,
            sirius_cold_s=cold,
            run_label=label,
        )
    return out


def collect_timings_from_glob(runs_glob: str) -> Dict[int, QueryTimings]:
    """Merge timings from multiple run directories (best warm per query)."""
    merged: Dict[int, QueryTimings] = {}
    for raw in sorted(glob.glob(runs_glob)):
        p = Path(raw)
        if not p.is_dir():
            continue
        part = collect_sirius_timings_runs_dir(p)
        for qn, qt in part.items():
            if qn not in merged:
                merged[qn] = qt
            else:
                if qt.sirius_warm_s is not None:
                    if (
                        merged[qn].sirius_warm_s is None
                        or qt.sirius_warm_s < merged[qn].sirius_warm_s
                    ):
                        merged[qn].sirius_warm_s = qt.sirius_warm_s
                        merged[qn].run_label = qt.run_label
    return merged


def nvtx_hash_join_seconds(sqlite_path: Path) -> Optional[float]:
    """Sum duration (seconds) of Sirius hash-join NVTX ranges in one nsys sqlite DB."""
    try:
        con = sqlite3.connect(str(sqlite_path))
    except sqlite3.Error:
        return None
    try:
        cur = con.execute(
            """
            SELECT name FROM sqlite_master WHERE type='table' AND name='NVTX_EVENTS'
            """
        )
        if cur.fetchone() is None:
            return None
        row = con.execute(
            """
            SELECT ROUND(SUM(end - start) / 1e9, 6)
            FROM NVTX_EVENTS
            WHERE domainId = 0 AND eventType = 59 AND end > start
              AND text LIKE '%sirius_physical_hash_join%'
            """
        ).fetchone()
        if not row or row[0] is None:
            return 0.0
        return float(row[0])
    except sqlite3.Error:
        return None
    finally:
        con.close()


def collect_nsys_nvtx(nsys_dir: Path) -> Dict[int, float]:
    """Map query number -> hash join NVTX seconds from q<N>.sqlite files."""
    out: Dict[int, float] = {}
    for p in sorted(nsys_dir.glob("q*.sqlite")):
        m = re.match(r"q(\d+)\.sqlite$", p.name, re.I)
        if not m:
            continue
        qn = int(m.group(1))
        t = nvtx_hash_join_seconds(p)
        if t is not None:
            out[qn] = t
    return out


def write_csv(
    path: Path,
    rows: Iterable[dict],
    fieldnames: List[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("Typical workflow")[0].strip()
    )
    ap.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repo root (default: parent of test/)",
    )
    ap.add_argument(
        "--runs-dir",
        type=Path,
        action="append",
        default=[],
        help="Benchmark run directory (timings). May be repeated.",
    )
    ap.add_argument(
        "--runs-glob",
        action="append",
        default=[],
        help="Glob of run dirs, e.g. runs/*_sf10_* (merged, best warm per query).",
    )
    ap.add_argument(
        "--nsys-dir",
        type=Path,
        default=None,
        help="Directory with q<N>.sqlite from profile_tpch_nsys.sh",
    )
    ap.add_argument(
        "--sirius-run-dir",
        type=Path,
        default=None,
        help="Benchmark run dir containing sirius/qN/sirius.log (trace). "
        "If omitted, each --runs-dir / runs-glob match is scanned for logs.",
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: test/tpch_performance/join_query_ranking.csv)",
    )
    args = ap.parse_args()
    root = args.project_root
    out_csv = args.output or (root / "test/tpch_performance/join_query_ranking.csv")

    timings: Dict[int, QueryTimings] = {}
    for d in args.runs_dir:
        if not d.is_dir():
            print(f"WARNING: runs-dir not a directory: {d}", file=sys.stderr)
            continue
        part = collect_sirius_timings_runs_dir(d.resolve())
        for qn, qt in part.items():
            if qn not in timings:
                timings[qn] = qt
            else:
                if qt.sirius_warm_s is not None:
                    if (
                        timings[qn].sirius_warm_s is None
                        or qt.sirius_warm_s < timings[qn].sirius_warm_s
                    ):
                        timings[qn].sirius_warm_s = qt.sirius_warm_s
                        timings[qn].run_label = qt.run_label
    for g in args.runs_glob:
        merged = collect_timings_from_glob(os.path.join(root, g))
        for qn, qt in merged.items():
            if qn not in timings:
                timings[qn] = qt
            else:
                if qt.sirius_warm_s is not None:
                    if (
                        timings[qn].sirius_warm_s is None
                        or qt.sirius_warm_s < timings[qn].sirius_warm_s
                    ):
                        timings[qn].sirius_warm_s = qt.sirius_warm_s
                        timings[qn].run_label = qt.run_label

    join_stats: Dict[int, HashJoinLogStats] = {}
    for d in args.runs_dir:
        if not d.is_dir():
            continue
        for qn, st in collect_hash_join_log_stats_run_dir(d.resolve()).items():
            join_stats[qn] = st
    for g in args.runs_glob:
        for raw in sorted(glob.glob(os.path.join(root, g))):
            p = Path(raw)
            if not p.is_dir():
                continue
            for qn, st in collect_hash_join_log_stats_run_dir(p.resolve()).items():
                join_stats[qn] = st
    if args.sirius_run_dir and args.sirius_run_dir.is_dir():
        for qn, st in collect_hash_join_log_stats_run_dir(
            args.sirius_run_dir.resolve()
        ).items():
            join_stats[qn] = st

    nsyx: Dict[int, float] = {}
    if args.nsys_dir and args.nsys_dir.is_dir():
        nsyx = collect_nsys_nvtx(args.nsys_dir.resolve())

    if not timings and not nsyx and not join_stats:
        print(
            "No data: pass --runs-dir / --runs-glob and/or --nsys-dir "
            "(and ensure sirius/qN/sirius.log exists with SIRIUS_LOG_LEVEL=trace).\n"
            "Example:\n"
            "  python3 test/tpch_performance/rank_tpch_joins.py "
            "--runs-glob 'runs/*_sf10_*' --nsys-dir nsys_profiles/sf100",
            file=sys.stderr,
        )
        return 1

    known = set(timings.keys()) | set(nsyx.keys()) | set(join_stats.keys())
    all_q = sorted(known if known else set(range(1, 23)))
    rows = []
    for qn in all_q:
        t = timings.get(qn)
        warm = t.sirius_warm_s if t else None
        cold = t.sirius_cold_s if t else None
        hj = nsyx.get(qn)
        frac_nvtx = None
        if hj is not None and warm and warm > 0:
            frac_nvtx = round(100.0 * hj / warm, 2)

        jst = join_stats.get(qn)
        log_total_ms = jst.total_exec_ms if jst else None
        log_n = jst.num_tasks if jst else None
        log_max = jst.max_single_task_ms if jst else None
        log_avg = (jst.total_exec_ms / jst.num_tasks) if jst and jst.num_tasks else None
        log_sum_s = (log_total_ms / 1000.0) if log_total_ms is not None else None
        frac_log = None
        if log_sum_s is not None and warm and warm > 0:
            frac_log = round(100.0 * log_sum_s / warm, 2)

        rows.append(
            {
                "query": f"Q{qn}",
                "sirius_warm_s": f"{warm:.4f}" if warm is not None else "",
                "sirius_cold_s": f"{cold:.4f}" if cold is not None else "",
                "hash_join_log_total_exec_ms": (
                    f"{log_total_ms:.2f}" if log_total_ms is not None else ""
                ),
                "hash_join_log_num_tasks": str(log_n) if log_n is not None else "",
                "hash_join_log_max_task_ms": (
                    f"{log_max:.2f}" if log_max is not None else ""
                ),
                "hash_join_log_avg_task_ms": (
                    f"{log_avg:.2f}" if log_avg is not None else ""
                ),
                "hash_join_log_sum_exec_s": (
                    f"{log_sum_s:.4f}" if log_sum_s is not None else ""
                ),
                "hash_join_log_pct_of_wall": (
                    f"{frac_log}" if frac_log is not None else ""
                ),
                "hash_join_nvtx_s": f"{hj:.4f}" if hj is not None else "",
                "hash_join_pct_of_wall": (
                    f"{frac_nvtx}" if frac_nvtx is not None else ""
                ),
                "run_label": t.run_label if t else "",
            }
        )

    def sort_key(r: dict) -> Tuple[float, float, float]:
        w = float(r["sirius_warm_s"]) if r["sirius_warm_s"] else 0.0
        log_ms = (
            float(r["hash_join_log_total_exec_ms"])
            if r["hash_join_log_total_exec_ms"]
            else 0.0
        )
        h = float(r["hash_join_nvtx_s"]) if r["hash_join_nvtx_s"] else 0.0
        return (-w, -log_ms, -h)

    rows.sort(key=sort_key)

    fields = [
        "query",
        "sirius_warm_s",
        "sirius_cold_s",
        "hash_join_log_total_exec_ms",
        "hash_join_log_num_tasks",
        "hash_join_log_max_task_ms",
        "hash_join_log_avg_task_ms",
        "hash_join_log_sum_exec_s",
        "hash_join_log_pct_of_wall",
        "hash_join_nvtx_s",
        "hash_join_pct_of_wall",
        "run_label",
    ]
    write_csv(out_csv, rows, fields)

    print(f"Wrote {out_csv}")
    print("\nTop queries by Sirius warm time (when timings present):")
    for r in rows[:8]:
        if r["sirius_warm_s"]:
            log_s = r["hash_join_log_sum_exec_s"]
            log_part = f"{log_s} s" if log_s else "n/a"
            print(
                f"  {r['query']}: warm={r['sirius_warm_s']}s "
                f"log_join_sum={log_part} "
                f"hash_join_nvtx={r['hash_join_nvtx_s'] or 'n/a'} "
                f"({r['hash_join_pct_of_wall'] or 'n/a'}% of wall)"
            )

    by_log = sorted(
        [r for r in rows if r["hash_join_log_total_exec_ms"]],
        key=lambda r: float(r["hash_join_log_total_exec_ms"]),
        reverse=True,
    )
    if by_log:
        print("\nTop queries by HASH_JOIN task execution time (from Sirius trace log):")
        for r in by_log[:8]:
            print(
                f"  {r['query']}: total={r['hash_join_log_total_exec_ms']} ms "
                f"({r['hash_join_log_num_tasks']} tasks, "
                f"max_single={r['hash_join_log_max_task_ms']} ms) "
                f"~{r['hash_join_log_pct_of_wall'] or 'n/a'}% of warm wall"
            )

    print("\nKnown high-impact join patterns (for extract_join_data / micro-bench):")
    for desc, qs in KNOWN_CRITICAL_JOINS:
        print(f"  - {desc}: {qs}")

    if not join_stats:
        print(
            "\n(No Sirius HASH_JOIN trace lines.) Use SIRIUS_LOG_LEVEL=trace and "
            "ensure sirius/qN/sirius.log exists under the run directory.",
            file=sys.stderr,
        )
    if not nsyx:
        print(
            "\n(No nsys data.) Profile join-heavy queries with:\n"
            "  export SIRIUS_CONFIG_FILE=$(pwd)/test/cpp/integration/integration.cfg\n"
            "  ./test/tpch_performance/profile_join_queries.sh <scale_factor> 3 5 7 8 9 10 17 21",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
