#!/usr/bin/env python3
# =============================================================================
# Copyright 2025, Sirius Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License.
# =============================================================================
"""Build TPC-H join catalog CSV (DuckDB physical plan @ SF10 + optional Sirius log).

Uses DuckDB built-in TPC-H generator (``CALL dbgen(sf=10)``) and ``EXPLAIN (FORMAT JSON)``
to list HASH_JOIN / *DELIM* / NESTED_LOOP_JOIN nodes with join type, conditions,
and human-readable left/right inputs.

Optional Sirius fields (filled when ``--sirius-log`` points at a log from a single-query
or multi-query run with ``SIRIUS_LOG_LEVEL=trace`` for per-operator ``produced`` lines,
and ``SIRIUS_LOG_LEVEL=debug`` for BUILD_PROBE switches):

* ``sirius_join_mode``: BUILD_PROBE if a debug line matches the mapped operator id,
  else MIXED_JOIN if join conditions contain a non-equality comparison, else STANDARD
  for HASH_JOIN rows (heuristic; delim rows use DELIM_JOIN).
* ``sirius_execute_task_count``: number of ``produced`` events for that operator id
  (execute completions) summed over the log spans attributed to each query.

Log attribution: pass ``--one-log-per-query`` if you concatenated 22 separate log files
in Q1..Q22 order; otherwise the script treats the whole log as applying to every query
(only useful for single-query logs).

Example::

    python3 test/tpch_performance/build_tpch_sf10_join_catalog.py \\
        -o test/tpch_performance/tpch_sf10_join_catalog.csv

    python3 test/tpch_performance/build_tpch_sf10_join_catalog.py \\
        -o out.csv --sirius-log /path/to/sirius_q5.trace.log --queries 5

"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Import repo TPC-H text (same as gpu runner)
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from queries import QUERIES  # noqa: E402

REAL_JOINS = frozenset(
    {
        "HASH_JOIN",
        "NESTED_LOOP_JOIN",
        "BLOCKWISE_NL_JOIN",
        "CROSS_PRODUCT",
        "ASOF_JOIN",
        "PIECEWISE_MERGE_JOIN",
        "RIGHT_DELIM_JOIN",
        "LEFT_DELIM_JOIN",
        "MARK_JOIN",
    }
)


def _strip_name(n: str) -> str:
    return (n or "").strip()


def _summarize_scan(node: Dict[str, Any]) -> str:
    ex = node.get("extra_info") or {}
    if "Table" in ex:
        parts = [str(ex["Table"])]
        if ex.get("Filters"):
            parts.append(f"filters={ex['Filters']}")
        return " ".join(parts)
    return _strip_name(node.get("name", "?"))


def _describe_child(node: Dict[str, Any]) -> str:
    """Short label for the subtree under one side of a join."""
    n = _strip_name(node.get("name", ""))

    if n == "SEQ_SCAN":
        return _summarize_scan(node)

    if n == "COLUMN_DATA_SCAN":
        return "column_data_scan"

    if n == "FILTER":
        kids = node.get("children") or []
        if kids:
            inner = _describe_child(kids[0])
            ex = node.get("extra_info") or {}
            if isinstance(ex, dict) and ex:
                return f"FILTER({inner})"
            return f"FILTER({inner})"
        return "FILTER"

    if n in REAL_JOINS or "JOIN" in n:
        return f"<{n}>"

    if n == "PROJECTION":
        kids = node.get("children") or []
        if kids:
            return _describe_child(kids[0])
        return "PROJECTION"

    if n in ("HASH_GROUP_BY", "PERFECT_HASH_GROUP_BY", "PARTITION_BY", "WINDOW"):
        kids = node.get("children") or []
        if kids:
            return _describe_child(kids[0])
        return n

    kids = node.get("children") or []
    if len(kids) == 1:
        return _describe_child(kids[0])
    if not kids:
        return n or "?"
    return f"{n}({len(kids)} children)"


def _projection_width_near_join(
    join_node: Dict[str, Any], parent: Optional[Dict[str, Any]]
) -> Optional[int]:
    if parent is None:
        return None
    pn = _strip_name(parent.get("name", ""))
    if pn == "PROJECTION":
        ex = parent.get("extra_info") or {}
        projs = ex.get("Projections")
        if isinstance(projs, list):
            return len(projs)
    return None


def _collect_joins(
    node: Dict[str, Any],
    parent: Optional[Dict[str, Any]],
    out: List[Tuple[Dict[str, Any], Optional[Dict[str, Any]]]],
) -> None:
    n = _strip_name(node.get("name", ""))
    if n in REAL_JOINS:
        out.append((node, parent))
    for ch in node.get("children") or []:
        _collect_joins(ch, node, out)


def _condition_looks_mixed(conditions: str) -> bool:
    if not conditions:
        return False
    # Inequality join predicate (heuristic; ignores string literals roughly)
    s = conditions
    s = re.sub(r"'(?:[^'\\]|\\.)*'", "''", s)
    return bool(re.search(r"(?<![=<>])<=|>=|(?<!=)<|(?!=)>", s))


def _parse_build_probe_ids(log_text: str) -> Dict[int, str]:
    """operator_id -> 'BUILD_PROBE' from debug lines."""
    pat = re.compile(
        r"sirius_physical_hash_join id (\d+) switching to BUILD_PROBE mode", re.I
    )
    m: Dict[int, str] = {}
    for mo in pat.finditer(log_text):
        m[int(mo.group(1))] = "BUILD_PROBE"
    return m


# Matches gpu_pipeline_task trace line (note ``num rows: N  , size:`` before ``execution time``).
_PRODUCED_RE = re.compile(
    r"Pipeline (\d+): operator (\w+) \(id=(\d+)\) produced (\d+) batches, num rows?: "
    r"([\d\s]+?)\s*, size:.*?execution time: ([\d.]+) ms"
)


def _parse_operator_produced_counts(log_text: str) -> Dict[Tuple[int, int, str], int]:
    """(pipeline_id, operator_id, operator_name) -> produced event count."""
    counts: Dict[Tuple[int, int, str], int] = {}
    for mo in _PRODUCED_RE.finditer(log_text):
        pipe = int(mo.group(1))
        opname = mo.group(2)
        opid = int(mo.group(3))
        k = (pipe, opid, opname)
        counts[k] = counts.get(k, 0) + 1
    return counts


def _split_log_per_query(log_text: str) -> List[str]:
    """Split after each RESULT_COLLECTOR ``produced`` line (one query per chunk)."""
    chunks: List[str] = []
    last = 0
    for mo in _PRODUCED_RE.finditer(log_text):
        if mo.group(2) != "RESULT_COLLECTOR":
            continue
        chunks.append(log_text[last : mo.end()])
        last = mo.end()
    tail = log_text[last:]
    if tail.strip():
        chunks.append(tail)
    return [c for c in chunks if c.strip()]


def _map_sirius_hash_join_by_order(
    log_snippet: str, num_joins: int
) -> List[Optional[int]]:
    """First-seen HASH_JOIN operator ids in log order (``produced`` lines)."""
    seen: List[int] = []
    for mo in _PRODUCED_RE.finditer(log_snippet):
        if mo.group(2) != "HASH_JOIN":
            continue
        oid = int(mo.group(3))
        if oid not in seen:
            seen.append(oid)
    out: List[Optional[int]] = []
    for i in range(num_joins):
        out.append(seen[i] if i < len(seen) else None)
    return out


def _leaf_projection_count(node: Dict[str, Any]) -> int:
    n = _strip_name(node.get("name", ""))
    ex = node.get("extra_info") or {}
    if n == "SEQ_SCAN":
        projs = ex.get("Projections")
        return len(projs) if isinstance(projs, list) else 0
    kids = node.get("children") or []
    if len(kids) == 1:
        return _leaf_projection_count(kids[0])
    if n == "FILTER" and kids:
        return _leaf_projection_count(kids[0])
    if n == "PROJECTION" and kids:
        return _leaf_projection_count(kids[0])
    return 0


def _sirius_mode_for_hash_join(
    op_id: Optional[int], build_probe_ids: Dict[int, str], conditions: str
) -> str:
    if op_id is not None and op_id in build_probe_ids:
        return "BUILD_PROBE"
    if _condition_looks_mixed(conditions):
        return "MIXED_JOIN"
    return "STANDARD"


def build_rows_for_query(
    con: Any,
    qnum: int,
    sql: str,
    sirius_log: Optional[str],
    log_slice: Optional[str],
    map_order: bool,
) -> List[Dict[str, str]]:
    plan_raw = con.execute("EXPLAIN (FORMAT JSON) " + sql).fetchone()[1]
    plan = json.loads(plan_raw)
    root = plan[0]
    collected: List[Tuple[Dict[str, Any], Optional[Dict[str, Any]]]] = []
    _collect_joins(root, None, collected)

    build_probe_ids: Dict[int, str] = {}
    produced_counts: Dict[Tuple[int, int, str], int] = {}
    if log_slice:
        build_probe_ids = _parse_build_probe_ids(log_slice)
        produced_counts = _parse_operator_produced_counts(log_slice)

    num_hash_joins = sum(
        1 for j, _ in collected if _strip_name(j.get("name", "")) == "HASH_JOIN"
    )
    hash_op_ids: List[Optional[int]] = []
    if log_slice and map_order:
        hash_op_ids = _map_sirius_hash_join_by_order(log_slice, num_hash_joins)

    rows: List[Dict[str, str]] = []
    hash_join_seq = 0
    for idx, (join_node, join_parent) in enumerate(collected, start=1):
        n = _strip_name(join_node.get("name", ""))
        ex = join_node.get("extra_info") or {}
        jt = ex.get("Join Type", "")
        cond = ex.get("Conditions", "")
        if isinstance(cond, list):
            cond = " AND ".join(str(c) for c in cond)
        else:
            cond = str(cond or "")

        kids = join_node.get("children") or []
        left_in = _describe_child(kids[0]) if len(kids) > 0 else ""
        right_in = _describe_child(kids[1]) if len(kids) > 1 else ""

        ncol = _projection_width_near_join(join_node, join_parent)
        if ncol is None:
            kids = join_node.get("children") or []
            if len(kids) >= 2:
                est = _leaf_projection_count(kids[0]) + _leaf_projection_count(kids[1])
                ncol = str(est) if est > 0 else ""
            else:
                ncol = ""
        else:
            ncol = str(ncol)

        op_id_for_mode: Optional[int] = None
        if n == "HASH_JOIN" and map_order:
            if hash_join_seq < len(hash_op_ids):
                op_id_for_mode = hash_op_ids[hash_join_seq]

        if n.endswith("DELIM_JOIN") or "DELIM" in n:
            sirius_mode = "DELIM_JOIN"
        elif n == "HASH_JOIN":
            sirius_mode = _sirius_mode_for_hash_join(
                op_id_for_mode, build_probe_ids, cond
            )
            if log_slice and not map_order:
                sirius_mode += (
                    "; use --match-order (+ per-query log) for BUILD_PROBE id + tasks"
                )
        else:
            sirius_mode = "OTHER_PHYSICAL_JOIN"

        op_id_str = ""
        exec_count = ""
        if n == "HASH_JOIN" and map_order:
            oid = op_id_for_mode
            if log_slice and oid is not None:
                op_id_str = str(oid)
                total = 0
                for (pipe, opid, opname), c in produced_counts.items():
                    if opid == oid and opname == "HASH_JOIN":
                        total += c
                exec_count = str(total)
            hash_join_seq += 1

        rows.append(
            {
                "query": f"Q{qnum}",
                "join_index": str(idx),
                "duckdb_operator": n,
                "join_type": str(jt),
                "join_key_or_conditions": cond,
                "left_input": left_in,
                "right_input": right_in,
                "output_column_count_estimate": ncol,
                "sirius_join_mode": sirius_mode,
                "sirius_operator_id": op_id_str,
                "sirius_execute_task_count": exec_count,
            }
        )
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=_SCRIPT_DIR / "tpch_sf10_join_catalog.csv",
    )
    ap.add_argument(
        "--sf",
        type=float,
        default=10.0,
        help="TPC-H scale factor for dbgen (default: 10)",
    )
    ap.add_argument(
        "--queries",
        type=int,
        nargs="*",
        default=list(range(1, 23)),
        help="Query numbers (default: 1-22)",
    )
    ap.add_argument(
        "--sirius-log",
        type=Path,
        default=None,
        help="Sirius log file (trace+ recommended; debug for BUILD_PROBE lines)",
    )
    ap.add_argument(
        "--one-log-per-query",
        action="store_true",
        help="Log file has one run per query in ascending Q order matching --queries",
    )
    ap.add_argument(
        "--match-order",
        action="store_true",
        help="Map HASH_JOIN to Sirius operator id by order of first HASH_JOIN "
        "in log produced events (fragile; single-pipeline queries work best)",
    )
    ap.add_argument(
        "--enrich-run-dir",
        type=Path,
        default=None,
        help="Benchmark run dir (e.g. runs/..._sf10_2iter) with sirius/qN/sirius.log "
        "per query; enables --match-order per query and BUILD_PROBE detection",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    try:
        import duckdb
    except ImportError:
        print("This script requires the Python duckdb package.", file=sys.stderr)
        return 1

    log_text = ""
    if args.sirius_log:
        log_text = args.sirius_log.read_text(errors="replace")

    log_chunks: List[str]
    if args.one_log_per_query:
        log_chunks = _split_log_per_query(log_text)
    else:
        log_chunks = [log_text]

    enrich_dir = args.enrich_run_dir.resolve() if args.enrich_run_dir else None

    con = duckdb.connect(":memory:")
    con.execute("INSTALL tpch; LOAD tpch")
    con.execute(f"CALL dbgen(sf={args.sf})")

    all_rows: List[Dict[str, str]] = []
    qnums = sorted(set(args.queries))
    for i, qn in enumerate(qnums):
        key = f"q{qn}"
        if key not in QUERIES:
            print(f"WARNING: missing {key} in queries.py", file=sys.stderr)
            continue
        sql = QUERIES[key]
        slice_log: Optional[str] = None
        use_match_order = args.match_order
        if enrich_dir:
            p = enrich_dir / f"sirius/q{qn}/sirius.log"
            if p.is_file():
                slice_log = p.read_text(errors="replace")
                use_match_order = True
        elif log_text:
            if args.one_log_per_query:
                slice_log = log_chunks[i] if i < len(log_chunks) else ""
            else:
                slice_log = log_text
        try:
            rows = build_rows_for_query(
                con, qn, sql, log_text, slice_log, use_match_order
            )
        except Exception as e:
            print(f"ERROR Q{qn}: {e}", file=sys.stderr)
            raise
        if not rows:
            all_rows.append(
                {
                    "query": f"Q{qn}",
                    "join_index": "0",
                    "duckdb_operator": "",
                    "join_type": "",
                    "join_key_or_conditions": "",
                    "left_input": "",
                    "right_input": "",
                    "output_column_count_estimate": "",
                    "sirius_join_mode": "",
                    "sirius_operator_id": "",
                    "sirius_execute_task_count": "",
                }
            )
        else:
            all_rows.extend(rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "query",
        "join_index",
        "duckdb_operator",
        "join_type",
        "join_key_or_conditions",
        "left_input",
        "right_input",
        "output_column_count_estimate",
        "sirius_join_mode",
        "sirius_operator_id",
        "sirius_execute_task_count",
    ]
    with args.output.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)

    print(f"Wrote {args.output} ({len(all_rows)} join rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
