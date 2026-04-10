#!/usr/bin/env python3
# =============================================================================
# Copyright 2025, Sirius Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License.
# =============================================================================
"""Extract pre-join build/probe tables from TPC-H (DuckDB CPU) as Parquet + JSON metadata.

Each *recipe* produces ``<name>_build.parquet``, ``<name>_probe.parquet``, and
``<name>_meta.json`` with row counts, join key column names, and dtypes.

Usage (repo root). Prefer ``--python`` if the Sirius-linked ``duckdb`` CLI fails to start::

    python3 test/tpch_performance/extract_join_data.py 10 --python \\
        --parquet-dir test_datasets/tpch_parquet_sf10 \\
        --output-dir test_datasets/join_extracts_sf10

List recipes::

    python3 test/tpch_performance/extract_join_data.py --list

The first ``num_key_columns`` columns in each output table are the equi-join keys
(in build / probe order) for use with ``join_batch_benchmark``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


@dataclass(frozen=True)
class JoinRecipe:
    """TPC-H-derived extraction: build and probe SQL (CPU), equi-join description."""

    name: str
    description: str
    build_sql: str
    probe_sql: str
    build_key_columns: List[str]
    probe_key_columns: List[str]


# Recipes mirror dominant large joins; SQL is valid DuckDB TPC-H.
JOIN_RECIPES: Dict[str, JoinRecipe] = {
    "q21_lineitem_orders": JoinRecipe(
        name="q21_lineitem_orders",
        description="Q21-style lineitem × orders (l_orderkey = o_orderkey); "
        "orders status F vs lineitem receipt > commit.",
        build_sql="""
            SELECT o_orderkey AS o_orderkey,
                   o_custkey AS o_custkey_payload
            FROM orders
            WHERE o_orderstatus = 'F'
        """,
        probe_sql="""
            SELECT l_orderkey AS l_orderkey,
                   l_suppkey AS l_suppkey_payload
            FROM lineitem
            WHERE l_receiptdate > l_commitdate
        """,
        build_key_columns=["o_orderkey"],
        probe_key_columns=["l_orderkey"],
    ),
    "q3_lineitem_orders": JoinRecipe(
        name="q3_lineitem_orders",
        description="Q3 customer segment + order date filter × lineitem ship date filter.",
        build_sql="""
            SELECT o.o_orderkey AS o_orderkey,
                   o.o_orderdate AS o_orderdate_payload,
                   o.o_shippriority AS o_shippriority_payload
            FROM orders o
            INNER JOIN customer c ON c.c_custkey = o.o_custkey
            WHERE c.c_mktsegment = 'HOUSEHOLD'
              AND o.o_orderdate < DATE '1995-03-25'
        """,
        probe_sql="""
            SELECT l_orderkey AS l_orderkey,
                   l_extendedprice AS l_extendedprice_payload,
                   l_discount AS l_discount_payload
            FROM lineitem
            WHERE l_shipdate > DATE '1995-03-25'
        """,
        build_key_columns=["o_orderkey"],
        probe_key_columns=["l_orderkey"],
    ),
    "q5_lineitem_supplier": JoinRecipe(
        name="q5_lineitem_supplier",
        description="Q5-style lineitem × European suppliers on l_suppkey = s_suppkey "
        "(lineitem restricted to orders in 1997).",
        build_sql="""
            SELECT s.s_suppkey AS s_suppkey,
                   s.s_nationkey AS s_nationkey_payload
            FROM supplier s
            INNER JOIN nation n ON s.s_nationkey = n.n_nationkey
            INNER JOIN region r ON n.n_regionkey = r.r_regionkey
            WHERE r.r_name = 'EUROPE'
        """,
        probe_sql="""
            SELECT l.l_suppkey AS l_suppkey,
                   l.l_orderkey AS l_orderkey_payload,
                   l.l_extendedprice AS l_extendedprice_payload,
                   l.l_discount AS l_discount_payload
            FROM lineitem l
            INNER JOIN orders o ON l.l_orderkey = o.o_orderkey
            WHERE o.o_orderdate >= DATE '1997-01-01'
              AND o.o_orderdate < DATE '1998-01-01'
        """,
        build_key_columns=["s_suppkey"],
        probe_key_columns=["l_suppkey"],
    ),
    "q9_partsupp_lineitem": JoinRecipe(
        name="q9_partsupp_lineitem",
        description="Q9-style partsupp (yellow parts) × lineitem on "
        "(ps_partkey, ps_suppkey) = (l_partkey, l_suppkey).",
        build_sql="""
            SELECT ps.ps_partkey AS ps_partkey,
                   ps.ps_suppkey AS ps_suppkey,
                   ps.ps_supplycost AS ps_supplycost_payload
            FROM partsupp ps
            INNER JOIN part p ON p.p_partkey = ps.ps_partkey
            WHERE p.p_name LIKE '%yellow%'
        """,
        probe_sql="""
            SELECT l.l_partkey AS l_partkey,
                   l.l_suppkey AS l_suppkey,
                   l.l_orderkey AS l_orderkey_payload,
                   l.l_quantity AS l_quantity_payload,
                   l.l_extendedprice AS l_extendedprice_payload,
                   l.l_discount AS l_discount_payload
            FROM lineitem l
        """,
        build_key_columns=["ps_partkey", "ps_suppkey"],
        probe_key_columns=["l_partkey", "l_suppkey"],
    ),
    "q8_lineitem_part": JoinRecipe(
        name="q8_lineitem_part",
        description="Q8-style lineitem × part on l_partkey = p_partkey; part type "
        "'PROMO BRUSHED COPPER'; lineitem via orders in 1995–1996 order-date window.",
        build_sql="""
            SELECT p.p_partkey AS p_partkey,
                   p.p_retailprice AS p_retailprice_payload
            FROM part p
            WHERE p.p_type = 'PROMO BRUSHED COPPER'
        """,
        probe_sql="""
            SELECT l.l_partkey AS l_partkey,
                   l.l_orderkey AS l_orderkey_payload,
                   l.l_suppkey AS l_suppkey_payload,
                   l.l_extendedprice AS l_extendedprice_payload,
                   l.l_discount AS l_discount_payload
            FROM lineitem l
            INNER JOIN orders o ON l.l_orderkey = o.o_orderkey
            WHERE o.o_orderdate >= DATE '1995-01-01'
              AND o.o_orderdate <= DATE '1996-12-31'
        """,
        build_key_columns=["p_partkey"],
        probe_key_columns=["l_partkey"],
    ),
    "q17_lineitem_part": JoinRecipe(
        name="q17_lineitem_part",
        description="Q17-style lineitem × part on l_partkey = p_partkey; "
        "part brand Brand#13 and container JUMBO CAN (large probe, tiny build).",
        build_sql="""
            SELECT p.p_partkey AS p_partkey,
                   p.p_retailprice AS p_retailprice_payload
            FROM part p
            WHERE p.p_brand = 'Brand#13'
              AND p.p_container = 'JUMBO CAN'
        """,
        probe_sql="""
            SELECT l.l_partkey AS l_partkey,
                   l.l_quantity AS l_quantity_payload,
                   l.l_extendedprice AS l_extendedprice_payload,
                   l.l_discount AS l_discount_payload
            FROM lineitem l
        """,
        build_key_columns=["p_partkey"],
        probe_key_columns=["l_partkey"],
    ),
    "q10_lineitem_orders": JoinRecipe(
        name="q10_lineitem_orders",
        description="Q10-style lineitem × orders on l_orderkey = o_orderkey; "
        "l_returnflag = 'R' and orders in 1994 Q2 window.",
        build_sql="""
            SELECT o.o_orderkey AS o_orderkey,
                   o.o_custkey AS o_custkey_payload,
                   o.o_orderdate AS o_orderdate_payload
            FROM orders o
            WHERE o.o_orderdate >= DATE '1994-03-01'
              AND o.o_orderdate < DATE '1994-06-01'
        """,
        probe_sql="""
            SELECT l.l_orderkey AS l_orderkey,
                   l.l_extendedprice AS l_extendedprice_payload,
                   l.l_discount AS l_discount_payload,
                   l.l_returnflag AS l_returnflag_payload
            FROM lineitem l
            WHERE l.l_returnflag = 'R'
        """,
        build_key_columns=["o_orderkey"],
        probe_key_columns=["l_orderkey"],
    ),
    "q7_lineitem_supplier": JoinRecipe(
        name="q7_lineitem_supplier",
        description="Q7-style lineitem × supplier on l_suppkey = s_suppkey; "
        "lineitem shipdate 1995–1996; supplier restricted to Egypt and United States.",
        build_sql="""
            SELECT s.s_suppkey AS s_suppkey,
                   s.s_nationkey AS s_nationkey_payload
            FROM supplier s
            INNER JOIN nation n ON s.s_nationkey = n.n_nationkey
            WHERE n.n_name IN ('EGYPT', 'UNITED STATES')
        """,
        probe_sql="""
            SELECT l.l_suppkey AS l_suppkey,
                   l.l_orderkey AS l_orderkey_payload,
                   l.l_extendedprice AS l_extendedprice_payload,
                   l.l_discount AS l_discount_payload
            FROM lineitem l
            WHERE l.l_shipdate >= DATE '1995-01-01'
              AND l.l_shipdate <= DATE '1996-12-31'
        """,
        build_key_columns=["s_suppkey"],
        probe_key_columns=["l_suppkey"],
    ),
}


def tpch_view_ddl(parquet_dir: Path) -> str:
    tables = [
        "customer",
        "lineitem",
        "nation",
        "orders",
        "part",
        "partsupp",
        "region",
        "supplier",
    ]
    lines: List[str] = []
    for name in tables:
        files: List[str] = []
        single = parquet_dir / f"{name}.parquet"
        if single.is_file():
            files.append(str(single.resolve()))
        files.extend(
            str(p.resolve()) for p in sorted(parquet_dir.glob(f"{name}_*.parquet"))
        )
        sub = parquet_dir / name
        if sub.is_dir():
            files.extend(str(p.resolve()) for p in sorted(sub.glob("*.parquet")))
        if not files:
            raise FileNotFoundError(
                f"No parquet files found for TPC-H table '{name}' under {parquet_dir}"
            )
        lst = ", ".join("'" + f.replace("'", "''") + "'" for f in sorted(set(files)))
        lines.append(
            f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet([{lst}]);"
        )
    return "\n".join(lines)


def run_duckdb_sql(duckdb_bin: str, sql: str) -> None:
    proc = subprocess.run(
        [duckdb_bin, "-c", sql],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"duckdb failed ({proc.returncode}):\n{proc.stderr or proc.stdout}"
        )


def run_duckdb_sql_in_process(con: Any, sql: str) -> None:
    """Run multi-statement SQL on an in-memory DuckDB connection (``duckdb`` PyPI)."""
    con.execute(sql)


def export_recipe(
    parquet_dir: Path,
    output_dir: Path,
    recipe: JoinRecipe,
    *,
    duckdb_bin: Optional[str] = None,
    duckdb_con: Any = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    build_path = output_dir / f"{recipe.name}_build.parquet"
    probe_path = output_dir / f"{recipe.name}_probe.parquet"
    meta_path = output_dir / f"{recipe.name}_meta.json"

    views = tpch_view_ddl(parquet_dir)
    # Single session: views + COPY
    esc_build = str(build_path.resolve()).replace("'", "''")
    esc_probe = str(probe_path.resolve()).replace("'", "''")
    full_sql = f"""
{views}
COPY ({recipe.build_sql.strip()}) TO '{esc_build}' (FORMAT PARQUET, COMPRESSION ZSTD);
COPY ({recipe.probe_sql.strip()}) TO '{esc_probe}' (FORMAT PARQUET, COMPRESSION ZSTD);
"""
    if duckdb_con is not None:
        run_duckdb_sql_in_process(duckdb_con, full_sql)
    elif duckdb_bin:
        run_duckdb_sql(duckdb_bin, full_sql)
    else:
        raise ValueError("export_recipe requires duckdb_bin or duckdb_con")

    def pq_lit(p: Path) -> str:
        return str(p.resolve()).replace("'", "''")

    def describe(path: Path) -> Dict[str, Any]:
        pl = pq_lit(path)
        cols: List[Dict[str, str]] = []
        if duckdb_con is not None:
            cnt = int(
                duckdb_con.execute(
                    f"SELECT count(*) FROM read_parquet('{pl}');"
                ).fetchone()[0]
            )
            rel = duckdb_con.execute(f"DESCRIBE SELECT * FROM read_parquet('{pl}');")
            names = [d[0] for d in rel.description]
            for row in rel.fetchall():
                dct = dict(zip(names, row))
                cols.append(
                    {
                        "name": str(dct.get("column_name", dct.get("name", ""))),
                        "type": str(dct.get("column_type", dct.get("type", ""))),
                    }
                )
        else:
            proc = subprocess.run(
                [
                    duckdb_bin,
                    "-csv",
                    "-noheader",
                    "-c",
                    f"SELECT count(*) FROM read_parquet('{pl}');",
                ],
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                raise RuntimeError(proc.stderr or proc.stdout)
            cnt = int(proc.stdout.strip() or "0")
            proc2 = subprocess.run(
                [
                    duckdb_bin,
                    "-json",
                    "-c",
                    f"DESCRIBE SELECT * FROM read_parquet('{pl}');",
                ],
                capture_output=True,
                text=True,
            )
            if proc2.returncode == 0 and proc2.stdout.strip():
                try:
                    rows = json.loads(proc2.stdout)
                    for row in rows:
                        cols.append(
                            {
                                "name": str(
                                    row.get("column_name", row.get("name", ""))
                                ),
                                "type": str(
                                    row.get("column_type", row.get("type", ""))
                                ),
                            }
                        )
                except json.JSONDecodeError:
                    pass
        return {"row_count": cnt, "columns": cols}

    meta: Dict[str, Any] = {
        "name": recipe.name,
        "description": recipe.description,
        "build_key_columns": recipe.build_key_columns,
        "probe_key_columns": recipe.probe_key_columns,
        "num_key_columns": len(recipe.build_key_columns),
        "build_parquet": str(build_path.resolve()),
        "probe_parquet": str(probe_path.resolve()),
        "build": describe(build_path),
        "probe": describe(probe_path),
    }
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")

    print(f"Wrote {build_path.name}, {probe_path.name}, {meta_path.name}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "scale_factor",
        nargs="?",
        default=None,
        help="Label only (e.g. 100); used in default output path if set.",
    )
    ap.add_argument(
        "--parquet-dir",
        type=Path,
        default=None,
        help="Directory with TPC-H parquet (default: test_datasets/tpch_parquet_sf<SF> under cwd)",
    )
    ap.add_argument("--output-dir", type=Path, required=False, default=None)
    ap.add_argument(
        "--duckdb",
        default=os.environ.get("DUCKDB", "duckdb"),
        help="DuckDB CLI binary (default: $DUCKDB or 'duckdb' on PATH). Ignored with --python.",
    )
    ap.add_argument(
        "--python",
        action="store_true",
        help="Run SQL via the ``duckdb`` Python package (in-memory). Use when the "
        "project's linked duckdb CLI fails to start (e.g. extension symbol mismatch).",
    )
    ap.add_argument(
        "--join",
        action="append",
        dest="joins",
        default=[],
        help="Recipe name (repeatable). Default: all recipes.",
    )
    ap.add_argument("--list", action="store_true", help="List recipe names and exit.")
    ap.add_argument(
        "--list-names",
        action="store_true",
        help="Print one recipe name per line and exit (for scripts).",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    if args.list_names:
        for k in sorted(JOIN_RECIPES):
            print(k)
        return 0

    if args.list:
        for k, r in sorted(JOIN_RECIPES.items()):
            print(f"{k}\n  {r.description}\n")
        return 0

    root = Path(__file__).resolve().parents[2]
    sf = args.scale_factor or ""
    pq = args.parquet_dir
    if pq is None:
        if not sf:
            ap.error("Provide scale_factor or --parquet-dir")
        pq = root / f"test_datasets/tpch_parquet_sf{sf}"
    pq = pq.resolve()
    if not pq.is_dir():
        ap.error(f"parquet dir not found: {pq}")

    out = args.output_dir
    if out is None:
        if not sf:
            ap.error("Provide scale_factor or --output-dir")
        out = root / f"test_datasets/join_extracts_sf{sf}"
    out = out.resolve()

    names = args.joins if args.joins else list(JOIN_RECIPES.keys())
    for n in names:
        if n not in JOIN_RECIPES:
            ap.error(f"Unknown recipe {n!r}; use --list")

    py_con: Any = None
    if args.python:
        try:
            import duckdb as ddb  # type: ignore[import-untyped]
        except ImportError:
            ap.error("Install the duckdb Python package: pip install duckdb")
        py_con = ddb.connect(":memory:")

    for n in names:
        export_recipe(
            pq,
            out,
            JOIN_RECIPES[n],
            duckdb_bin=None if args.python else args.duckdb,
            duckdb_con=py_con,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
