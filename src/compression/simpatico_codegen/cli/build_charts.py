import math
from pathlib import Path

BLUE = "#2a78d6"
ORANGE = "#eb6834"
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3de"

# name, dtype, plan, decode_only_gbps (parquet, device-staged source, 100M rows), decompress_gbps (simpatico),
# parquet_ratio (100M rows, from footer), simpatico_ratio
LINEITEM = [
    ("l_orderkey", "i64", "Δ+bp", 118.37, 2077.9, 25.662, 12.3933),
    ("l_partkey", "i32", "bp", 52.96, 1499.17, 1.082, 1.13999),
    ("l_suppkey", "i32", "bp", 58.59, 1525.14, 1.278, 1.32944),
    ("l_linenumber", "i32", "bp", 66.47, 2119.11, 15.176, 10.4224),
    ("l_quantity", "dec64", "bp", 145.27, 2885.5, 10.611, 4.88491),
    ("l_extendedprice", "dec64", "bp", 120.84, 2644.36, 2.620, 2.65543),
    ("l_discount", "dec64", "bp", 157.34, 3294.73, 15.885, 15.6038),
    ("l_tax", "dec64", "bp", 159.95, 3300.82, 15.886, 15.6038),
    ("l_returnflag", "str", "dict+bp", 58.14, 927.139, 20.374, 19.3207),
    ("l_linestatus", "str", "dict+bp", 64.51, 971.594, 31.795, 37.3722),
    ("l_shipdate", "date", "bp", 65.68, 1712.84, 2.644, 2.65113),
    ("l_commitdate", "date", "bp", 67.23, 1711.2, 2.644, 2.65113),
    ("l_receiptdate", "date", "bp", 68.05, 1717.79, 2.644, 2.65113),
    ("l_shipinstruct", "str", "dict+bp x2", 213.42, 290.728, 63.114, 61.8232),
    ("l_shipmode", "str", "spl+bp", 116.53, 1235.48, 21.892, 1.39974),
    ("l_comment", "str", "spl+Δ+snpy+ans", 134.8, 228.05, 2.603, 2.44778),
]

ORDERS = [
    ("o_orderkey", "i64", "Δ+bp", 147.86, 2106.68, 242.049, 12.3933),
    ("o_custkey", "i32", "bp", 41.92, 1508.2, 0.974, 1.13999),
    ("o_orderstatus", "str", "dict+bp", 77.76, 913.1, 19.708, 19.3207),
    ("o_totalprice", "dec64", "bp", 136.79, 2710.02, 2.423, 2.45196),
    ("o_orderdate", "date", "bp", 77.56, 1706.94, 2.648, 2.65113),
    ("o_orderpriority", "str", "dict+bp x2", 184.4, 249.587, 32.764, 32.3101),
    ("o_clerk", "str", "dict+bp+ans+Δ+rle", 58.0, 486.344, 2.601, 7.34597),
    ("o_shippriority", "i32", "bp", 109.21, 3471.23, 2715.767, 455.101),
    ("o_comment", "str", "identity", 161.04, 2916.0, 2.915, 0.929204),
]


def logy(v, vmin, vmax, y0, y1):
    """v -> pixel y, log scale, y0 at top (vmin) .. y1 at bottom is inverted by caller."""
    v = min(max(v, vmin), vmax)
    return y1 - (y1 - y0) * (math.log10(v) - math.log10(vmin)) / (math.log10(vmax) - math.log10(vmin))


def bar_ratio_chart(rows, total, title, out_path, ratio_min, ratio_max, ratio_ticks):
    # `total` is a single (name, dtype, plan, decode, decomp, pq_ratio, sp_ratio) tuple for the
    # whole file, appended as an extra, visually separated group after the per-column ones.
    rows = list(rows) + [total]
    total_idx = len(rows) - 1

    W, H = 1060 + 60, 640
    ML, MR, MT = 74, 74, 56
    plot_h = 380
    plot_y0, plot_y1 = MT + 30, MT + 30 + plot_h
    plot_x0, plot_x1 = ML, W - MR

    thr_min, thr_max = 20, 4500

    n = len(rows)
    plot_w = plot_x1 - plot_x0
    group_w = plot_w / n
    bar_w = group_w * 0.30
    pad = group_w * 0.08

    svg = []
    svg.append(
        f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
        f'font-family="Helvetica, Arial, sans-serif">'
    )
    svg.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="{SURFACE}"/>')
    svg.append(
        f'<text x="{ML}" y="28" font-size="17" font-weight="600" fill="{TEXT_PRIMARY}">{title}</text>'
    )
    svg.append(
        f'<text x="{ML}" y="{MT+20}" font-size="12" fill="{TEXT_SECONDARY}">'
        f"Bars = decompression throughput (left axis, GB/s, log). "
        f"Diamonds = compression ratio (right axis, x, log), same column as their bar.</text>"
    )

    for gb in (20, 50, 100, 200, 500, 1000, 2000, 4000):
        y = logy(gb, thr_min, thr_max, plot_y0, plot_y1)
        svg.append(
            f'<line x1="{plot_x0}" y1="{y:.1f}" x2="{plot_x1}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>'
        )
        svg.append(
            f'<text x="{plot_x0-8}" y="{y+4:.1f}" font-size="10" fill="{TEXT_SECONDARY}" text-anchor="end">{gb}</text>'
        )
    svg.append(
        f'<text x="18" y="{(plot_y0+plot_y1)/2:.1f}" font-size="11" fill="{TEXT_SECONDARY}" '
        f'transform="rotate(-90 18 {(plot_y0+plot_y1)/2:.1f})" text-anchor="middle">Throughput GB/s (log)</text>'
    )

    for r in ratio_ticks:
        y = logy(r, ratio_min, ratio_max, plot_y0, plot_y1)
        svg.append(
            f'<text x="{plot_x1+8}" y="{y+4:.1f}" font-size="10" fill="{TEXT_SECONDARY}">{r}x</text>'
        )
    svg.append(
        f'<text x="{W-16}" y="{(plot_y0+plot_y1)/2:.1f}" font-size="11" fill="{TEXT_SECONDARY}" '
        f'transform="rotate(90 {W-16} {(plot_y0+plot_y1)/2:.1f})" text-anchor="middle">Compression ratio (log)</text>'
    )

    svg.append(
        f'<line x1="{plot_x0}" y1="{plot_y1}" x2="{plot_x1}" y2="{plot_y1}" stroke="{TEXT_SECONDARY}" stroke-width="1"/>'
    )

    # Highlight band + divider setting the TOTAL group apart from the per-column ones.
    divider_x = plot_x0 + total_idx * group_w
    svg.append(
        f'<rect x="{divider_x:.1f}" y="{plot_y0-30:.1f}" width="{plot_x1-divider_x:.1f}" '
        f'height="{plot_y1-plot_y0+30:.1f}" fill="#c3c2b7" fill-opacity="0.18"/>'
    )
    svg.append(
        f'<line x1="{divider_x:.1f}" y1="{plot_y0-30:.1f}" x2="{divider_x:.1f}" y2="{plot_y1:.1f}" '
        f'stroke="{TEXT_SECONDARY}" stroke-width="1" stroke-dasharray="3,3"/>'
    )

    for i, (name, dtype, plan, decode, decomp, pq_ratio, sp_ratio) in enumerate(rows):
        gx0 = plot_x0 + i * group_w
        x1 = gx0 + pad
        x2 = x1 + bar_w + pad

        y1v = logy(decode, thr_min, thr_max, plot_y0, plot_y1)
        h1 = plot_y1 - y1v
        svg.append(
            f'<rect x="{x1:.1f}" y="{y1v:.1f}" width="{bar_w:.1f}" height="{h1:.1f}" rx="3" fill="{BLUE}">'
            f"<title>{name}: parquet decode (ex. metadata) {decode:.1f} GB/s</title></rect>"
        )
        y2v = logy(decomp, thr_min, thr_max, plot_y0, plot_y1)
        h2 = plot_y1 - y2v
        svg.append(
            f'<rect x="{x2:.1f}" y="{y2v:.1f}" width="{bar_w:.1f}" height="{h2:.1f}" rx="3" fill="{ORANGE}">'
            f"<title>{name}: simpatico decompress {decomp:.1f} GB/s</title></rect>"
        )

        cx1 = x1 + bar_w / 2
        cy1 = logy(pq_ratio, ratio_min, ratio_max, plot_y0, plot_y1)
        svg.append(
            f'<path d="M {cx1:.1f} {cy1-6:.1f} L {cx1+6:.1f} {cy1:.1f} L {cx1:.1f} {cy1+6:.1f} '
            f'L {cx1-6:.1f} {cy1:.1f} Z" fill="{SURFACE}" stroke="{BLUE}" stroke-width="2.5">'
            f"<title>{name}: parquet on-disk ratio {pq_ratio:.2f}x</title></path>"
        )
        cx2 = x2 + bar_w / 2
        cy2 = logy(sp_ratio, ratio_min, ratio_max, plot_y0, plot_y1)
        svg.append(
            f'<path d="M {cx2:.1f} {cy2-6:.1f} L {cx2+6:.1f} {cy2:.1f} L {cx2:.1f} {cy2+6:.1f} '
            f'L {cx2-6:.1f} {cy2:.1f} Z" fill="{SURFACE}" stroke="{ORANGE}" stroke-width="2.5">'
            f"<title>{name}: simpatico ratio {sp_ratio:.2f}x</title></path>"
        )

    for i, (name, dtype, plan, *_ ) in enumerate(rows):
        gx0 = plot_x0 + i * group_w
        cx = gx0 + group_w / 2
        short = name.split("_", 1)[1] if "_" in name else name
        label = f"{dtype} · {plan}" if dtype else plan
        svg.append(
            f'<text x="{cx:.1f}" y="{plot_y1+14}" font-size="10" font-weight="600" fill="{TEXT_PRIMARY}" '
            f'text-anchor="end" transform="rotate(-35 {cx:.1f} {plot_y1+14})">{short}</text>'
        )
        svg.append(
            f'<text x="{cx:.1f}" y="{plot_y1+30}" font-size="8" fill="{TEXT_SECONDARY}" '
            f'text-anchor="end" transform="rotate(-35 {cx:.1f} {plot_y1+30})">{label}</text>'
        )

    lx = plot_x0
    ly = plot_y1 + 96
    svg.append(f'<rect x="{lx}" y="{ly-10}" width="10" height="10" rx="2" fill="{BLUE}"/>')
    svg.append(f'<text x="{lx+16}" y="{ly}" font-size="11" fill="{TEXT_PRIMARY}">Parquet decode throughput (bar, left axis)</text>')
    svg.append(
        f'<path d="M {lx+5} {ly+18-6} L {lx+11} {ly+18} L {lx+5} {ly+18+6} L {lx-1} {ly+18} Z" '
        f'fill="{SURFACE}" stroke="{BLUE}" stroke-width="2"/>'
    )
    svg.append(f'<text x="{lx+16}" y="{ly+22}" font-size="11" fill="{TEXT_PRIMARY}">Parquet on-disk ratio (diamond, right axis)</text>')

    lx2 = lx + 340
    svg.append(f'<rect x="{lx2}" y="{ly-10}" width="10" height="10" rx="2" fill="{ORANGE}"/>')
    svg.append(f'<text x="{lx2+16}" y="{ly}" font-size="11" fill="{TEXT_PRIMARY}">Simpatico decompress throughput (bar, left axis)</text>')
    svg.append(
        f'<path d="M {lx2+5} {ly+18-6} L {lx2+11} {ly+18} L {lx2+5} {ly+18+6} L {lx2-1} {ly+18} Z" '
        f'fill="{SURFACE}" stroke="{ORANGE}" stroke-width="2"/>'
    )
    svg.append(f'<text x="{lx2+16}" y="{ly+22}" font-size="11" fill="{TEXT_PRIMARY}">Simpatico ratio (diamond, right axis)</text>')

    svg.append("</svg>")

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{title}</title></head>
<body style="background:{SURFACE};margin:0;padding:24px;">
{''.join(svg)}
</body></html>"""
    with open(out_path, "w") as f:
        f.write(html)


def scatter_chart(out_path):
    W, H = 940, 640
    ML, MR, MT, MB = 74, 30, 56, 60
    plot_x0, plot_x1 = ML, W - MR
    plot_y0, plot_y1 = MT, H - MB

    ratio_min, ratio_max = 0.6, 3000
    thr_min, thr_max = 20, 4500

    svg = [
        f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" font-family="Helvetica, Arial, sans-serif">'
    ]
    svg.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="{SURFACE}"/>')
    svg.append(
        f'<text x="{ML}" y="28" font-size="17" font-weight="600" fill="{TEXT_PRIMARY}">'
        f"Compression ratio vs. decompression throughput</text>"
    )

    for gb in (20, 50, 100, 200, 500, 1000, 2000, 4000):
        y = logy(gb, thr_min, thr_max, plot_y0, plot_y1)
        svg.append(f'<line x1="{plot_x0}" y1="{y:.1f}" x2="{plot_x1}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        svg.append(
            f'<text x="{plot_x0-8}" y="{y+4:.1f}" font-size="10" fill="{TEXT_SECONDARY}" text-anchor="end">{gb}</text>'
        )
    svg.append(
        f'<text x="16" y="{(plot_y0+plot_y1)/2:.1f}" font-size="12" fill="{TEXT_SECONDARY}" '
        f'transform="rotate(-90 16 {(plot_y0+plot_y1)/2:.1f})" text-anchor="middle">Throughput GB/s (log)</text>'
    )

    for r in (1, 3, 10, 30, 100, 300, 1000, 3000):
        x = plot_x0 + (plot_x1 - plot_x0) * (math.log10(r) - math.log10(ratio_min)) / (
            math.log10(ratio_max) - math.log10(ratio_min)
        )
        svg.append(f'<line x1="{x:.1f}" y1="{plot_y0}" x2="{x:.1f}" y2="{plot_y1}" stroke="{GRID}" stroke-width="1"/>')
        svg.append(
            f'<text x="{x:.1f}" y="{plot_y1+16}" font-size="10" fill="{TEXT_SECONDARY}" text-anchor="middle">{r}x</text>'
        )
    svg.append(
        f'<text x="{(plot_x0+plot_x1)/2:.1f}" y="{H-14}" font-size="12" fill="{TEXT_SECONDARY}" '
        f'text-anchor="middle">Compression ratio (x, log)</text>'
    )

    def px(ratio):
        rv = min(max(ratio, ratio_min), ratio_max)
        return plot_x0 + (plot_x1 - plot_x0) * (math.log10(rv) - math.log10(ratio_min)) / (
            math.log10(ratio_max) - math.log10(ratio_min)
        )

    # Color = engine (matches the bar charts: blue = parquet, orange = simpatico).
    # Shape = table (circle = lineitem, square = orders) so table is still distinguishable.
    def plot_series(rows, engine, color, table, shape):
        out = []
        for name, dtype, plan, decode, decomp, pq_ratio, sp_ratio in rows:
            ratio, thr = (pq_ratio, decode) if engine == "parquet" else (sp_ratio, decomp)
            x = px(ratio)
            y = logy(thr, thr_min, thr_max, plot_y0, plot_y1)
            title = f"{name} ({engine}, {table}): {ratio:.1f}x, {thr:.0f} GB/s"
            if shape == "circle":
                out.append(
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{color}" fill-opacity="0.85" '
                    f'stroke="{SURFACE}" stroke-width="2"><title>{title}</title></circle>'
                )
            else:
                out.append(
                    f'<rect x="{x-5.5:.1f}" y="{y-5.5:.1f}" width="11" height="11" rx="2" fill="{color}" '
                    f'fill-opacity="0.85" stroke="{SURFACE}" stroke-width="2"><title>{title}</title></rect>'
                )
        return out

    svg += plot_series(LINEITEM, "parquet", BLUE, "lineitem", "circle")
    svg += plot_series(ORDERS, "parquet", BLUE, "orders", "square")
    svg += plot_series(LINEITEM, "simpatico", ORANGE, "lineitem", "circle")
    svg += plot_series(ORDERS, "simpatico", ORANGE, "orders", "square")

    # TOTAL points: larger, dark-outlined, always direct-labeled.
    def plot_total(total, engine, color, table, shape):
        name, dtype, plan, decode, decomp, pq_ratio, sp_ratio = total
        ratio, thr = (pq_ratio, decode) if engine == "parquet" else (sp_ratio, decomp)
        x = px(ratio)
        y = logy(thr, thr_min, thr_max, plot_y0, plot_y1)
        title = f"{table} TOTAL ({engine}): {ratio:.1f}x, {thr:.0f} GB/s"
        out = []
        if shape == "circle":
            out.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="9" fill="{color}" '
                f'stroke="{TEXT_PRIMARY}" stroke-width="2"><title>{title}</title></circle>'
            )
        else:
            out.append(
                f'<rect x="{x-8:.1f}" y="{y-8:.1f}" width="16" height="16" rx="2" fill="{color}" '
                f'stroke="{TEXT_PRIMARY}" stroke-width="2"><title>{title}</title></rect>'
            )
        out.append(
            f'<text x="{x+13:.1f}" y="{y+4:.1f}" font-size="10" font-weight="600" '
            f'fill="{TEXT_PRIMARY}">{table} TOTAL</text>'
        )
        return out

    svg += plot_total(LINEITEM_TOTAL, "parquet", BLUE, "lineitem", "circle")
    svg += plot_total(ORDERS_TOTAL, "parquet", BLUE, "orders", "square")
    svg += plot_total(LINEITEM_TOTAL, "simpatico", ORANGE, "lineitem", "circle")
    svg += plot_total(ORDERS_TOTAL, "simpatico", ORANGE, "orders", "square")

    lx = plot_x1 - 300
    ly = MT - 6
    svg.append(f'<rect x="{lx-5}" y="{ly-5}" width="10" height="10" rx="2" fill="{BLUE}"/>')
    svg.append(f'<text x="{lx+12}" y="{ly+4}" font-size="11" fill="{TEXT_PRIMARY}">Parquet decode</text>')
    svg.append(f'<rect x="{lx+145}" y="{ly-5}" width="10" height="10" rx="2" fill="{ORANGE}"/>')
    svg.append(f'<text x="{lx+162}" y="{ly+4}" font-size="11" fill="{TEXT_PRIMARY}">Simpatico decompress</text>')
    ly2 = ly + 20
    svg.append(f'<circle cx="{lx}" cy="{ly2}" r="5" fill="{TEXT_SECONDARY}"/>')
    svg.append(f'<text x="{lx+12}" y="{ly2+4}" font-size="11" fill="{TEXT_PRIMARY}">lineitem</text>')
    svg.append(f'<rect x="{lx+145-5}" y="{ly2-5}" width="10" height="10" rx="2" fill="{TEXT_SECONDARY}"/>')
    svg.append(f'<text x="{lx+162}" y="{ly2+4}" font-size="11" fill="{TEXT_PRIMARY}">orders</text>')

    svg.append("</svg>")

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>ratio vs throughput</title></head>
<body style="background:{SURFACE};margin:0;padding:24px;">
{''.join(svg)}
</body></html>"""
    with open(out_path, "w") as f:
        f.write(html)


# TOTAL: whole-file, single read_parquet/compress_with_plan call (100M rows, matches the
# per-column runs' scale). decode/decompress from the full-table benchmark mode; ratios from
# that same full-table run (parquet) and simpatico's full-table TOTAL row.
LINEITEM_TOTAL = ("TOTAL", "", "whole file, full-table read", 206.71, 512.69, 4.364, 3.501)
ORDERS_TOTAL = ("TOTAL", "", "whole file, full-table read", 154.74, 907.76, 3.350, 1.703)

if __name__ == "__main__":
    base = Path(__file__).parent / "charts_out"
    base.mkdir(exist_ok=True)
    bar_ratio_chart(
        LINEITEM,
        LINEITEM_TOTAL,
        "lineitem — decompress throughput &amp; compression ratio",
        f"{base}/lineitem_decompress.html",
        ratio_min=0.7, ratio_max=100, ratio_ticks=(1, 2, 5, 10, 20, 50, 100),
    )
    bar_ratio_chart(
        ORDERS,
        ORDERS_TOTAL,
        "orders — decompress throughput &amp; compression ratio",
        f"{base}/orders_decompress.html",
        ratio_min=0.7, ratio_max=3000, ratio_ticks=(1, 5, 25, 125, 625, 3000),
    )
    scatter_chart(f"{base}/ratio_vs_throughput_scatter.html")
    print("done")
