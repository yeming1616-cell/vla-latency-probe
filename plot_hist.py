#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_hist.py — 从 latency_probe.py 输出的 CSV 生成延迟分布直方图（SVG）。

仅使用标准库，不依赖 matplotlib。产物是矢量 SVG，可直接放进论文或
GitHub README（浏览器原生支持 SVG）。

用法
----
    python plot_hist.py results/latency.csv
    python plot_hist.py results/latency.csv --out hist.svg --bins 20

作者：原壹铭　（东南大学物理学院）
"""

import argparse
import csv
import math
import os
import statistics
import sys


def percentile(sorted_vals, q):
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def load(path):
    """返回 (labels->list) 的字典，按出现顺序保留。"""
    series = {}
    order = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            lab = row.get("label") or "run"
            try:
                val = float(row["latency_ms"])
            except (KeyError, TypeError, ValueError):
                continue
            if lab not in series:
                series[lab] = []
                order.append(lab)
            series[lab].append(val)
    return series, order


def esc(s):
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def build_svg(vals, label, bins, width, height):
    vals = sorted(vals)
    n = len(vals)
    vmin, vmax = vals[0], vals[-1]
    if vmax <= vmin:
        vmax = vmin + 1.0

    # 直方图分箱
    bw = (vmax - vmin) / bins
    counts = [0] * bins
    for v in vals:
        idx = int((v - vmin) / bw)
        if idx >= bins:
            idx = bins - 1
        counts[idx] += 1
    cmax = max(counts) if counts else 1

    # 画布
    ml, mr, mt, mb = 70, 24, 46, 62
    pw = width - ml - mr
    ph = height - mt - mb

    p50 = percentile(vals, 0.50)
    p95 = percentile(vals, 0.95)
    p99 = percentile(vals, 0.99)
    mean = statistics.fmean(vals)
    std = statistics.pstdev(vals) if n > 1 else 0.0

    def x_of(v):
        return ml + (v - vmin) / (vmax - vmin) * pw

    o = []
    o.append('<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" '
             'viewBox="0 0 %d %d" font-family="Microsoft YaHei, DejaVu Sans, sans-serif">'
             % (width, height, width, height))
    o.append('<rect width="%d" height="%d" fill="#ffffff"/>' % (width, height))

    # 标题
    o.append('<text x="%d" y="26" font-size="17" font-weight="bold" fill="#1a4d8f">'
             '推理延迟分布　%s</text>' % (ml, esc(label)))
    o.append('<text x="%d" y="44" font-size="11.5" fill="#555">'
             'n = %d　mean = %.2f ms　std = %.2f ms　p50 = %.2f　p95 = %.2f　p99 = %.2f ms'
             '</text>' % (ml, n, mean, std, p50, p95, p99))

    # Y 轴网格
    steps = 4
    for k in range(steps + 1):
        yy = mt + ph - ph * k / steps
        cnt = cmax * k / steps
        o.append('<line x1="%d" y1="%.2f" x2="%d" y2="%.2f" stroke="#e2e6ea" stroke-width="1"/>'
                 % (ml, yy, ml + pw, yy))
        o.append('<text x="%d" y="%.2f" font-size="10.5" fill="#666" text-anchor="end">%d</text>'
                 % (ml - 8, yy + 3.5, round(cnt)))

    # 柱
    for i, c in enumerate(counts):
        if c == 0:
            continue
        x0 = ml + pw * i / bins
        x1 = ml + pw * (i + 1) / bins
        h = ph * c / cmax
        y0 = mt + ph - h
        o.append('<rect x="%.2f" y="%.2f" width="%.2f" height="%.2f" fill="#4a7eb5" '
                 'stroke="#2f5d8c" stroke-width="0.6"/>' % (x0, y0, max(x1 - x0 - 1.2, 0.8), h))

    # 坐标轴
    o.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#333" stroke-width="1.4"/>'
             % (ml, mt + ph, ml + pw, mt + ph))
    o.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#333" stroke-width="1.4"/>'
             % (ml, mt, ml, mt + ph))
    o.append('<text x="%d" y="%d" font-size="11.5" fill="#333" text-anchor="middle">'
             '单次推理延迟 (ms)</text>' % (ml + pw // 2, height - 26))
    o.append('<text x="16" y="%d" font-size="11.5" fill="#333" text-anchor="middle" '
             'transform="rotate(-90 16 %d)">频次</text>' % (mt + ph // 2, mt + ph // 2))

    # X 轴刻度
    for k in range(6):
        v = vmin + (vmax - vmin) * k / 5.0
        xx = x_of(v)
        o.append('<line x1="%.2f" y1="%d" x2="%.2f" y2="%d" stroke="#333" stroke-width="1"/>'
                 % (xx, mt + ph, xx, mt + ph + 5))
        o.append('<text x="%.2f" y="%d" font-size="10.5" fill="#666" text-anchor="middle">'
                 '%.1f</text>' % (xx, mt + ph + 19, v))

    # 分位数参考线
    marks = [("p50", p50, "#2e7d32"), ("p95", p95, "#ef6c00"), ("p99", p99, "#c62828")]
    for name, v, color in marks:
        xx = x_of(v)
        o.append('<line x1="%.2f" y1="%d" x2="%.2f" y2="%d" stroke="%s" '
                 'stroke-width="1.4" stroke-dasharray="5 3"/>'
                 % (xx, mt, xx, mt + ph, color))
        o.append('<text x="%.2f" y="%d" font-size="10.5" fill="%s" text-anchor="middle">'
                 '%s</text>' % (xx, mt - 6, color, name))

    o.append('</svg>')
    return "\n".join(o), p50, p95, p99, mean, std


def main(argv=None):
    ap = argparse.ArgumentParser(description="由延迟 CSV 生成 SVG 直方图")
    ap.add_argument("csvpath", help="latency_probe.py 输出的 CSV")
    ap.add_argument("--out", default="", help="输出 SVG 路径（默认与 CSV 同目录）")
    ap.add_argument("--bins", type=int, default=18, help="分箱数（默认 18）")
    ap.add_argument("--width", type=int, default=880)
    ap.add_argument("--height", type=int, default=470)
    ap.add_argument("--label", default="", help="覆盖图中标签")
    args = ap.parse_args(argv)

    if not os.path.isfile(args.csvpath):
        print("错误：找不到文件 %s" % args.csvpath, file=sys.stderr)
        return 2

    series, order = load(args.csvpath)
    if not order:
        print("错误：CSV 中没有有效的 latency_ms 数据", file=sys.stderr)
        return 2

    for lab in order:
        vals = series[lab]
        if len(vals) < 2:
            print("跳过 %s：样本数不足（%d）" % (lab, len(vals)))
            continue
        svg, p50, p95, p99, mean, std = build_svg(
            vals, args.label or lab, args.bins, args.width, args.height)
        out = args.out or os.path.join(
            os.path.dirname(os.path.abspath(args.csvpath)), "latency_hist.svg")
        with open(out, "w", encoding="utf-8") as f:
            f.write(svg)
        print("已生成 %s" % out)
        print("  n=%d  mean=%.2f ms  std=%.2f ms  p50=%.2f  p95=%.2f  p99=%.2f"
              % (len(vals), mean, std, p50, p95, p99))
    return 0


if __name__ == "__main__":
    sys.exit(main())
