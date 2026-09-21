#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
latency_probe.py — 测量推理调用的逐步延迟与抖动（jitter）。

背景
----
机器人 AI 必须运行在持续闭环中，控制质量对延迟 *及其方差* 都高度敏感。
均值延迟相同的两个实现，抖动大的那个会让控制更差。因此本工具在测量
每次调用耗时的基础上，额外报告分位数与相邻调用间隔的波动。

设计原则
--------
* 仅使用 Python 标准库，无需 numpy / torch，可在任何环境直接运行。
* 通过 --model 选择被测对象：
    - dummy         内置合成负载（默认，用于验证工具本身）
    - plugin:模块:函数  加载你自己的模型调用（例如 VLA 单步推理）
* 输出 CSV 原始数据，便于后续统计与绘图。

用法
----
    python latency_probe.py --runs 50 --warmup 5
    python latency_probe.py --runs 50 --model plugin:my_model:step --out results
    python plot_hist.py results/latency.csv

模型插件约定
------------
被加载的函数必须是无参数可调用对象，执行一次「一次推理」并返回任意值。
例如：

    # my_model.py
    def step():
        return model.infer(observation)

作者：原壹铭　（东南大学物理学院）
"""

import argparse
import csv
import importlib
import math
import os
import random
import statistics
import sys
import time


# --------------------------------------------------------------------------
# 被测对象
# --------------------------------------------------------------------------

def _dummy_callable(work_ms: float, jitter: float, seed: int):
    """返回一个合成调用，用于验证测量链路。

    每次调用的目标耗时在 work_ms * (1 ± jitter) 之间均匀抖动，
    以模拟真实推理中由显存分配、内核调度、批大小变化等引入的波动。
    固定 seed 时结果可复现。
    """
    rng = random.Random(seed)

    def _step():
        target_ms = work_ms * (1.0 + rng.uniform(-jitter, jitter))
        target = max(target_ms, 0.01) / 1000.0
        t0 = time.perf_counter()
        # 用纯计算循环消耗时间，避免依赖 sleep 的调度粒度
        acc = 0.0
        i = 0
        while time.perf_counter() - t0 < target:
            acc += math.sqrt(i + 1.0)
            i += 1
        return acc

    return _step


def load_plugin(spec: str):
    """解析 'plugin:module:attr'，返回可调用对象。"""
    parts = spec.split(":")
    if len(parts) != 3:
        raise ValueError(
            "插件格式应为 plugin:模块名:函数名，例如 plugin:my_model:step"
        )
    _, mod_name, attr = parts
    module = importlib.import_module(mod_name)
    fn = getattr(module, attr, None)
    if fn is None:
        raise AttributeError("模块 %s 中找不到 %s" % (mod_name, attr))
    if not callable(fn):
        raise TypeError("%s.%s 不是可调用对象" % (mod_name, attr))
    return fn


# --------------------------------------------------------------------------
# 统计
# --------------------------------------------------------------------------

def percentile(sorted_vals, q):
    """线性插值分位数；sorted_vals 必须已升序。q 取值 0~1。"""
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


def summarize(latencies_ms, gaps_ms):
    """返回统计摘要字典。"""
    s = sorted(latencies_ms)
    out = {
        "n": len(s),
        "mean_ms": statistics.fmean(latencies_ms),
        "std_ms": statistics.pstdev(latencies_ms) if len(s) > 1 else 0.0,
        "min_ms": s[0],
        "max_ms": s[-1],
        "p50_ms": percentile(s, 0.50),
        "p95_ms": percentile(s, 0.95),
        "p99_ms": percentile(s, 0.99),
    }
    # 抖动指标：相邻调用间隔的标准差（越小越平稳）
    if gaps_ms:
        out["gap_mean_ms"] = statistics.fmean(gaps_ms)
        out["gap_std_ms"] = (
            statistics.pstdev(gaps_ms) if len(gaps_ms) > 1 else 0.0
        )
        out["gap_max_ms"] = max(gaps_ms)
    else:
        out["gap_mean_ms"] = float("nan")
        out["gap_std_ms"] = float("nan")
        out["gap_max_ms"] = float("nan")
    return out


def print_summary(s, label):
    def fmt(v):
        """nan 表示样本不足（例如只有一次调用时不存在调用间隔），显示为破折号。"""
        return "       —" if isinstance(v, float) and math.isnan(v) else "%8.3f" % v

    print("")
    print("=" * 58)
    print("  延迟统计  (%s)" % label)
    print("=" * 58)
    print("  样本数            n        = %d" % s["n"])
    print("  平均延迟          mean     = %8.3f ms" % s["mean_ms"])
    print("  标准差            std      = %8.3f ms" % s["std_ms"])
    print("  最小 / 最大       min/max  = %8.3f / %.3f ms" % (s["min_ms"], s["max_ms"]))
    print("  分位数            p50      = %8.3f ms" % s["p50_ms"])
    print("                    p95      = %8.3f ms" % s["p95_ms"])
    print("                    p99      = %8.3f ms" % s["p99_ms"])
    print("  ------------------------------------------------------------")
    print("  调用间隔均值      gap_mean = %s ms" % fmt(s["gap_mean_ms"]))
    print("  调用间隔标准差    gap_std  = %s ms   <-- 抖动指标" % fmt(s["gap_std_ms"]))
    print("  调用间隔最大      gap_max  = %s ms" % fmt(s["gap_max_ms"]))
    if math.isnan(s["gap_mean_ms"]):
        print("  （样本数 < 2，无法计算调用间隔）")
    print("=" * 58)
    print("")
    print("  提示：控制质量通常受 *尾部* 影响更大，请优先关注 p95 / p99")
    print("        与 gap_std，而不只是平均值。")
    print("")


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="测量推理调用的逐步延迟与抖动",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--runs", type=int, default=50, help="测量次数（默认 50）")
    ap.add_argument("--warmup", type=int, default=5, help="预热次数，不计入统计（默认 5）")
    ap.add_argument(
        "--model",
        default="dummy",
        help="dummy（默认，合成负载）或 plugin:模块:函数",
    )
    ap.add_argument(
        "--dummy-ms", type=float, default=8.0,
        help="dummy 模式每次调用的目标耗时（毫秒，默认 8）",
    )
    ap.add_argument(
        "--jitter", type=float, default=0.45,
        help="dummy 模式耗时的相对抖动幅度，0~1（默认 0.45）",
    )
    ap.add_argument(
        "--seed", type=int, default=20260922,
        help="dummy 模式随机种子，保证可复现（默认固定值）",
    )
    ap.add_argument("--out", default="", help="输出目录；给出则写 CSV")
    ap.add_argument("--label", default="", help="写入 CSV 的标签，便于对比不同配置")
    args = ap.parse_args(argv)

    if args.runs < 1:
        print("错误：--runs 必须 >= 1", file=sys.stderr)
        return 2
    if not 0.0 <= args.jitter < 1.0:
        print("错误：--jitter 必须在 [0, 1) 区间内", file=sys.stderr)
        return 2

    # 构造被测对象
    if args.model == "dummy":
        step = _dummy_callable(args.dummy_ms, args.jitter, args.seed)
        label = args.label or ("dummy-%.1fms-j%.2f" % (args.dummy_ms, args.jitter))
    elif args.model.startswith("plugin:"):
        step = load_plugin(args.model)
        label = args.label or args.model
    else:
        print("错误：--model 只能是 dummy 或 plugin:模块:函数", file=sys.stderr)
        return 2

    print("被测对象 : %s" % (label,))
    print("预热 %d 次，随后测量 %d 次 …" % (args.warmup, args.runs))

    for _ in range(args.warmup):
        step()

    latencies_ms = []
    gaps_ms = []
    prev_end = None

    for i in range(args.runs):
        t0 = time.perf_counter()
        step()
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)
        if prev_end is not None:
            gaps_ms.append((t0 - prev_end) * 1000.0)
        prev_end = t1
        if (i + 1) % 10 == 0 or (i + 1) == args.runs:
            print("  已完成 %d / %d" % (i + 1, args.runs))

    s = summarize(latencies_ms, gaps_ms)
    print_summary(s, label)

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        path = os.path.join(args.out, "latency.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["label", "index", "latency_ms", "gap_ms"])
            for i, lat in enumerate(latencies_ms):
                gap = gaps_ms[i - 1] if i > 0 else ""
                w.writerow([label, i, "%.6f" % lat,
                            ("%.6f" % gap) if gap != "" else ""])
        print("原始数据已写入 : %s" % path)
        print("生成直方图     : python plot_hist.py %s" % path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
