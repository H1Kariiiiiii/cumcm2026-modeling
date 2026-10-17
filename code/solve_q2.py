# -*- coding: utf-8 -*-
"""问题 2：第二检测点的最优选择策略与候选区域

模型（题目分析报告 §5.2）
------------------------
* 几何设定：S1 处测得示向度 theta1，源 G 在楔形 W1 内、距 S1 为 r（未知），
  R_eff ∈ [1000,1500] 未知（R6）。
* 第二测点取在垂直于首条示向线的方向：S2 = S1 + t·u(theta1 ± 90°)。
* 条带近似闭式解（报告 §5.2.2）：
    面积准则  Area(t) = 4 r (r²+t²) α² / t          =>  t* = r
    直径准则  d(t)   = 2α sqrt((r²+t²)(4r²+t²)/t²)   =>  t* = √2 r,  d_min = 6 r tan1°
  其中 α = tan1°。
* 精确复核（报告 §5.2.2 口径声明）：精确楔形交 + 三分搜索给出
    直径准则精确 t*/r = 1.416366（与 √2 偏 +0.1522%）
    面积准则精确 t*/r = 1.002433（与 1 偏 +0.2433%）
  本脚本独立复算这两组值。
* 可探测约束（R6）：|S2-G| = sqrt(r²+t²) ≤ R_eff，最坏 R_eff=1000（r≤1000 时）
  => t ≤ sqrt(10⁶ - r²)；与 t*=√2 r 联立 => r ≤ 1000/√3 ≈ 577.35 m。
* 稳健化：r 未知 => 给出 minimax 与 Bayes 两种稳健 t，及 r>577.35 的两步策略。

输出（results/）
--------------
* q2_exact_optimum.csv / q2_curves.csv / q2_strategy.json
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geo import (EPS_ERR_DEG, azi_deg, diameter_calipers, localization_region,
                 polygon_area, unit_vec, wrap_deg)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

ALPHA = math.tan(math.radians(EPS_ERR_DEG))   # tan 1°


# ---------------------------------------------------------------- 精确目标函数
def exact_diam_area(r: float, t: float):
    """精确楔形交：S1=(0,0), G=(r,0), theta1=0, S2=(0,t)。返回 (直径, 面积)。"""
    s1 = np.array([0.0, 0.0])
    s2 = np.array([0.0, float(t)])
    g = np.array([float(r), 0.0])
    theta1 = azi_deg(g, s1)            # = 0
    theta2 = azi_deg(g, s2)
    poly, unb = localization_region([(s1, theta1), (s2, theta2)], EPS_ERR_DEG,
                                    arena_radius=None)
    if poly is None or len(poly) < 3 or unb:
        return float("inf"), float("inf")
    d, _, _ = diameter_calipers(poly)
    a = polygon_area(poly)
    return d, a


def strip_diam(r: float, t: float) -> float:
    return 2.0 * ALPHA * math.sqrt((r * r + t * t) * (4.0 * r * r + t * t) / (t * t))


def strip_area(r: float, t: float) -> float:
    return 4.0 * r * (r * r + t * t) * ALPHA * ALPHA / t


def ternary_min(f, lo: float, hi: float, iters: int = 200):
    """三分搜索（确定性），返回 (argmin, min)。"""
    for _ in range(iters):
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        if f(m1) <= f(m2):
            hi = m2
        else:
            lo = m1
    x = 0.5 * (lo + hi)
    return x, f(x)


def exact_optimum(r: float):
    """对给定 r 精确求直径/面积准则的最优 t。"""
    fd = lambda t: exact_diam_area(r, t)[0]
    fa = lambda t: exact_diam_area(r, t)[1]
    t_d, d_min = ternary_min(fd, 1.0, 4000.0)
    t_a, a_min = ternary_min(fa, 1.0, 4000.0)
    return {
        "r": r,
        "t_star_diam_exact": round(t_d, 4),
        "t_star_diam_ratio": round(t_d / r, 6),
        "t_star_area_exact": round(t_a, 4),
        "t_star_area_ratio": round(t_a / r, 6),
        "d_min_exact": round(d_min, 4),
        "d_min_strip_formula": round(6.0 * r * ALPHA, 4),
        "d_min_rel_dev": round((d_min - 6.0 * r * ALPHA) / (6.0 * r * ALPHA), 6),
        "area_min_exact": round(a_min, 4),
    }


# ---------------------------------------------------------------- 稳健策略
def t_detect_max(r: float, r_eff_worst: float | None = None) -> float:
    """可探测约束的上界 t ≤ sqrt(R_eff_worst² - r²)。

    r ≤ 1000 时最坏 R_eff=1000；r>1000 时由「S1 处已测得信号」知 R_eff ≥ r，
    最坏 R_eff = r（此时无 t>0 可行），按报告 §5.2.3/§9-R2 处理。
    """
    reff = max(1000.0, r) if r_eff_worst is None else r_eff_worst
    if r >= reff:
        return 0.0
    return math.sqrt(reff * reff - r * r)


def robust_t(r_lo: float, r_hi: float, criterion: str = "minimax",
             grid: int = 60, t_grid: int = 400):
    """先验 r∈[r_lo,r_hi] 上的稳健 t（单步、保证可探测的可行带内）。

    可行带：单 t 须对全部 r∈先验可探测 => t ≤ min_r t_detect_max(r) = t_detect_max(r_hi)
    （r>1000 的先验无正可行 t，返回 None，走两步策略）。
    * minimax：min_t max_r d(t;r)（条带近似）
    * bayes  ：min_t E_{r~U} d(t;r)
    """
    if r_hi > 1000.0:
        return None
    t_cap = t_detect_max(r_hi)
    if t_cap <= 20.0:
        return None
    rs = np.linspace(r_lo, r_hi, grid)
    ts = np.linspace(10.0, t_cap, t_grid)
    best = None
    for t in ts:
        vals = [strip_diam(r, t) for r in rs]
        obj = max(vals) if criterion == "minimax" else float(np.mean(vals))
        if best is None or obj < best[1]:
            best = (float(t), float(obj))
    return best


def two_step_plan(r_lo: float, r_hi: float):
    """两步策略（r 先验上界超过临界值 577.35 m 时）。

    第一步沿示向线前移 m，使剩余距离 r'=r-m 的上界降到临界值以下；
    第二步在垂直方向按 t*=√2·r' 布点。
    """
    r_crit = 1000.0 / math.sqrt(3.0)
    m = max(0.0, r_hi - r_crit)
    r_lo2 = max(5.0, r_lo - m)
    r_hi2 = min(r_hi - m, r_crit)
    return {
        "advance_m": round(m, 2),
        "r_after": [round(r_lo2, 2), round(r_hi2, 2)],
        "t_second_step": [round(math.sqrt(2.0) * r_lo2, 2),
                          round(math.sqrt(2.0) * r_hi2, 2)],
        "note": "先沿示向线前移 m=r_hi-577.35 m，再于垂直方向取 t=√2·r'（r' 为前移后距离）",
    }


def candidate_region(s1, theta1, r_prior=(100.0, 1000.0), r_grid=None):
    """候选区域 C（报告 §5.2.3）：以 S1 为极点、沿 theta1±90° 两条射线的参数化线段族。

    输出：
    * per_r —— 对先验内网格 r 的推荐 t 区间 [0.5r, t_use(r)]，
      其中 t_use(r)=min(√2·1500 上端按 t*(r)=√2r 截断, 可探测上界 sqrt(10⁶-r²))；
    * union —— 全族并集的 t 范围（候选线段的总范围）；
    * 两支射线与目标圆域的截断。
    """
    r_lo, r_hi = r_prior
    if r_grid is None:
        r_grid = [round(x, 1) for x in np.linspace(r_lo, min(r_hi, 1000.0), 10)]
    per_r = []
    for r in r_grid:
        t_ideal = math.sqrt(2.0) * r
        t_det = t_detect_max(r) if r <= 1000.0 else 0.0
        t_use = min(t_ideal, t_det)
        per_r.append({
            "r": round(float(r), 1),
            "t_ideal_sqrt2_r": round(t_ideal, 2),
            "t_detect_cap": round(t_det, 2),
            "t_use": round(t_use, 2),
            "feasible": bool(t_use >= 0.5 * r),
        })
    t_uses = [p["t_use"] for p in per_r if p["t_use"] > 0]
    union = [round(0.5 * r_lo, 2), round(max(t_uses), 2)] if t_uses else [0.0, 0.0]

    s1 = np.asarray(s1, float)
    branches = []
    for sgn in (+90.0, -90.0):
        u = unit_vec(theta1 + sgn)
        a = float(u @ u)
        b = 2.0 * float(s1 @ u)
        cc = float(s1 @ s1) - 1800.0 ** 2
        roots = [float(np.real(x)) for x in np.roots([a, b, cc]) if np.isreal(x)]
        pos = [x for x in roots if x > 0]
        t_arena = min(pos) if pos else float("inf")
        branches.append({
            "direction_deg": round((theta1 + sgn) % 360.0, 2),
            "t_arena_cap": round(t_arena, 2) if math.isfinite(t_arena) else None,
        })
    return {
        "s1": [float(s1[0]), float(s1[1])],
        "theta1_deg": theta1,
        "r_prior": list(r_prior),
        "per_r": per_r,
        "union_t_range": union,
        "branches": branches,
        "critical_r": round(1000.0 / math.sqrt(3.0), 4),
        "note": "C 为两支垂线射线上的参数化线段族；r>577.35 m 的支线按两步策略处理",
    }


def main():
    os.makedirs(RESULTS, exist_ok=True)

    # 1) 精确最优 t* 复核（报告冻结值：直径 1.416366 / 面积 1.002433）
    rs = [200.0, 400.0, 600.0, 800.0, 1000.0, 1200.0, 1450.0]
    opts = [exact_optimum(r) for r in rs]
    diam_ratios = [o["t_star_diam_ratio"] for o in opts]
    area_ratios = [o["t_star_area_ratio"] for o in opts]
    with open(os.path.join(RESULTS, "q2_exact_optimum.csv"), "w",
              encoding="utf-8") as f:
        f.write("r,t_star_diam_exact,t_star_diam_ratio,t_star_area_exact,"
                "t_star_area_ratio,d_min_exact,d_min_strip_formula,d_min_rel_dev\n")
        for o in opts:
            f.write(f"{o['r']},{o['t_star_diam_exact']},{o['t_star_diam_ratio']},"
                    f"{o['t_star_area_exact']},{o['t_star_area_ratio']},"
                    f"{o['d_min_exact']},{o['d_min_strip_formula']},"
                    f"{o['d_min_rel_dev']}\n")

    # 2) 性能曲线（r=600 与 r=1200 两组；精确 + 条带近似）
    curve_rows = []
    for r in (600.0, 1200.0):
        for t in range(50, 3001, 25):
            d_ex, a_ex = exact_diam_area(r, float(t))
            curve_rows.append({
                "r": r, "t": t,
                "d_exact": round(d_ex, 4), "d_strip": round(strip_diam(r, t), 4),
                "area_exact": round(a_ex, 4), "area_strip": round(strip_area(r, t), 4),
            })
    with open(os.path.join(RESULTS, "q2_curves.csv"), "w", encoding="utf-8") as f:
        f.write("r,t,d_exact,d_strip,area_exact,area_strip\n")
        for c in curve_rows:
            f.write(f"{c['r']},{c['t']},{c['d_exact']},{c['d_strip']},"
                    f"{c['area_exact']},{c['area_strip']}\n")

    # 3) 稳健策略（r 先验未知）
    #    注：robust_t 在 r_hi > 1000 m 时返回 None（单步无可行 t，须走两步策略）。
    #    此处显式分支处理，避免对 None 解包。
    robust = {}
    for prior in [(100.0, 577.35), (100.0, 1000.0), (300.0, 1500.0)]:
        for crit in ("minimax", "bayes"):
            key = f"r[{prior[0]:.0f},{prior[1]:.0f}]_{crit}"
            res = robust_t(*prior, criterion=crit)
            if res is None:
                robust[key] = {
                    "t_robust": None,
                    "objective_m": None,
                    "note": "r 先验上界 > 1000 m：最坏 R_eff 下单步无可行 t，"
                            "应改走两步策略 two_step_plan()",
                    "two_step": two_step_plan(*prior),
                }
                continue
            t, obj = res
            robust[key] = {"t_robust": round(t, 1), "objective_m": round(obj, 3)}

    # 4) 候选区域（示例：S1=(-300,200)，theta1=65°，先验 r∈[200,1000]）
    cand = candidate_region(np.array([-300.0, 200.0]), 65.0, (200.0, 1000.0))

    out = {
        "problem": "Q2 第二检测点的最优选择",
        "closed_form": {
            "area_criterion_t_star": "r",
            "diameter_criterion_t_star": "sqrt(2)*r",
            "d_min_formula": "6*r*tan(1°)",
        },
        "exact_verification": {
            "t_star_diam_ratio_mean": round(float(np.mean(diam_ratios)), 6),
            "t_star_diam_ratio_all": diam_ratios,
            "t_star_diam_ratio_reference": 1.416366,
            "t_star_area_ratio_mean": round(float(np.mean(area_ratios)), 6),
            "t_star_area_ratio_all": area_ratios,
            "t_star_area_ratio_reference": 1.002433,
            "d_min_rel_dev_range": [min(o["d_min_rel_dev"] for o in opts),
                                    max(o["d_min_rel_dev"] for o in opts)],
        },
        "robust_choices": robust,
        "candidate_region_example": cand,
        "strategy": {
            "rule": "S2 = S1 + t·u(theta1±90°)；直径准则 t*=√2·r（精确 1.416366r），"
                    "面积准则 t*=r（精确 1.002433r）",
            "unknown_r": "先验 r∈[r_lo,r_hi] 时取 minimax 或 Bayes 稳健 t；"
                         "r>577.35 m 时最坏 R_eff 下无可行 t，先沿示向线前移再垂直布点",
            "detectability": "t ≤ sqrt(max(1000,r)²-r²)（最坏 R_eff；r≤1000 即 sqrt(10⁶-r²)）",
        },
    }
    with open(os.path.join(RESULTS, "q2_strategy.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("[Q2] 精确最优与稳健策略完成")
    print(f"  直径准则 t*/r = {np.mean(diam_ratios):.6f}（参考 1.416366，√2={math.sqrt(2):.6f}）")
    print(f"  面积准则 t*/r = {np.mean(area_ratios):.6f}（参考 1.002433）")
    print(f"  d_min 相对 6r·tan1° 偏差范围: "
          f"[{out['exact_verification']['d_min_rel_dev_range'][0]:.4%}, "
          f"{out['exact_verification']['d_min_rel_dev_range'][1]:.4%}]")
    for k, v in robust.items():
        print(f"  稳健 t {k}: t={v['t_robust']} m (obj={v['objective_m']} m)")
    for b in cand["branches"]:
        print(f"  候选区域支线 {b['direction_deg']}°: 目标圆域截断上限 "
              f"t ≤ {b['t_arena_cap']} m")


if __name__ == "__main__":
    main()
