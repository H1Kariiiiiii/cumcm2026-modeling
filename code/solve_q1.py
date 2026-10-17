# -*- coding: utf-8 -*-
"""问题 1：交会定位法 · 多边形定位区域直径算法与覆盖判定

模型（题目分析报告 §5.1，楔形交模型为读图裁定后的正式模型）
----------------------------------------------------------
1. 楔形 W_i = {P : |wrap(azi(P,S_i) - theta_i)| <= 1°}，等价两个半平面之交；
2. 定位区域 K = ∩_i W_i（凸多边形，边数至多 2n）；无界时与目标圆域取交并标记；
3. 直径 diam(K)：主算法 = 半平面交 + 旋转卡壳 O(m)；对照 = 顶点对暴力 O(m^2)；
4. 覆盖判定：直径圆（半径 d/2）覆盖 K  <=>  最小包围圆半径 R* <= d/2；
   理论界（Jung）：d/2 <= R* <= d/sqrt(3)，等边三角形达到上界 2/sqrt(3)=1.1547。

输出（results/）
--------------
* q1_examples.json ：算例（含图 2 复现算例、覆盖反例、退化算例）
* q1_mc_cover.csv  ：蒙特卡洛覆盖比例随测点数 n
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geo import (EPS_ERR_DEG, R_ARENA, azi_deg, diameter_bruteforce,
                 diameter_calipers, localization_region, mec, unit_vec,
                 wrap_deg)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")


def solve_case(bearings, clip_arena=False):
    """单算例：返回 K 顶点、直径（两种算法）、R*、覆盖判定、放大系数。"""
    poly, unbounded = localization_region(bearings, EPS_ERR_DEG,
                                          arena_radius=R_ARENA,
                                          clip_arena=clip_arena)
    if poly is None or len(poly) == 0:
        return {"empty": True}
    d_cal, p_cal, q_cal = diameter_calipers(poly)
    d_bf, _, _ = diameter_bruteforce(poly)
    c_star, r_star = mec(poly)
    covered = bool(r_star <= d_cal / 2.0 + 1e-9)
    return {
        "empty": False,
        "unbounded_clipped": bool(unbounded),
        "vertices": [[float(v[0]), float(v[1])] for v in poly],
        "n_vertices": len(poly),
        "diameter_calipers": round(d_cal, 6),
        "diameter_bruteforce": round(d_bf, 6),
        "algorithms_agree": bool(abs(d_cal - d_bf) < 1e-6),
        "antipodal_pair": [[float(p_cal[0]), float(p_cal[1])],
                           [float(q_cal[0]), float(q_cal[1])]],
        "mec_center": [float(c_star[0]), float(c_star[1])],
        "mec_radius": round(float(r_star), 6),
        "diameter_circle_covers": covered,
        "radius_ratio": round(float(r_star / (d_cal / 2.0)), 6) if d_cal > 0 else None,
        "jung_ratio": 2.0 / math.sqrt(3.0),
    }


def _bearing_to(g, s, err_deg=0.0):
    """真值方位角 + 确定性误差 -> 示向度。"""
    return wrap_deg(azi_deg(g, s) + err_deg)


def build_examples():
    """三个代表性算例（确定性构造）。"""
    ex = {}

    # 算例 A：复现题目图 2 的两测点交会（有界四边形）
    g = np.array([420.0, 660.0])
    s1 = np.array([-560.0, -380.0])
    s2 = np.array([980.0, -540.0])
    b1 = _bearing_to(g, s1, +0.6)
    b2 = _bearing_to(g, s2, -0.4)
    ex["A_two_point_intersection"] = {
        "description": "两测点交会定位（图 2 情形）：有界四边形定位区域",
        "source": [420.0, 660.0],
        "bearings": [
            {"S": [float(s1[0]), float(s1[1])], "svd_deg": round(b1, 2)},
            {"S": [float(s2[0]), float(s2[1])], "svd_deg": round(b2, 2)},
        ],
        "result": solve_case([(s1, b1), (s2, b2)]),
    }

    # 算例 B：三测点、示向线呈锐角汇聚 -> 直径圆不可覆盖的反例
    g = np.array([0.0, 0.0])
    pts = [np.array([900.0 * math.cos(math.radians(a)),
                     900.0 * math.sin(math.radians(a))]) for a in (90.0, 210.0, 330.0)]
    bears = [(p, _bearing_to(g, p, 0.0)) for p in pts]
    ex["B_cover_counterexample"] = {
        "description": "三测点对称汇聚：定位区域近等边三角形，直径圆不覆盖（Jung 上界）",
        "source": [0.0, 0.0],
        "bearings": [{"S": [float(p[0]), float(p[1])], "svd_deg": round(t, 2)}
                     for p, t in bears],
        "result": solve_case(bears),
    }

    # 算例 C：两示向线近并行 -> 无界，与目标圆域取交并标记
    s1 = np.array([-900.0, -300.0])
    s2 = np.array([-860.0, -250.0])
    t1 = 35.0
    t2 = 36.2     # 近并行
    ex["C_unbounded_clipped"] = {
        "description": "示向线近并行（交会角过小）：定位区域无界，与半径 1800 m 圆域取交",
        "bearings": [
            {"S": [float(s1[0]), float(s1[1])], "svd_deg": t1},
            {"S": [float(s2[0]), float(s2[1])], "svd_deg": t2},
        ],
        "result": solve_case([(s1, t1), (s2, t2)], clip_arena=True),
    }
    return ex


def mc_cover_study(trials: int = 3000, seed: int = 42,
                   ns=(2, 3, 4, 6), r_lo: float = 100.0, r_hi: float = 1500.0):
    """蒙特卡洛：随机检测点 + U(-1,1) 示向度误差，统计直径圆覆盖比例。

    检测点围绕源随机布设（距离 U(r_lo, r_hi)，方位均匀），与报告 §5.1.4 同口径。
    同时记录半径比最大的算例（确定性复现的覆盖反例）。
    """
    rng = np.random.default_rng(seed)
    rows = []
    worst = {"ratio": 0.0, "n_points": None, "bearings": None}
    for n in ns:
        ratios = []
        covered = 0
        unbounded_cnt = 0
        for _ in range(trials):
            g = np.array([0.0, 0.0])
            dist = rng.uniform(r_lo, r_hi, size=n)
            ang = rng.uniform(0.0, 360.0, size=n)
            # 保证任意两点与源不共线过近（交会角病态时跳过该样本）
            angs = np.sort(ang)
            gaps = np.diff(np.concatenate([angs, angs[:1] + 360.0]))
            if gaps.min() < 1.0:
                continue
            bears = []
            for k in range(n):
                s = g + dist[k] * unit_vec(float(ang[k]))
                theta = _bearing_to(g, s, float(rng.uniform(-1.0, 1.0)))
                bears.append((s, theta))
            poly, unb = localization_region(bears, EPS_ERR_DEG, R_ARENA,
                                            clip_arena=False, arena_edges=180)
            if poly is None or len(poly) < 3:
                continue
            if unb:
                unbounded_cnt += 1
            d, _, _ = diameter_calipers(poly)
            _, r_star = mec(poly)
            if d <= 0:
                continue
            ratio = float(r_star / (d / 2.0))
            ratios.append(ratio)
            if r_star <= d / 2.0 + 1e-9:
                covered += 1
            if ratio > worst["ratio"]:
                worst = {
                    "ratio": ratio,
                    "n_points": n,
                    "bearings": [(s.tolist(), float(t)) for s, t in bears],
                }
        ratios = np.asarray(ratios)
        rows.append({
            "n_points": n,
            "trials": int(len(ratios)),
            "cover_count": int(covered),
            "cover_ratio": round(float(covered) / len(ratios), 6),
            "ratio_mean": round(float(ratios.mean()), 6),
            "ratio_p95": round(float(np.percentile(ratios, 95)), 6),
            "ratio_max": round(float(ratios.max()), 6),
            "unbounded_clipped": int(unbounded_cnt),
            "jung_upper": round(2.0 / math.sqrt(3.0), 6),
        })
    return rows, worst


def verify_jung_tightness():
    """等边三角形达到 Jung 上界（回归测试）。"""
    tri = [np.array([0.0, 0.0]),
           np.array([1000.0, 0.0]),
           np.array([500.0, 500.0 * math.sqrt(3.0)])]
    d, _, _ = diameter_bruteforce(tri)
    _, r = mec(tri)
    return {
        "equilateral_diameter": round(d, 6),
        "equilateral_mec_radius": round(r, 6),
        "ratio_vs_half_diameter": round(r / (d / 2.0), 6),
        "theory_2_over_sqrt3": round(2.0 / math.sqrt(3.0), 6),
        "reaches_jung_bound": bool(abs(r / (d / 2.0) - 2.0 / math.sqrt(3.0)) < 1e-6),
    }


def main():
    os.makedirs(RESULTS, exist_ok=True)
    examples = build_examples()
    for name, e in examples.items():
        r = e["result"]
        assert r.get("algorithms_agree", True), f"算例 {name} 两算法不一致"

    mc, worst = mc_cover_study()
    jung = verify_jung_tightness()

    # MC 最差算例作为确定性覆盖反例（seed=42 可复现）
    worst_bears = [(np.asarray(s), t) for s, t in worst["bearings"]]
    worst_res = solve_case(worst_bears)
    examples["D_mc_worst_case"] = {
        "description": "蒙特卡洛中半径比最大的算例（确定性复现）：直径圆不覆盖的实测反例",
        "bearings": [{"S": [float(s[0]), float(s[1])], "svd_deg": round(t, 2)}
                     for s, t in worst_bears],
        "result": worst_res,
        "mc_ratio": round(worst["ratio"], 6),
    }

    out = {
        "problem": "Q1 交会定位区域直径与覆盖判定",
        "epsilon_deg": EPS_ERR_DEG,
        "examples": examples,
        "monte_carlo_cover": mc,
        "jung_regression": jung,
        "conclusion": {
            "diameter_algorithm": "半平面交(Sutherland-Hodgman) + 旋转卡壳 O(m)；"
                                  "与顶点对暴力 O(m^2) 全部算例逐位一致",
            "coverage_verdict": "以定位区域直径为直径的圆不能保证覆盖定位区域",
            "criterion": "直径圆覆盖 K <=> R*(K) <= d/2（最小包围圆半径不超过半直径）",
            "worst_case_amplification": "最坏需放大 2/sqrt(3)≈1.154701 倍（Jung 定理，等边三角形取等）",
        },
    }
    with open(os.path.join(RESULTS, "q1_examples.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # CSV：蒙特卡洛覆盖统计
    csv_path = os.path.join(RESULTS, "q1_mc_cover.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("n_points,trials,cover_count,cover_ratio,ratio_mean,ratio_p95,"
                "ratio_max,unbounded_clipped,jung_upper\n")
        for r in mc:
            f.write(f"{r['n_points']},{r['trials']},{r['cover_count']},"
                    f"{r['cover_ratio']},{r['ratio_mean']},{r['ratio_p95']},"
                    f"{r['ratio_max']},{r['unbounded_clipped']},{r['jung_upper']}\n")

    print("[Q1] 算例与蒙特卡洛完成")
    for name, e in examples.items():
        r = e["result"]
        if r.get("empty"):
            print(f"  {name}: 区域为空")
            continue
        print(f"  {name}: 顶点数={r['n_vertices']} 直径={r['diameter_calipers']:.3f} m "
              f"R*={r['mec_radius']:.3f} m 直径圆覆盖={r['diameter_circle_covers']} "
              f"半径比={r['radius_ratio']}")
    for r in mc:
        print(f"  MC n={r['n_points']}: 覆盖比例={r['cover_ratio']*100:.2f}% "
              f"p95={r['ratio_p95']:.6f} max={r['ratio_max']:.6f}")
    print(f"  Jung 回归: 等边三角形半径比={jung['ratio_vs_half_diameter']} "
          f"理论={jung['theory_2_over_sqrt3']} 达界={jung['reaches_jung_bound']}")


if __name__ == "__main__":
    main()
