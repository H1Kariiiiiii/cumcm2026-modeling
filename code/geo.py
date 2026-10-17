# -*- coding: utf-8 -*-
"""2026 CUMCM B 题 · 共享计算几何模块（Q1/Q2/Q3/Q4 公用）

内容
----
* 楔形（可行角域）-> 半平面表示；半平面交（Sutherland-Hodgman 裁剪）
* 凸多边形直径：旋转卡壳（主算法）+ 顶点对暴力（对照）
* 最小包围圆（Welzl，固定种子洗牌，确定性）
* 圆域的正外切多边形近似（用于无界定位区域与圆域取交）
* 楔形交定位区域 K 的构造（含无界检测与圆域截断）

依据：题目分析报告 §5.1（楔形交模型、直径算法、覆盖判定）。
符号与术语遵循《术语表格.md》：定位区域 K、直径 d、最小包围圆半径 R*、
直径圆（半径 d/2）——三者不得混用。
"""
from __future__ import annotations

import math
import numpy as np

EPS_ERR_DEG = 1.0      # 示向度误差上界（R5，附录2(1)）
R_ARENA = 1800.0       # 目标区域半径（R12）
CLEAR_R = 20.0         # 清除半径（R10）
NEAR_R = 5.0           # 近距离阈值（R11）

_TOL = 1e-9


def cross2(e, v) -> float:
    """二维叉积 e×v（numpy>=2 移除了 2-D np.cross，此实现等价且更快）。"""
    return float(e[0] * v[1] - e[1] * v[0])


# ---------------------------------------------------------------- 角度与向量
def wrap_deg(a: float) -> float:
    """归一化到 [0, 360)。"""
    return float(a) % 360.0


def ang_diff_deg(a: float, b: float) -> float:
    """带符号角度差 a-b，落在 (-180, 180]。"""
    d = (a - b) % 360.0
    if d > 180.0:
        d -= 360.0
    return d


def unit_vec(a_deg: float) -> np.ndarray:
    a = math.radians(a_deg)
    return np.array([math.cos(a), math.sin(a)], dtype=float)


def azi_deg(p: np.ndarray, s: np.ndarray) -> float:
    """方位角 azi(P,S)：S 指向 P 的向量方位角（附件2 §1.2）。"""
    return wrap_deg(math.degrees(math.atan2(p[1] - s[1], p[0] - s[0])))


# ---------------------------------------------------------------- 半平面交
def clip_halfplane(poly: list, n: np.ndarray, c: float, tol: float = _TOL) -> list:
    """用半平面 {P : n·P >= c} 裁剪凸多边形 poly（顶点列表，np.ndarray）。

    Sutherland-Hodgman 逐边裁剪；返回新顶点列表（可能为空）。
    """
    if not poly:
        return []
    out: list = []
    m = len(poly)
    for i in range(m):
        a = poly[i]
        b = poly[(i + 1) % m]
        da = float(n @ a) - c
        db = float(n @ b) - c
        if da >= -tol:
            out.append(a)
        if (da > tol and db < -tol) or (da < -tol and db > tol):
            t = da / (da - db)
            out.append(a + t * (b - a))
    return out


def wedge_halfplanes(s: np.ndarray, theta_deg: float, eps_deg: float = EPS_ERR_DEG):
    """楔形 W = {P : |wrap(azi(P,S)-theta)| <= eps} 的两个半平面 (n, c) 表示。

    推导（报告 §5.1.1）：
      左边界（射线角 theta-eps）：P 在射线左侧 <=> cross(u(theta-eps), P-S) >= 0
        <=> n1·P >= n1·S,  n1 = u(theta-eps+90)
      右边界（射线角 theta+eps）：P 在射线右侧 <=> cross(u(theta+eps), P-S) <= 0
        <=> n2·P >= n2·S,  n2 = u(theta+eps-90)
    """
    n1 = unit_vec(theta_deg - eps_deg + 90.0)
    n2 = unit_vec(theta_deg + eps_deg - 90.0)
    s = np.asarray(s, dtype=float)
    return [(n1, float(n1 @ s)), (n2, float(n2 @ s))]


def circumscribed_polygon_halfplanes(radius: float, n_edges: int = 3600):
    """圆 { |P| <= radius } 的正外切 n 边形半平面表示（外切 => 包含圆域）。

    用于定位区域无界时与目标圆域取交（报告 §5.1.2/§5.1.3）。
    """
    halfplanes = []
    for k in range(n_edges):
        phi = 360.0 * k / n_edges            # 外法线方向
        n = -unit_vec(phi)                   # 约定 n·P >= c：-u·P >= -radius
        halfplanes.append((n, -radius))
    return halfplanes


def polygon_area(poly) -> float:
    if len(poly) < 3:
        return 0.0
    p = np.asarray(poly, dtype=float)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def polygon_centroid(poly) -> np.ndarray:
    """多边形面心（凸多边形，非退化时）。退化时返回顶点均值。"""
    p = np.asarray(poly, dtype=float)
    if len(p) < 3:
        return p.mean(axis=0)
    x, y = p[:, 0], p[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cross = x * yn - xn * y
    a = 0.5 * float(np.sum(cross))
    if abs(a) < 1e-12:
        return p.mean(axis=0)
    cx = float(np.sum((x + xn) * cross)) / (6.0 * a)
    cy = float(np.sum((y + yn) * cross)) / (6.0 * a)
    return np.array([cx, cy])


def normalize_convex(poly, tol: float = 1e-7) -> list:
    """去除近似重复与共线顶点，输出 CCW 序。"""
    if not poly:
        return []
    pts = [np.asarray(p, dtype=float) for p in poly]
    # 去重
    dedup: list = []
    for p in pts:
        if not dedup or np.linalg.norm(p - dedup[-1]) > tol:
            dedup.append(p)
    if len(dedup) > 1 and np.linalg.norm(dedup[0] - dedup[-1]) <= tol:
        dedup.pop()
    if len(dedup) < 3:
        return dedup
    # 保证 CCW
    p = np.asarray(dedup, dtype=float)
    x, y = p[:, 0], p[:, 1]
    area2 = float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    if area2 < 0:
        dedup = dedup[::-1]
    # 去共线
    out: list = []
    m = len(dedup)
    for i in range(m):
        a, b, cpt = dedup[i - 1], dedup[i], dedup[(i + 1) % m]
        cross = float((b[0] - a[0]) * (cpt[1] - a[1]) - (b[1] - a[1]) * (cpt[0] - a[0]))
        if abs(cross) > tol * max(1.0, float(np.linalg.norm(b - a)) * float(np.linalg.norm(cpt - b))):
            out.append(b)
    if len(out) >= 3:
        return out
    return dedup


def localization_region(bearings, eps_deg: float = EPS_ERR_DEG,
                        arena_radius: float | None = R_ARENA,
                        clip_arena: bool = False,
                        arena_edges: int = 3600,
                        extra_disks=None,
                        disk_edges: int = 64):
    """定位区域 K = ∩_i W_i（报告 §5.1.1 楔形交模型）。

    Parameters
    ----------
    bearings : list of (S, theta_deg) —— 检测点与示向度
    clip_arena : 无界（或显式要求）时与目标圆域取外切多边形交并标记
    arena_edges : 圆域外切多边形边数（越多越精确；批量统计可用小值提速）
    extra_disks : list of (center, radius) —— 附加圆域约束（如 B(P_i,1500)：
                  源被 P_i 探测到 ⟹ |G-P_i| ≤ r_eff ≤ 1500），外切多边形保守近似
    disk_edges : 附加圆域外切多边形边数

    Returns
    -------
    (poly, unbounded) ：poly 为顶点列表（CCW，已归一化），无界时为圆域截断结果；
                        区域为空返回 (None, False)。
    """
    B = 4.0e6
    poly = [np.array([-B, -B]), np.array([B, -B]), np.array([B, B]), np.array([-B, B])]
    for s, theta in bearings:
        for n, c in wedge_halfplanes(s, theta, eps_deg):
            poly = clip_halfplane(poly, n, c)
            if not poly:
                return None, False
    unbounded = any(float(np.max(np.abs(v))) >= 0.75 * B for v in poly)
    if (unbounded or clip_arena) and arena_radius is not None:
        for n, c in circumscribed_polygon_halfplanes(arena_radius, arena_edges):
            poly = clip_halfplane(poly, n, c)
            if not poly:
                return None, unbounded
    if extra_disks:
        for ctr, rad in extra_disks:
            ctr = np.asarray(ctr, dtype=float)
            for n, c0 in circumscribed_polygon_halfplanes(rad, disk_edges):
                poly = clip_halfplane(poly, n, c0 + float(n @ ctr))
                if not poly:
                    return None, unbounded
    return normalize_convex(poly), unbounded


# ---------------------------------------------------------------- 直径
def diameter_bruteforce(poly):
    """顶点对暴力直径 O(m^2)（对照算法）。"""
    if poly is None or len(poly) == 0:
        return 0.0, None, None
    p = np.asarray(poly, dtype=float)
    if len(p) == 1:
        return 0.0, p[0], p[0]
    best = -1.0
    bp = bq = p[0]
    for i in range(len(p)):
        d = np.linalg.norm(p[i + 1:] - p[i], axis=1)
        if len(d):
            j = int(np.argmax(d))
            if d[j] > best:
                best = float(d[j])
                bp, bq = p[i], p[i + 1 + j]
    return best, bp, bq


def diameter_calipers(poly):
    """旋转卡壳求凸多边形直径 O(m)（主算法，Toussaint 1983）。

    输入须为 CCW 凸多边形（normalize_convex 的输出可直接使用）。
    """
    if poly is None or len(poly) == 0:
        return 0.0, None, None
    p = np.asarray(poly, dtype=float)
    m = len(p)
    if m < 3:
        return diameter_bruteforce(poly)
    j = 1
    best = -1.0
    bp = bq = p[0]
    for i in range(m):
        i1 = (i + 1) % m
        e = p[i1] - p[i]
        while True:
            j1 = (j + 1) % m
            cur = cross2(e, p[j] - p[i])
            nxt = cross2(e, p[j1] - p[i])
            if nxt > cur + 1e-12:
                j = j1
            else:
                break
        d = float(np.linalg.norm(p[i] - p[j]))
        if d > best:
            best, bp, bq = d, p[i], p[j]
        d2 = float(np.linalg.norm(p[i1] - p[j]))
        if d2 > best:
            best, bp, bq = d2, p[i1], p[j]
    return best, bp, bq


# ---------------------------------------------------------------- 最小包围圆
def _circumcircle(a, b, cpt):
    """三角形外接圆（圆心, 半径）；退化返回 None。"""
    d = 2.0 * (a[0] * (b[1] - cpt[1]) + b[0] * (cpt[1] - a[1]) + cpt[0] * (a[1] - b[1]))
    if abs(d) < 1e-12:
        return None
    ux = ((a[0] ** 2 + a[1] ** 2) * (b[1] - cpt[1])
          + (b[0] ** 2 + b[1] ** 2) * (cpt[1] - a[1])
          + (cpt[0] ** 2 + cpt[1] ** 2) * (a[1] - b[1])) / d
    uy = ((a[0] ** 2 + a[1] ** 2) * (cpt[0] - b[0])
          + (b[0] ** 2 + b[1] ** 2) * (a[0] - cpt[0])
          + (cpt[0] ** 2 + cpt[1] ** 2) * (b[0] - a[0])) / d
    c = np.array([ux, uy])
    return c, float(np.linalg.norm(a - c))


def mec(points, seed: int = 12345):
    """最小包围圆（Welzl 随机增量，固定种子 => 确定性）。返回 (圆心, 半径)。"""
    pts = np.asarray(points, dtype=float)
    if len(pts) == 0:
        return np.zeros(2), 0.0
    if len(pts) == 1:
        return pts[0].copy(), 0.0
    rng = np.random.default_rng(seed)
    pts = pts[rng.permutation(len(pts))]
    c = pts[0].copy()
    r = 0.0
    for i in range(1, len(pts)):
        if np.linalg.norm(pts[i] - c) > r + 1e-12:
            c = pts[i].copy()
            r = 0.0
            for j in range(i):
                if np.linalg.norm(pts[j] - c) > r + 1e-12:
                    c = (pts[i] + pts[j]) / 2.0
                    r = float(np.linalg.norm(pts[i] - c))
                    for k in range(j):
                        if np.linalg.norm(pts[k] - c) > r + 1e-12:
                            cc = _circumcircle(pts[i], pts[j], pts[k])
                            if cc is not None:
                                c, r = cc
    return c, r


# ---------------------------------------------------------------- 演示/校验
def point_in_polygon(p: np.ndarray, poly, tol: float = 1e-9) -> bool:
    """凸多边形包含测试（按半平面约定：所有边 n·(P-A) <= 0，CCW 多边形）。"""
    if not poly or len(poly) < 3:
        return False
    q = np.asarray(p, dtype=float)
    m = len(poly)
    for i in range(m):
        a = np.asarray(poly[i], dtype=float)
        b = np.asarray(poly[(i + 1) % m], dtype=float)
        e = b - a
        if cross2(e, q - a) < -tol:
            return False
    return True


def point_poly_distance(p, poly) -> float:
    """点到凸多边形的距离（点在内或边界上为 0；poly 为空返回 inf）。"""
    if not poly:
        return float("inf")
    q = np.asarray(p, dtype=float)
    if len(poly) == 1:
        return float(np.linalg.norm(q - np.asarray(poly[0], dtype=float)))
    if len(poly) >= 3 and point_in_polygon(q, poly, tol=1e-12):
        return 0.0
    pts = np.asarray(poly, dtype=float)
    m = len(pts)
    best = float("inf")
    for i in range(m):
        a, b = pts[i], pts[(i + 1) % m]
        e = b - a
        L2 = float(e @ e)
        if L2 < 1e-18:
            d = float(np.linalg.norm(q - a))
        else:
            t = float(np.clip((q - a) @ e / L2, 0.0, 1.0))
            d = float(np.linalg.norm(q - (a + t * e)))
        if d < best:
            best = d
    return best
