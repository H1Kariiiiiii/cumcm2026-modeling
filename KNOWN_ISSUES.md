# 已知问题

汇总在整理本仓库时实际运行代码发现的问题。所有条目都经复现验证，非推测。

---

## 1. `solve_q2.py` 与 `q2_strategy.json` 版本漂移（重要）

**现象**

提交包中的 `solve_q2.py` 无法完整运行，且即便修好也无法完全复现 `q2_strategy.json`。

**原始报错**

```
File "solve_q2.py", line 249, in main
    t, obj = robust_t(*prior, criterion=crit)
TypeError: cannot unpack non-iterable NoneType object
```

**原因**

`robust_t()` 在以下两种情况返回 `None`：

```python
if r_hi > 1000.0:      # 先验上界超过 1000 m
    return None
t_cap = t_detect_max(r_hi)
if t_cap <= 20.0:      # 可行带过窄
    return None
```

但 `main()` 中对 `(100.0, 1000.0)` 与 `(300.0, 1500.0)` 两组先验直接解包返回值。
其中 `r_hi = 1000` 时 `t_cap = √(1000² − 1000²) = 0`，同样触发 `return None`。

**与提交结果的不一致**

提交版 `q2_strategy.json` 的 `robust_choices` 中，这四组分支**都有值**：

| 分支 | 提交版 | 当前代码 |
|---|---|---|
| `r[100,577]_minimax` | 814.8 | 816.5 |
| `r[100,577]_bayes` | 516.7 | 517.3 |
| `r[100,1000]_minimax` | 173.9 | `None` |
| `r[100,1000]_bayes` | 382.6 | `None` |
| `r[300,1500]_minimax` | 171.0 | `None` |
| `r[300,1500]_bayes` | 364.9 | `None` |

`r[100,577]` 两组虽能算出，数值也与提交版略有差异（816.5 vs 814.8）。
说明**提交包里的这份脚本不是生成该 JSON 的那一版**——打包时代码与结果发生了分离。

**已做的修正**

为使脚本可运行，补了两处（均在 `solve_q2.py` 内，有中文注释标注）：

1. `main()` 中对 `robust_t()` 返回 `None` 的情况显式分支，记录 `two_step_plan()` 给出的两步策略，
   不再对 `None` 解包。
2. 结尾打印语句引用 `b['t_min']` / `b['t_max']`，但 `candidate_region()` 实际返回的是
   `direction_deg` / `t_arena_cap`，改为后者。

**影响范围**

- 不影响 `exact_verification`（问题 2 的核心结论 `t*/r = 1.416366`、`1.002433`、`d_min` 偏差）
  —— 该字段重跑后与提交版**完全一致**。
- 不影响 `closed_form` 与 `strategy` 字段。
- 仅影响 `robust_choices` 与 `candidate_region_example` 两个字段的部分条目。

**未做的事**

没有反向还原"能生成提交版"的那一版算法，因为无法确定当时用的是哪套网格参数，
强行拟合会引入无法验证的推测。此处如实记录差异。

---

## 2. 早期本地结果与最终口径不一致（已归档）

`archive/early-local-runs/` 中的两个文件来自更早的本地运行，覆盖比例与最终提交版不同：

| 来源 | n=2 | n=3 | n=4 | n=6 |
|---|---|---|---|---|
| `early-local-runs/q1_geometry.json` | 99.167% | 89.70% | 82.53% | 75.00% |
| `results/q1_examples.json`（最终版） | 98.59% | 84.12% | 77.77% | 67.49% |

**最终版口径以 `results/` 为准**，与论文表 4 一致。归档文件仅用于说明迭代过程，
不应引用其数值。

---

## 3. 图件非脚本生成

`figures/` 中的 PNG/SVG 是提交包里的成品，生成脚本 `make_figures.py` 依赖模拟器结果
（问题 3/4 的真机数据），未纳入本仓库。因此这些图**无法从本仓库直接重绘**。

图表数据来源见各图对应的 `results/` 文件与论文正文。

---

## 4. 未纳入的内容

以下内容属于策略与工程实现，不属于建模阶段，未包含：

- `strategy.py`、`solve_q3_official.py`、`solve_q4_official.py`：扫描布点、频道剪枝、三段式猎杀
- 官方模拟器交互层（`official_api.py`、`robot_api.py`）
- 12 局真机行为日志（含参赛队号，不适合公开）
