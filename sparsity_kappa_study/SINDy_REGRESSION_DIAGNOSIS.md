# SINDy 回归失效诊断：为什么 `sindy-ssa` 的 SINDy 比 `FYP_Development_V8` 差得多，以及如何修回

> 面向对象：接手修复的工程师 / AI agent。
> 目标：把 `sindy-ssa` 的 `SINDy`（对观测 MEE 导数做稀疏辨识的那一路）从 ideal 档的
> “重建推力超执行机构上限 ~7 500 倍、99.9 % 采样被截断、EI 钉在饱和地板”恢复到
> `FYP_Development_V8` 老方法的水平（甚至更好）。
>
> 本文所有数字都可以用 `sparsity_kappa_study/` 下的脚本复现；**诊断阶段没有修改 `src/`
> 任何文件**。

---

## 0. TL;DR（English summary for the implementing agent）

1. `SINDy` 的**库函数、中心差分求导、GVE 反演在新旧两版中完全一致**；唯一致命的差异是
   **选定支撑集之后的“系数重拟合”**：
   * 老代码调用 `ps.SSR(max_iter=20, alpha=1)`，PySINDy 的默认 `unbias=True` 会把选中的支撑集
     **用 OLS 重新解一遍**（`FYP_Development_V8/SINDy/SINDy_Ori.py:66`）。
   * 新代码 `fit_ssr_path`（`src/sindy_ssa/sindy/regression.py:80`）每一步只用 **ridge**
     重拟合（`_ridge_refit`，同文件 `:56`），**从不做 OLS 无偏化**。
2. 叠加第二处改动：新代码对原库打开 `normalize_columns=True`
   （`configs/config.yaml:45`，`config.py:176`），而 `original_alpha = 1.0`
   （`configs/config.yaml:38`）。ridge 的单系数收缩因子是
   `fnorm² / (fnorm² + alpha)`：老代码列是原始尺度（`fnorm² ≈ 10³–10⁵`）→ 因子 ≈ 1（ridge≈OLS）；
   新代码列是单位范数（`fnorm² = 1`）→ 因子 = `1/(1+1) = 0.5`，**每个保留系数被系统性砍半**。
3. 一个“整体偏小一半”的导数模型，其残差**与自然动力学同量级且强相关**，而自然背景比推力引起的
   导数大 10²–10⁶ 倍（`L_dot` 5.9e6）。这条结构化残差经病态 GVE 反解放大成 ~10⁻³ km/s² 的
   虚假推力（真值 ~1.3e-7，执行机构上限 2.125e-7），**每个采样点都被截断**成 ±`T_max/m` 方波，
   EI 被压到 ≈24.6 m/s 的饱和地板。
4. **不是原因的东西**：论文的 κ（`sparsity_tol`）选择规则、`p/6371` 缩放、库函数集合、
   中心差分、`MEEdot2Acc`、观测频率。加入无偏化后，新的 κ-L 曲线选择反而是所有组合里最好的。
5. **修复**：在 `fit_ssr_auto` 里对 κ 选出的支撑集做一次 OLS 重拟合（≈8 行，等价于
   `unbias=True`）；可选地把 `sindy.normalize_columns` 设为 `false`。
   预期：ideal 档 `SINDy` 的 EI 由 **19 040 → 1.96 m/s**（截断后 24.6 → 1.96），
   `max|u_n|` 由 1.6e-3 → 1.1e-7，**不再触发执行机构饱和**。

---

## 1. 症状（可复现）

场景：`stationkeeping`，`ideal`（无噪声、满采样率 10 s），**全长弧辨识**（`fit_fraction = 1.0`），
用流程自己的 `run_reconstruction` 口径。真值冲量 3.751 m/s；真值 `max|u_r| = 1.347e-7 km/s²`，
`u_n ≡ 0`；执行机构上限 `T_max/m = 0.17/800/1000 = 2.125e-7 km/s² (= 0.2125 mm/s²)`。

| 方法 | EI [m/s] | EPE 3D [km] |
|---|---|---|
| MEEFD-SG | 0.0961 | 0.2687 |
| MEEFD-Spline | 0.1390 | 0.2689 |
| TIME-OLS | 0.0958 | 0.2068 |
| **SINDy** | **24.5923** | **18.6954** |
| SINDy-TIME | 0.1166 | 0.2794 |
| SINDy-UMEE | 0.4328 | 0.1478 |
| ORACLE（用真导数走同一条反演链） | 4.4e-4 | — |

规避档同型：SINDy 9.149 / 17.32 km，而 SINDy-TIME 0.3503 / 0.739 km、SINDy-UMEE 0.5805 / 0.146 km。

SINDy 的原始（未截断）重建：

| 分量 | max | 说明 |
|---|---|---|
| `u_r` | 1.28e-05 | 真值的 ~95 倍 |
| `u_t` | 3.38e-06 | 真值的 ~67 倍 |
| `u_n` | **1.60e-03** | 真值恒为 0；超执行机构上限 **7 500 倍** |

截断比例 **99.9 %**；未截断 EI = 1.9e4 m/s（≈ 真值冲量的 5 000 倍）。
因此截图里那条 ±0.21 mm/s² 的方波**不是辨识噪声，而是"模型输出爆表后被执行机构限幅"的结果**，
而报告出的 EI ≈ 24.6 m/s 就是"饱和地板"（`Σ_c ∫(u_max − |u_true|) dt ≈ 24.6`），
对任何超限模型都相同，**在这个区间内 EI 没有区分度**。

---

## 2. 新旧 SINDy 逐步对照

老实现：`sparsity_kappa_study/FYP_Development_V8/SINDy/SINDy_Ori.py`（约 40 行主体）。
新实现：`src/sindy_ssa/sindy/sindy_original.py`（`run_sindy_original`）+ `regression.py`。

| # | 步骤 | 老 (`FYP_Development_V8`) | 新 (`sindy-ssa`) | 是否致因 |
|---|---|---|---|---|
| 1 | 观测频率 | `ObserveSat(SamplingRate=12)` → 120 s | 噪声档 ×6 → 60 s；`ideal` ×1 → 10 s | 否 |
| 2 | 设计矩阵 | `np.column_stack((p/6371, f, g, h, k, L))`（`SINDy_Ori.py:55`） | 原始 `p`（km），`normalize_columns=True`（`sindy_original.py:54`） | 见 #4，**放大器** |
| 3 | 稀疏路径 | `ps.SSR(max_iter=20, alpha=1)`（`SINDy_Ori.py:66`），走 PySINDy 内部的逐步删除路径 | `fit_ssr_path`（`regression.py:80`）自定义路径，`max_iter` 被忽略，走完整条路径（κ 选择） | 否（选择规则本身无害） |
| 4 | **系数重拟合** | **`unbias=True`（PySINDy 默认）：对该支撑集做 OLS 重解** | **`_ridge_refit`（`regression.py:56`）只用 ridge，`alpha=1`，从不无偏化** | ✅ **主因** |
| 5 | 稀疏度选择 | PySINDy 自身误差拐点（未传 `kappa`） | 论文 κ-L 曲线（`select_elbow`，`regression.py:157`；κ=`sparsity_tol`=0.02） | 否（见 §5） |
| 6 | 推力后处理 | `MinusAcc(...).smooth(window=51)`（`SINDy_Ori.py:94`；`Sequence.smooth` 用 SG，`Sequence.py:95`） | 无（只有 MEEFD-SG 平滑） | 次要（≈2.3×） |
| 7 | 执行机构饱和 | **不存在**（`Evaluate.py` 直接由 CSV 算 EI/EPE） | `saturate_thrust` 裁到 `T_max/m`，EI 与重推演前都裁（`experiments/run_sindy.py`） | 否，但**掩盖并放大症状** |
| 8 | 评价窗口 | 整个观测弧；`Evaluate.py` 分 R/T/N 报告，另有一版 "Excluding N" | 只取机动区间，三分量合并 | 否（口径差异） |
| 9 | 库函数集合 | 36 项（Eq. 23） | **逐项相同**（`libraries.py:original_library`） | 否 |
| 10 | 求导/反演 | `FD_MEE` 中心差分；`MEEdot2Acc` plain `lstsq` | 同名同式 | 否 |

> `p/6371` 其实是**空操作**：SINDy 库（Eq. 23）里不含 `p` 项，缩放不影响任何库列。
> 真正承重的两个开关是 **`unbias`** 与 **`normalize_columns`**，再加上新增的**截断**。

---

## 3. 根因机制

### 3.1 缺失的 OLS 无偏重拟合（主因）

`fit_ssr_path` 在每一步删除后调用

```python
A = theta_j.T @ theta_j + alpha * np.eye(cols.sum())
Xi[cols, j] = np.linalg.solve(A, theta_j.T @ Y[:, j])      # regression.py::_ridge_refit
```

返回的就是**带 ridge 偏置的系数**；`fit_ssr_auto` 直接把该点作为最终模型
（`Xi = next(p["Xi"] for p in path if p["max_iter"] == selected_iter)`），
没有任何 `unbias`/OLS 步骤。

PySINDy 的 SSR 默认 `unbias=True`，在 `_reduce` 选点之后调用 `_unbias`：
对每个输出列，在其支撑集上用 `LinearRegression(fit_intercept=False)` 重新求解。
老代码用的就是这条默认路径。

### 3.2 为什么 `normalize_columns=True` 会把它放大 ~40 倍

ridge 对单个系数的收缩因子：

```
k_j = ||theta_j||² / (||theta_j||² + alpha)
```

| 配置 | `||theta_j||²` 量级 | `alpha` | `k_j` | 后果 |
|---|---|---|---|---|
| 老：原始列 | `10³ ~ 10⁵`（n≈4000 个样本） | 1.0 | ≈ 1 | ridge ≈ OLS，**alpha 实际无效** |
| 新：列 L2 归一化 | 1 | 1.0 | **0.5** | **每个保留系数被砍半** |

所以老方法“碰巧”是无偏的（`alpha=1` 在原始尺度下不起作用，且还有 `unbias=True` 兜底），
而新方法在单位列范数下把同一个 `alpha=1.0` 变成“半系数”偏置，而且没有兜底。

### 3.3 为什么“砍一半”会变成 10⁴ 倍的虚假推力

推力构造（`preprocessing.subtract_acceleration` + `MEEdot2Acc`）是线性映射：

```
u = MEEdot2Acc(X, Xdot_model) - MEEdot2Acc(X, Xdot_natural)
  = MEEdot2Acc(X, (Xdot_model - Xdot_obs) + (Xdot_obs - Xdot_natural))
                     └── 模型残差 ──┘   └── 真实推力项 ──┘
```

模型输出整体偏小一半 ⇒ `Xdot_model − Xdot_obs ≈ −Xdot/2`，**与自然背景同量级、且与自然动力学强相关**，
而自然背景远大于推力引起的导数（拟合窗口 RMS，`out_tol_obs/*/derivative_scales.json`）：

| 导数 | 自然背景 RMS | 推力信号 RMS | 背景/信号 |
|---|---|---|---|
| `p_dot` | 6.94e-03 | 4.60e-05 | 1.5e+02 |
| `f_dot` | 1.32e-06 | 1.04e-08 | 1.3e+02 |
| `g_dot` | 1.56e-06 | 1.05e-08 | 1.5e+02 |
| `h_dot` | 3.14e-07 | 3.24e-11 | 9.7e+03 |
| `k_dot` | 5.41e-07 | 2.56e-11 | 2.1e+04 |
| `L_dot` | 1.10e-03 | 1.88e-10 | **5.9e+06** |

要复原 3.75 m/s 的冲量，模型必须把 `L_dot` 算准到 ~1e-11 rad/s（相对自然背景 ~1e-8）；
“整体砍半”带来的残差比这个要求大 4–6 个数量级，反解后自然变成 10⁻³ km/s² 的虚假推力。

**结构性残差 ≠ 拟合残差。** 一个真正无偏的模型即使支撑集很小，残差也是“拟合残差”（小、近似不相关）；
被系统性压缩的模型残差是“信号相关残差”，量级等同信号本身——这正是新旧两版的本质差别。

### 3.4 饱和与 EI 地板（症状掩盖）

SINDy 的原始输出 1.6e-3 km/s² 超上限 7 500 倍，`saturate_thrust` 把它裁成 ±`T_max/m` 方波；
EI 随之收敛到 `Σ_c ∫(u_max − |u_true|)dt ≈ 24.6 m/s`，**与模型好坏无关**。
老流程不做截断，因此同样的模型问题会以“未截断的大 EI”暴露（本诊断里 1.9e4 m/s），
不会呈现为方波。做修复验证时请同时看 `ei_raw_total`（未截断）与 `clipped_frac`。

---

## 4. 证据：同一份数据上的开关消融

数据：`stationkeeping` / `ideal` / 满弧 / 观测状态评估；`alpha=1`，`max_iter=20`；
EI 单位 m/s，`max|u_n|` 单位 km/s²（上限 2.125e-7）。
（表源：`sparsity_kappa_study/out_tol_obs/stationkeeping/legacy_ablation.csv`）

| 选择规则 | 无偏重拟合 `unbias` | 列归一化 | n_a | max\|u_n\| | EI（未截断） | EI（截断后） |
|---|---|---|---|---|---|---|
| 拐点（老默认） | **是** | 否 | 102 | 6.2e-07 | 16.7 | 12.7 |
| 拐点 | 是 | 是 | 114 | 1.1e-06 | 17.9 | 11.1 |
| 拐点 | **否** | 否 | 102 | 3.1e-05 | 506 | 22.7 |
| 拐点 | **否** | 是 | 114 | 1.6e-03 | 18 870 | 24.6 |
| κ-L 曲线 | **是** | 否 | 102 | **1.1e-07** | **1.96** | **1.96** |
| κ-L 曲线 | 是 | 是 | 144 | 1.3e-07 | 3.66 | 3.35 |
| κ-L 曲线 | **否** | 否 | 102 | 3.1e-05 | 506 | 22.7 |
| κ-L 曲线 ← **当前代码** | **否** | 是 | 144 | **1.6e-03** | **19 040** | 24.6 |

三条结论：

* **单变量对照**：第 6 行 vs 第 8 行**支撑集完全相同（144 项）**，只差一次 OLS 重拟合 →
  EI 19 040 → 3.66（**5 200 倍**）。这就是责任锁定。
* **协同效应**：只丢无偏化 → 506（约 30× 变差）；只加列归一化且无偏化 → 也 506（无害）；
  两者同时 → 19 040（再放大 37×）。**两处改动单独都“不算致命”，合起来才是灾难**，
  这解释了为什么“改变点不多”却后果巨大。
* **κ 选择规则清白**：补回无偏化后，新规则（1.96）比老的拐点规则（16.7）好 8 倍；
  老方法的表现并不是靠拐点规则，而是靠无偏化。

旁证（老流程自己保存的产物 `FYP_Development_V8/ReportFigs/FreqBatch/2min/Output/SINDyOri/U_SINDyOri.csv`）：
`max u_r = 5.83e-07`、`max u_t = 3.91e-07`（与真值同量级）、`max u_n = 8.07e-05`（超限 380 倍但平滑），
与 `FYP_Development_V8/Figures/AccCompare1.png` 里 SINDy 在 ±1e-6 内摆动的观感一致。
新模型的 `u_n` 又大了 20 倍，再被裁成方波——这就是“新旧观感差异”的直接来源。

---

## 5. 无辜的改动（不要在这上面找原因）

1. **论文的 κ（`sparsity_tol`）与 L 曲线选择**：它是 `select_elbow` 的 `tol`，由
   `cfg.sindy.sparsity_tol` 传入（0.02），**是活的**；对 ideal 档 κ 从 0.001 扫到 2.0，
   SINDy 的 `n_a` 从 198 变到 6，但 EI 一直钉在饱和地板 / 未截断 EI 1.7e4–3.9e4。
   加入无偏化后它是最好的选择规则。
2. **`p/6371`**：空操作（库中无 `p` 项）。
3. **库函数集合 / 中心差分 / `MEEdot2Acc`**：新旧逐项一致。
4. **执行机构截断**：是“症状放大器”，不是致因；修复后重建不再超限，截断自动失效。
5. **老流程的 SG 平滑**（`window=51` 样本 ≈ 1.7 h @120 s）：次要（≈2.3×，16.7→7.1），
   修复后不需要它。
6. **观测频率**（120 s vs 60/10 s）与**评价窗口**：会造成口径差异，但不是本故障来源。

---

## 6. 修复方案（施工单）

### 6.1 目标

在**保持论文 κ-L 曲线选择规则不变**的前提下，把“选定支撑集 → 系数”这一步换成 OLS 无偏解，
使 SINDy 的导数模型不再被系统性压缩。

### 6.2 补丁规格

文件：`src/sindy_ssa/sindy/regression.py`

1) 新增（放在 `_ridge_refit` 附近，`regression.py:56` 之后）：

```python
def _ols_refit(theta: np.ndarray, Y: np.ndarray, Xi: np.ndarray) -> np.ndarray:
    """OLS refit on the support of ``Xi`` — PySINDy's ``unbias=True`` semantics.

    ``theta`` must be the library matrix in the *original* (un-normalised) units,
    which is what :func:`fit_ssr_path` returns.
    """
    out = np.zeros_like(Xi)
    for j in range(Y.shape[1]):
        cols = np.abs(Xi[:, j]) > 0.0
        if cols.any():
            out[cols, j] = np.linalg.lstsq(theta[:, cols], Y[:, j], rcond=None)[0]
    return out
```

2) 在 `fit_ssr_auto` 内（`regression.py:194-210`）把选点后的三行改成：

```python
    library, theta, path = fit_ssr_path(spec, X, Y, normalize_columns=normalize_columns)
    selected_iter = select_elbow(path, tol=tol)
    Xi = next(p["Xi"] for p in path if p["max_iter"] == selected_iter)
    Xi = _ols_refit(theta, Y, Xi)          # <-- 新增：无偏化（unbias=True）
    optimizer = _PathOptimizer(Xi)
    return library, optimizer, Xi, theta, selected_iter, path
```

   `theta` 在这里就是 `library.transform(X)` 的原始（未归一化）矩阵，所以对
   `normalize_columns` 取 `True`/`False` 都成立：`fit_ssr_path` 返回的 `Xi` 已按
   `1/feat_norms` 还原到原尺度，用原尺度 `theta` 重解即可。

3) （推荐，可选）把行为显式化，便于 A/B 与回退：

```python
@dataclass
class LibrarySpec:
    ...
    unbias: bool = True        # 新增；True = PySINDy 语义（默认）
```

   并让 `fit_ssr_auto(..., unbias: bool = True)` 与 `fit_ssr` 一起尊重该开关。
   **默认必须是 `True`**（对齐 PySINDy，也即老方法的行为）；仅在需要复现历史结果时置 `False`。

### 6.3 建议同时调整的配置（可选，二次收益）

`configs/config.yaml`：

```yaml
sindy:
  normalize_columns: false      # 原库（Eq.23）列尺度差异大，配合 alpha=1 会额外收缩
  # original_alpha 保持 1.0 即可：无偏化后它只影响删除顺序，不再压缩最终系数
```

* 只加无偏化、不改配置：`EI 19 040 → 3.66`（截断后 24.6 → 3.35），**已经不饱和**。
* 再加 `normalize_columns: false`：`→ 1.96`（最好，且优于老方法的 16.7）。
* `time_normalize_columns` 保持 `false`（SINDy-TIME 用 `time_alpha = 1e-3`，本来就≈OLS）。

### 6.4 不要做的事

* 不要动 κ 选择规则 / `select_elbow` / `sparsity_tol`——它没问题，改了反而变差。
* 不要为了“看起来平滑”而把 SG 平滑加到 SINDy 推力上——那是掩盖，不是修复。
* 不要靠关掉 `auto_select` 走密集 OLS 解（`fit_ssr` + `kappa=0`）当长期方案：它确实能把未截断 EI
  从 1.9e4 降到 ~29，但那是“全 36 项密集回归”，论文的 κ 稀疏选择失效，且仍有 ~74 % 采样被截断
  （EI≈16.7），比正确修复差一个数量级。
* 不要删除 `saturate_thrust`：它是物理装置约束。修复后它自然不再触发；同时建议把
  `ei_raw_total` 与 `clipped_frac` 作为诊断列一起报告。

### 6.5 对其他两个稀疏方法的预期影响（必须核对）

`fit_ssr_auto` 被三个方法共用，但只有 SINDy 同时满足 `normalize_columns=True` 与 `alpha≈1`：

| 方法 | `alpha` | `normalize_columns` | 现状 | 加无偏化后的预期 |
|---|---|---|---|---|
| SINDy | 1.0 | True | 饱和、EI≈24.6 | **大幅改善（→ ~2–4）** |
| SINDy-TIME | 1e-3 | False | EI 0.117 | 几乎不变（本来就是≈OLS；实测 0.115/0.117 同档） |
| SINDy-UMEE | 1e-4 | True | EI 0.433 | 几乎不变（收缩因子≈1；实测 0.44 同档） |

先在 ideal 档核对这三个数，确认 TIME/UMEE 的变化 < 2 % 再推广到 batch。

### 6.6 验收标准（数值，stationkeeping / ideal / 满弧）

| 指标 | 修复前 | 目标 |
|---|---|---|
| `SINDy` EI（截断后） | 24.5923 | ≤ 4（预期 1.9–3.7，取决于 `normalize_columns`） |
| `SINDy` EI（未截断） | 19 040 | ≤ 4 |
| `clipped_frac` | 0.999 | 0（不再触发饱和） |
| `max \|u_n\|` | 1.6e-03 | ≲ 2e-07 |
| `SINDy` EPE 3D | 18.70 km | 显著下降；以与 SINDy-TIME（0.279）/ SINDy-UMEE（0.148）同量级为目标（需实测确认） |
| SINDy-TIME / SINDy-UMEE EI | 0.1166 / 0.4328 | 变化 < 2 % |
| MEEFD-SG / MEEFD-Spline / TIME-OLS | 0.0961 / 0.1390 / 0.0958 | 完全不变（不走 `fit_ssr_auto`） |

### 6.7 修复后需要重跑/更新的一切

1. `sindy-ssa sindy run`（两个场景 × ideal/high/low/reduced）→ `data/**/typical/**`；
2. `sindy-ssa batch run`（两个场景 × 各噪声档 × 100 次）→ `batch_archive/**`；
3. 由 1、2 生成的论文表格/图（SINDy 的 EI/EPE、Xi 热图、推力对比图、重推演图）都需要重绘；
   其它五个方法不应变化（可用于回归核对）。
4. 若论文正文引用过 ideal 档 SINDy 的 24.59 / 18.70，需要同步更新。

---

## 7. 复现与核对（命令与产物）

工作解释器说明：`.venv` 指向已失效的 Microsoft Store Python，需通过
`sparsity_kappa_study/run.py` 启动器（自动注入 venv 的 `site-packages` 与 `src`）或直接指定可用解释器。

| 目的 | 命令 | 产物 |
|---|---|---|
| 新旧配方对照 + 消融 | `python sparsity_kappa_study/legacy_vs_current.py --scenario stationkeeping` | `out_tol_obs/stationkeeping/legacy_vs_current.csv`、`legacy_ablation.csv`、`legacy_*.csv` |
| 端到端修复预览（EI+EPE） | `python sparsity_kappa_study/fix_preview.py --scenario stationkeeping evasive` | `out_tol_obs/<scenario>/fix_preview/*`（另：**该脚本尚未跑完，EPE 待确认**） |
| κ 扫描（论文口径） | `python sparsity_kappa_study/run.py --scenario stationkeeping evasive --case ideal --fit-fraction 1.0 --future-state observed --groups tol --reprop --out sparsity_kappa_study/out_tol_obs` | `out_tol_obs/<scenario>/tol_scan.csv`、`report.md` |
| 流程自校验（必须与归档一致） | `... --groups auto --verify`（同口径） | `pipeline_check.csv`（应复现 24.5923 / 18.6954） |
| 最优 κ 的 RTN 重推演曲线 | `python sparsity_kappa_study/plot_best_rtn.py --out sparsity_kappa_study/out_tol_obs` | `best_kappa/best_reprop_rtn_*.png`、`best_reprop_TR_plane_*.png` |
| 结论汇总 | — | `sparsity_kappa_study/FINDINGS.md`（§2 两个 κ 的区分、§6.1 EPE 最优 κ、§7 新旧对照） |

**验证锚点**（修改前后的对照组）：

* 修改前：`pipeline_check.csv` 必须给出 SINDy EI 24.5923 / EPE 18.6954 —— 证明环境与口径正确。
* 修改后：同一命令应给出 SINDy EI ≈ 2–4、`clipped_frac = 0`；TIME/UMEE 变化 < 2 %。

---

## 8. 附录：关键数字速查

| 量 | 值 |
|---|---|
| 执行机构上限 `T_max/m` | 2.125e-7 km/s² (= 0.2125 mm/s²) |
| 真值冲量（stationkeeping / evasive） | 3.751 / 2.597 m/s |
| 真值 `max|u_r|` | 1.347e-7 km/s² |
| ORACLE（真导数）EI | 4.4e-4 m/s |
| SINDy 现状 EI（截断 / 未截断） | 24.59 / 1.9e4 |
| SINDy 现状 `max|u_n|` / `clipped_frac` | 1.6e-3 / 0.999 |
| 修复后（预期）EI（截断 / 未截断） | 1.96 / 1.96（`normalize_columns=false`）或 3.35 / 3.66（保持 true） |
| 老方法对照 EI（截断 / 未截断） | 12.7 / 16.7（老 + SG 平滑 ≈ 7.1） |
| 同一 144 项支撑集：无偏化前后 | 3.66 vs 19 040（5 200×） |

关键文件与行号：

* 新：`src/sindy_ssa/sindy/regression.py`（`_ssr_kwargs:17`、`fit_ssr:34`、`_ridge_refit:56`、
  `_PathOptimizer:70`、`fit_ssr_path:80`、`select_elbow:157`、`fit_ssr_auto:194`）；
  `sindy_original.py:50-61`；`configs/config.yaml:38/45/47`；`config.py:165/176/179`。
* 老：`sparsity_kappa_study/FYP_Development_V8/SINDy/SINDy_Ori.py:55/58/66/94`、
  `SINDy/Sequence.py:95`（SG 平滑）、`Evaluate.py`（无截断的评价）、
  `ReportFigs/FreqBatch/*/Output/SINDyOri/U_SINDyOri.csv`（老产物）。

> **结论**：请只在 `fit_ssr_auto` 里补回 OLS 无偏重拟合（并可选地把原库的
> `normalize_columns` 设为 `false`）。这两处之外的一切（κ 规则、库、求导、反演、平滑、
> 截断）都不要改动，改完按 §6.6/§7 验收即可把 SINDy 恢复到老方法水平并略有超越。
