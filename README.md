# sindy-ssa — SINDy-Inspired Low-Thrust Manoeuvre Reconstruction

Companion code for the manuscript
**“Satellite Low-Thrust Maneuver Reconstruction using SINDy-Inspired Sparse
Identification Methods”** (Aerospace Science and Technology, manuscript
AESCTE-D-26-04832).

本仓库是论文 **《Satellite Low-Thrust Maneuver Reconstruction using SINDy-Inspired
Sparse Identification Methods》**（Aerospace Science and Technology）的配套代码。

---

## 1. Overview / 概述

**English.** The pipeline reconstructs an unknown continuous low-thrust manoeuvre
of a LEO satellite from noisy, high-cadence ephemeris data. Six methods are
compared:

1. **MEEFD-SG** — finite-difference inversion of the MEE variational equations + Savitzky–Golay smoothing (baseline).
2. **MEEFD-Spline** — smoothing-spline fit of the (unsmoothed) MEEFD inversion (baseline).
3. **TIME-OLS** — dense (non-sparse) least-squares fit on the time library (baseline).
4. **SINDy** — the SINDy framework applied to manoeuvre reconstruction: sparse identification on the observed MEE derivatives.
5. **SINDy-TIME** — sparse regression of the MEEFD inversion on a time library.
6. **SINDy-UMEE** — sparse regression of the MEEFD inversion on an MEE library.

All six share a common first stage (a finite-difference inversion of the MEE
variational equations, `MEEFD`); they differ in how the resulting unsmoothed
thrust estimate `u_rec` — or, for `SINDy`, the observed derivatives — is processed.
Two scenarios are provided: drag-compensating **station-keeping** and an
emergency **evasive** manoeuvre. Both are processed together.

**中文。** 本代码从含噪、高重访率的星历数据中重建 LEO 卫星的未知连续低推力机动，
比较六种方法（SG / Spline / OLS 三个基线，以及 SINDy、SINDy-TIME、SINDy-UMEE）。
六种方法共用同一反演前端（MEE 变分方程有限差分反演，即 MEEFD），区别在于对反演得到
的未平滑推力估计 `u_rec`（SINDy 为观测导数）的后续处理。提供两个场景：补偿阻力的
**站位保持**机动与紧急**规避**机动，二者一起处理。

---

## 2. Data and archived results / 数据与归档结果

**English.**

- `data/Analysis90/` — **input data.** 90 operator-published Starlink MEME/ITC
  ephemeris files used to calibrate the observation-noise tiers. Required by
  `sindy-ssa noise run`.
- `batch_archive/` — **archived Monte-Carlo results.** The full 100-realisation
  campaign for both scenarios at every noise tier, stored as
  `<window>/<scenario>/<tier>/` with:
  - `batch_evaluation.csv` and `batch_summary.csv` — the per-run and
    median/IQR metrics actually used in the manuscript;
  - `methods/` and `observation/` — the per-run intermediate thrust and
    observation files.
  The script `plot_batch_tiers.py` reads the `batch_evaluation.csv` tables and
  regenerates the batch-tier, versus-window and summary-heatmap figures of the
  paper.
- `data/<scenario>/{truth,typical,batch}` and `data/noise/` — generated CSV
  outputs (regenerable with the CLI).
- `results/` — generated figures (regenerable; safe to delete).

**中文。**

- `data/Analysis90/` —— **输入数据**：90 份 Starlink 官方 MEME/ITC 星历文件，用于
  标定观测噪声档位，`sindy-ssa noise run` 需要它。
- `batch_archive/` —— **蒙特卡洛归档结果**：两个场景、各噪声档、100 次实现，
  目录为 `<window>/<scenario>/<tier>/`，其中 `batch_evaluation.csv` /
  `batch_summary.csv` 是论文所用的逐次与中位数/IQR 指标，`methods/` 与
  `observation/` 为逐次中间文件。`plot_batch_tiers.py` 读取评估表即可复现论文的
  batch-tier、vs-window 与汇总热图。
- `data/<scenario>/{truth,typical,batch}`、`data/noise/` —— 生成的 CSV，可由 CLI 重跑。
- `results/` —— 生成的图件，可删除后重跑。

> **Data provenance / 数据来源.** The `Analysis90` files were obtained from the
> Starlink Space Safety platform / operator-published ephemeris. Please respect
> their terms of use when redistributing. / `Analysis90` 中的星历文件来自 Starlink
> Space Safety 平台公开数据，再分发时请遵守其使用条款。

---

## 3. Repository layout / 目录结构

```
sindy-ssa/
├── configs/
│   ├── config.yaml              # unified config (both scenarios + settings)
│   └── batch_run.yaml           # batch config; inherits config.yaml via "base:"
├── data/
│   ├── Analysis90/              # input: 90 Starlink MEME/ITC files (noise calibration)
│   ├── noise/                   # generated: global noise-calibration CSVs
│   ├── <scenario>/              # generated: stationkeeping | evasive
│   │   ├── truth/               #   truth dynamics CSV/JSON
│   │   ├── typical/<case>/      #   ideal / high / low / reduced reconstruction
│   │   └── batch/<case>/        #   Monte-Carlo batch
│   └── studies/ballistic/       # generated: Cd-uncertainty CSVs (legacy runs)
├── batch_archive/               # archived Monte-Carlo results (see §2)
├── results/                     # generated figures only (safe to delete)
├── src/sindy_ssa/               # installable Python package
│   ├── cli.py                   # unified CLI entry
│   ├── config.py                # typed YAML config + path helpers
│   ├── constants.py             # physical constants / column names
│   ├── registry.py              # scenario/case/method/default-config registry
│   ├── data/                    # DataSeries / DataTable containers
│   ├── dynamics/                # orbital dynamics (frames, elements, perturbations, propagation)
│   ├── noise/                   # RTN calibration + MEE Gaussian injection
│   ├── sindy/                   # libraries, SSR regression, the six methods
│   ├── metrics/                 # EPE and EI
│   ├── plotting/                # trajectory / thrust / heatmap / violin / noise figures
│   ├── experiments/             # pipeline drivers (dynamics, sindy, batch, plotting)
│   └── studies/                 # independent studies (noise, ballistic)
├── plot_batch_tiers.py          # standalone: batch-tier/heatmap figures from batch_archive
├── plot_dynamics.py             # standalone: truth/noise figures (Tavotto entry)
├── tavotto_registry.json        # Tavotto figure-editor registry (optional)
├── tests/test_smoke.py          # smoke tests
├── pyproject.toml
└── requirements.txt
```

---

## 4. Requirements and installation / 环境与安装

**English.** Python ≥ 3.10. Dependencies: `numpy`, `scipy`, `pandas`,
`matplotlib`, `seaborn`, `sympy`, `pysindy>=2.0`, `pyyaml`.

```bash
cd sindy-ssa
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

pip install -e ".[dev]"          # editable install + pytest/ruff
```

**中文。** 需要 Python ≥ 3.10。建议用虚拟环境并以可编辑模式安装，`[dev]` 会额外安装
`pytest` 与 `ruff`：

```bash
pip install -e ".[dev]"
```

安装后即可使用 `sindy-ssa` 命令，或用 `python -m sindy_ssa`。

---

## 5. Quick start / 快速开始

```bash
# 1) global noise calibration + truth dynamics
sindy-ssa noise run --input-dir data/Analysis90
sindy-ssa dynamics run
sindy-ssa dynamics plot

# 2) typical reconstruction: ideal / high / low / reduced (both scenarios)
sindy-ssa sindy run
sindy-ssa evaluate
sindy-ssa sindy plot

# 3) Monte-Carlo batch study (100 runs per noise tier by default)
sindy-ssa batch run
sindy-ssa batch plot

# 4) drag-coefficient (Cd) uncertainty study — prints to the console only
sindy-ssa ballistic run

# 5) regenerate the archived batch figures from batch_archive/
python plot_batch_tiers.py --archive batch_archive --out results/paper
```

**中文。** 依次执行：噪声标定与真值动力学 → 四种典型工况重建 → 100 次蒙特卡洛批处理 →
弹道系数不确定性研究（仅控制台输出）→ 由 `batch_archive/` 重建论文批量图。

---

## 6. Configuration / 配置

**English.** All options live in `configs/config.yaml` (annotated). Key entries:

| Key | Meaning |
|-----|---------|
| `satellite.*` | mass, cross-section, `C_D`, `Isp`, max thrust |
| `scenarios.<name>` | scenario blocks: `stationkeeping` \| `evasive` |
| `observation.sampling_rate` | down-sampling factor (`6` → 60 s cadence) |
| `observation.noise.<case>` | `{mode, tier, seed}`; `mode: gaussian` (MEE-space) \| `none` |
| `observation.noise.<case>.tier` | `low` \| `high` \| `reduced` (selects the σ table) |
| `sindy.smoothing_window_seconds` | Savitzky–Golay window duration `T_w` (s) |
| `sindy.meefd_spline_smoothing_factor` | MEEFD-Spline smoothing coefficient (`s = factor·n·var(y)`; larger = smoother) |
| `sindy.normalize_columns` | L2-normalise MEE library columns before ridge/SSR |
| `sindy.sparsity_tol` | relative RSS tolerance for the L-curve sparsity selection |
| `sindy.prediction.enabled` | enable partial-window future prediction |
| `sindy.prediction.fit_fraction` | fraction of the arc used for identification (e.g. `0.75`) |
| `sindy.prediction.future_state` | MEE grid of the state-based methods: `reference` (non-thrusting orbit) \| `propagated` (self-consistent re-propagation, causal) |

`configs/batch_run.yaml` inherits everything from `config.yaml` through
`base: config.yaml` and only overrides the `batch:` block (`runs`, `seed0`,
`cases`, and an optional `prediction:` override).

**中文。** 所有参数集中在 `configs/config.yaml`（含注释）。`batch_run.yaml` 通过
`base: config.yaml` 继承主配置，仅覆盖 `batch:` 段（`runs`、`seed0`、`cases`，以及
可选的 `prediction:` 覆盖项）。

---

## 7. Pipeline / 数据流

```
Data_real.csv ──► observation (downsample + noise)
                     ├─► MEEFD inversion (u_rec, unsmoothed)
                     │        ├─► MEEFD-SG     — SG smoothing + up-sampling
                     │        ├─► MEEFD-Spline — smoothing spline
                     │        ├─► TIME-OLS     — dense LS on time library
                     │        ├─► SINDy-TIME   — sparse regression on time library
                     │        └─► SINDy-UMEE   — sparse regression on MEE library
                     └─► SINDy (on observed MEE derivatives)
                          reconstructed thrust ──► re-propagation ──► metrics (EPE, EI)
```

---

## 8. CLI reference / 命令参考

Everything is invoked through `sindy-ssa` (or `python -m sindy_ssa`).

**Global options / 全局选项**

| Option | Default | Meaning |
|--------|---------|---------|
| `--config PATH` | `configs/config.yaml` | unified config (single-scenario file → only that scenario) |
| `--case NAME` | none | restrict to `ideal` / `high` / `low` / `reduced` |
| `--data-root PATH` | `data` | root for CSV/JSON output |
| `--figures-root PATH` | `results` | root for figure output |

### 8.1 Truth dynamics / 真值动力学

```bash
sindy-ssa dynamics run
sindy-ssa dynamics plot
```

Output: `data/<scenario>/truth/Data_{real,ref,LVLH,mano}.csv`, `SatParam.json`,
`TimeParam.json`; `results/<scenario>/truth/*.{pdf,png}`.

### 8.2 Typical reconstruction / 典型重建

```bash
sindy-ssa sindy run        # observation + six methods + re-propagation for all four cases
sindy-ssa evaluate         # write evaluation.csv from the outputs
sindy-ssa sindy plot       # thrust / Xi heatmaps / reprop / error bars / selection paths
```

Output: `data/<scenario>/typical/<case>/{observation,methods}/...`,
`evaluation.csv`; `results/<scenario>/typical/<case>/...` (including
`IdentifiedExpressions_<Method>.txt`).

### 8.3 Partial-window future prediction / 部分窗口预测

Enabled through `sindy.prediction` in the config (or the batch override). `sindy run`
then fits on the first `fit_fraction` of the manoeuvre and predicts the remainder:

- **Predictive** — `SINDy`, `SINDy-TIME`, `TIME-OLS`, `SINDy-UMEE`: re-propagated
  over the full arc and evaluated (EPE / EI).
- **Non-predictive** — `MEEFD-SG`, `MEEFD-Spline`: the thrust curve stops at the fit
  horizon; they do not participate in re-propagation / EPE / EI.
- State-based methods (`SINDy` / `SINDy-UMEE`) are evaluated on an MEE grid chosen by
  `future_state`: `reference` (the non-thrusting reference orbit) or `propagated`
  (self-consistent: observed states inside the fit window, then the trajectory
  re-propagated with the identified thrust itself — a causal forecast; the
  fixed-point sweep and its under-relaxation are internal pipeline constants).
  Per-sweep previews are written to `methods/<method>/propagation_iter/`.
- The `thrust_comparison` figure marks the fit horizon with a red dashed line.

### 8.4 Monte-Carlo batch / 蒙特卡洛批处理

```bash
sindy-ssa batch run                       # configs/batch_run.yaml, both scenarios
sindy-ssa batch plot
sindy-ssa batch run --case ideal          # deterministic ideal reference only
sindy-ssa batch run --runs 5 --case high --data-root data/_pilot --figures-root results/_pilot
```

Output: `data/<scenario>/batch/<case>/batch_evaluation.csv` (long table with a
`run` column), `batch_summary.csv` (median + IQR); `results/<scenario>/batch/*.{pdf,png}`.

### 8.5 Noise calibration / 噪声标定

```bash
sindy-ssa noise run --input-dir data/Analysis90
sindy-ssa noise plot
```

Output: `data/noise/{rtn_uncertainty,rtn_mee_tiers,snr_summary}.csv`,
`mee_samples_{reduced,low,high}.csv`; `results/noise/*.{pdf,png}`.
Calibration is scenario-agnostic and is auto-run once if `data/noise/rtn_mee_tiers.csv`
is missing.

### 8.6 Drag-coefficient (Cd) uncertainty study / 弹道系数不确定性研究

```bash
sindy-ssa ballistic run --factors 0.6 0.8 1.0 1.2 1.4
```

This is a purely analytical bias study: for each Cd factor it recomputes the drag
acceleration with the perturbed coefficient and integrates the resulting thrust bias
over the manoeuvre. **Results are printed to the console only; no figure or data file
is written.** / 该研究为纯解析偏差分析，**仅在控制台打印结果，不产生任何图件或数据文件**。

### 8.7 Re-evaluate existing outputs / 复用已有结果

```bash
sindy-ssa evaluate --case high
```

Reads existing method outputs and rewrites `evaluation.csv`.

---

## 9. Reproducing the paper / 复现论文

```bash
pip install -e .
sindy-ssa noise run --input-dir data/Analysis90
sindy-ssa dynamics run

# Typical + statistical results
sindy-ssa sindy run                 # ideal / high / low / reduced, six methods
sindy-ssa evaluate
sindy-ssa sindy plot

# 100-realisation Monte-Carlo campaign (full window)
sindy-ssa batch run --runs 100
sindy-ssa batch plot

# Partial-window prediction: set sindy.prediction.enabled: true in batch_run.yaml,
# then re-run "sindy-ssa batch run --runs 100" for 25 / 50 / 75 % windows.

# Archived batch figures (batch_tiers / vs_window / summary heatmaps)
python plot_batch_tiers.py --archive batch_archive --out results/paper
```

The archived 100-realisation results of the manuscript are shipped in
`batch_archive/`, so the batch figures can be regenerated without re-running the
campaign.

论文使用的 100 次蒙特卡洛结果已随仓库提供（`batch_archive/`），无需重跑即可复现批量图。

---

## 10. Figures and the Tavotto editor / 图件与 Tavotto 编辑器

All `plot` subcommands emit PDF (for the paper) plus a PNG preview via
`sindy_ssa.plotting.style.save_figure`.

`plot_dynamics.py` at the repository root is a **top-level entry** used by the
Tavotto figure editor to re-render the truth-dynamics and noise figures. Tavotto
imports the script recorded in `tavotto_registry.json` and calls its `main()`:

```bash
python plot_dynamics.py
```

Point the Tavotto panel’s `script` at `plot_dynamics.py`; do **not** use
`src/sindy_ssa/cli.py` (as a package module it uses relative imports and cannot be
imported standalone).

---

## 11. Tests / 测试

```bash
pytest -q
ruff check src tests
```

The smoke tests check that (i) both noise models give physically reasonable MEE
standard deviations, and (ii) the end-to-end pipeline runs for both noise modes.

---

## 12. Citation / 引用

Please cite the manuscript once published. / 论文发表后请引用原文。

## 13. License / 许可

MIT — see `LICENSE`.
