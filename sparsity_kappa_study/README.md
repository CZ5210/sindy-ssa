# Sparsity / `kappa` sensitivity study for the SINDy family

Standalone diagnostic study (safe to delete; nothing in `src/` depends on it).

## Why

In the manuscript pipeline (`sindy_ssa.experiments.run_reconstruction`) every
SINDy-family method is fitted through `fit_ssr_auto`, i.e. the **custom greedy
deletion path + relative-RSS L-curve** in `sindy_ssa.sindy.regression`.  That
code path

* always passes `kappa = 0.0` to PySINDy (and the custom path never consumes
  `kappa` at all), and
* picks the sparsity level from a single relative-RSS tolerance
  (`sindy.sparsity_tol`, default `0.02`) computed on the **total** RSS of all
  target columns, whose scales differ by orders of magnitude.

Two questions are open:

1. Does that auto-selection explain why `SINDy` (identification on the observed
   MEE derivatives) is far worse than the baselines in the **ideal** (noise-free)
   case?
2. Does the SSR sparsity parameter `kappa` (the one that differs between the
   `non_sparse` / `sparse` / `very_sparse` time-library presets) change anything?

## What it does

For each scenario (`stationkeeping`, `evasive`) it rebuilds the ideal noise-free
case in `out/<scenario>/` and scores many model-selection variants against the
truth in `data/<scenario>/truth`:

| group | what it varies |
|-------|----------------|
| `auto` | the pipeline's own auto-selection, swept over `sparsity_tol` |
| `ssr` | PySINDy SSR through the project wrapper (`fit_ssr`), `max_iter × kappa` grid |
| `raw` | PySINDy SSR called directly, which permits `kappa=None` (PySINDy's own error-inflection rule) |
| `preset` | the three named time-library sparsity presets (`non_sparse`, `sparse`, `very_sparse`) |
| `path` | the **full** deletion path scored in thrust space (EI vs number of retained terms) |

Metrics are the manuscript ones: `EI` (thrust error integral) always, plus `EPE`
(terminal re-propagation error) when `--reprop` is given.

## Run

```bash
python sparsity_kappa_study/run.py --scenario stationkeeping --reprop --verify
python sparsity_kappa_study/run.py --scenario evasive       --reprop --verify
# archived (full-arc identification) mode:
python sparsity_kappa_study/run.py --scenario stationkeeping --fit-fraction 1.0 --out sparsity_kappa_study/out_full
# any noisy typical case (this one is the case of the paper's selection-path figure):
python sparsity_kappa_study/run.py --scenario evasive --case reduced --groups auto --out sparsity_kappa_study/out_case
# focused scan of the manuscript kappa (sparsity_tol) on SINDy, ideal + full arc:
python sparsity_kappa_study/run.py --scenario stationkeeping evasive --case ideal \
    --fit-fraction 1.0 --groups tol --reprop --out sparsity_kappa_study/out_tol
python sparsity_kappa_study/make_figures.py --out sparsity_kappa_study/out
# EPE-optimal kappa + its RTN thrust curves (pipeline full-arc convention):
python sparsity_kappa_study/run.py --scenario stationkeeping evasive --case ideal \
    --fit-fraction 1.0 --future-state observed --groups tol --reprop --out sparsity_kappa_study/out_tol_obs
python sparsity_kappa_study/plot_best_rtn.py --out sparsity_kappa_study/out_tol_obs
```

`run.py` is only a launcher: this machine's `.venv` points at a Microsoft Store
interpreter that can no longer be started, so `run.py` injects the (intact)
virtual-env `site-packages` and `src/` into `sys.path` before importing the study
module. Use `run_kappa_study.py` directly if your interpreter works normally.

Outputs land in `<out>/<scenario>/`:

* `summary.csv` / `summary.json` — one row per variant (EI, pre-clipping EI,
  clipped fraction, fit residual, EPE, …),
* `path_scans.csv` — EI along the full deletion path,
* `derivative_scales.json` — natural-background vs thrust-signal RMS per element,
* `pipeline_check.csv` — metrics of a fresh `run_reconstruction` run (with `--verify`),
* `variants/<id>/` — per-variant `U_*.csv`, `Xi.csv`, `meta.json`,
* `report.md`, `sparsity_kappa_<scenario>.png/.pdf`.

## Results

See [FINDINGS.md](FINDINGS.md) for the conclusions and the numbers behind them.
Short version: two different parameters are both named `kappa` — the
manuscript's κ is `sindy.sparsity_tol` (live, `0.02`, drives all three sparse
methods), while PySINDy's SSR `kappa` is unreachable in the auto path. Neither
the sparsity level nor `kappa` causes the ideal-case failure: the reported EI is
a saturation artefact, and the real cause is that the Eq. 23 library cannot
represent the natural MEE dynamics (`L_dot` background is ~6e6x the thrust
signal), so the residual becomes a spurious thrust.
