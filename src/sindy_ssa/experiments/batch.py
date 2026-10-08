"""Monte-Carlo batch study. / 蒙特卡洛批量统计。"""

from __future__ import annotations

import copy
import os
import time

import pandas as pd

from ..config import AppConfig
from ..plotting import summarise
from ..registry import METHODS
from ..utils import clear_folder, ensure_dir, format_duration, progress
from .run_sindy import ensure_dynamics, run_reconstruction


def run_batch(config: AppConfig, runs: int = 100, seed0: int = 0) -> pd.DataFrame:
    """Repeat the reconstruction over ``runs`` noise realisations.

    This function writes only CSVs:
    - ``batch_evaluation.csv`` (long table, one row per method per run)
    - ``batch_summary.csv``    (median + IQR per method)

    Figures are produced separately by ``experiments.plot_batch.plot_batch_figures``.

    Progress and per-run timing (with an ETA) are printed to the console.
    """
    t_start = time.perf_counter()
    progress(f"批处理开始：共 {runs} 次", f"batch started: {runs} runs")
    ensure_dynamics(config)
    rows = []
    obs_dir = config.paths.observation_dir
    ensure_dir(obs_dir)
    # Remove stale per-run observation files from previous Monte-Carlo runs so
    # the output directory reflects exactly the requested number of runs.
    clear_folder(obs_dir)

    for i in range(runs):
        t_run = time.perf_counter()
        run_cfg = copy.deepcopy(config)
        run_cfg.observation.noise.seed = seed0 + i
        obs_path = os.path.join(obs_dir, f"Observe_{i + 1}.csv")
        results = run_reconstruction(run_cfg, obs_path=obs_path)
        for metrics in results.values():
            metrics = dict(metrics)
            metrics["run"] = i + 1
            rows.append(metrics)

        done = i + 1
        elapsed = time.perf_counter() - t_start
        eta = elapsed / done * (runs - done)
        progress(f"第 {done}/{runs} 次完成", f"run {done}/{runs} done", time.perf_counter() - t_run)
        print(
            f"          已用 / elapsed {format_duration(elapsed)}"
            f"  |  预计剩余 / ETA {format_duration(eta)}",
            flush=True,
        )

    df = pd.DataFrame(rows)
    ensure_dir(os.path.dirname(config.paths.evaluation_file))
    df.to_csv(config.paths.evaluation_file, index=False)

    # Median / IQR summary table (reviewer-friendly), restricted to the methods
    # actually present so a partial-window (prediction) batch omits the
    # non-predictive MEEFD baselines instead of writing empty rows.
    summary_path = config.paths.evaluation_file.replace("batch_evaluation.csv", "batch_summary.csv")
    present = [name for name in METHODS if name in set(df["method"].astype(str))]
    summary = summarise(df, "epe", present)
    summary.to_csv(summary_path, index=False)

    total = time.perf_counter() - t_start
    progress("批处理完成", "batch done", total)
    print(
        f"    平均每次 / mean per run: {format_duration(total / runs if runs else 0.0)}",
        flush=True,
    )
    return df
