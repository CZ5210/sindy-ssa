"""The paper's full Monte-Carlo campaign in a single call.

一条命令跑完论文的全部蒙特卡洛批量结果。

``sindy-ssa batch paper`` replays the batch study once per archive window
(``full`` = fully observed, ``windowNN`` = identification on the first ``NN``
percent of the manoeuvre arc) and archives every case into::

    <archive_root>/<window>/<scenario>/<case>/batch_evaluation.csv
    <archive_root>/<window>/<scenario>/<case>/batch_summary.csv

which is exactly the layout :mod:`sindy_ssa.experiments.archive_figures` (and
therefore ``plot_batch_tiers.py``) reads.  Each case is archived as soon as its
Monte-Carlo loop finishes, so an interrupted campaign still leaves a complete
archive for every finished case.
"""

from __future__ import annotations

import copy
import os
import shutil
import time

from ..config import AppConfig, NoiseConfig, apply_task_paths
from ..registry import (
    BATCH_CASES,
    BATCH_IDEAL_CASE,
    PAPER_WINDOWS,
    paper_window,
)
from ..studies._common import ensure_noise_calibrated
from ..utils import clear_folder, ensure_dir, progress
from .archive_figures import plot_archive_figures
from .batch import run_batch


def resolve_windows(names) -> list[tuple[str, float | None]]:
    """Validate and order the requested paper windows.

    Accepts archive directory names (``full``/``window70``/``window30``) and
    returns them in the canonical campaign order, so a subset such as
    ``["window30", "full"]`` still runs fully observed first.
    """
    requested = {paper_window(name)[0] for name in names}
    windows = [(name, fraction) for name, fraction in PAPER_WINDOWS if name in requested]
    unknown = requested.difference({name for name, _ in windows})
    if unknown:
        raise ValueError(f"unknown paper windows: {sorted(unknown)}")
    return windows


def _window_config(config: AppConfig, fit_fraction: float | None) -> AppConfig:
    """Copy ``config`` with the partial-window prediction of one campaign window."""
    base = config.batch.prediction or config.sindy.prediction
    prediction = copy.deepcopy(base)
    prediction.enabled = fit_fraction is not None
    if fit_fraction is not None:
        prediction.fit_fraction = fit_fraction
    window_cfg = copy.deepcopy(config)
    window_cfg.sindy.prediction = prediction
    window_cfg.batch.prediction = prediction
    return window_cfg


def archive_case(config: AppConfig, archive_root: str, window: str, scenario: str,
                 case: str, copy_intermediates: bool = False) -> list[str]:
    """Copy one finished batch case into the archive layout.

    The evaluation tables are always stored; ``copy_intermediates`` also copies
    the per-run ``methods/`` and ``observation/`` trees (much larger).  Returns
    the archived paths.
    """
    case_dir = os.path.dirname(config.paths.evaluation_file)
    dest = os.path.join(archive_root, window, scenario, case)
    ensure_dir(dest)
    saved: list[str] = []

    if copy_intermediates:
        # Refresh from scratch so the archive cannot keep stale intermediates
        # from an earlier campaign.
        clear_folder(dest)
        for name in ("methods", "observation"):
            src = os.path.join(case_dir, name)
            if os.path.isdir(src):
                shutil.copytree(src, os.path.join(dest, name), dirs_exist_ok=True)

    for name in ("batch_evaluation.csv", "batch_summary.csv"):
        src = os.path.join(case_dir, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dest, name))
            saved.append(os.path.join(dest, name))
    return saved


def run_paper_batch(config: AppConfig, *, runs: int | None = None,
                    seed0: int | None = None, windows=None,
                    cases=None, archive_root: str | None = None,
                    copy_intermediates: bool | None = None) -> list[str]:
    """Run every campaign window of one scenario and archive the results.

    ``config`` holds the batch settings of a single scenario (see
    :func:`sindy_ssa.cli.cmd_batch_paper`, which iterates the scenarios).  The
    output of each case is archived immediately, so the caller can also derive
    the figures once all scenarios are done.  Returns the archived CSV paths.

    ``config.batch.paper`` supplies the windows, the archive root and whether to
    copy the intermediate files; the keyword arguments override it.
    """
    paper = config.batch.paper
    runs = config.batch.runs if runs is None else runs
    seed0 = config.batch.seed0 if seed0 is None else seed0
    archive_root = paper.archive_root if archive_root is None else archive_root
    if copy_intermediates is None:
        copy_intermediates = paper.copy_intermediates
    windows = resolve_windows(paper.windows if windows is None else windows)
    cases = list(config.batch.cases if cases is None else cases)
    scenario = config.scenario.name

    archived: list[str] = []
    t_start = time.perf_counter()
    for window, fit_fraction in windows:
        window_cfg = _window_config(config, fit_fraction)
        observed = "100%" if fit_fraction is None else f"{fit_fraction:.0%}"
        progress(
            f"论文批量窗口 {window}（观测弧长 {observed}）",
            f"paper batch window {window} ({observed} observed)",
        )
        for case in cases:
            if case == BATCH_IDEAL_CASE:
                mode, tier, case_runs = "none", None, 1
            elif case in dict(BATCH_CASES):
                mode, tier, case_runs = "gaussian", dict(BATCH_CASES)[case], runs
            else:
                print(f"[skip] unknown batch case {case!r}", flush=True)
                continue

            case_cfg = copy.deepcopy(window_cfg)
            apply_task_paths(case_cfg, scenario, "batch", case)
            case_cfg.observation.noise = NoiseConfig(mode=mode, tier=tier or "high")
            # Every case keeps the configured cadence (including the noise-free
            # reference), so the comparison isolates the absence of noise.
            case_cfg.sindy.interpolation_rate = case_cfg.observation.sampling_rate
            ensure_noise_calibrated(case_cfg.paths.noise_data_dir)

            print(
                f"=== paper batch: {window} / {scenario} / {case} ({case_runs} runs) ===",
                flush=True,
            )
            run_batch(case_cfg, runs=case_runs, seed0=seed0)
            archived += archive_case(
                case_cfg, archive_root, window, scenario, case, copy_intermediates
            )
            progress(
                f"已归档 {scenario}/{case} 到 {archive_root}/{window}/",
                f"archived {scenario}/{case} under {archive_root}/{window}/",
            )

    progress(
        "论文批量全部窗口完成",
        "paper batch finished",
        time.perf_counter() - t_start,
    )
    return archived


def make_archive_figures(config: AppConfig, *, archive_root: str | None = None,
                         out_dir: str | None = None, log_scale: bool | None = None,
                         png: bool | None = None) -> list[str]:
    """Regenerate the paper's archive figures from ``config.batch.paper`` settings."""
    paper = config.batch.paper
    archive_root = paper.archive_root if archive_root is None else archive_root
    out_dir = out_dir or paper.figures_dir or None
    log_scale = paper.figures_log_scale if log_scale is None else log_scale
    png = paper.figures_png if png is None else png
    saved = plot_archive_figures(archive_root, out_dir, log_scale=log_scale, png=png)
    for path in saved:
        print(f"wrote {path}")
    return saved
