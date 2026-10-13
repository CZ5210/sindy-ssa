"""Unified command-line interface for sindy-ssa.

All entry points live here.  The old root-level ``run_*.py`` / ``plot_*.py``
scripts have been removed; use the installed ``sindy-ssa`` console command or
``python -m sindy_ssa`` instead.

Examples::

    sindy-ssa dynamics run
    sindy-ssa sindy run --case high
    sindy-ssa batch run --runs 100
    sindy-ssa noise run --input-dir data/Analysis90
"""

from __future__ import annotations

import argparse
import copy
import os

from .config import AppConfig, NoiseConfig, load_config, scenario_names
from .experiments import (
    evaluate_existing,
    make_result_figures,
    plot_batch_figures,
    plot_batch_overview,
    plot_batch_tiers,
    plot_scenario_truth,
    run_batch,
    run_dynamics,
    run_reconstruction,
)
from .registry import (
    BATCH_CASES,
    BATCH_IDEAL_CASE,
    DEFAULT_BATCH_CONFIG,
    DEFAULT_CONFIG,
    TYPICAL_CASES,
)
from .studies import ballistic, noise
from .studies._common import ensure_noise_calibrated


def _iter_scenarios(args) -> list[tuple[str, str]]:
    """Return (scenario_name, config_path) tuples to operate on."""
    path = args.config or DEFAULT_CONFIG
    return [(name, path) for name in scenario_names(path)]



def _set_roots(config: AppConfig, args) -> None:
    config.paths.data_root = args.data_root
    config.paths.figures_root = args.figures_root
    config.paths.noise_data_dir = os.path.join(args.data_root, "noise")
    config.paths.noise_figures_dir = os.path.join(args.figures_root, "noise")


def _truth_paths(config: AppConfig, scenario: str):
    """Set config paths for the truth-dynamics task."""
    config.paths.data_dir = os.path.join(config.paths.data_root, scenario, "truth")
    config.paths.figures_dir = os.path.join(config.paths.figures_root, scenario, "truth")


def _task_paths(config: AppConfig, scenario: str, task: str, case: str):
    """Set config paths for typical/batch tasks."""
    from .config import apply_task_paths

    return apply_task_paths(config, scenario, task, case)


# --------------------------------------------------------------------------- #
# dynamics
# --------------------------------------------------------------------------- #
def cmd_dynamics_run(args) -> int:
    for scenario, cfg_path in _iter_scenarios(args):
        cfg = load_config(cfg_path, scenario=scenario)
        _set_roots(cfg, args)
        _truth_paths(cfg, scenario)
        ensure_noise_calibrated(cfg.paths.noise_data_dir)
        run_dynamics(cfg)
    return 0


def cmd_dynamics_plot(args) -> int:
    for scenario, cfg_path in _iter_scenarios(args):
        cfg = load_config(cfg_path, scenario=scenario)
        _set_roots(cfg, args)
        _truth_paths(cfg, scenario)
        for path in plot_scenario_truth(cfg):
            print(path)
    return 0


# --------------------------------------------------------------------------- #
# sindy (typical ideal/high/low)
# --------------------------------------------------------------------------- #
def _sindy_cases(cfg: AppConfig, requested_case: str | None) -> list[tuple[str, NoiseConfig, int]]:
    """Return (case_name, noise_config, sampling_rate) tuples for a typical run.

    If the config defines ``observation.noise`` as a mapping, iterate its cases.
    Every case shares the configured observation cadence, so the noise-free
    ``ideal`` case differs from the noisy tiers only by the absence of noise.
    The ``ideal`` case is synthesised when explicitly requested and absent from
    the mapping.  Legacy single noise configs fall back to the registry's
    ``TYPICAL_CASES``.
    """
    if isinstance(cfg.observation.noise, dict):
        cases: list[tuple[str, NoiseConfig, int]] = []
        for case_name in cfg.observation.noise:
            noise_cfg = cfg.observation.get_noise(case_name)
            cases.append((case_name, noise_cfg, cfg.observation.sampling_rate))
        if requested_case == "ideal" and "ideal" not in cfg.observation.noise:
            cases.append(("ideal", NoiseConfig(mode="none"), cfg.observation.sampling_rate))
        if requested_case is None:
            return cases
        return [(n, c, s) for n, c, s in cases if n == requested_case]

    # Legacy single-noise config: use the registry case list.
    cases = []
    for case_name, noise_mode, noise_tier, sampling_rate in TYPICAL_CASES:
        if requested_case and requested_case != case_name:
            continue
        noise_cfg = NoiseConfig(mode=noise_mode, tier=noise_tier or "high")
        cases.append((case_name, noise_cfg, sampling_rate))
    return cases


def _sindy_case_names(cfg: AppConfig, requested_case: str | None) -> list[str]:
    """Return typical case names for plotting/evaluation."""
    if isinstance(cfg.observation.noise, dict):
        names = list(cfg.observation.noise.keys())
    else:
        names = [name for name, _, _, _ in TYPICAL_CASES]
    if requested_case:
        names = [n for n in names if n == requested_case]
        # The ideal case may be synthesised on demand (see ``_sindy_cases``).
        if requested_case == "ideal" and "ideal" not in names:
            names.append("ideal")
    return names


def cmd_sindy_run(args) -> int:
    for scenario, cfg_path in _iter_scenarios(args):
        cfg = load_config(cfg_path, scenario=scenario)
        _set_roots(cfg, args)
        ensure_noise_calibrated(cfg.paths.noise_data_dir)
        for case_name, noise_cfg, sampling_rate in _sindy_cases(cfg, args.case):
            print(f"=== sindy run: {scenario} / {case_name} ===", flush=True)
            case_cfg = copy.deepcopy(cfg)
            _task_paths(case_cfg, scenario, "typical", case_name)
            case_cfg.observation.noise = noise_cfg
            case_cfg.observation.sampling_rate = sampling_rate
            case_cfg.sindy.interpolation_rate = sampling_rate
            run_reconstruction(case_cfg, verbose=True)
            evaluate_existing(case_cfg)
    return 0


def cmd_sindy_plot(args) -> int:
    for scenario, cfg_path in _iter_scenarios(args):
        cfg = load_config(cfg_path, scenario=scenario)
        _set_roots(cfg, args)
        for case_name in _sindy_case_names(cfg, args.case):
            _task_paths(cfg, scenario, "typical", case_name)
            for path in make_result_figures(cfg):
                print(path)
    return 0


# --------------------------------------------------------------------------- #
# batch
# --------------------------------------------------------------------------- #
def cmd_batch_run(args) -> int:
    tiers = dict(BATCH_CASES)
    for scenario, cfg_path in _iter_scenarios(args):
        cfg = load_config(cfg_path, scenario=scenario)
        runs = args.runs if args.runs is not None else cfg.batch.runs
        seed0 = args.seed0 if args.seed0 is not None else cfg.batch.seed0
        if cfg.batch.prediction is not None:
            cfg.sindy.prediction = cfg.batch.prediction
        cases = [c for c in cfg.batch.cases if args.case is None or c == args.case]
        for case_name in cases:
            if case_name == BATCH_IDEAL_CASE:
                mode, tier, case_runs = "none", None, 1
            elif case_name in tiers:
                mode, tier, case_runs = "gaussian", tiers[case_name], runs
            else:
                print(f"[skip] unknown batch case {case_name!r}", flush=True)
                continue
            print(f"=== batch run: {scenario} / {case_name} ({case_runs} runs) ===", flush=True)
            _set_roots(cfg, args)
            _task_paths(cfg, scenario, "batch", case_name)
            cfg.observation.noise = NoiseConfig(mode=mode, tier=tier or "high")
            # Every case keeps the configured cadence (including the noise-free
            # reference), so the comparison isolates the absence of noise.
            cfg.sindy.interpolation_rate = cfg.observation.sampling_rate
            ensure_noise_calibrated(cfg.paths.noise_data_dir)
            run_batch(cfg, runs=case_runs, seed0=seed0)
    return 0


def cmd_batch_plot(args) -> int:
    for scenario, cfg_path in _iter_scenarios(args):
        cfg = load_config(cfg_path, scenario=scenario)
        cases = [c for c in cfg.batch.cases if args.case is None or c == args.case]
        # Per-case merged violin figures (noise tiers only; ideal is deterministic).
        for case_name in cases:
            if case_name == BATCH_IDEAL_CASE:
                continue
            _set_roots(cfg, args)
            _task_paths(cfg, scenario, "batch", case_name)
            for path in plot_batch_figures(cfg):
                print(path)
        # Cross-noise overview combining every tier plus the ideal reference.
        _set_roots(cfg, args)
        for path in plot_batch_overview(cfg, cases):
            print(path)
        for path in plot_batch_tiers(cfg, cases):
            print(path)
    return 0


# --------------------------------------------------------------------------- #
# noise (independent study)
# --------------------------------------------------------------------------- #
def cmd_noise_run(args) -> int:
    noise.run(output_dir=args.noise_data_dir, input_dir=getattr(args, "input_dir", None),
              data_root=getattr(args, "data_root", "data"))
    return 0


def cmd_noise_plot(args) -> int:
    for path in noise.plot(data_dir=args.noise_data_dir, figures_dir=args.noise_figures_dir):
        print(path)
    return 0


# --------------------------------------------------------------------------- #
# ballistic (independent study)
# --------------------------------------------------------------------------- #
def cmd_ballistic_run(args) -> int:
    for scenario, cfg_path in _iter_scenarios(args):
        cfg = load_config(cfg_path, scenario=scenario)
        _set_roots(cfg, args)
        cfg.paths.data_dir = os.path.join(args.data_root, scenario, "truth")
        ballistic.run(cfg, cd_factors=args.factors)
    return 0


# --------------------------------------------------------------------------- #
# evaluate
# --------------------------------------------------------------------------- #
def cmd_evaluate(args) -> int:
    for scenario, cfg_path in _iter_scenarios(args):
        cfg = load_config(cfg_path, scenario=scenario)
        _set_roots(cfg, args)
        for case_name in _sindy_case_names(cfg, args.case):
            _task_paths(cfg, scenario, "typical", case_name)
            print(f"# {scenario} / {case_name}")
            print(evaluate_existing(cfg).to_string(index=False))
    return 0


# --------------------------------------------------------------------------- #
# parser construction
# --------------------------------------------------------------------------- #
def _add_common(p):
    p.add_argument("--config", default=None, help="single-scenario YAML (omit for both)")
    p.add_argument("--data-root", default="data", help="data output root")
    p.add_argument("--figures-root", default="results", help="figures output root")


def _add_case(p, default=None):
    p.add_argument("--case", default=default, help="case name (e.g. ideal/high/low)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sindy-ssa", description="SINDy-inspired low-thrust manoeuvre reconstruction")
    sub = parser.add_subparsers(dest="command", required=True)

    # dynamics
    dyn = sub.add_parser("dynamics", help="truth dynamics")
    dyn_sub = dyn.add_subparsers(dest="subcommand", required=True)
    p_dyn_run = dyn_sub.add_parser("run", help="propagate truth dynamics")
    _add_common(p_dyn_run)
    p_dyn_plot = dyn_sub.add_parser("plot", help="plot truth dynamics")
    _add_common(p_dyn_plot)
    p_dyn_run.set_defaults(func=cmd_dynamics_run)
    p_dyn_plot.set_defaults(func=cmd_dynamics_plot)

    # sindy (typical)
    sindy = sub.add_parser("sindy", help="typical reconstruction (ideal/high/low)")
    sindy_sub = sindy.add_subparsers(dest="subcommand", required=True)
    p_sindy_run = sindy_sub.add_parser("run", help="run reconstruction")
    _add_common(p_sindy_run)
    _add_case(p_sindy_run)
    p_sindy_plot = sindy_sub.add_parser("plot", help="plot reconstruction results")
    _add_common(p_sindy_plot)
    _add_case(p_sindy_plot)
    p_sindy_run.set_defaults(func=cmd_sindy_run)
    p_sindy_plot.set_defaults(func=cmd_sindy_plot)

    # batch
    batch = sub.add_parser("batch", help="Monte-Carlo batch study")
    batch_sub = batch.add_subparsers(dest="subcommand", required=True)
    p_batch_run = batch_sub.add_parser("run", help="run Monte-Carlo batch")
    _add_common(p_batch_run)
    _add_case(p_batch_run)
    p_batch_run.add_argument("--runs", type=int, default=None, help="number of Monte-Carlo runs (default: batch.runs in the config)")
    p_batch_run.add_argument("--seed0", type=int, default=None, help="base seed (default: batch.seed0 in the config)")
    p_batch_plot = batch_sub.add_parser("plot", help="plot batch boxplots")
    _add_common(p_batch_plot)
    _add_case(p_batch_plot)
    # Default to the dedicated batch config so prediction stays off there.
    p_batch_run.set_defaults(func=cmd_batch_run, config=DEFAULT_BATCH_CONFIG)
    p_batch_plot.set_defaults(func=cmd_batch_plot, config=DEFAULT_BATCH_CONFIG)

    # noise
    noise_p = sub.add_parser("noise", help="RTN noise calibration")
    noise_sub = noise_p.add_subparsers(dest="subcommand", required=True)
    p_noise_run = noise_sub.add_parser("run", help="run calibration")
    p_noise_run.add_argument("--input-dir", default=None, help="Analysis90 directory")
    p_noise_run.add_argument("--noise-data-dir", default="data/noise")
    p_noise_run.add_argument("--data-root", default="data", help="scenario truth data root (for the SNR summary)")
    p_noise_plot = noise_sub.add_parser("plot", help="plot calibration figures")
    p_noise_plot.add_argument("--noise-data-dir", default="data/noise")
    p_noise_plot.add_argument("--noise-figures-dir", default="results/noise")
    p_noise_run.set_defaults(func=cmd_noise_run)
    p_noise_plot.set_defaults(func=cmd_noise_plot)

    # ballistic
    bal = sub.add_parser("ballistic", help="Cd uncertainty study (console output only)")
    bal_sub = bal.add_subparsers(dest="subcommand", required=True)
    p_bal_run = bal_sub.add_parser("run", help="run Cd sensitivity")
    _add_common(p_bal_run)
    p_bal_run.add_argument("--factors", nargs="+", type=float, default=[0.6, 0.8, 1.0, 1.2, 1.4])
    p_bal_run.set_defaults(func=cmd_ballistic_run)

    # evaluate

    p_eval = sub.add_parser("evaluate", help="evaluate existing reconstruction outputs")
    _add_common(p_eval)
    _add_case(p_eval)
    p_eval.set_defaults(func=cmd_evaluate)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
