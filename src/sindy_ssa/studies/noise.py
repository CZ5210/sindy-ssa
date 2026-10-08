"""RTN noise calibration study. / RTN 噪声标定研究。

This is an independent, scenario-agnostic study: it converts the anisotropic
Starlink RTN uncertainty tiers (from ``Analysis90``) into MEE-space standard
deviations used by the observation noise model, and closes with a rough SNR
summary: the maximum manoeuvre thrust compared against the equivalent RTN
acceleration noise of the raw differenced ephemeris
(:func:`sindy_ssa.noise.equivalent_accel_noise`).
"""

from __future__ import annotations

import os

import pandas as pd

from ..config import AppConfig, load_scenarios
from ..noise import TIER_ORDER, equivalent_accel_noise, run_noise_study
from ..plotting import make_mee_tier_figure, make_rtn_tier_figure
from ..registry import DEFAULT_CONFIG
from ..utils import ensure_dir

_COMPONENTS = ["u_r", "u_t", "u_n"]


def run(config: AppConfig | None = None, *, output_dir: str | None = None,
        input_dir: str | None = None, data_root: str = "data") -> str:
    """Run the global RTN → MEE noise calibration plus the rough SNR summary.

    ``output_dir`` overrides ``config.paths.noise_data_dir`` when provided.
    ``input_dir`` overrides the bundled ``data/Analysis90`` source directory.
    ``data_root`` locates the scenario truth CSVs used by the SNR summary.
    Returns the directory containing the calibration CSVs.
    """
    noise_dir = output_dir or (config.paths.noise_data_dir if config else "data/noise")
    ensure_dir(noise_dir)
    run_noise_study(input_dir=input_dir, output_dir=noise_dir)
    snr_summary(data_root=data_root, noise_dir=noise_dir)
    return noise_dir


def snr_summary(*, data_root: str = "data", noise_dir: str | None = None,
                config_path: str = DEFAULT_CONFIG) -> pd.DataFrame:
    """三档噪声下的粗略 SNR 表：最大机动推力 vs 原始差分星历噪声。

    信号取各场景机动区间逐分量最大推力加速度 (mm/s^2)；噪声取
    :func:`sindy_ssa.noise.equivalent_accel_noise`，即
    ``sigma_u = sigma_v / (sqrt(2) * dt)``，其中 ``dt`` 为观测采样间隔
    （``step_seconds * sampling_rate``）。无平滑、无加权，只给量级关系。
    """
    rows: list[dict] = []
    for scenario, cfg in load_scenarios(config_path).items():
        dt = cfg.scenario.step_seconds * cfg.observation.sampling_rate
        truth_csv = os.path.join(data_root, scenario, "truth", "Data_real.csv")
        if not os.path.exists(truth_csv):
            print(f"warning: {truth_csv} not found, skipping {scenario}")
            continue
        df = pd.read_csv(truth_csv)
        man = df[df["Flag"] == "manoeuvre"]
        if man.empty:
            print(f"warning: no manoeuvre interval in {truth_csv}, skipping {scenario}")
            continue
        signal = {
            comp: float(man[comp].astype(float).abs().max()) * 1e6
            for comp in _COMPONENTS
        }
        for tier in TIER_ORDER:
            noise = equivalent_accel_noise(tier, dt)
            for idx, comp in enumerate(_COMPONENTS):
                sig = signal[comp]
                rows.append({
                    "scenario": scenario,
                    "noise_tier": tier,
                    "component": comp,
                    "signal_max_mm_s2": sig,
                    "noise_floor_mm_s2": float(noise[idx]),
                    "snr": sig / float(noise[idx]) if sig > 0 else float("nan"),
                })

    table = pd.DataFrame(rows)
    print("\nrough SNR summary (max thrust vs raw differenced-ephemeris noise):")
    print(table.to_string(index=False))
    if noise_dir:
        out_csv = os.path.join(noise_dir, "snr_summary.csv")
        table.to_csv(out_csv, index=False, float_format="%.4f")
        print(f"wrote {out_csv}")
    return table


def plot(config: AppConfig | None = None, *, data_dir: str | None = None,
         figures_dir: str | None = None) -> list[str]:
    """Plot RTN-tier histogram and MEE-tier violin from calibration CSVs."""
    noise_dir = data_dir or (config.paths.noise_data_dir if config else "data/noise")
    fig_dir = figures_dir or (config.paths.noise_figures_dir if config else "results/noise")
    ensure_dir(fig_dir)

    saved: list[str] = []
    uncertainty_csv = os.path.join(noise_dir, "rtn_uncertainty.csv")
    if os.path.exists(uncertainty_csv):
        saved.append(
            make_rtn_tier_figure(
                uncertainty_csv,
                os.path.join(fig_dir, "noise_rtn_selected_histogram"),
            )
        )
        # violin 只画实测两档（low/high）：reduced 是 low 的 1/10 假设档，RTN→MEE
        # 传播近似线性，其分布形态与 low 相同、仅尺度缩小，图中重复展示无新信息，
        # 且三档跨度约两个数量级会导致 low/reduced 不可读；reduced 仅在表格呈现。
        tiers = [(tier, os.path.join(noise_dir, f"mee_samples_{tier}.csv")) for tier in ("low", "high")]
        tiers = [(tier, path) for tier, path in tiers if os.path.exists(path)]
        if tiers:
            saved.append(
                make_mee_tier_figure(
                    [path for _, path in tiers],
                    [tier.capitalize() for tier, _ in tiers],
                    os.path.join(fig_dir, "noise_mee_tier_violin"),
                )
            )
    return saved


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="RTN→MEE noise calibration")
    parser.add_argument("--noise-data-dir", default="data/noise")
    parser.add_argument("--noise-figures-dir", default="results/noise")
    parser.add_argument("--input-dir", default=None, help="Analysis90 directory")
    parser.add_argument("--data-root", default="data",
                        help="scenario truth data root (for the SNR summary)")
    args = parser.parse_args(argv)

    run(output_dir=args.noise_data_dir, input_dir=args.input_dir,
        data_root=args.data_root)
    plot(data_dir=args.noise_data_dir, figures_dir=args.noise_figures_dir)
    return 0