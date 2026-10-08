"""噪声标定结果绘图（只读 CSV，只出图）。

对应论文 2.3 节的两张图：

- ``noise_rtn_selected_histogram.png`` —— 过滤后的 RTN 标准差直方图，
  并标注低/高两档；
- ``noise_mee_tier_violin.png`` —— 两档 RTN 噪声传播到 MEE 后的偏差小提琴图。

输入 CSV 由 :func:`sindy_ssa.noise.calibration.run_noise_study` 生成。
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import ScalarFormatter
from scipy.stats import norm

from ..noise import MEE0, STARLINK_RTN_TIERS
from . import style
from .style import save_figure

# 列名 -> (显示符号, 单位, 显示单位换算因子)
RTN_COMPONENTS = [
    ("sigma_r_km", r"$\sigma_r$", "km", 1.0),
    ("sigma_t_km", r"$\sigma_t$", "km", 1.0),
    ("sigma_n_km", r"$\sigma_n$", "km", 1.0),
    ("sigma_vr_kms", r"$\sigma_{v_r}$", "m/s", 1000.0),
    ("sigma_vt_kms", r"$\sigma_{v_t}$", "m/s", 1000.0),
    ("sigma_vn_kms", r"$\sigma_{v_n}$", "m/s", 1000.0),
]

MEE_PLOT_LABELS = [
    r"$\Delta p$ [km]",
    r"$\Delta f$",
    r"$\Delta g$",
    r"$\Delta h$",
    r"$\Delta k$",
    r"$\Delta L$ [rad]",
]


def _chosen_rtn_sigma() -> dict[str, dict[str, float]]:
    """把 ``STARLINK_RTN_TIERS`` 展成绘图需要的低/高档取值。"""
    low = STARLINK_RTN_TIERS["low"]
    high = STARLINK_RTN_TIERS["high"]
    return {
        "sigma_r_km": {"low": low["position_km"][0], "high": high["position_km"][0]},
        "sigma_t_km": {"low": low["position_km"][1], "high": high["position_km"][1]},
        "sigma_n_km": {"low": low["position_km"][2], "high": high["position_km"][2]},
        "sigma_vr_kms": {"low": low["velocity_m_s"][0], "high": high["velocity_m_s"][0]},
        "sigma_vt_kms": {"low": low["velocity_m_s"][1], "high": high["velocity_m_s"][1]},
        "sigma_vn_kms": {"low": low["velocity_m_s"][2], "high": high["velocity_m_s"][2]},
    }


# 直方图里标注两档时允许的半宽（与旧版保持一致，单位同显示单位）。
CHOSEN_RTN_SIGMA_TOL = {
    "sigma_r_km": {"low": 0.02, "high": 0.02},
    "sigma_t_km": {"low": 0.15, "high": 0.1},
    "sigma_n_km": {"low": 0.005, "high": 0.05},
    "sigma_vr_kms": {"low": 0.1, "high": 0.15},
    "sigma_vt_kms": {"low": 0.01, "high": 0.01},
    "sigma_vn_kms": {"low": 0.008, "high": 0.02},
}


def _save(fig, save_path: str):
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        save_figure(fig, save_path)
    plt.close(fig)
    return save_path


def make_rtn_tier_figure(uncertainty_csv: str, save_path: str) -> str:
    """从 ``rtn_uncertainty.csv`` 画 RTN 直方图并标注两档。"""
    filtered_df = pd.read_csv(uncertainty_csv)
    chosen = _chosen_rtn_sigma()
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    axes = axes.flatten()

    for ax, (col, symbol, unit, scale) in zip(axes, RTN_COMPONENTS):
        empirical = filtered_df[col].to_numpy(dtype=float) * scale
        empirical = np.maximum(empirical, 1e-12)
        bins = np.histogram_bin_edges(empirical, bins=80)

        ax.hist(empirical, bins=bins, density=False, color="#7f7f7f", alpha=0.35)

        low_val = chosen[col]["low"]
        high_val = chosen[col]["high"]
        low_half = CHOSEN_RTN_SIGMA_TOL[col]["low"]
        high_half = CHOSEN_RTN_SIGMA_TOL[col]["high"]
        low_mask = (empirical >= low_val - low_half) & (empirical <= low_val + low_half)
        high_mask = (empirical >= high_val - high_half) & (empirical <= high_val + high_half)

        for mask, color in ((low_mask, "#1f77b4"), (high_mask, "#d62728")):
            selected = empirical[mask]
            if selected.size == 0:
                continue
            ax.hist(selected, bins=bins, density=False, color=color, alpha=0.7)

        ax.set_yscale("linear")
        ax.set_xlabel(f"{symbol} [{unit}]")
        ax.set_ylabel("sample count")
        ax.grid(True, axis="y", alpha=0.25, linewidth=0.7)
        ax.set_axisbelow(True)

    fig.legend(
        handles=[
            Patch(facecolor="#7f7f7f", edgecolor="#7f7f7f", alpha=0.5, label="Filtered samples"),
            Patch(facecolor="#1f77b4", edgecolor="#1f77b4", alpha=0.7, label="Low uncertainty"),
            Patch(facecolor="#d62728", edgecolor="#d62728", alpha=0.7, label="High uncertainty"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=3,
        frameon=False,
    )
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    return _save(fig, save_path)


def make_mee_tier_figure(sample_csvs: list[str], tier_labels: list[str], save_path: str) -> str:
    """从各档蒙特卡洛样本 CSV 画 MEE 偏差小提琴图。

    ``sample_csvs`` 与 ``tier_labels`` 一一对应，按不确定度递增排列
    （通常为 Reduced / Low / High），同一分量各档同色、透明度递增。
    """
    deltas = [
        pd.read_csv(csv, usecols=["p", "f", "g", "h", "k", "L"]).to_numpy(dtype=float) - MEE0
        for csv in sample_csvs
    ]
    n_tiers = len(deltas)
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    alphas = np.linspace(0.30, 0.70, n_tiers)

    def draw(ax, j: int, label: str) -> None:
        data = [delta[:, j] for delta in deltas]
        positions = list(range(1, n_tiers + 1))

        gauss_params = [
            (pos, float(np.mean(d)), float(np.std(d, ddof=1)))
            for pos, d in zip(positions, data)
        ]

        parts = ax.violinplot(
            data,
            positions=positions,
            widths=0.7,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for body, alpha in zip(parts["bodies"], alphas):
            body.set_facecolor(palette[j])
            body.set_edgecolor(palette[j])
            body.set_alpha(alpha)

        y_lo, y_hi = ax.get_ylim()
        pad = 0.05 * (y_hi - y_lo)
        y_min = min(float(np.min(d)) for d in data) - pad
        y_max = max(float(np.max(d)) for d in data) + pad
        ax.set_ylim(y_min, y_max)
        y_grid = np.linspace(y_min, y_max, 400)

        half_width = 0.35
        gauss_style = {"color": "#9a9a9a", "linewidth": 0.9, "linestyle": "--", "alpha": 0.9, "zorder": 5}
        for pos_i, mean, std in gauss_params:
            pdf = norm.pdf(y_grid, loc=mean, scale=max(std, 1e-12))
            pdf_norm = pdf / pdf.max() * half_width
            ax.plot(pos_i + pdf_norm, y_grid, **gauss_style)
            ax.plot(pos_i - pdf_norm, y_grid, **gauss_style)

        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((0, 0))
        ax.yaxis.set_major_formatter(formatter)
        ax.yaxis.get_offset_text().set_size(style.FONT_SIZE)

        ax.set_xticks(positions)
        ax.set_xticklabels(tier_labels)
        ax.set_xlim(0.4, n_tiers + 0.6)
        ax.set_xlabel(label)
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_axisbelow(True)

    fig, axes = plt.subplots(1, 6, figsize=(12, 5), sharey=False)
    for j, (ax, label) in enumerate(zip(axes, MEE_PLOT_LABELS)):
        draw(ax, j, label)
        if j > 0:
            ax.set_ylabel("")

    fig.legend(
        handles=[plt.Line2D([], [], color="#9a9a9a", linestyle="--", linewidth=0.9, label="Fitted Gaussian")],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=1,
        frameon=False,
        fontsize=style.FONT_SIZE - 2,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    return _save(fig, save_path)
