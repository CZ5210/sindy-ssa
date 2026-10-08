"""Starlink RTN 观测噪声标定与 MEE 传播（只生成数据，不绘图）。

本模块只负责“标定/分析”这一半：

1. 读取运营商发布的 MEME/ITC 协方差文件（``data/Analysis90``），提取 RTN 六个
   标准差（R/T/N 位置与 vR/vT/vN 速度）；
2. 用 ``median + 3·IQR`` 规则剔除异常历元；
 3. 选取论文 Table ``tab:rtn_sigma`` 中的低/高两档，并附加 ``reduced`` 假设档
    （low 档的 1/10，代表下一代高精度定轨能力）；
 4. 通过非线性 ``RV2MEE`` 映射做蒙特卡洛，把 RTN 噪声传播为 MEE 噪声，得到论文
    Table ``tab:MEE_std`` 的六个标准差（``sigma_p ... sigma_L``）。


仿真观测里的“加噪”请使用 :mod:`sindy_ssa.noise.injection`。本模块把中间结果写成
CSV，绘图逻辑放在 :mod:`sindy_ssa.plotting.noise_plots`，从而实现“生成数据 /
基于 CSV 绘图”的分离。
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import NoiseConfig
from ..dynamics.elements import MEE2RV, RV2MEE
from ..dynamics.frames import LVLH2ECI

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ANALYSIS90_DIR = REPO_ROOT / "data" / "Analysis90"

# --------------------------------------------------------------------------- #
# 实测 RTN 两档（来自 Analysis90 直方图的人工选取）
# 位置单位 km，速度单位 m/s（可读性好），经 :func:`tier_sigma` 转成 km/s。
# 与论文 Table tab:rtn_sigma 一一对应。
# --------------------------------------------------------------------------- #
STARLINK_RTN_TIERS: dict[str, dict[str, list[float]]] = {
    "low": {
        "position_km": [0.05, 0.90, 0.02],
        "velocity_m_s": [0.30, 0.05, 0.01],
    },
    "high": {
        "position_km": [0.40, 6.00, 0.60],
        "velocity_m_s": [1.58, 0.10, 0.10],
    },
    "reduced": {
        "position_km": [0.005, 0.090, 0.002],
        "velocity_m_s": [0.030, 0.005, 0.001],
    },
}

# RTN→MEE 蒙特卡洛使用的参考近圆轨道：530 km 高度、43° 倾角。
# MEE 顺序为 [p (km), f, g, h, k, L (rad)]。
MEE0 = np.array([6901.0, 0.0, 0.0, 0.3939, 0.0, 0.0])

RTN_COLUMNS = [
    "satellite",
    "epoch",
    "sigma_r_km",
    "sigma_t_km",
    "sigma_n_km",
    "sigma_vr_kms",
    "sigma_vt_kms",
    "sigma_vn_kms",
]

# MEME/ITC 下三角协方差（21 个元素）中对角线位置，取出即六个方差。
DIAG_INDICES = (0, 2, 5, 9, 14, 20)

MEE_LABELS = ["sigma_p_km", "sigma_f", "sigma_g", "sigma_h", "sigma_k", "sigma_L_rad"]


def _velocity_km_s(velocity_m_s: list[float]) -> list[float]:
    """把 m/s 速度标准差转为 km/s。"""
    return [float(v) / 1000.0 for v in velocity_m_s]


def tier_sigma(tier: str) -> dict[str, list[float]]:
    """返回某档的 RTN 位置 (km) 与速度 (km/s) 标准差。"""
    try:
        tier_data = STARLINK_RTN_TIERS[tier]
    except KeyError as exc:
        raise ValueError(f"unknown RTN noise tier {tier!r}; expected one of {sorted(STARLINK_RTN_TIERS)}") from exc
    return {
        "position_km": list(tier_data["position_km"]),
        "velocity_km_s": _velocity_km_s(tier_data["velocity_m_s"]),
    }


def apply_noise_tier(noise: NoiseConfig, tier: str) -> NoiseConfig:
    """把 ``noise.tier`` 设为指定档位（新配置模型只记录档名）。"""
    if tier not in STARLINK_RTN_TIERS:
        raise ValueError(f"unknown RTN noise tier {tier!r}")
    noise.tier = tier
    return noise


def _read_one_file(path: Path) -> list[dict]:
    """从单个 MEME/ITC 文件读取每个历元的 RTN 标准差记录。"""
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return []

    start = None
    for i, line in enumerate(lines):
        if line.strip() == "UVW":
            start = i + 1
            break
    if start is None:
        return []

    satellite = path.stem.split("_", 2)[2] if len(path.stem.split("_", 2)) > 2 else path.stem
    records: list[dict] = []
    for i in range(start, len(lines), 4):
        if i + 3 >= len(lines):
            break
        try:
            state = lines[i].split()
            epoch = float(state[0])
            cov_values: list[float] = []
            for j in (1, 2, 3):
                cov_values.extend(float(x) for x in lines[i + j].split())
            if len(cov_values) != 21:
                continue
            std = np.sqrt(np.maximum(np.asarray(cov_values)[list(DIAG_INDICES)], 0.0))
            records.append(
                {
                    "satellite": satellite,
                    "epoch": epoch,
                    "sigma_r_km": std[0],
                    "sigma_t_km": std[1],
                    "sigma_n_km": std[2],
                    "sigma_vr_kms": std[3],
                    "sigma_vt_kms": std[4],
                    "sigma_vn_kms": std[5],
                }
            )
        except (IndexError, ValueError):
            continue
    return records


def parse_analysis90(input_dir: str | Path) -> pd.DataFrame:
    """解析 Analysis90 目录下所有 MEME/ITC 文件，返回长表。"""
    input_dir = Path(input_dir)
    files = sorted(input_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"no MEME/ITC files found in {input_dir}")

    rows: list[dict] = []
    for idx, path in enumerate(files, start=1):
        rows.extend(_read_one_file(path))
        if idx % 15 == 0 or idx == len(files):
            print(f"parsed {idx}/{len(files)} files; {len(rows):,} epochs")
    return pd.DataFrame(rows, columns=RTN_COLUMNS)


def filter_outliers(df: pd.DataFrame, k: float = 3.0) -> tuple[pd.DataFrame, dict[str, float]]:
    """按分量剔除超过 ``median + k * IQR`` 的异常历元。"""
    mask = pd.Series(False, index=df.index)
    thresholds: dict[str, float] = {}
    for col in RTN_COLUMNS[2:]:
        values = df[col].to_numpy(dtype=float)
        med = float(np.median(values))
        iqr = float(np.percentile(values, 75) - np.percentile(values, 25))
        threshold = med + k * iqr
        thresholds[col] = threshold
        mask |= df[col] > threshold
    return df.loc[~mask].reset_index(drop=True), thresholds


def _wrap_longitude(samples: np.ndarray, reference: float) -> np.ndarray:
    """把经度 L 重绕为相对 ``reference`` 的有符号角度偏差，避免 2π 跳变。"""
    delta = samples[:, 5] - reference
    samples[:, 5] = np.arctan2(np.sin(delta), np.cos(delta))
    return samples


def sample_tier_to_mee(tier: str, n_samples: int = 5000, seed: int = 12345) -> np.ndarray:
    """对某一 RTN 档做蒙特卡洛，返回 ``(n_samples, 6)`` 的 MEE 样本。"""
    sigma = tier_sigma(tier)
    sigma_pos_km = np.asarray(sigma["position_km"], dtype=float)
    sigma_vel_km_s = np.asarray(sigma["velocity_km_s"], dtype=float)

    R0, V0 = MEE2RV(MEE0)
    rng = np.random.default_rng(seed)
    delta_rtn = rng.normal(
        loc=0.0,
        scale=np.concatenate((sigma_pos_km, sigma_vel_km_s)),
        size=(n_samples, 6),
    )

    mee_samples = np.empty((n_samples, 6), dtype=float)
    for k in range(n_samples):
        dR = LVLH2ECI(R0, V0, delta_rtn[k, 0:3])
        dV = LVLH2ECI(R0, V0, delta_rtn[k, 3:6])
        mee_samples[k] = RV2MEE(R0 + dR, V0 + dV)

    L_ref = RV2MEE(R0, V0)[5]
    return _wrap_longitude(mee_samples, L_ref)


def propagate_tier_to_mee(tier: str, n_samples: int = 5000, seed: int = 12345) -> np.ndarray:
    """把 RTN 档传播为 MEE 标准差（论文 Table ``tab:MEE_std``）。"""
    return sample_tier_to_mee(tier, n_samples=n_samples, seed=seed).std(axis=0, ddof=1)


@cache
def mee_sigma_for_tier(tier: str) -> np.ndarray:
    """返回某档的 MEE 标准差 ``[sigma_p, sigma_f, sigma_g, sigma_h, sigma_k, sigma_L]``。

    结果由确定种子蒙特卡洛复现论文 Table ``tab:MEE_std``，并缓存避免重复计算。
    """
    return propagate_tier_to_mee(tier)


def mee_weights_for_tier(tier: str) -> np.ndarray:
    """返回加权最小二乘权重 ``1 / sigma^2``（对角 W 的对角元）。"""
    sigma = mee_sigma_for_tier(tier)
    return 1.0 / np.square(sigma)


# 论文叙述与表格中使用的档位顺序（不确定度递增）。
TIER_ORDER = ("reduced", "low", "high")


def equivalent_accel_noise(tier: str, dt_seconds: float) -> np.ndarray:
    """原始差分星历的等效 RTN 加速度噪声标准差 ``[sigma_ur, sigma_ut, sigma_un]`` (mm/s^2)。

    对含噪星历做中心差分 ``a_i = (v_{i+1} - v_{i-1}) / (2 dt)``，零均值白噪声的
    方差被放大为 ``sigma_v^2 / (2 dt^2)``，即

        sigma_u = sigma_v / (sqrt(2) * dt)

    这是未经任何平滑/加权的"原始"噪声级：RTN->MEE 映射与加权反解在线性化下
    近似坍缩回 RTN 速度噪声（``(A^T Sigma_MEE^-1 A)^-1 ≈ Sigma_v,RTN``），
    因此可以直接用它与最大机动推力比较，给出粗略的 SNR 量级评估。
    """
    sigma_v_km_s = np.asarray(tier_sigma(tier)["velocity_km_s"], dtype=float)
    return sigma_v_km_s / (np.sqrt(2.0) * dt_seconds) * 1e6


def run_noise_study(
    input_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    n_samples: int = 5000,
    seed: int = 12345,
) -> pd.DataFrame:
    """完成 RTN 档选取 + MEE 传播，并把结果写成 CSV（不绘图）。

    写出三组 CSV：

    - ``rtn_uncertainty.csv`` —— 过滤后的 RTN 实测标准差长表；
    - ``mee_samples_{reduced,low,high}.csv`` —— 三档蒙特卡洛 MEE 样本；
    - ``rtn_mee_tiers.csv`` —— 三档的 RTN 与 MEE 标准差汇总（供注入与权重读取）。
    """
    input_dir = Path(input_dir or DEFAULT_ANALYSIS90_DIR)
    output_dir = Path(output_dir or (REPO_ROOT / "results" / "noise"))
    output_dir.mkdir(parents=True, exist_ok=True)

    df = parse_analysis90(input_dir)
    filtered_df, thresholds = filter_outliers(df)
    print(
        f"filtered outliers: removed {len(df) - len(filtered_df):,}/{len(df):,} epochs, "
        f"kept {len(filtered_df):,}"
    )
    print("component thresholds (median + 3 IQR):")
    for col, thr in thresholds.items():
        print(f"  {col}: {thr:.6g}")

    filtered_df.to_csv(output_dir / "rtn_uncertainty.csv", index=False)

    rows: list[dict] = []
    for level in STARLINK_RTN_TIERS:
        sigma = tier_sigma(level)
        mee_samples = sample_tier_to_mee(level, n_samples=n_samples, seed=seed)
        mee_stds = mee_samples.std(axis=0, ddof=1)

        pd.DataFrame(mee_samples, columns=["p", "f", "g", "h", "k", "L"]).to_csv(
            output_dir / f"mee_samples_{level}.csv", index=False
        )

        rtn_row = {"tier": level, "space": "RTN"}
        rtn_row.update(
            {
                "sigma_r_km": sigma["position_km"][0],
                "sigma_t_km": sigma["position_km"][1],
                "sigma_n_km": sigma["position_km"][2],
                "sigma_vr_m_s": STARLINK_RTN_TIERS[level]["velocity_m_s"][0],
                "sigma_vt_m_s": STARLINK_RTN_TIERS[level]["velocity_m_s"][1],
                "sigma_vn_m_s": STARLINK_RTN_TIERS[level]["velocity_m_s"][2],
            }
        )
        mee_row = {"tier": level, "space": "MEE"}
        mee_row.update(dict(zip(MEE_LABELS, mee_stds)))
        rows.append(rtn_row)
        rows.append(mee_row)

    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "rtn_mee_tiers.csv", index=False)
    print(f"wrote noise calibration outputs under {output_dir}")
    print(table.to_string(index=False))
    return table
