"""观测加噪：MEE 空间直接零均值高斯注入。

论文在 ``eq:mee_noise_model`` 里指出，RTN 噪声经非线性映射后的 MEE 偏差近似高斯，
因此在仿真中直接给观测 MEE 状态加零均值高斯噪声：

    x_obs = x_MEE + delta_x_MEE,   delta_x_MEE ~ N(0, diag(sigma^2))

其中 ``sigma`` 取标定出的 Table ``tab:MEE_std``（见 :mod:`sindy_ssa.noise.calibration`）。
``none`` 模式表示无噪声的理想情形。
"""

from __future__ import annotations

import numpy as np

from ..config import NoiseConfig
from ..data import DataSeries, DataTable
from .calibration import mee_sigma_for_tier

MEE_COLUMNS = ["p", "f", "g", "h", "k", "L"]
OBSERVATION_COLUMNS = ["Time", "p", "f", "g", "h", "k", "L", "Apn_r", "Apn_t", "Apn_n", "Flag"]


def _replace_mee(obs: DataTable, mee: np.ndarray) -> DataTable:
    """用含噪 MEE 覆盖观测表中的六个 MEE 分量。"""
    index = {name: i for i, name in enumerate(obs.columns)}
    columns = list(obs.series)
    for j, col in enumerate(MEE_COLUMNS):
        columns[index[col]] = DataSeries(mee[:, j], col)
    return DataTable(columns, name=obs.name)


def _add_mee_gaussian(obs: DataTable, sigma: np.ndarray, rng) -> DataTable:
    """在 MEE 空间直接加独立零均值高斯噪声。"""
    mee = np.column_stack([obs.get(c).data for c in MEE_COLUMNS]).astype(float)
    delta = rng.normal(loc=0.0, scale=np.asarray(sigma, dtype=float), size=mee.shape)
    mee = mee + delta
    # L 是角度，绕回 [0, 2π) 保持与 MEE 定义一致。
    mee[:, 5] = np.mod(mee[:, 5], 2.0 * np.pi)
    return _replace_mee(obs, mee)


def add_observation_noise(table: DataTable, noise: NoiseConfig, rng=None) -> DataTable:
    """按配置给观测表加噪；``mode`` 为 ``none`` 时原样返回。"""
    rng = rng or np.random.default_rng(noise.seed)
    if noise.mode in ("none", "ideal", "off"):
        return table
    if noise.mode == "gaussian":
        sigma = np.asarray(noise.mee_sigma, dtype=float) if noise.mee_sigma else mee_sigma_for_tier(noise.tier)
        return _add_mee_gaussian(table, sigma, rng)
    raise ValueError(f"unknown noise mode: {noise.mode!r}")


def simulate_observation(truth_path: str, observation_cfg, out_path: str, rng=None) -> DataTable:
    """把真值数据降采样到观测频率并加噪，写出观测 CSV。"""
    truth = DataTable.from_csv(truth_path)
    obs = truth.sample(observation_cfg.sampling_rate)
    obs = add_observation_noise(obs, observation_cfg.noise, rng=rng)
    obs.select(OBSERVATION_COLUMNS).save(out_path)
    return obs
