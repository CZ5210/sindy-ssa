"""观测噪声：标定与分析、MEE 高斯注入。

本包把原来的 ``observations/noise.py`` 与顶层 ``noise.py`` 合并为两个职责清晰的
子模块：

- :mod:`sindy_ssa.noise.calibration` —— 从 Analysis90 实测协方差标定 RTN 各档，
  并通过非线性 RTN→MEE 蒙特卡洛得到论文 Table ``tab:MEE_std`` 的六个标准差；
  同时给出原始差分星历的等效 RTN 加速度噪声（:func:`equivalent_accel_noise`），
  供粗略 SNR 评估使用。
- :mod:`sindy_ssa.noise.injection` —— 实际仿真时直接在 MEE 空间注入零均值高斯
  噪声（使用标定出的 MEE 标准差），不再逐状态做 RTN 采样。
"""

from .calibration import (
    DEFAULT_ANALYSIS90_DIR,
    MEE0,
    STARLINK_RTN_TIERS,
    TIER_ORDER,
    equivalent_accel_noise,
    filter_outliers,
    mee_sigma_for_tier,
    mee_weights_for_tier,
    parse_analysis90,
    propagate_tier_to_mee,
    run_noise_study,
    sample_tier_to_mee,
    tier_sigma,
)
from .injection import (
    OBSERVATION_COLUMNS,
    add_observation_noise,
    simulate_observation,
)

__all__ = [
    "DEFAULT_ANALYSIS90_DIR",
    "MEE0",
    "OBSERVATION_COLUMNS",
    "STARLINK_RTN_TIERS",
    "TIER_ORDER",
    "add_observation_noise",
    "equivalent_accel_noise",
    "filter_outliers",
    "mee_sigma_for_tier",
    "mee_weights_for_tier",
    "parse_analysis90",
    "propagate_tier_to_mee",
    "run_noise_study",
    "sample_tier_to_mee",
    "simulate_observation",
    "tier_sigma",
]