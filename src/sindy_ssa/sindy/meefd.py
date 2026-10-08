"""MEEFD: finite-difference inversion of the variational equations.

MEEFD：对摄动方程做有限差分反解，作为基线方法。

The unsmoothed inversion ``u'_rec`` is the shared input of every downstream
method; it is persisted on the observation grid as ``U_MEEFD_inv.csv``.  The
``MEEFD-SG`` baseline additionally applies a Savitzky-Golay filter to that
inversion and up-samples it to ``U_MEEFD_SG.csv``.
"""

from __future__ import annotations

import os

from ..config import AppConfig
from ..data import DataTable
from ..noise import mee_weights_for_tier
from . import preprocessing as pre


def _inversion(observation: DataTable, cfg: AppConfig) -> DataTable:
    """Unsmoothed RTN thrust inversion over the full observation series."""
    natural = pre.gve_meedot(observation)
    derivatives = pre.compute_derivatives(observation)
    noise = cfg.observation.noise
    weights = None if noise.mode == "none" else mee_weights_for_tier(noise.tier)
    u_natural = pre.acceleration_from_meedot(observation, natural, weights=weights)
    u_total = pre.acceleration_from_meedot(observation, derivatives, weights=weights)
    return pre.subtract_acceleration(u_total, u_natural)


def meefd_thrust(observation: DataTable, cfg: AppConfig) -> DataTable:
    """Unsmoothed MEEFD inversion on the manoeuvre interval.

    返回在观测网格（未上采样）上、裁剪到首个机动区间的**未平滑** RTN 推力估计
    ``u'_rec``。它是所有反演相关方法的共享输入。
    """
    intervals = pre.manoeuvre_intervals(observation)
    if not intervals:
        raise ValueError("no manoeuvre interval found in the observation")
    return _inversion(observation, cfg).filter(intervals[0])


def run_meefd_sg(observe_path: str, cfg: AppConfig, raw_path: str, thrust_path: str) -> DataTable:
    """Reconstruct thrust with the MEEFD inversion + Savitzky-Golay baseline.

    Persists the unsmoothed inversion (observation grid) to ``raw_path`` and the
    SG-smoothed, up-sampled thrust to ``thrust_path``.
    """
    observation = DataTable.from_csv(observe_path)
    intervals = pre.manoeuvre_intervals(observation)
    if not intervals:
        raise ValueError("no manoeuvre interval found in the observation")

    inversion = _inversion(observation, cfg)
    os.makedirs(os.path.dirname(raw_path) or ".", exist_ok=True)
    inversion.filter(intervals[0]).save(raw_path)

    window = pre.smoothing_window_samples(
        cfg.sindy.smoothing_window_seconds, observation.get("Time").data
    )
    smoothed = (
        inversion.smooth(window, exclude="Time")
        .filter(intervals[0])
        .linear_interp(cfg.sindy.interpolation_rate)
    )
    smoothed.save(thrust_path)
    return smoothed
