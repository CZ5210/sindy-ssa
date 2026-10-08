"""Smoke tests for the refactored pipeline.

重构流水线的冒烟测试。
"""

from __future__ import annotations

import numpy as np
import pytest

from sindy_ssa import AppConfig
from sindy_ssa.experiments import run_dynamics, run_reconstruction
from sindy_ssa.noise import mee_sigma_for_tier
from sindy_ssa.registry import METHODS


def _tiny_config(tmp_path, mode="gaussian"):
    return AppConfig.from_dict(
        {
            "scenario": {
                "name": "evasive",
                "duration_hours": 0.4,
                "step_seconds": 10.0,
                "time_of_flight_hours": 0.25,
            },
            "observation": {"sampling_rate": 6, "noise": {"mode": mode, "tier": "high", "seed": 1}},
            "sindy": {"interpolation_rate": 6, "smoothing_window_seconds": 300},
            "paths": {
                "data_dir": str(tmp_path / "data"),
                "observation_dir": str(tmp_path / "observation"),
                "methods_dir": str(tmp_path / "methods"),
                "figures_dir": str(tmp_path / "figures"),
                "evaluation_file": str(tmp_path / "evaluation.csv"),
            },
        }
    )


def test_noise_model_sane():
    """标定的 MEE 标准差物理量级合理：σ_L ≈ σ_T / r ≈ 8.7e-4 rad。"""
    high = mee_sigma_for_tier("high")
    low = mee_sigma_for_tier("low")
    assert np.all(high > 0.0)
    assert np.all(low > 0.0)
    # 沿迹位置误差 σ_T = 6.0 km，轨道半径约 6901 km → σ_L ≈ 8.7e-4 rad。
    assert 1e-4 < high[5] < 1e-3
    # 低档应当整体小于高档（位置/速度不确定性更低）。
    assert np.all(low < high)


def test_end_to_end_pipeline(tmp_path):
    config = _tiny_config(tmp_path)
    run_dynamics(config)
    results = run_reconstruction(config)
    assert set(results) == set(METHODS)
    for metrics in results.values():
        assert np.isfinite(metrics["epe"])
        assert np.isfinite(metrics["ei_total"])
        assert metrics["ei_total"] >= 0.0


@pytest.mark.parametrize("mode", ["gaussian", "none"])
def test_noise_modes_run(tmp_path, mode):
    config = _tiny_config(tmp_path, mode=mode)
    run_dynamics(config)
    results = run_reconstruction(config)
    assert len(results) == len(METHODS)
