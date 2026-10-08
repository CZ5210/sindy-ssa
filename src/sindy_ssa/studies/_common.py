"""Common helpers for independent study scripts. / 独立研究脚本的公共助手。

This module is kept intentionally thin: it only handles the lazy global noise
calibration so that each study module can focus on its own logic.
"""

from __future__ import annotations

import os

from ..noise import run_noise_study
from ..utils import ensure_dir


def ensure_noise_calibrated(noise_dir: str) -> None:
    """Run the global RTN→MEE noise calibration if it has not been done yet."""
    ensure_dir(noise_dir)
    if not os.path.exists(os.path.join(noise_dir, "rtn_mee_tiers.csv")):
        run_noise_study(output_dir=noise_dir)

