"""SINDy-inspired low-thrust manoeuvre reconstruction (companion code).

基于 SINDy 启发的低推力机动重建（论文配套代码）。

This package accompanies the manuscript
"Satellite Low-Thrust Maneuver Reconstruction using SINDy-Inspired Sparse
Identification Methods". It refactors the original research scripts into a
clean, reproducible pipeline (dynamics simulation, observation noise, four
reconstruction methods, evaluation and plotting).

本包为论文配套代码，将原始研究脚本重构为可复现的流水线：
动力学仿真、观测噪声、四种重建方法、评价与绘图。
"""

from __future__ import annotations

__version__ = "1.0.0"

from .config import AppConfig, load_config, save_config

__all__ = ["AppConfig", "__version__", "load_config", "save_config"]
