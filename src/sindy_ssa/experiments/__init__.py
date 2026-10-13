"""Experiment drivers. / 实验驱动。"""

from ..registry import METHODS
from .archive_figures import plot_archive_figures
from .batch import run_batch
from .evaluate import evaluate_existing
from .make_figures import make_result_figures
from .paper_batch import make_archive_figures, resolve_windows, run_paper_batch
from .plot_batch import plot_batch_figures, plot_batch_overview, plot_batch_tiers
from .plot_dynamics import plot_scenario_truth
from .run_dynamics import build_satellite, run_dynamics
from .run_sindy import ensure_dynamics, generate_observation, run_reconstruction

__all__ = [
    "METHODS",
    "build_satellite",
    "ensure_dynamics",
    "evaluate_existing",
    "generate_observation",
    "make_archive_figures",
    "make_result_figures",
    "plot_archive_figures",
    "plot_batch_figures",
    "plot_batch_overview",
    "plot_batch_tiers",
    "plot_scenario_truth",
    "resolve_windows",
    "run_batch",
    "run_dynamics",
    "run_paper_batch",
    "run_reconstruction",
]
