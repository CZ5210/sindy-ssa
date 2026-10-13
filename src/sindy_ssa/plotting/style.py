"""Shared plotting style constants. / 绘图样式常量。"""

from __future__ import annotations

import os

import matplotlib

FONT_SIZE = 18
LEGEND_FONT_SIZE = 16
matplotlib.rcParams.update(
    {
        "font.size": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "legend.fontsize": LEGEND_FONT_SIZE,
    }
)

EARTH_COLOR = "#2a68bc"
EARTH_ALPHA = 0.05
AXIS_COLOR = ["r", "g", "b"]

TRAJ_COLOR = ["#3498db", "#cb4335"]
TRAJ_ALPHA = [1.0, 1.0]
TRAJ_STYLE = ["--", "-"]
TRAJ_WIDTH = [0.4, 0.4]
POINT_COLOR = ["#229954", "#d4ac0d", "purple"]
POINT_MARKER = ["o", "d", "*"]
POINT_SIZE = 100

LVLH_TRAJ_WIDTH = [2, 3]
LVLH_TRAJ_ALPHA = [0.8, 1.0]

REPO_TRAJ_COLOR = ["r", "g", "b", "c", "m", "orange"]
REPO_TRAJ_STYLE = ["--", ":", "-.", "-", "--", ":"]
REPO_MARKER = ["d", "s", "p", "P", "*", "h"]
REPO_MARKER_SIZE = 100
REPO_ALPHA = 0.7
REPO_LINEWIDTH = 2

TRUE_TRAJ_COLOR = "k"
TRUE_TRAJ_ALPHA = 0.5
TRUE_TRAJ_STYLE = "-"
TRUE_TRAJ_WIDTH = 3
TRUE_MARKER_COLOR = ["k", "purple"]
TRUE_MARKER_SIZE = 100

# Figure geometry for the T-R re-propagation comparison (width, height) in inches.
# Flatter than square so it can share a column with the error-metrics panel in
# the composite typical-results figure (reprop on top, error metrics below,
# thrust facet as a full-height right column).
REPRO_FIGSIZE = (10, 5.0)
# Thrust facet grid: (per-column width, per-row height) in inches.
THRUST_FACET_FIGSIZE = (11.0, 2.5)

# Fixed RTN axis limits ((R_min, R_max), (T_min, T_max), (N_min, N_max)) in km,
# per scenario. Centralised here so the re-propagation and LVLH trajectory plots
# share one place to tune the view.
STATIONKEEPING_RTN_LIMITS = ((-2.0, 2.0), (-10.0, 30.0), (-5.0, 5.0))
EVASIVE_RTN_LIMITS = ((-1.0, 3.0), (-25.0, 5.0), (-5.0, 5.0))
SCENARIO_RTN_LIMITS = {
    "stationkeeping": STATIONKEEPING_RTN_LIMITS,
    "evasive": EVASIVE_RTN_LIMITS,
}

ACC_METHODS = ["True", "MEEFD-SG", "MEEFD-Spline", "TIME-OLS", "SINDy", "SINDy-TIME", "SINDy-UMEE"]
ACC_COLOR = ["k", "r", "g", "b", "c", "m", "orange"]
ACC_STYLE = ["-", "--", "-.", ":", "-", "--", ":"]
ACC_LABELS = [r"$u_r$", r"$u_t$", r"$u_n$"]
ACC_ALPHA = [1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

# Canonical per-method colour shared by the RTN, thrust and error figures, so
# the same method keeps the same main tone across all panels.
METHOD_COLORS = {m: c for m, c in zip(ACC_METHODS, ACC_COLOR) if m != "True"}

# Canonical method order (truth excluded).  The per-method colour/style/marker
# lists are aligned to this order; resolving them by method *name* rather than by
# position keeps each curve's appearance fixed when a method is skipped from a
# figure (e.g. the non-predictive MEEFD baselines in partial-window mode).
METHOD_ORDER = tuple(m for m in ACC_METHODS if m != "True")

# Canonical marker per method for the versus-window charts, keyed by name like
# METHOD_COLORS so the four predictive methods keep the glyph they are drawn with
# when the two non-predictive MEEFD baselines are added to the figure.  Those two
# only ever appear as a single-window box glyph at 100% observed.
METHOD_MARKERS = {
    "MEEFD-SG": "v",
    "MEEFD-Spline": "P",
    "TIME-OLS": "o",
    "SINDy": "s",
    "SINDy-TIME": "^",
    "SINDy-UMEE": "D",
}


def method_palette_index(label, fallback):
    """Return the canonical palette index for ``label`` or ``fallback``."""
    return METHOD_ORDER.index(label) if label in METHOD_ORDER else fallback

ELEMENT_LABELS = [r"$\dot{p}$", r"$\dot{f}$", r"$\dot{g}$", r"$\dot{h}$", r"$\dot{k}$", r"$\dot{L}$"]
XI_CMAP = "plasma"


def save_figure(fig, save_path: str, *, png: bool = True, **kwargs) -> list[str]:
    """Save a figure as PDF (default) and optionally PNG.

    ``save_path`` may be given with or without extension; the function strips any
    extension and always writes ``<path>.pdf``.  When ``png=True`` it also writes
    ``<path>.png`` for quick preview.
    """
    base, _ = os.path.splitext(save_path)
    saved: list[str] = []

    pdf_path = f"{base}.pdf"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", **kwargs)
    saved.append(pdf_path)

    if png:
        png_path = f"{base}.png"
        fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight", **kwargs)
        saved.append(png_path)

    return saved
