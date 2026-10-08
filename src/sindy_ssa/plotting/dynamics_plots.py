"""Trajectory and geometry plots. / 轨迹与几何绘图。"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from ..data import DataTable
from . import style
from .style import save_figure


def _save(fig, save_path, show):
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        saved = save_figure(fig, save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return saved[0] if save_path else None


def _plot_earth(ax):
    radius = 6371.0
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color=style.EARTH_COLOR, alpha=style.EARTH_ALPHA)


def _plot_axis(ax, length=8000, buffer=500, label="R,T,N"):
    ax.plot([0, length], [0, 0], [0, 0], color=style.AXIS_COLOR[0], linewidth=1.5, zorder=-5)
    ax.plot([0, 0], [0, length], [0, 0], color=style.AXIS_COLOR[1], linewidth=1.5, zorder=-5)
    ax.plot([0, 0], [0, 0], [0, length], color=style.AXIS_COLOR[2], linewidth=1.5, zorder=-5)
    if label == "X,Y,Z":
        first, second, third = "X", "Y", "Z"
    else:
        first, second, third = "R", "T", "N"
    ax.text(length + buffer, 0, 0, first, color=style.AXIS_COLOR[0], fontsize=style.FONT_SIZE, zorder=-5)
    ax.text(0, length + buffer, 0, second, color=style.AXIS_COLOR[1], fontsize=style.FONT_SIZE, zorder=-5)
    ax.text(0, 0, length + buffer, third, color=style.AXIS_COLOR[2], fontsize=style.FONT_SIZE, zorder=-5)


def _plot_axis_2d(ax, length=3, buffer=0.7):
    ax.plot([0, length], [0, 0], color=style.AXIS_COLOR[1], linewidth=1.5, zorder=-5)
    ax.plot([0, 0], [0, length], color=style.AXIS_COLOR[0], linewidth=1.5, zorder=-5)
    ax.text(0, length + buffer, "R", color=style.AXIS_COLOR[0], fontsize=style.FONT_SIZE, zorder=-5)
    ax.text(length + buffer, 0, "T", color=style.AXIS_COLOR[1], fontsize=style.FONT_SIZE, zorder=-5)


def _plot_box(ax, box_size, shift):
    x = [-box_size[0] / 2, box_size[0] / 2]
    y = [-box_size[1] / 2 - shift, box_size[1] / 2 - shift]
    z = [-box_size[2] / 2, box_size[2] / 2]
    corners = [(i, j, k) for i in (0, 1) for j in (0, 1) for k in (0, 1)]
    edges = [((0, 0, 0), (1, 0, 0)), ((0, 1, 0), (1, 1, 0)), ((0, 0, 1), (1, 0, 1)), ((0, 1, 1), (1, 1, 1)),
             ((0, 0, 0), (0, 1, 0)), ((1, 0, 0), (1, 1, 0)), ((0, 0, 1), (0, 1, 1)), ((1, 0, 1), (1, 1, 1)),
             ((0, 0, 0), (0, 0, 1)), ((1, 0, 0), (1, 0, 1)), ((0, 1, 0), (0, 1, 1)), ((1, 1, 0), (1, 1, 1))]
    del corners
    for a, b in edges:
        ax.plot([x[a[0]], x[b[0]]], [y[a[1]], y[b[1]]], [z[a[2]], z[b[2]]], color="black", linewidth=1.0, alpha=0.5)


def _plot_box_2d(ax, box_size, shift):
    half_r = box_size[0] / 2
    half_t = box_size[1] / 2
    t_min = -half_t - shift
    t_max = half_t - shift
    r_min = -half_r
    r_max = half_r
    corners = [(t_min, r_min), (t_max, r_min), (t_max, r_max), (t_min, r_max), (t_min, r_min)]
    ax.plot(
        [c[0] for c in corners],
        [c[1] for c in corners],
        color="black",
        linewidth=1.0,
        alpha=0.5,
        zorder=-4,
    )


def _limits_3d(rx, ry, rz, box_size=None, shift=0.0, target=None, pad_frac=0.06, min_pad=0.2):
    """Compute RTN axis limits that contain the trajectory and the control box."""
    rx = np.asarray(rx)
    ry = np.asarray(ry)
    rz = np.asarray(rz)
    rmin, rmax = (float(np.min(rx)), float(np.max(rx))) if rx.size else (0.0, 0.0)
    tmin, tmax = (float(np.min(ry)), float(np.max(ry))) if ry.size else (0.0, 0.0)
    nmin, nmax = (float(np.min(rz)), float(np.max(rz))) if rz.size else (0.0, 0.0)

    if box_size is not None:
        rmin = min(rmin, -box_size[0] / 2)
        rmax = max(rmax, box_size[0] / 2)
        tmin = min(tmin, -box_size[1] / 2 - shift)
        tmax = max(tmax, box_size[1] / 2 - shift)
        nmin = min(nmin, -box_size[2] / 2)
        nmax = max(nmax, box_size[2] / 2)

    if target is not None:
        rmin = min(rmin, target[0])
        rmax = max(rmax, target[0])
        tmin = min(tmin, target[1])
        tmax = max(tmax, target[1])
        nmin = min(nmin, target[2])
        nmax = max(nmax, target[2])

    def pad(lo, hi):
        span = hi - lo
        p = max(span * pad_frac, min_pad)
        return lo - p, hi + p

    rmin, rmax = pad(rmin, rmax)
    tmin, tmax = pad(tmin, tmax)
    nmin, nmax = pad(nmin, nmax)
    return (rmin, rmax), (tmin, tmax), (nmin, nmax)


def _limits_2d(rx, ry, box_size=None, shift=0.0, target=None, pad_frac=0.06, min_pad=0.2):
    """Compute limits for the T-R orbital-plane projection."""
    (rmin, rmax), (tmin, tmax), _ = _limits_3d(
        rx, ry, np.array([0.0]), box_size=box_size, shift=shift, target=target,
        pad_frac=pad_frac, min_pad=min_pad
    )
    return (tmin, tmax), (rmin, rmax)


def _set_rtn_limits(ax3, ax2, rx, ry, rz, box_size=None, shift=0.0, target=None, limits=None):
    """Apply consistent limits to the 3D and T-R projection axes."""
    if limits is not None:
        (rmin, rmax), (tmin, tmax), (nmin, nmax) = limits
    else:
        (rmin, rmax), (tmin, tmax), (nmin, nmax) = _limits_3d(
            rx, ry, rz, box_size=box_size, shift=shift, target=target
        )
    ax3.set_xlim3d(rmin, rmax)
    ax3.set_ylim3d(tmin, tmax)
    ax3.set_zlim3d(nmin, nmax)

    x_span = rmax - rmin
    y_span = tmax - tmin
    z_span = nmax - nmin
    if x_span > 0 and y_span > 0 and z_span > 0:
        scale = max(x_span, y_span, z_span)
        ax3.set_box_aspect((x_span / scale, y_span / scale, z_span / scale))
    else:
        ax3.set_box_aspect((1.0, 1.0, 1.0))

    if limits is not None:
        tmin2, tmax2 = tmin, tmax
        rmin2, rmax2 = rmin, rmax
    else:
        (tmin2, tmax2), (rmin2, rmax2) = _limits_2d(
            rx, ry, box_size=box_size, shift=shift, target=target
        )
    ax2.set_xlim(tmin2, tmax2)
    ax2.set_ylim(rmin2, rmax2)


def scenario_rtn_limits(name: str):
    """Return the fixed RTN limits for a scenario name (or ``None``)."""
    return style.SCENARIO_RTN_LIMITS.get(name)


def _tr_limits(limits):
    """Convert 3D RTN ``((r), (t), (n))`` limits to 2D T-R axis limits."""
    (rmin, rmax), (tmin, tmax), _ = limits
    return (tmin, tmax), (rmin, rmax)


def _new_rtn_figure(figsize=(8, 10)):
    """Create the 2x1 RTN figure: top 3D, bottom T-R orbital plane."""
    fig = plt.figure(figsize=figsize)
    ax3 = fig.add_subplot(2, 1, 1, projection="3d")
    ax2 = fig.add_subplot(2, 1, 2)
    return fig, ax3, ax2


def _plot_disjoint_3d(ax, x, y, z, mask, label, **kwargs):
    """Plot masked 3D points as separate contiguous segments."""
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    mask = np.asarray(mask, dtype=bool)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return
    boundaries = np.flatnonzero(np.diff(idx) > 1) + 1
    for i, group in enumerate(np.split(idx, boundaries)):
        sl = slice(group[0], group[-1] + 1)
        ax.plot(x[sl], y[sl], z[sl], label=label if i == 0 else None, **kwargs)


def _plot_disjoint_2d(ax, x, y, mask, label, **kwargs):
    """Plot masked 2D points as separate contiguous segments."""
    x = np.asarray(x)
    y = np.asarray(y)
    mask = np.asarray(mask, dtype=bool)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return
    boundaries = np.flatnonzero(np.diff(idx) > 1) + 1
    for i, group in enumerate(np.split(idx, boundaries)):
        sl = slice(group[0], group[-1] + 1)
        ax.plot(x[sl], y[sl], label=label if i == 0 else None, **kwargs)


def plot_trajectory_eci(real_path, save_path=None, show=False):
    _, rx, ry, rz, flag = DataTable.from_csv(
        real_path, columns=["Time", "R_x", "R_y", "R_z", "Flag"]
    ).to_tuple()
    free = flag != "manoeuvre"
    mano = flag == "manoeuvre"

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    _plot_earth(ax)
    _plot_axis(ax, length=9000, buffer=1500, label="X,Y,Z")
    ax.plot(rx[free], ry[free], rz[free], color=style.TRAJ_COLOR[0], linestyle=style.TRAJ_STYLE[0],
            linewidth=style.TRAJ_WIDTH[0], label="Free Flight Trajectory")
    ax.plot(rx[mano], ry[mano], rz[mano], color=style.TRAJ_COLOR[1], linestyle=style.TRAJ_STYLE[1],
            linewidth=style.TRAJ_WIDTH[1], label="Maneuver Segment")
    ax.scatter(rx[0], ry[0], rz[0], color=style.POINT_COLOR[0], s=style.POINT_SIZE, marker=style.POINT_MARKER[0], label="Start")
    ax.scatter(rx[-1], ry[-1], rz[-1], color=style.POINT_COLOR[1], s=style.POINT_SIZE, marker=style.POINT_MARKER[1], label="End")
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_zlabel("Z (km)")
    lim = 8000
    ax.set_xlim3d([-lim, lim])
    ax.set_ylim3d([-lim, lim])
    ax.set_zlim3d([-lim, lim])
    ax.set_box_aspect([1, 1, 1])
    ax.grid(False)
    ax.legend()
    return _save(fig, save_path, show)


def plot_lvlh_stationkeeping(lvlh_path, box_size, shift=0.0, equal_aspect=True, save_path=None, show=False):
    _, rx, ry, rz, flag = DataTable.from_csv(
        lvlh_path, columns=["Time", "R_x", "R_y", "R_z", "Flag"]
    ).to_tuple()
    mano = flag == "manoeuvre"

    fig, ax3, ax2 = _new_rtn_figure()

    free = ~mano
    _plot_disjoint_3d(ax3, rx, ry, rz, free, "Free Flight Trajectory", color=style.TRAJ_COLOR[0],
                      linestyle=style.TRAJ_STYLE[0], linewidth=style.LVLH_TRAJ_WIDTH[0],
                      alpha=style.LVLH_TRAJ_ALPHA[0])
    _plot_disjoint_3d(ax3, rx, ry, rz, mano, "Maneuver Segment", color=style.TRAJ_COLOR[1],
                      linestyle=style.TRAJ_STYLE[1], linewidth=style.LVLH_TRAJ_WIDTH[1],
                      alpha=style.LVLH_TRAJ_ALPHA[1])
    ax3.scatter(rx[0], ry[0], rz[0], color=style.POINT_COLOR[0], s=style.POINT_SIZE, label="Start")
    ax3.scatter(rx[-1], ry[-1], rz[-1], color=style.POINT_COLOR[1], s=style.POINT_SIZE, label="End")
    ax3.set_xlabel("R (km)")
    ax3.set_ylabel("T (km)")
    ax3.set_zlabel("N (km)")
    _plot_axis(ax3, length=5, buffer=0.7)
    _plot_box(ax3, box_size, shift)
    box_handle = Patch(facecolor="none", edgecolor="black", linewidth=1.0, alpha=0.5,
                       label="Station-keeping box")
    _set_rtn_limits(ax3, ax2, rx, ry, rz, limits=style.STATIONKEEPING_RTN_LIMITS)
    ax3.view_init(elev=22, azim=-50)
    handles, labels = ax3.get_legend_handles_labels()
    handles.append(box_handle)
    labels.append(box_handle.get_label())
    ax3.legend(handles, labels, loc="upper left")
    ax3.grid(False)

    _plot_disjoint_2d(ax2, ry, rx, free, "Free Flight Trajectory", color=style.TRAJ_COLOR[0],
                      linestyle=style.TRAJ_STYLE[0], linewidth=style.LVLH_TRAJ_WIDTH[0],
                      alpha=style.LVLH_TRAJ_ALPHA[0])
    _plot_disjoint_2d(ax2, ry, rx, mano, "Maneuver Segment", color=style.TRAJ_COLOR[1],
                      linestyle=style.TRAJ_STYLE[1], linewidth=style.LVLH_TRAJ_WIDTH[1],
                      alpha=style.LVLH_TRAJ_ALPHA[1])
    ax2.scatter(ry[0], rx[0], color=style.POINT_COLOR[0], s=style.POINT_SIZE, label="Start")
    ax2.scatter(ry[-1], rx[-1], color=style.POINT_COLOR[1], s=style.POINT_SIZE, label="End")
    ax2.set_xlabel("T (km)")
    ax2.set_ylabel("R (km)")
    _plot_axis_2d(ax2)
    _plot_box_2d(ax2, box_size, shift)
    ax2.grid(True, alpha=0.3)
    if equal_aspect:
        ax2.set_aspect("equal")

    fig.tight_layout()
    return _save(fig, save_path, show)


def plot_lvlh_evasive(lvlh_path, target, equal_aspect=True, save_path=None, show=False):
    _, rx, ry, rz, flag = DataTable.from_csv(
        lvlh_path, columns=["Time", "R_x", "R_y", "R_z", "Flag"]
    ).to_tuple()
    mano = flag == "manoeuvre"

    fig, ax3, ax2 = _new_rtn_figure()

    free = ~mano
    _plot_disjoint_3d(ax3, rx, ry, rz, free, "Free Flight Trajectory", color=style.TRAJ_COLOR[0],
                      linestyle=style.TRAJ_STYLE[0], linewidth=style.LVLH_TRAJ_WIDTH[0],
                      alpha=style.LVLH_TRAJ_ALPHA[0])
    _plot_disjoint_3d(ax3, rx, ry, rz, mano, "Maneuver Segment", color=style.TRAJ_COLOR[1],
                      linestyle=style.TRAJ_STYLE[1], linewidth=style.LVLH_TRAJ_WIDTH[1],
                      alpha=style.LVLH_TRAJ_ALPHA[1])
    ax3.scatter(rx[0], ry[0], rz[0], color=style.POINT_COLOR[0], s=style.POINT_SIZE, label="Start")
    ax3.scatter(rx[-1], ry[-1], rz[-1], color=style.POINT_COLOR[1], s=style.POINT_SIZE, label="End")
    ax3.scatter(target[0], target[1], target[2], color=style.POINT_COLOR[2], s=style.POINT_SIZE,
                marker=style.POINT_MARKER[2], label="Target")
    ax3.set_xlabel("R (km)")
    ax3.set_ylabel("T (km)")
    ax3.set_zlabel("N (km)")
    _plot_axis(ax3, length=5, buffer=0.7)
    _set_rtn_limits(ax3, ax2, rx, ry, rz, limits=style.EVASIVE_RTN_LIMITS)
    ax3.view_init(elev=22, azim=-50)
    ax3.legend(loc="upper left")
    ax3.grid(False)

    _plot_disjoint_2d(ax2, ry, rx, free, "Free Flight Trajectory", color=style.TRAJ_COLOR[0],
                      linestyle=style.TRAJ_STYLE[0], linewidth=style.LVLH_TRAJ_WIDTH[0],
                      alpha=style.LVLH_TRAJ_ALPHA[0])
    _plot_disjoint_2d(ax2, ry, rx, mano, "Maneuver Segment", color=style.TRAJ_COLOR[1],
                      linestyle=style.TRAJ_STYLE[1], linewidth=style.LVLH_TRAJ_WIDTH[1],
                      alpha=style.LVLH_TRAJ_ALPHA[1])
    ax2.scatter(ry[0], rx[0], color=style.POINT_COLOR[0], s=style.POINT_SIZE, label="Start")
    ax2.scatter(ry[-1], rx[-1], color=style.POINT_COLOR[1], s=style.POINT_SIZE, label="End")
    ax2.scatter(target[1], target[0], color=style.POINT_COLOR[2], s=style.POINT_SIZE,
                marker=style.POINT_MARKER[2], label="Target")
    ax2.set_xlabel("T (km)")
    ax2.set_ylabel("R (km)")
    _plot_axis_2d(ax2)
    ax2.grid(True, alpha=0.3)
    if equal_aspect:
        ax2.set_aspect("equal")

    fig.tight_layout()
    return _save(fig, save_path, show)


def plot_repropagated_rt(directory, path_list, labels, limits=None, figsize=None,
                         save_path=None, show=False):
    """True vs re-propagated trajectories in the T-R orbital plane.

    ``limits`` is an RTN limit tuple (see
    :data:`sindy_ssa.plotting.style.SCENARIO_RTN_LIMITS`); when
    provided the axes are fixed to it so that a diverging reconstruction does
    not stretch the plot away from the truth trajectory.  ``figsize`` defaults
    to :data:`sindy_ssa.plotting.style.REPRO_FIGSIZE`.
    """
    fig, ax = plt.subplots(figsize=figsize or style.REPRO_FIGSIZE)
    lvlh_path = os.path.join(directory, "Data_LVLH.csv")
    _, rx, ry, flag = DataTable.from_csv(
        lvlh_path, columns=["Time", "R_x", "R_y", "Flag"]
    ).to_tuple()
    mano = flag == "manoeuvre"

    ax.plot(ry[mano], rx[mano], color=style.TRUE_TRAJ_COLOR, linestyle=style.TRUE_TRAJ_STYLE,
            linewidth=style.TRUE_TRAJ_WIDTH, alpha=style.TRUE_TRAJ_ALPHA, label="True Maneuver")
    ax.scatter(ry[mano][0], rx[mano][0], color=style.POINT_COLOR[0], marker=style.POINT_MARKER[0],
               s=style.POINT_SIZE, label="Start", zorder=5)
    ax.scatter(ry[mano][-1], rx[mano][-1], color=style.POINT_COLOR[1], marker=style.POINT_MARKER[1],
               s=style.POINT_SIZE, label="End", zorder=5)

    all_r = [rx[mano]]
    all_t = [ry[mano]]

    for idx, rel_path in enumerate(path_list):
        label = labels[idx]
        pidx = style.method_palette_index(label, idx)
        table = DataTable.from_csv(
            os.path.join(directory, rel_path), columns=["R_x", "R_y", "R_z"]
        )
        rxr = np.asarray(table.get("R_x").data, dtype=float)
        ryr = np.asarray(table.get("R_y").data, dtype=float)

        ax.plot(ryr, rxr, color=style.REPO_TRAJ_COLOR[pidx], linestyle=style.REPO_TRAJ_STYLE[pidx],
                linewidth=style.REPO_LINEWIDTH, alpha=style.REPO_ALPHA, label=f"{label}")
        ax.scatter(ryr[-1], rxr[-1], color=style.REPO_TRAJ_COLOR[pidx], marker=style.REPO_MARKER[pidx],
                   s=style.REPO_MARKER_SIZE)

        all_r.append(rxr)
        all_t.append(ryr)

    ax.set_xlabel("T (km)", fontsize=style.FONT_SIZE + 2)
    ax.set_ylabel("R (km)", fontsize=style.FONT_SIZE + 2)
    ax.tick_params(axis="both", labelsize=style.FONT_SIZE + 2)
    ax.grid(True, alpha=0.3)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        fontsize=style.LEGEND_FONT_SIZE + 2,
    )

    if limits is not None:
        (tmin, tmax), (rmin, rmax) = _tr_limits(limits)
    else:
        (tmin, tmax), (rmin, rmax) = _limits_2d(
            np.concatenate(all_r), np.concatenate(all_t)
        )
    ax.set_xlim(tmin, tmax)
    ax.set_ylim(rmin, rmax)
    fig.tight_layout()
    return _save(fig, save_path, show)


def _manoeuvre_window(path):
    """Return ``(t0, t1)`` seconds spanning the truth ``manoeuvre`` flag, or None."""
    table = DataTable.from_csv(path, columns=["Time", "Flag"])
    time = np.asarray(table.get("Time").data, dtype=float)
    flag = np.asarray(table.get("Flag").data, dtype=str)
    idx = np.flatnonzero(flag == "manoeuvre")
    if idx.size == 0:
        return None
    return float(time[idx[0]]), float(time[idx[-1]])


def plot_thrust_comparison(paths, labels, window=None, figsize=None, vline=None,
                           vline_label=None, save_path=None, show=False,
                           ylim=None, x_tick_interval=None, x_tick_start=None):
    """Faceted thrust comparison: one row per method, one column per component.

    Each cell overlays the truth thrust (black) with the reconstructed method
    curve, restricted to the manoeuvre window.  When the first entry is the
    truth (``labels[0] == "True"``) its ``Flag`` column defines the window and
    no separate row is created for it.  ``figsize`` is ``(width, row_height)``
    and defaults to :data:`sindy_ssa.plotting.style.THRUST_FACET_FIGSIZE`.

    ``vline`` (seconds) draws a dashed vertical marker in every cell, e.g. the
    fit horizon of a partial-window prediction.  ``ylim`` fixes the
    acceleration window of every R/T/N panel.  When ``x_tick_interval`` (hours)
    is given the x ticks are spaced accordingly, starting at ``x_tick_start``
    (hours) or at the first multiple of the interval inside the window.
    """
    tables = [DataTable.from_csv(p, columns=["Time", "u_r", "u_t", "u_n"]) for p in paths]
    has_true = bool(labels) and str(labels[0]).strip().lower() == "true"
    ref = tables[0] if has_true else None
    if window is None and has_true:
        try:
            window = _manoeuvre_window(paths[0])
        except (OSError, KeyError, ValueError):
            window = None

    row_keys = list(range(1, len(tables))) if has_true else list(range(len(tables)))
    if not row_keys:
        row_keys = [0]

    components = ("u_r", "u_t", "u_n")
    palette = [c for c in style.ACC_COLOR if c != "k"]
    col_w, row_h = figsize or style.THRUST_FACET_FIGSIZE
    fig, axes = plt.subplots(
        len(row_keys), 3, figsize=(col_w, row_h * len(row_keys)), sharex=True, squeeze=False
    )

    for r, k in enumerate(row_keys):
        table = tables[k]
        time = np.asarray(table.get("Time").data, dtype=float) / 3600.0
        color = palette[r % len(palette)]
        is_truth_row = ref is not None and k == 0
        for c, component in enumerate(components):
            ax = axes[r][c]
            if ref is not None and not is_truth_row:
                ref_time = np.asarray(ref.get("Time").data, dtype=float) / 3600.0
                ref_data = np.asarray(ref.get(component).data, dtype=float) * 1e6
                ax.plot(ref_time, ref_data, color="k", linewidth=1.4, alpha=0.65, zorder=1)
            data = np.asarray(table.get(component).data, dtype=float) * 1e6
            ax.plot(time, data, color="k" if is_truth_row else color, linewidth=1.0, zorder=2)
            ax.grid(True, alpha=0.3)
            if r == 0:
                ax.set_title(style.ACC_LABELS[c], fontsize=style.FONT_SIZE + 2)
            if c == 0:
                ax.set_ylabel(f"{labels[k]}\n(mm/s$^2$)", fontsize=style.FONT_SIZE + 2)
            if r == len(row_keys) - 1:
                ax.set_xlabel("Time (h)", fontsize=style.FONT_SIZE + 2)
            if window is not None:
                ax.set_xlim(window[0] / 3600.0, window[1] / 3600.0)
                if x_tick_interval:
                    t0h, t1h = window[0] / 3600.0, window[1] / 3600.0
                    start = (x_tick_start if x_tick_start is not None
                             else np.ceil(t0h / x_tick_interval) * x_tick_interval)
                    ax.set_xticks(np.arange(start, t1h + 1e-9, x_tick_interval))
            if ylim is not None:
                ax.set_ylim(*ylim)
            if vline is not None:
                ax.axvline(vline / 3600.0, color="red", linestyle="--", linewidth=1.0, alpha=0.6)

    fig.tight_layout()
    return _save(fig, save_path, show)


def plot_true_thrust(real_path, save_path=None, show=False):
    """Plot the true thrust (u_r, u_t, u_n) over the full time window.

    Three stacked panels (one RTN component per row), full time axis, no legend.
    """
    table = DataTable.from_csv(real_path, columns=["Time", "u_r", "u_t", "u_n"])
    time = np.asarray(table.get("Time").data, dtype=float) / 3600.0

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    for ax, component, label in zip(axes, ("u_r", "u_t", "u_n"), style.ACC_LABELS):
        data = np.asarray(table.get(component).data, dtype=float) * 1e6  # km/s^2 -> mm/s^2
        ax.plot(time, data, color="k", linewidth=1.0)
        ax.set_ylabel(rf"{label} (mm/s$^2$)")
        ax.grid(True, alpha=0.4)
    axes[-1].set_xlabel("Time (h)")
    fig.tight_layout()
    return _save(fig, save_path, show)
