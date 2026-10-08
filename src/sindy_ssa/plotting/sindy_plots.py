"""SINDy-specific plots (coefficient heatmaps). / SINDy 系数热图。"""

from __future__ import annotations

import functools
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from . import style
from .style import save_figure


@functools.cache
def feature_math_label(name) -> str:
    """Render a SymPy-style feature name (e.g. ``(f^2 + g^2)*sin(L)``) as mathtext.

    把库函数名（SymPy 的 ``str`` 形式，``**`` 已写成 ``^``）转成 matplotlib
    mathtext，供 Xi 热图的 y 轴刻度使用。解析失败时原样返回。
    """
    text = str(name)
    import sympy as sp

    try:
        expr = sp.sympify(text.replace("^", "**"))
    except (sp.SympifyError, SyntaxError, TypeError, ValueError):
        return text
    return f"${sp.latex(expr)}$"


def plot_xi_heatmap(path, save_path=None, x_labels=None, feature_labels=None,
                    textsize=12, figsize=(12, 18), annotate=True, show=False):
    """Plot the absolute coefficient matrix ``Xi`` as a heatmap.

    ``x_labels`` labels the output columns (thrust components or MEE
    derivatives); ``feature_labels`` overrides the y-axis library terms.
    Both default to LaTeX (mathtext) rendering.

    ``x_labels`` 标注输出列，``feature_labels`` 覆盖 y 轴的库函数项；两者默认
    渲染为 LaTeX（mathtext）公式。
    """
    df = pd.read_csv(path)
    feature_names = df.iloc[:, 0].values
    xi = df.iloc[:, 1:].values
    xi_abs = abs(xi)
    mask = xi_abs < 1e-90

    x_ticklabels = x_labels if x_labels is not None else list(df.columns[1:])
    if annotate:
        if feature_labels is not None:
            y_ticklabels = list(feature_labels)
        else:
            y_ticklabels = [feature_math_label(name) for name in feature_names]
    else:
        y_ticklabels = []

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        xi_abs,
        annot=annotate,
        fmt=".3e",
        mask=mask,
        cmap=style.XI_CMAP,
        cbar=True,
        cbar_kws={"label": "Coefficients Absolute Value"},
        linewidths=0,
        xticklabels=x_ticklabels,
        yticklabels=y_ticklabels,
        annot_kws={"size": textsize, "va": "center_baseline"},
        ax=ax,
    )
    plt.xticks(rotation=45, ha="right", fontsize=16)
    plt.yticks(rotation=0, fontsize=14)
    ax.set_ylabel("Features", fontsize=16)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        save_figure(fig, save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return save_path
