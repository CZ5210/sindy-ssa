"""Candidate-function libraries for the sparse-regression stage.

稀疏回归阶段使用的候选函数库。

Each library is described by a :class:`LibrarySpec`, which bundles the
lambdified functions, human-readable names and the SSR hyper-parameters.

每个库用 :class:`LibrarySpec` 描述：包含数值化函数、可读名称和 SSR 超参数。
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp


@dataclass
class LibrarySpec:
    functions: list
    names: list
    max_iter: int
    alpha: float
    kappa: float | None = None
    label: str = ""


def _lambdify(symbols, expressions, modules="numpy"):
    functions = [sp.lambdify(symbols, expr, modules=modules) for expr in expressions]
    names = [str(expr).replace("**", "^") for expr in expressions]
    return functions, names


def original_library(max_iter: int = 20, alpha: float = 1.0, kappa: float | None = None) -> LibrarySpec:
    """SINDy library over the MEE states (manuscript Eq. 23)."""
    p, f, g, h, k, L = sp.symbols("p f g h k L")
    expressions = [
        1,
        f, g, h, k, L,
        f**2, g**2, h**2, k**2,
        f * g, f * h, f * k, f * L,
        g * h, g * k, h * k,
        f**3, g**3, h**3, k**3, L**3,
        f * g * h, f * g * k, f * h * k, g * h * k,
        h * sp.sin(L), h * sp.cos(L),
        f * sp.sin(L), g * sp.cos(L),
        sp.sin(L), sp.cos(L),
        sp.sin(L) ** 2, sp.cos(L) ** 2,
        sp.sin(L) ** 3, sp.cos(L) ** 3,
    ]
    functions, names = _lambdify((p, f, g, h, k, L), expressions)
    return LibrarySpec(functions, names, max_iter, alpha, kappa, label="SINDy")


def time_library(level: str = "sparse", alpha=None, max_iter=None, kappa=None) -> LibrarySpec:
    """Time-only library for SINDy-TIME (manuscript Eq. 26).

    Levels: ``non_sparse`` (5, 1e-4, 0), ``sparse`` (25, 1e-3, 1e-19),
    ``very_sparse`` (50, 1e-1, 1e-15).
    """
    presets = {
        "non_sparse": (5, 1e-4, 0.0),
        "sparse": (25, 1e-3, 1e-19),
        "very_sparse": (50, 1e-1, 1e-15),
    }
    if level not in presets:
        raise KeyError(f"unknown sparsity level: {level!r}; choose from {list(presets)}")
    n_default, alpha_default, kappa_default = presets[level]

    t = sp.symbols("t")
    eps = 1e-7
    powers = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    expressions = [1, t, t**2, t**3]
    expressions += [(t + eps) ** q for q in powers]
    expressions += [sp.sin((t + eps) ** q) for q in powers]
    expressions += [sp.cos((t + eps) ** q) for q in powers]
    functions, names = _lambdify(t, expressions)
    return LibrarySpec(
        functions,
        names,
        max_iter if max_iter is not None else n_default,
        alpha if alpha is not None else alpha_default,
        kappa if kappa is not None else kappa_default,
        label=f"SINDy-TIME[{level}]",
    )


def umee_library(max_iter: int = 30, alpha: float = 1e-3, kappa: float | None = None) -> LibrarySpec:
    """MEE-based library for SINDy-UMEE (manuscript Eq. 29).

    The caller passes the raw MEE elements; the scale disparity between ``p``
    (km) and the dimensionless elements is handled by ``normalize_columns`` in
    the regression stage. / 调用方传入原始 MEE 元素，尺度差异由回归阶段的列归一化处理。
    """
    p, f, g, h, k, L = sp.symbols("p f g h k L")
    expressions = [
        p**0, p, f, g, h, k, L,
        p**2, f**2, g**2, h**2, k**2, L**2,
        (f * f + g * g) * sp.sin(L),
        sp.sin(L), sp.cos(L),
        sp.sin(2 * L), sp.cos(2 * L), sp.sin(2 * L) * sp.cos(2 * L),
        p * sp.sin(L), f * sp.sin(L), g * sp.cos(L), h * sp.sin(L), k * sp.cos(L), L * sp.sin(L),
        p * sp.sin(2 * L), f * sp.sin(2 * L), g * sp.cos(2 * L), h * sp.sin(2 * L), k * sp.cos(2 * L), L * sp.sin(2 * L),
    ]
    functions, names = _lambdify((p, f, g, h, k, L), expressions)
    return LibrarySpec(functions, names, max_iter, alpha, kappa, label="SINDy-UMEE")
