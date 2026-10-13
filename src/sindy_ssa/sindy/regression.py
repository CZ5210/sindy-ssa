"""Sparse regression wrappers around PySINDy's SSR optimiser.

对 PySINDy SSR 优化器的封装。
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from .libraries import LibrarySpec


def _ssr_kwargs(spec: LibrarySpec, max_iter: int, normalize_columns: bool = False):
    """Build PySINDy SSR keyword dict."""
    return {
        "max_iter": max_iter,
        "alpha": spec.alpha,
        "kappa": 0.0 if spec.kappa is None else spec.kappa,
        "normalize_columns": normalize_columns,
    }


def _column_norms(theta: np.ndarray):
    """L2 norm of each library column (for ``normalize_columns``)."""
    norms = np.linalg.norm(theta, axis=0)
    norms[norms == 0] = 1.0
    return norms


def fit_ssr(spec: LibrarySpec, X: np.ndarray, Y: np.ndarray, normalize_columns: bool = False):
    """Fit a custom library with SSR; returns ``(library, optimizer, Xi, Theta)``."""
    import pysindy as ps

    library = ps.CustomLibrary(library_functions=spec.functions).fit(X)
    theta = library.transform(X)
    optimizer = ps.SSR(**_ssr_kwargs(spec, spec.max_iter, normalize_columns))
    optimizer.fit(theta, Y)
    Xi = optimizer.coef_.T
    return library, optimizer, Xi, theta


def _ridge_coeffs(theta: np.ndarray, Y: np.ndarray, alpha: float) -> np.ndarray:
    """Ridge solution matching PySINDy SSR with ``normalize_columns=False``.

    Returns ``Xi`` of shape ``(n_features, n_outputs)``.
    """
    A = theta.T @ theta + alpha * np.eye(theta.shape[1])
    B = theta.T @ Y
    return np.linalg.solve(A, B)


def _ridge_refit(theta: np.ndarray, Y: np.ndarray, alpha: float,
                 supports: np.ndarray) -> np.ndarray:
    """Ridge refit restricted to the per-column ``supports`` mask."""
    Xi = np.zeros((theta.shape[1], Y.shape[1]))
    for j in range(Y.shape[1]):
        cols = supports[:, j]
        if not cols.any():
            continue
        theta_j = theta[:, cols]
        A = theta_j.T @ theta_j + alpha * np.eye(cols.sum())
        Xi[cols, j] = np.linalg.solve(A, theta_j.T @ Y[:, j])
    return Xi


def _ols_refit(theta: np.ndarray, Y: np.ndarray, Xi: np.ndarray) -> np.ndarray:
    """Re-solve the support of ``Xi`` by ordinary least squares, per output column.

    This is exactly PySINDy's ``unbias=True`` post-processing
    (``LinearRegression(fit_intercept=False)`` on the retained terms), and is
    what the legacy ``ps.SSR(...)`` path did by default.  Without it the ridge
    penalty keeps shrinking the retained coefficients after the support has been
    chosen; with unit-normalised columns and ``alpha = 1`` every retained
    coefficient would be halved.

    ``theta`` must be the library matrix in the *original* (un-normalised) units,
    which is what :func:`fit_ssr_path` returns, so the result is directly
    comparable with ``Xi``.
    """
    out = np.zeros_like(Xi)
    for j in range(Y.shape[1]):
        cols = np.abs(Xi[:, j]) > 0.0
        if not cols.any():
            continue
        out[cols, j] = np.linalg.lstsq(theta[:, cols], Y[:, j], rcond=None)[0]
    return out


class _PathOptimizer:
    """Minimal optimizer exposing ``predict`` for a path-selected model."""

    def __init__(self, Xi: np.ndarray):
        self.coef_ = Xi.T

    def predict(self, theta: np.ndarray) -> np.ndarray:
        return theta @ self.coef_.T


def fit_ssr_path(spec: LibrarySpec, X: np.ndarray, Y: np.ndarray,
                 normalize_columns: bool = False):
    """Generate the complete SSR greedy deletion path.

    Mirrors the SSR algorithm: starting from the full ridge fit, each step
    deletes the smallest-magnitude coefficient **per output column** and
    refits ridge on the surviving support. Unlike repeated calls to
    ``ps.SSR(max_iter=k)`` (which stops when one term per column remains and
    returns an inflection/argmin model instead of the k-th model), this
    yields the full path from ``n_f * n_u`` down to ``n_u`` coefficients.

    When ``normalize_columns=True`` the library columns are scaled to unit L2
    norm for *both* the ridge fit and the greedy magnitude test (matching
    PySINDy's ``normalize_columns`` semantics); the returned ``Xi`` entries are
    rescaled to the original feature units so predictions stay valid.

    Every ``Xi`` on the path is a **ridge** solution; de-biasing the selected
    support (OLS) is the caller's job — see :func:`_ols_refit` and
    :func:`fit_ssr_auto`.

    Returns ``(library, Theta, path)`` with entries ``max_iter`` (deletions
    per column), ``rss``, ``n_active`` and ``Xi``.
    """
    import pysindy as ps

    library = ps.CustomLibrary(library_functions=spec.functions).fit(X)
    theta = np.asarray(library.transform(X))
    n_f = theta.shape[1]

    feat_norms = None
    theta_fit = theta
    if normalize_columns:
        feat_norms = _column_norms(theta)
        theta_fit = theta / feat_norms

    def original(Xi):
        return Xi if feat_norms is None else Xi / feat_norms[:, None]

    supports = np.ones((n_f, Y.shape[1]), dtype=bool)
    Xi = _ridge_refit(theta_fit, Y, spec.alpha, supports)
    path = [{
        "max_iter": 0,
        "rss": float(np.sum((Y - theta_fit @ Xi) ** 2)),
        "n_active": int(supports.sum()),
        "Xi": original(Xi),
    }]

    for k in range(1, n_f):
        for j in range(Y.shape[1]):
            magnitude = np.abs(Xi[:, j]).copy()
            magnitude[~supports[:, j]] = np.inf
            supports[int(np.argmin(magnitude)), j] = False
        Xi = _ridge_refit(theta_fit, Y, spec.alpha, supports)
        path.append({
            "max_iter": k,
            "rss": float(np.sum((Y - theta_fit @ Xi) ** 2)),
            "n_active": int(supports.sum()),
            "Xi": original(Xi),
        })
    return library, theta, path

DEFAULT_SPARSITY_TOL = 0.02


def pareto_frontier(path: list[dict]):
    """Lower frontier of ``(n_active, rss)``: best RSS with at most k terms.

    Because PySINDy SSR refits retained coefficients with OLS (``unbias=True``),
    a denser model can occasionally have a *worse* RSS than a sparser one;
    such points are dominated and removed by a running minimum.
    Returns ``(n_active_sorted, rss_frontier, order)``.
    """
    n_active = np.array([p["n_active"] for p in path], dtype=float)
    rss = np.array([p["rss"] for p in path], dtype=float)
    order = np.argsort(n_active)
    n_active = n_active[order]
    rss = rss[order]
    rss = np.minimum.accumulate(rss)
    return n_active, rss, order


def select_elbow(path: list[dict], min_active: int = 3,
                 tol: float = DEFAULT_SPARSITY_TOL) -> int:
    """Sparsest-tolerant L-curve selection on the SSR deletion path.

    On the Pareto frontier, pick the *sparsest* model whose RSS is within
    ``tol`` of the achievable minimum::

        rss <= rss_min + tol * (rss_max - rss_min)

    i.e. we accept a slight RSS increase in exchange for a sparser model,
    which selects the beginning of the plateau of the L-curve.
    """
    n_active, rss, _ = pareto_frontier(path)

    valid = n_active >= min_active
    if not np.any(valid):
        valid = n_active >= 1.0
    n_active_v = n_active[valid]
    rss_v = rss[valid]
    if len(n_active_v) == 0:
        return max(path[0]["max_iter"], 1)

    rss_min = rss_v.min()
    rss_max = rss_v.max()
    threshold = rss_min + tol * (rss_max - rss_min)
    eligible = n_active_v[rss_v <= threshold]
    selected_n_active = float(eligible.min()) if len(eligible) else float(n_active_v[-1])

    candidates = [p for p in path if p["n_active"] == selected_n_active]
    if not candidates:
        all_n = np.array([p["n_active"] for p in path])
        closest = int(np.argmin(np.abs(all_n - selected_n_active)))
        candidates = [path[closest]]
    candidates.sort(key=lambda p: p["rss"])
    return max(candidates[0]["max_iter"], 1)


def fit_ssr_auto(spec: LibrarySpec, X: np.ndarray, Y: np.ndarray,
                 normalize_columns: bool = False,
                 tol: float = DEFAULT_SPARSITY_TOL,
                 unbias: bool | None = None):
    """Fit SSR and select the sparsity level automatically via the L-curve elbow.

    The support is selected exactly as before: the greedy deletion path from
    :func:`fit_ssr_path` (ridge refits only) drives :func:`select_elbow`, so the
    plotted curve, the manuscript's κ rule and the selected iteration are
    unchanged.  What *is* new is the last step: the coefficients on the selected
    support are re-solved by ordinary least squares (:func:`_ols_refit`), i.e.
    PySINDy's ``unbias=True`` semantics (the ``ps.SSR`` default used by the
    legacy pipeline).  Skipping it left the ridge penalty shrinking the retained
    coefficients, which is what made the identified derivative model systematically
    too small.

    ``unbias`` overrides :attr:`LibrarySpec.unbias`; ``None`` (default) uses the
    spec's setting (``True`` for every library defined in this package).

    ``tol`` is the relative RSS tolerance of :func:`select_elbow`.

    Returns ``(library, optimizer, Xi, Theta, selected_iter, path)``.  ``Xi`` is
    the de-biased model on the selected support; ``path`` still holds the ridge
    deletion path (that is the L-curve, so it is left untouched).
    """
    if unbias is None:
        unbias = getattr(spec, "unbias", True)
    library, theta, path = fit_ssr_path(spec, X, Y, normalize_columns=normalize_columns)
    selected_iter = select_elbow(path, tol=tol)
    Xi = next(p["Xi"] for p in path if p["max_iter"] == selected_iter)
    if unbias:
        Xi = _ols_refit(theta, Y, Xi)
    optimizer = _PathOptimizer(Xi)
    return library, optimizer, Xi, theta, selected_iter, path


def save_selection(path: list[dict], selected_iter: int, out_path: str,
                   tol: float = DEFAULT_SPARSITY_TOL, alpha: float | None = None) -> None:
    """Save the L-curve path, selected point and the selection settings as JSON."""
    payload = {
        "selected_max_iter": selected_iter,
        "sparsity_tol": tol,
        "ridge_alpha": alpha,
        "path": [
            {"max_iter": p["max_iter"], "rss": p["rss"], "n_active": p["n_active"]}
            for p in path
        ],
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def predict(optimizer, library, X: np.ndarray):
    """Evaluate the fitted sparse model on new states."""
    theta = library.transform(X)
    return optimizer.predict(theta), theta


def save_matrix(matrix, path: str, row_headers=None, column_headers=None, precision: int = 8) -> None:
    """Save a coefficient/feature matrix to CSV.

    保存系数/特征矩阵到 CSV。
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    df = pd.DataFrame(
        matrix,
        index=row_headers if row_headers is not None else None,
        columns=column_headers if column_headers is not None else None,
    )
    df.to_csv(path, encoding="utf-8-sig", float_format=f"%.{precision}e", index=row_headers is not None)
