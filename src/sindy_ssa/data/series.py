"""Column-wise data containers used across the pipeline.

贯穿整个流程的按列数据容器。

`DataSeries` is a single named 1-D channel; `DataTable` is a collection of
equal-length channels. They intentionally mirror the operations of the original
`Sequence` / `SequenceData` classes (sampling, smoothing, interpolation, noise,
normalisation) but with clearer naming and explicit random-number handling.

`DataSeries` 表示单个带名字的一维通道，`DataTable` 表示等长的多通道集合。
它们保留了原 `Sequence` / `SequenceData` 的操作语义，但命名更清晰，并显式
处理随机数生成器。
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd


class DataSeries:
    """A single named 1-D channel.

    单个带名字的一维通道。
    """

    def __init__(self, data, name: str = "series"):
        self.data = np.asarray(data).ravel()
        self.name = str(name)

    # -- basic -------------------------------------------------------------
    @property
    def n(self) -> int:
        return int(self.data.size)

    def copy(self) -> DataSeries:
        return DataSeries(self.data.copy(), self.name)

    def renamed(self, name: str) -> DataSeries:
        return DataSeries(self.data, name)

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"DataSeries(name={self.name!r}, n={self.n})"

    # -- transforms --------------------------------------------------------
    def sample(self, rate: int) -> DataSeries:
        """Down-sample by taking every `rate`-th point."""
        return DataSeries(self.data[::rate], self.name)

    def filter(self, interval) -> DataSeries:
        """Keep the half-open index interval ``[start, end)``."""
        start, end = interval
        return DataSeries(self.data[start:end], self.name)

    def smooth(self, window_size: int, polyorder: int = 3) -> DataSeries:
        """Savitzky-Golay smoothing (window forced odd); ``0`` is a no-op."""
        if window_size in (0, None):
            return self.copy()
        n = self.n
        if window_size > n:
            raise ValueError(
                f"smoothing window ({window_size}) exceeds series length ({n})"
            )
        if window_size % 2 == 0:
            window_size += 1
        from scipy.signal import savgol_filter

        smoothed = savgol_filter(
            np.asarray(self.data, dtype=float),
            window_length=window_size,
            polyorder=polyorder,
            mode="mirror",
        )
        return DataSeries(smoothed, self.name)

    def linear_interp(self, rate: int) -> DataSeries:
        """Upsample by ``rate`` using linear interpolation on the index axis."""
        n = self.n
        if n < 2:
            return self.copy()
        original = np.arange(n)
        new_length = (n - 1) * rate + 1
        new_index = np.linspace(0.0, n - 1.0, new_length)
        interp = np.interp(new_index, original, np.asarray(self.data, dtype=float))
        return DataSeries(interp, self.name)

    def add(self, other: DataSeries, k: float = 1.0, new_name: str | None = None) -> DataSeries:
        return DataSeries(self.data + k * other.data, new_name or self.name)

    def add_noise(self, mu: float, sigma: float, rng: np.random.Generator | None = None) -> DataSeries:
        rng = rng or np.random.default_rng()
        noise = rng.normal(loc=mu, scale=sigma, size=self.n)
        return DataSeries(self.data + noise, self.name)

    def normalize(self):
        """Scale by the maximum absolute value; returns ``(coeff, scaled)``."""
        coeff = float(np.max(np.abs(self.data)))
        if coeff == 0.0:
            coeff = 1.0
        return coeff, DataSeries(self.data / coeff, self.name)


class DataTable:
    """A collection of equal-length :class:`DataSeries`.

    等长 :class:`DataSeries` 的集合。
    """

    def __init__(self, series, name: str = "table"):
        self.series = list(series)
        self.name = str(name)
        lengths = {s.n for s in self.series}
        if len(lengths) > 1:
            detail = ", ".join(f"{s.name}:{s.n}" for s in self.series)
            raise ValueError(f"columns have inconsistent lengths -> {detail}")
        self.n = self.series[0].n if self.series else 0

    # -- constructors ------------------------------------------------------
    @classmethod
    def from_arrays(cls, names, arrays, name: str = "table") -> DataTable:
        return cls([DataSeries(a, n) for n, a in zip(names, arrays)], name=name)

    @classmethod
    def single(cls, name: str, data, table_name: str = "table") -> DataTable:
        return cls([DataSeries(data, name)], name=table_name)

    @classmethod
    def from_csv(cls, path, columns=None, name: str = "table") -> DataTable:
        df = pd.read_csv(path)
        if columns is not None:
            if isinstance(columns, str):
                columns = [columns]
            missing = [c for c in columns if c not in df.columns]
            if missing:
                raise ValueError(
                    f"columns not found in {path}: {missing}; available: {list(df.columns)}"
                )
            df = df[columns]
        return cls([DataSeries(df[c].to_numpy(), c) for c in df.columns], name=name)

    @classmethod
    def from_concat(cls, *tables: DataTable, name: str = "table") -> DataTable:
        merged: list[DataSeries] = []
        seen: set[str] = set()
        for t in tables:
            for s in t.series:
                if s.name in seen:
                    raise ValueError(f"duplicate column name on concat: {s.name}")
                seen.add(s.name)
                merged.append(s)
        return cls(merged, name=name)

    # -- access ------------------------------------------------------------
    @property
    def columns(self) -> list[str]:
        return [s.name for s in self.series]

    def get(self, name: str) -> DataSeries:
        for s in self.series:
            if s.name == name:
                return s
        raise KeyError(f"column {name!r} not found; available: {self.columns}")

    def has(self, name: str) -> bool:
        return any(s.name == name for s in self.series)

    def to_tuple(self):
        return tuple(s.data for s in self.series)

    def to_dict(self) -> dict:
        return {s.name: s.data for s in self.series}

    def select(self, names) -> DataTable:
        if isinstance(names, str):
            names = [names]
        return DataTable([self.get(n) for n in names], name=self.name)

    def without(self, names) -> DataTable:
        if isinstance(names, str):
            names = [names]
        return DataTable([s for s in self.series if s.name not in names], name=self.name)

    # -- transforms --------------------------------------------------------
    def sample(self, rate: int) -> DataTable:
        return DataTable([s.sample(rate) for s in self.series], name=self.name)

    def filter(self, interval) -> DataTable:
        return DataTable([s.filter(interval) for s in self.series], name=self.name)

    def linear_interp(self, rate: int) -> DataTable:
        return DataTable([s.linear_interp(rate) for s in self.series], name=self.name)

    def smooth(self, window: int, exclude: str | int | None = None, polyorder: int = 3) -> DataTable:
        """Smooth every column except ``exclude`` (name or index)."""
        if isinstance(exclude, int):
            exclude = self.columns[exclude] if self.columns else None
        out = []
        for s in self.series:
            out.append(s if s.name == exclude else s.smooth(window, polyorder=polyorder))
        return DataTable(out, name=self.name)

    def add_noise(self, indices, mus, sigmas, rng: np.random.Generator | None = None) -> DataTable:
        """Add Gaussian noise to selected columns (by index)."""
        rng = rng or np.random.default_rng()
        out = list(self.series)
        for i, idx in enumerate(indices):
            out[idx] = self.series[idx].add_noise(mus[i], sigmas[i], rng=rng)
        return DataTable(out, name=self.name)

    def normalize_columns(self, names) -> tuple[dict, DataTable]:
        """Normalise the given columns; returns ``(coeffs, table)``."""
        coeffs = {}
        out = list(self.series)
        for name in names:
            idx = self.columns.index(name)
            coeff, scaled = out[idx].normalize()
            coeffs[name] = coeff
            out[idx] = scaled
        return coeffs, DataTable(out, name=self.name)

    # -- io ----------------------------------------------------------------
    def save(self, path: str, precision: int = 8) -> None:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        df = pd.DataFrame({s.name: s.data for s in self.series})
        df.to_csv(path, index=False, float_format=f"%.{precision}e")
