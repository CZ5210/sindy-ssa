"""Typed configuration objects with YAML (de)serialisation.

带类型的配置对象，支持 YAML 读写。

The configuration covers: satellite properties, scenario definition,
observation/noise model, SINDy hyper-parameters and output paths.

配置涵盖：卫星参数、场景、观测/噪声模型、SINDy 超参数与输出路径。
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field, fields
from typing import Any

import yaml


# --------------------------------------------------------------------------- #
# physical / scenario
# --------------------------------------------------------------------------- #
@dataclass
class SatelliteConfig:
    drag_coefficient: float = 2.5
    cross_section_area: float = 120.0
    exhaust_velocity: float = 2500.0 * 9.81
    dry_mass: float = 720.0
    initial_mass: float = 800.0
    max_thrust: float = 0.17  # N


@dataclass
class LQRControlConfig:
    """LQR weights exposed in the configuration.

    Only the state and control weights ``q_diag`` and ``r_diag`` are user-facing.
    The remaining controller constants (state/control scaling, trailing-window
    length and termination tolerances) are fixed implementation details defined
    in :mod:`sindy_ssa.dynamics.satellite`.
    """

    q_diag: list = field(default_factory=lambda: [10.0, 5.0, 0.01, 20.0, 30.0, 20.0])
    r_diag: list = field(default_factory=lambda: [1.0, 1.0, 1.0])


# Backward-compatible alias (the class used to be named after the station-keeping scene).
StationKeepingControlConfig = LQRControlConfig


@dataclass
class ScenarioConfig:
    name: str = "stationkeeping"  # "stationkeeping" | "evasive"
    duration_hours: float = 50.0
    step_seconds: float = 10.0
    orbit_altitude: float = 530.0
    orbit_inclination: float = 43.0
    # station-keeping
    box_size: list = field(default_factory=lambda: [5.0, 40.0, 5.0])
    box_shift: float = 0.0
    stationkeeping_control: LQRControlConfig = field(
        default_factory=LQRControlConfig
    )
    # evasive
    target_position: list = field(default_factory=lambda: [0.0, -20.0, 0.0])
    time_of_flight_hours: float = 4.0
    # When set, the evasive scene mirrors the station-keeping scaled-LQR law;
    # ``None`` (default) keeps the native bang-bang controller.
    evasive_control: LQRControlConfig | None = None


# --------------------------------------------------------------------------- #
# observation / noise
# --------------------------------------------------------------------------- #
@dataclass
class NoiseConfig:
    """MEE 空间直接高斯注入的观测噪声模型。

    由于 RTN 噪声经非线性映射后的 MEE 偏差近似高斯，仿真中直接在 MEE 空间注入
    零均值高斯噪声：

        x_obs = x_MEE + N(0, diag(sigma_p^2, ..., sigma_L^2))

    ``mode`` 为 ``gaussian``（按档位注入）或 ``none``（无噪声理想情形）。``tier``
    选择低/高档（对应论文 Table ``tab:MEE_std``）；也可用 ``mee_sigma`` 显式覆盖
    六个 MEE 标准差。
    """

    mode: str = "gaussian"  # "gaussian" | "none"
    tier: str = "high"      # "low" | "high"
    # 可选：显式指定 [sigma_p, sigma_f, sigma_g, sigma_h, sigma_k, sigma_L]。
    mee_sigma: list | None = None
    seed: int = 0


@dataclass
class ObservationConfig:
    # Down-sampling factor applied to the truth data (10 s -> 60 s means 6).
    sampling_rate: int = 6
    # Single noise config (legacy/backward-compatible) or a mapping from case
    # name to NoiseConfig.  When a mapping is provided, ``sindy run`` iterates
    # all cases and writes one output tree per case.
    noise: NoiseConfig | dict[str, NoiseConfig] = field(default_factory=NoiseConfig)

    def noise_is_multi(self) -> bool:
        return isinstance(self.noise, dict)

    def noise_cases(self) -> list[str]:
        """Return configured case names, or an empty list for a single noise."""
        if isinstance(self.noise, dict):
            return list(self.noise.keys())
        return []

    def get_noise(self, case: str | None = None) -> NoiseConfig:
        """Return the NoiseConfig for ``case``.

        For a single (legacy) noise config, ``case`` is ignored.  For a mapping,
        the matching case is returned; a missing ``ideal`` case is synthesised as
        ``mode="none"`` so the noise-free reference works even without an
        explicit entry in the config.
        """
        if isinstance(self.noise, NoiseConfig):
            return self.noise
        if case is None:
            if len(self.noise) == 1:
                return next(iter(self.noise.values()))
            raise ValueError("case name is required when multiple noise configs are defined")
        if case in self.noise:
            return self.noise[case]
        if case == "ideal":
            return NoiseConfig(mode="none")
        raise KeyError(f"no noise config defined for case {case!r}")


# --------------------------------------------------------------------------- #
# SINDy
# --------------------------------------------------------------------------- #
@dataclass
class PredictionConfig:
    """Partial-window prediction: fit on the first ``fit_fraction`` of the
    manoeuvre arc, then predict the rest.

    部分窗口预测：用机动弧前 ``fit_fraction`` 拟合，预测剩余部分。

    All four predictive methods are identified on the first ``fit_fraction`` of
    the arc; only ``future_state`` differs in how the two **state-based** methods
    (SINDy / SINDy-UMEE) obtain the MEE states they are evaluated on:

    - ``reference`` — the reference (non-thrusting) orbit.  No future
      observation is used, but those states belong to another trajectory, which
      the derivative-based SINDy is extremely sensitive to.
    - ``propagated`` — self-consistent: inside the fit window the observed states
      are used, beyond it the trajectory is re-propagated with the identified
      thrust itself, and the model is re-evaluated on the propagated states
      (causal, fixed-point sweep).  The sweep length and its under-relaxation are
      internal constants of the pipeline, not user-facing.
    """

    enabled: bool = False
    fit_fraction: float = 0.5
    future_state: str = "reference"  # "reference" | "propagated"


@dataclass
class SindyConfig:
    smoothing_window_seconds: float = 1428.0  # T_w ≈ 23.8 min（95.1 min 轨道的 1/4）
    # MEEFD-Spline baseline: smoothing coefficient of the SciPy
    # ``UnivariateSpline``. It sets the spline smoothing factor
    # ``s = factor * n * var(y)``; larger values give fewer knots (smoother),
    # ``0`` yields an interpolating spline.
    meefd_spline_smoothing_factor: float = 0.5
    interpolation_rate: int = 12
    original_alpha: float = 1.0
    original_max_iter: int = 20
    time_alpha: float = 1e-3
    time_max_iter: int = 25
    time_kappa: float = 1e-19
    umee_alpha: float = 1e-4
    umee_max_iter: int = 30
    # Column-wise L2 normalisation of the library before SSR/ridge (PySINDy
    # ``normalize_columns``). Needed for the MEE libraries whose columns span
    # ~7 orders of magnitude; the time library is already well scaled, so it is
    # controlled separately via ``time_normalize_columns``.
    normalize_columns: bool = True
    time_normalize_columns: bool = False
    # Relative RSS tolerance for the L-curve sparsity selection (``select_elbow``).
    sparsity_tol: float = 0.02
    prediction: PredictionConfig = field(default_factory=PredictionConfig)


@dataclass
class BatchConfig:
    """Monte-Carlo batch study settings.

    蒙特卡洛批量统计设置。 由 ``sindy-ssa batch run`` / ``batch plot`` 读取；
    命令行 ``--runs`` / ``--seed0`` / ``--case`` 可覆盖此处的值。

    ``prediction`` overrides :attr:`SindyConfig.prediction` for the batch run only:
    ``None`` (default) keeps whatever the base config defines; when present it lets
    the batch config enable/disable partial-window prediction independently.
    """

    runs: int = 100
    seed0: int = 0
    cases: list = field(default_factory=lambda: ["ideal", "high", "low"])
    prediction: PredictionConfig | None = None
    # Cross-noise overview: use a log y-axis (per panel) instead of linear.
    overview_log: bool = True


@dataclass
class PathsConfig:
    # 顶层根目录：data 只放 CSV/JSON，figures 只放图片。
    data_root: str = "data"
    figures_root: str = "results"
    noise_data_dir: str = "data/noise"
    noise_figures_dir: str = "results/noise"

    # 解析后的单次运行路径，由 ``apply_task_paths`` 填充。
    data_dir: str = "data"
    observation_dir: str = "data"
    methods_dir: str = "data"
    figures_dir: str = "results"
    evaluation_file: str = "data/evaluation.csv"


# --------------------------------------------------------------------------- #
# top level
# --------------------------------------------------------------------------- #
@dataclass
class AppConfig:
    satellite: SatelliteConfig = field(default_factory=SatelliteConfig)
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    sindy: SindyConfig = field(default_factory=SindyConfig)
    batch: BatchConfig = field(default_factory=BatchConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    # -- construction ------------------------------------------------------
    @classmethod
    def from_dict(cls, data: dict[str, Any] | None, scenario: str | None = None) -> AppConfig:
        """Build an :class:`AppConfig` from a YAML mapping.

        ``scenario`` selects one block from a unified file's ``scenarios:``
        mapping; when ``None`` a legacy top-level ``scenario:`` block is used.
        """
        data = data or {}

        def scalars(klass, sub):
            names = {f.name for f in fields(klass)}
            return {k: v for k, v in (sub or {}).items() if k in names}

        obs = data.get("observation") or {}
        obs_kwargs = scalars(ObservationConfig, obs)
        raw_noise = obs.get("noise", {})
        if isinstance(raw_noise, dict) and any(
            isinstance(v, dict) for v in raw_noise.values()
        ):
            # New style: one noise config per case, e.g. {high: {mode: ...}, ...}
            obs_kwargs["noise"] = {
                str(case): NoiseConfig(**scalars(NoiseConfig, noise_data))
                for case, noise_data in raw_noise.items()
            }
        else:
            # Legacy/backward-compatible single noise config.
            obs_kwargs["noise"] = NoiseConfig(**scalars(NoiseConfig, raw_noise))

        if scenario is not None:
            scenario_block = (data.get("scenarios") or {}).get(scenario) or {}
            scenario_block = dict(scenario_block)
            scenario_block.setdefault("name", scenario)
        else:
            scenario_block = data.get("scenario") or {}
        sk_ctrl = scenario_block.get("stationkeeping_control") or {}
        scenario_kwargs = scalars(ScenarioConfig, scenario_block)
        scenario_kwargs["stationkeeping_control"] = LQRControlConfig(
            **scalars(LQRControlConfig, sk_ctrl)
        )
        ev_ctrl = scenario_block.get("evasive_control")
        scenario_kwargs["evasive_control"] = (
            None
            if ev_ctrl is None
            else LQRControlConfig(**scalars(LQRControlConfig, ev_ctrl))
        )

        sindy = data.get("sindy") or {}
        sindy_kwargs = scalars(SindyConfig, sindy)
        sindy_kwargs["prediction"] = PredictionConfig(
            **scalars(PredictionConfig, sindy.get("prediction", {}))
        )

        batch = data.get("batch") or {}
        batch_kwargs = scalars(BatchConfig, batch)
        # ``prediction`` is optional here: omit it to inherit ``sindy.prediction``.
        batch_kwargs["prediction"] = (
            PredictionConfig(**scalars(PredictionConfig, batch["prediction"]))
            if "prediction" in batch
            else None
        )

        return cls(
            satellite=SatelliteConfig(**scalars(SatelliteConfig, data.get("satellite"))),
            scenario=ScenarioConfig(**scenario_kwargs),
            observation=ObservationConfig(**obs_kwargs),
            sindy=SindyConfig(**sindy_kwargs),
            batch=BatchConfig(**batch_kwargs),
            paths=PathsConfig(**scalars(PathsConfig, data.get("paths"))),
        )

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


TASKS = ("typical", "batch")
_EVAL_FILES = {
    "typical": "evaluation.csv",
    "batch": "batch_evaluation.csv",
}


def apply_task_paths(config: AppConfig, scenario: str, task: str, case: str) -> AppConfig:
    """为一次运行填充解析后的输出路径，并返回 ``config`` 便于链式调用。

    目录约定：真值每个场景一份、三任务共用；两类任务各自独立子树。

    - ``data_root/<scenario>/truth/``          —— 真值 CSV
    - ``data_root/<scenario>/<task>/<case>/``  —— 观测 / 方法 / 评估 CSV
    - ``figures_root/<scenario>/<task>/<case>/`` —— 对应图片
    """
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; expected one of {TASKS}")

    p = config.paths
    scenario_data = os.path.join(p.data_root, scenario)
    task_data = os.path.join(scenario_data, task, case)

    p.data_dir = os.path.join(scenario_data, "truth")
    p.observation_dir = os.path.join(task_data, "observation")
    p.methods_dir = os.path.join(task_data, "methods")
    p.figures_dir = os.path.join(p.figures_root, scenario, task, case)
    p.evaluation_file = os.path.join(task_data, _EVAL_FILES[task])
    return config


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` on top of ``base`` (override wins)."""
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _resolve_base(data: dict, base_dir: str) -> dict:
    """Resolve an optional top-level ``base:`` config-inheritance reference."""
    base_ref = data.pop("base", None)
    if base_ref is None:
        return data
    base_path = base_ref if os.path.isabs(base_ref) else os.path.join(base_dir, base_ref)
    with open(base_path, "r", encoding="utf-8") as handle:
        base_data = _resolve_base(yaml.safe_load(handle) or {}, os.path.dirname(base_path))
    return _deep_merge(base_data, data)


def _load_raw(path: str) -> dict:
    """Load a YAML config, following an optional ``base:`` inheritance chain.

    A config may declare ``base: <path>`` (relative to itself) to inherit every
    key from another config and override only what it needs, e.g. the batch config
    inherits ``config.yaml`` and only sets its ``batch:`` block.
    """
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return _resolve_base(data, os.path.dirname(os.path.abspath(path)))


def load_config(path: str | None = None, scenario: str | None = None) -> AppConfig:
    """Load an :class:`AppConfig` from YAML, or return defaults if ``None``.

    For a unified file with a ``scenarios:`` mapping, pass ``scenario`` to pick
    the block.  A ``base:`` reference in the file is resolved first.
    """
    if path is None:
        return AppConfig()
    data = _load_raw(path)
    return AppConfig.from_dict(data, scenario=scenario)


def scenario_names(path: str) -> list[str]:
    """Return the scenario names available in a config file.

    Unified files expose a ``scenarios:`` mapping; a legacy single-scenario file
    falls back to its top-level ``scenario.name``.  ``base:`` inheritance is
    resolved so an inheriting config still reports the base's scenarios.
    """
    data = _load_raw(path)
    scenarios = data.get("scenarios")
    if scenarios:
        return list(scenarios.keys())
    name = (data.get("scenario") or {}).get("name", "stationkeeping")
    return [name]


def load_scenarios(path: str) -> dict[str, AppConfig]:
    """Load every scenario from a unified (or legacy) config file."""
    return {name: load_config(path, scenario=name) for name in scenario_names(path)}



def save_config(config: AppConfig, path: str) -> None:
    import os

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False, allow_unicode=True)
