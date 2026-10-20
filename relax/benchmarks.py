"""Named environment suites understood by ``scripts/launch.py``."""
from __future__ import annotations

from relax.humanoidbench import HUMANOIDBENCH_MAIN_ENVS

MUJOCO_V3_ENVS = (
    "Walker2d-v3",
    "Hopper-v3",
    "Humanoid-v3",
    "Ant-v3",
    "HalfCheetah-v3",
    "Swimmer-v3",
)

_STATIC_BENCHMARKS = {
    "mujoco-v3": MUJOCO_V3_ENVS,
    "humanoidbench": HUMANOIDBENCH_MAIN_ENVS,
}
_ALIASES = {
    "hb": "humanoidbench",
    "humanoidbench-main": "humanoidbench",
    "mujoco": "mujoco-v3",
}


def benchmark_names() -> tuple[str, ...]:
    return tuple((*_STATIC_BENCHMARKS, "dmc"))


def benchmark_envs(name: str) -> tuple[str, ...]:
    """Resolve a benchmark name to the environment ids used as a hard sweep axis."""
    key = _ALIASES.get(name.lower(), name.lower())
    if key in _STATIC_BENCHMARKS:
        return _STATIC_BENCHMARKS[key]
    if key == "dmc":
        import gymnasium
        from relax.dmc import PREFIX, register_dmc_envs

        if register_dmc_envs() == 0:
            raise RuntimeError("The dmc benchmark requires shimmy and dm_control")
        return tuple(sorted(
            env_id for env_id in gymnasium.registry
            if env_id.startswith(f"{PREFIX}/")
        ))
    choices = ", ".join(benchmark_names())
    raise ValueError(f"Unknown benchmark {name!r}; expected one of: {choices}")
