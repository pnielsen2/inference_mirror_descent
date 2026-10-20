"""Gymnasium aliases and benchmark metadata for HumanoidBench."""
from __future__ import annotations

import importlib

import gymnasium
import numpy as np

PREFIX = "humanoidbench"

HUMANOIDBENCH_MAIN_NATIVE_ENVS = (
    "h1hand-walk-v0",
    "h1hand-reach-v0",
    "h1hand-hurdle-v0",
    "h1hand-crawl-v0",
    "h1hand-maze-v0",
    "h1hand-push-v0",
    "h1hand-cabinet-v0",
    "h1strong-highbar_hard-v0",
    "h1hand-door-v0",
    "h1hand-truck-v0",
    "h1hand-cube-v0",
    "h1hand-bookshelf_simple-v0",
    "h1hand-bookshelf_hard-v0",
    "h1hand-basketball-v0",
    "h1hand-window-v0",
    "h1hand-spoon-v0",
    "h1hand-kitchen-v0",
    "h1hand-package-v0",
    "h1hand-powerlift-v0",
    "h1hand-room-v0",
    "h1hand-stand-v0",
    "h1hand-run-v0",
    "h1hand-sit_simple-v0",
    "h1hand-sit_hard-v0",
    "h1hand-balance_simple-v0",
    "h1hand-balance_hard-v0",
    "h1hand-stair-v0",
    "h1hand-slide-v0",
    "h1hand-pole-v0",
    "h1hand-insert_normal-v0",
    "h1hand-insert_small-v0",
)
HUMANOIDBENCH_MAIN_ENVS = tuple(
    f"{PREFIX}/{env_id}" for env_id in HUMANOIDBENCH_MAIN_NATIVE_ENVS
)

_done = False


def is_humanoidbench_id(name: str) -> bool:
    return isinstance(name, str) and name.startswith(f"{PREFIX}/")


class _IsolatedNumpyRandom(gymnasium.Wrapper):
    """Give each env its own stream for HumanoidBench's legacy np.random calls."""
    def __init__(self, env):
        super().__init__(env)
        self._legacy_rng = np.random.RandomState()

    def _call(self, fn, *args, **kwargs):
        process_state = np.random.get_state()
        np.random.set_state(self._legacy_rng.get_state())
        try:
            return fn(*args, **kwargs)
        finally:
            self._legacy_rng.set_state(np.random.get_state())
            np.random.set_state(process_state)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._legacy_rng.seed(seed)
        return self._call(self.env.reset, seed=seed, options=options)

    def step(self, action):
        return self._call(self.env.step, action)


def _make_humanoidbench(native_id: str, **kwargs):
    kwargs.setdefault("render_mode", None)
    env = gymnasium.make(native_id, disable_env_checker=True, **kwargs)
    return _IsolatedNumpyRandom(env)


def register_humanoidbench_envs(name: str | None = None) -> int:
    """Lazily import HumanoidBench and register ``humanoidbench/<native-id>`` aliases."""
    global _done
    if name is not None and not is_humanoidbench_id(name):
        return 0
    if not _done:
        try:
            importlib.import_module("humanoid_bench")
        except ImportError as exc:
            if name is None:
                return 0
            raise ImportError(
                "HumanoidBench is required for humanoidbench/* environments; "
                "install carlosferrazza/humanoid-bench in the launch venv"
            ) from exc
        for native_id, spec in list(gymnasium.registry.items()):
            if not str(spec.entry_point).startswith("humanoid_bench.env:"):
                continue
            alias = f"{PREFIX}/{native_id}"
            if alias not in gymnasium.registry:
                gymnasium.register(
                    id=alias,
                    entry_point=_make_humanoidbench,
                    max_episode_steps=spec.max_episode_steps,
                    kwargs={"native_id": native_id},
                )
        _done = True
    if name is not None and name not in gymnasium.registry:
        native_id = name.removeprefix(f"{PREFIX}/")
        raise ValueError(
            f"Installed HumanoidBench does not register {native_id!r}; "
            "check the package version or benchmark task list"
        )
    return sum(1 for env_id in gymnasium.registry if is_humanoidbench_id(env_id))
