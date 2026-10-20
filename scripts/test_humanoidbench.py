#!/usr/bin/env python3
"""Regression checks for HumanoidBench registration and benchmark expansion."""
import importlib.util
from pathlib import Path
from unittest import mock

import gymnasium
import numpy as np

import relax.humanoidbench as hb
import relax.env.vector.process3 as process3
from relax.benchmarks import benchmark_envs


def load_launcher():
    path = Path(__file__).with_name("launch.py")
    spec = importlib.util.spec_from_file_location("launch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_rng_isolation():
    class LegacyEnv(gymnasium.Env):
        def reset(self, *, seed=None, options=None):
            return np.array([np.random.random()]), {}

        def step(self, action):
            return np.array([np.random.random()]), 0.0, False, False, {}

    def standalone(seed):
        env = hb._IsolatedNumpyRandom(LegacyEnv())
        values = [env.reset(seed=seed)[0][0]]
        values.extend(env.step(None)[0][0] for _ in range(3))
        return values

    first = hb._IsolatedNumpyRandom(LegacyEnv())
    second = hb._IsolatedNumpyRandom(LegacyEnv())
    first_values = [first.reset(seed=3)[0][0]]
    second_values = [second.reset(seed=7)[0][0]]
    for _ in range(3):
        first_values.append(first.step(None)[0][0])
        second_values.append(second.step(None)[0][0])
    assert np.array_equal(first_values, standalone(3))
    assert np.array_equal(second_values, standalone(7))


def check_registration():
    native_id = "hb-test-native-v0"
    alias = f"{hb.PREFIX}/{native_id}"
    old_done = hb._done
    gymnasium.register(
        id=native_id,
        entry_point="humanoid_bench.env:HumanoidEnv",
        max_episode_steps=321,
    )
    try:
        hb._done = False
        with mock.patch.object(hb.importlib, "import_module", return_value=object()):
            assert hb.register_humanoidbench_envs(alias) >= 1
        assert alias in gymnasium.registry
        assert gymnasium.spec(alias).max_episode_steps == 321
    finally:
        gymnasium.registry.pop(alias, None)
        gymnasium.registry.pop(native_id, None)
        hb._done = old_done


def check_worker_failure_status():
    env = object.__new__(process3.ProcessVectorEnv)
    env.signal_pointer = 0
    env.num_workers = 2
    env.status = np.array([0, 1], np.uint8)
    env.workers = [mock.Mock(), mock.Mock()]
    env.workers[1].poll.return_value = 17
    try:
        with mock.patch.object(process3, "futex_server_wait"):
            try:
                env._wait()
            except RuntimeError as exc:
                assert "1 (exit=17)" in str(exc)
            else:
                raise AssertionError("worker failure status did not raise")
    finally:
        env.workers = None


def main():
    envs = hb.HUMANOIDBENCH_MAIN_ENVS
    assert len(envs) == len(set(envs)) == 31
    assert all(hb.is_humanoidbench_id(env_id) for env_id in envs)
    assert benchmark_envs("hb") == benchmark_envs("humanoidbench-main") == envs

    launcher = load_launcher()
    ablations, oat = launcher.expand_benchmark_ablations("humanoidbench", None, None)
    assert ablations == [["env", *envs]] and oat == []
    try:
        launcher.expand_benchmark_ablations("hb", [["env", "x"]], None)
    except ValueError:
        pass
    else:
        raise AssertionError("benchmark plus explicit env ablation was accepted")

    packs = launcher.generate_packs(
        "python scripts/train_mujoco.py --alg mgmd --T 0 --eta 64",
        None, ablations, oat, 2, seeds_per_config=2,
    )
    assert len(packs) == 31 and sum(pack["pack_size"] for pack in packs) == 62
    assert all(
        launcher.venv_for_command(pack["cmd"])
        == launcher.HUMANOIDBENCH_VENV.resolve()
        for pack in packs
    )

    check_registration()
    check_rng_isolation()
    check_worker_failure_status()
    print("HumanoidBench registration and benchmark expansion checks passed")


if __name__ == "__main__":
    main()
