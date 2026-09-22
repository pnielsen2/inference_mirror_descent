"""Flat-observation gymnasium ids for the DeepMind Control Suite.

Registers ``dmc/<domain>-<task>-v0`` for every dm_control suite task, e.g.
``dmc/humanoid-walk-v0``, ``dmc/humanoid_CMU-run-v0``, ``dmc/dog-run-v0``.

Why this module exists rather than a wrapper at the call site: dm_control
observations are a ``Dict`` of named arrays (humanoid-walk is ``com_velocity``,
``extremities``, ``head_height``, ``joint_angles``, ``torso_vertical``,
``velocity``), which ``ProcessVectorEnv`` rejects -- it requires a 1-D ``Box``.
Every shimmy example fixes that with ``FlattenObservation(gym.make(...))``, but
this codebase has no such call site to wrap: ``worker3.py`` builds its envs in a
fresh subprocess from a bare ``--env`` STRING, and never imports shimmy, so the
``dm_control/`` namespace does not even exist there. Both the flattening and the
shimmy import therefore have to live behind a registered id.

Kept at the top of ``relax`` rather than in ``relax.env`` on purpose: importing
``relax.env.*`` would pull in ``relax.env.vector.process3``, which is itself one
of this module's importers, i.e. a package-init cycle.

Safe to import when dm_control/shimmy are absent -- registration is then a no-op
and the existing ``*-v3`` runs are unaffected.
"""
from __future__ import annotations

import gymnasium
from gymnasium.wrappers import FlattenObservation

PREFIX = "dmc"

# Every dm_control suite task runs exactly 1000 control steps and ends by
# TRUNCATION -- there are no terminal states, because base.Task.get_termination
# returns None (verified on humanoid-walk, humanoid_CMU-run, dog-run, dog-fetch
# and cheetah-run). Declaring it means gymnasium.spec(id).max_episode_steps
# reports the real bound, which is what `EvaluationRunner` reads to size its
# eval loop; the TimeLimit this adds fires on the same step as dm_control's own
# limit, so it changes no behaviour and only makes the spec self-describing.
EPISODE_STEPS = 1000

_done = False


def is_dmc_id(name: str) -> bool:
    return isinstance(name, str) and name.startswith(f"{PREFIX}/")


def _make_flat(domain: str, task: str, **kwargs):
    """Shimmy's own env, with its Dict observation flattened to a 1-D Box.

    Goes through the shimmy-registered id rather than reimplementing its
    ``dm_control.suite.load`` + ``DmControlCompatibilityV0`` call, so this
    tracks whatever shimmy does. ``disable_env_checker`` avoids paying for a
    second per-step passive checker underneath the outer ``make``.
    """
    env = gymnasium.make(f"dm_control/{domain}-{task}-v0",
                         disable_env_checker=True, **kwargs)
    return FlattenObservation(env)


def register_dmc_envs(name: str | None = None) -> int:
    """Register the flat ids; return how many are registered.

    ``name`` is the env about to be built. When it is not a ``dmc/`` id this
    returns immediately WITHOUT importing shimmy, because dm_control is a slow
    import and this runs once per env subprocess -- a v3 run must not pay for it.
    """
    global _done
    if name is not None and not is_dmc_id(name):
        return 0
    if _done:
        return sum(1 for k in gymnasium.registry if k.startswith(f"{PREFIX}/"))
    _done = True
    try:
        import shimmy  # noqa: F401  importing it registers the dm_control/ ids
        from shimmy.utils.envs_configs import DM_CONTROL_SUITE_ENVS
    except ImportError:
        return 0
    n = 0
    for domain, task in DM_CONTROL_SUITE_ENVS:
        env_id = f"{PREFIX}/{domain}-{task}-v0"
        n += 1
        if env_id in gymnasium.registry:
            continue
        gymnasium.register(id=env_id, entry_point=_make_flat,
                           max_episode_steps=EPISODE_STEPS,
                           kwargs={"domain": domain, "task": task})
    return n
