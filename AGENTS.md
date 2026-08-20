# Agent notes

## Environment

Use the shared venv: `source ~/.venvs/general/bin/activate`.

`v3` MuJoCo envs need `mujoco-py`/`mujoco210` at runtime. When running training
outside the sbatch wrapper, replicate what `scripts/launch.py` sets:

```bash
export LD_LIBRARY_PATH="$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia:/lib64:$LD_LIBRARY_PATH"
export CPATH="$HOME/.local/glew/glew-2.1.0/include:$CPATH"
export WANDB_DIR=<writable dir>   # default /tmp/wandb may be another user's and unwritable
export JAX_PLATFORMS=cpu          # login nodes have no GPU
```

`--alg mgmd` requires **exactly two** of `--alpha / --beta / --T / --eta`.

## Disk

`/n/home09/$USER` (repo + `logs/`) runs near 100% full. Write run artifacts,
snapshots, and anything bulky to `/n/netscratch/kdbrantley_lab/Lab/$USER/`
(`relax.utils.fs.WANDB_OFFLINE_BASE` points there).

## Reading sweep results: prefer local files over the wandb API

Curves are mirrored to disk by `SampleMetricsRecorder` *specifically* so
analysis does not depend on wandb (rate limits / dropped points):

```
<codebase>/logs/<env>/<exp_dir.name>/episode_returns.csv   # seed,step,episode_return/<env>
```

where `seed` is the vmap slot index. Per-slot config comes from
`$WANDB_OFFLINE_BASE/sweep_<N>/job_*/wandb/offline-run-*/files/config.yaml`, and
`<codebase>` is recovered from that run dir's `wandb-metadata.json` `program`
field. `scripts/build_sweep_local_metrics.py` has reusable helpers
(`_cfg_value`, `_read_run_basename`, `_codebase_dir_from_metadata`).

This path is ~30s for a 240-run sweep. Going through `wandb.Api` instead means
one `api.run()` per run (stub `.config` is empty for this project, so a full
fetch is required just to read hyperparameters) plus a paginated
`scan_history()` over ~1M steps each — minutes to tens of minutes. Only use the
API for sweeps whose offline dirs are gone.

Note `api.runs(..., filters={"config.<key>": v})` *does* filter server-side
correctly even though the returned stubs have empty `.config`.

### Evaluation-episode curve

`--eval_every` (default 1000000 = the default `--total_step`, so one end-of-run
score and no curve unless you lower it; 0 disables) runs separate best-of-N
evaluation episodes in their own envs (`relax/trainer/evaluation.py`). They are
mirrored to

```
<codebase>/logs/<env>/<exp_dir.name>/eval_episode_returns.csv
# seed_index,step,eval_best_of_n_actions,episode_index,episode_return,episode_length
```

next to the rollout `episode_returns.csv`, and logged to wandb as
`eval/episode_return_{mean,std,min,max}` / `eval/episode_length_mean` per run.
Evaluation writes nothing to the train state and has its own envs and PRNG
stream, so a run's training curve is bit-identical with it on or off (verified by
diffing `episode_returns.csv` between `--eval_every 0` and `--eval_every 300`
runs); only wall clock changes. `build_sweep_local_metrics.py` does **not** read
the eval CSV yet — it aggregates `episode_returns.csv` only.

### Heatmap / fraction-of-baseline pipeline

`build_sweep_local_metrics.py --sweep-id N` writes
`scripts/topsis_out/sweep_N/local/{per_slot_metrics,per_config_env_metrics,config_index,overall_scores}.csv`,
which every heatmap script reads. A hyperparameter only reaches
`config_index.csv` if it is listed in `HP_KEYS` or `EXTRA_KEYS` in that script —
add it there and rebuild the sweep, otherwise it cannot be used as a plot axis.

`plot_distillation_sweeps.py` (sweeps 166, 174, and 175+177 pooled) skips that
pipeline and recomputes the tail metric from the episode-return mirrors itself,
so it also renders sweeps that are still running; add a `SweepSpec` to plot a
new one, keyed by the tuple of sweep ids that share the design (sweeps in one
key are pooled into single figures, valid only when they cover disjoint cells).

Two things to keep in mind when reading these figures:
- **Fraction of baseline is capped at 1.0.** `compute_topsis.log_score` does
  `np.log2(np.clip(ratio, 0.01, 1.0))`, so a config that beats the LSAC
  baseline is indistinguishable from one that merely matches it. On easy envs
  this saturates (sweep 125 @80: Swimmer 14/20 cells at 1.00, HalfCheetah 5/20),
  so "Overall" is effectively driven by the hard envs.
- Slots below `COMPLETION_FRAC` (0.95) of their `total_step` are dropped, and
  `fraction_of_baseline` is NaN unless a config covers all 6 envs.

## launch.py

`_snapshot_codebase()` captures the codebase with `git stash create`, which
**ignores untracked files**. `git add` any new file before launching or the
submitted jobs will run from a snapshot missing it and crash on import.
