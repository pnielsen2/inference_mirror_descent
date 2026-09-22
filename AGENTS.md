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

## Snapshot action-selection studies

`evaluate_snapshot_action_selection.py` replays a saved MGMD state under two
protocols: `tilted_eta` (N=1, sweeping the sampling eta) and `base_best_of_n`
(eta = beta = 0, i.e. the untilted base policy, best of N by
`--q-agg-sample`). Snapshot dirs, and the manifest naming the 24 configs
(6 envs x diffusion_steps 40/80 x training eta 32/64) with their 1M checkpoints,
live under `$SNAPSHOT_ROOT = /n/netscratch/kdbrantley_lab/Lab/$USER/diagnostic_snapshots`:

- `action_selection_eval_100ep_eta32_eta64_1m/` — `manifest.txt` plus
  `{slug}_traineta{32,64}_diff{40,80}_{mean,min}[_step{S}].csv`, N/eta <= 256.
  The 1M files carry no `_step` suffix.
- `action_selection_eval_100ep_bestofn_min_extended/` — `..._bestmin_step{S}.csv`,
  `base_best_of_n` at N = 512, 1024 (2048/4096 for a few configs).
- `action_selection_eval_100ep_final_k_ddpm/` — `..._k{K}_step{S}.csv`.
- `action_selection_eval_100ep_ddim_bestmin/` — `..._ddimmin_step{S}.csv`,
  protocol `ddim_best_of_n`, N = 1..1024.
- `action_selection_eval_100ep_dipo_eval/` — `..._dipo_step{S}.csv`, protocol
  `dipo_ddpm_mean`, one row group at N = 1 (the protocol has no N).

All of them are 100 episodes at `--eval-seed 0`, and env seeds come from
`default_rng(eval_seed).integers(...)` sized by the episode count, so any two
files with the same `--episodes` are paired episode-for-episode. All 24 configs
x 10 snapshot steps (100k..1M) are complete for `base_best_of_n` at N <= 1024.

**Old snapshots need `_repair_legacy_state`.** These states were pickled before
`HParams` gained `mala_target_acceptance_rate` / `mala_step_size_max`, and a
NamedTuple unpickles positionally, so those fields arrive as the class-default
*scalars* (0.574 / 0.5 — deliberately the values those runs used) instead of
per-slot arrays, which every sampler's `vmap` over seeds rejects. The loader
broadcasts rank-0 leaves onto the slot axis; with that, the stored `eta=0`
best-of-N returns reproduce bit-for-bit at HEAD (verified per episode by
`scripts/sbatch_ddim_smoke.sh`).

### Sampling the base policy without the MALA chain

`relax/algorithm/base_policy_sampler.py` holds the unguided chains (x_T ~ N(0,I),
`timesteps` transitions, no MCMC, no Q guidance), selected by `transition` and
reachable as `MGMD.get_eval_action_vmap(..., sampler_kind=...)` — analysis only,
training never uses them:

- `"ddim"` — deterministic DDIM, no x0 clip.
- `"ddpm_mean"` — DDPM posterior mean over x0 clipped to
  `model.x_recon_clip_radius`. **This is DIPO's eval sampler**: its
  `Diffusion.sample(..., eval=True)` sets `noise_ratio = 0`, which drops the
  ancestral noise from every `p_sample` and leaves exactly this mean, one action
  per state, critic never consulted (no best-of-N). DIPO evaluates over 10
  episodes; we use 100 so the episodes pair with the best-of-N runs.

Each reproduces `build_mala_sampler` at the matching `denoising_predictor` with
`mala_steps=0`, `beta=0`, `alpha=1` (`scripts/test_base_policy_samplers.py`;
exactly for DDIM, to ~2 float32 ulp for DDPM-mean, which shares the same
`ddpm_mean_from_eps` and differs only in XLA's fma contraction) at ~2.4x lower
cost, because that build still evaluates the guidance gradient at every level
before multiplying it by a traced zero, which XLA cannot fold away.

Runners, both sharding over the manifest and reusing one compiled sampler across
a config's ten steps (state shapes are constant over a run, so a new checkpoint
is a pickle read plus a pointer swap), and both skipping settings already on disk
so a timed-out task resumes on resubmit:

- `evaluate_snapshot_ddim_best_of_n.py` + `sbatch_ddim_best_of_n.sh`
  (`--array=0-95`: task = config x step-shard). Measured on ant/d40 at 100
  episodes: N=1 12s, N=256 24s, N=1024 65s, against 18s / 158s / 595s for the
  eta=0 MALA chain (~9x at N=1024). ~10 min per task.
- `evaluate_snapshot_dipo_eval.py` + `sbatch_dipo_eval.sh` (`--array=0-5`:
  task = config-shard). One setting per snapshot at ~15s, so ~1 GPU-hour total.

`scripts/plot_snapshot_base_sampler_comparison.py` draws all three against each
other, one figure per (snapshot step, training eta); the DIPO line is
N-independent so it spans the axis as a reference level.

Figures land in `figures/action_selection/bestofn_vs_dipo/`, one per (snapshot
step, training eta), 20 in all.

**Keep array tasks short.** This partition's GPUs run ~fully allocated and its
free ones are usually held by the backfill scheduler for a higher-priority
reservation, so only a job that fits in the gap runs: a 24-task `-t 3:00` array
sat at `(Priority)` for 45 min, and the same work as 96 tasks at `-t 0:40`
started within a minute. One task per (config, step) at `-t 0:20` scheduled
best of all — 12 concurrent. `kempner_base` QOS also caps a user at **16 GPUs**
total, so concurrent training runs directly limit how many tasks can run.

Two scheduling notes for retargeting a stalled array:
- `scancel <arrayjob>` on a partly-run array is safe here: `append_results`
  writes a setting's 100 episodes in one call, so a killed task can only lose
  the setting in flight, and the runners skip what is on disk. Recompute the
  remaining `(config, step)` pairs from the CSVs and resubmit them as an
  explicit `--array=<id list>` (with `STEP_SHARDS=10`, id = `config*10 +
  step_index`), which wastes no no-op tasks. Resubmitting does reset queue age.
- `kempner_requeue` covers the *same* H100 nodes but only gets genuinely idle
  capacity — it cannot preempt the priority partitions. With the partition full
  a 44-task requeue array got nothing for an hour while the same tasks on
  `kempner_h100` ran 12 at a time. Not a way around a saturated partition.

## launch.py

`_snapshot_codebase()` captures the codebase with `git stash create`, which
**ignores untracked files**. `git add` any new file before launching or the
submitted jobs will run from a snapshot missing it and crash on import.
