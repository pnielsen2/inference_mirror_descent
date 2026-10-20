# Evaluation-episode tables

One schema for every method's evaluation episodes, so any comparison is a
groupby rather than a per-source special case.

Regenerate everything (rewrites the `*__eval_episodes.csv` files and `index.csv`)
with:

```bash
python scripts/build_baseline_index.py --verify
```

`--verify` re-reads each written file and checks its row count and total return
against the source it came from.

## Layout

```
baselines/
  index.csv                                  inventory: one row per (algorithm, variant, env, eval_method)
  DIPO/DIPO__<env>__eval_episodes.csv         baseline
  DIPO/source/                                upstream export these were built from
  DPMD/DPMD__<env>__eval_episodes.csv         baseline
  DPMD/source/                                upstream export these were built from
  SAC/SAC__<env>__eval_episodes.csv           baseline
  MGMD/MGMD__<env>__eval_episodes.csv         ours, not a baseline (see below)
```

MGMD is our method, not a baseline. It lives here anyway so that every curve in
the comparison figure shares one schema and one loader; move it out if that
becomes confusing.

## Schema

One row per evaluation episode.

| column | meaning |
| --- | --- |
| `algorithm` | `DIPO`, `DPMD`, `SAC`, `MGMD` |
| `variant` | training config within an algorithm (see below) |
| `env` | Gym id, e.g. `Ant-v3` |
| `seed` | the run's own seed as launched |
| `seed_index` | 0-9 within `(algorithm, variant, env, eval_method)` |
| `step` | env step of the evaluation checkpoint |
| `eval_method` | canonical, cross-algorithm action-selection rule |
| `eval_detail` | the algorithm-specific mechanism behind `eval_method` |
| `best_of_n` | candidate actions scored per state (1 when the critic is not consulted) |
| `episode_index` | 0-9 within a checkpoint |
| `episode_return` | undiscounted return |
| `episode_length` | steps survived (empty for DIPO, whose export omitted it) |
| `source_path` | file this row was read from |

### `eval_method`

Only two things are ever done at evaluation time, so there are two canonical
names. `eval_detail` keeps what differs underneath — this matters, because the
same canonical name is not produced the same way by every algorithm.

| `eval_method` | `eval_detail` | meaning |
| --- | --- | --- |
| `best_of_32` | `critic_best_of_32` | DIPO / DPMD: 32 policy samples, critic picks one |
| `best_of_32` | `critic_best_of_32_minQ` | SAC: 32 samples scored by `min(Q1,Q2)`, the value its actor loss maximises |
| `best_of_32` | `tilted_mala_best_of_32_meanQ` | MGMD: tilted MALA sampler, guidance and scoring use the ensemble mean |
| `mean_action` | `ddpm_posterior_mean` | DIPO native: `Diffusion.sample(eval=True)`, i.e. `noise_ratio=0`, so each reverse step returns its posterior mean; critic never consulted |
| `mean_action` | `tanh_gaussian_mean` | SAC: `tanh(mean) * scale + bias`, critic never consulted |

DPMD and MGMD have no `mean_action` curve; nothing was ever run for them.

### `variant`

| algorithm | variant | note |
| --- | --- | --- |
| DIPO | `default` | |
| DPMD | `original` | the canonical DPMD baseline |
| DPMD | `trainN1_no_noise` | ablation; retained here but excluded from the comparison figure |
| SAC | `default` | |
| MGMD | `sweep244_rollout_alpha1` | sweep 244, `rollout_alpha=1`, `eta=64`, `buffer_size=250000` |

## Coverage

Every group is 10 seeds x 100 checkpoints (every 10k env steps) x 10 episodes,
**except** DPMD on `Humanoid-v3`, which ran to 2M steps in 200 checkpoints for
both variants. Anything comparing final performance must therefore clip to a
common horizon; `scripts/plot_eval_method_comparison.py` defaults to
`--max-step 1000000`.

`index.csv` records `n_seeds`, `n_checkpoints`, `episodes_per_checkpoint` and the
step range per group, so this is checkable rather than assumed.

## Upstream sources

| algorithm | origin |
| --- | --- |
| DIPO | `baselines/DIPO/source/` (exported from `/n/home09/pnielsen/DIPO` runs) |
| DPMD | `baselines/DPMD/source/` |
| SAC | `/n/netscratch/kdbrantley_lab/Lab/pnielsen/sac_v3_gpu_best32_mean_baselines/20260922_212513/sac`, SLURM array `47876289`, launched by `scripts/submit_mujoco_v5_standard_baselines.py --env-version 3 --eval-every 10000 --eval-episodes 10 --eval-best-of-n 32 --eval-q-agg min --eval-mean-action` |
| MGMD | sweep 244 offline wandb runs under `/n/holylabs/kdbrantley_lab/Lab/pnielsen/wandb/sweep_244`, eval curves mirrored to `logs/slurm/20260919_115738/codebase/logs/<env>/<run>/eval_episode_returns.csv` |

Note `*.csv` is in `.gitignore`, so these tables are not tracked by git. They are
reproducible from the command at the top of this file as long as the upstream
sources above are intact.
