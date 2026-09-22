# Could a normalizer have replaced the per-env eta of row 2?

Row 2 of `adaptive_oracle.md`: base config `buffer_size=400000 gamma=0.99 rollout_alpha=1`, with **eta 64 for Ant/Hopper/Humanoid and 128 for HalfCheetah/Swimmer/Walker2d** (oracle 0.9904 vs 0.9693 fixed).

These runs have `T=0`, so `alpha=1` and `beta=eta`, and the energy is `U = E_theta - (eta / N) Q`: the effective mirror-descent step on the raw Q is **`eta / N`**, `N` = guidance-Q divisor. They ran `--ema_advantage_normalization`, so `N = S =` `Critic/adv_norm_running_std`. The step the row prescribes is therefore `Z = eta_pres / S`, and swapping in divisor `N'` with ONE global eta `g` reproduces it per env iff

```
g_req(env) = Z(env) * N'(env)   is constant across envs
```

Aggregation: geometric mean over logged steps (800k-1M window) then over seeds (54 runs: 6 seeds/env at eta=64 incl. the sweep-233 pool, 3 at eta=128).

`RMSE` is the root-mean-square of `log2(g_req / eta*)` -- the per-env error in eta, **in octaves (doublings)**, if you shipped the single best `eta*`. **The benchmark to beat is the status quo, whose `g_req` IS 64/128, giving exactly 0.500 octaves.** A normalizer only earns its place by scoring BELOW that.

The miss splits in two. The row asks for exactly one thing -- the eta-64 envs at half the eta of the eta-128 envs -- so `group ratio` = geomean(`g_req`) over the eta-64 envs divided by that over the eta-128 envs measures whether the divisor delivers it: **1.00 absorbs the whole split, 0.50 absorbs none of it** (the status quo, by construction). `within` is the extra per-env scatter the divisor introduces on top, which is pure cost.

With only three envs per group one env can manufacture the whole group signal, so `group ratio` is also quoted as a **leave-one-env-out range**; a range straddling 0.50 means nothing robust is absorbed.

| divisor `N'` | best single eta* | RMSE (octaves) | boot 5-95% | group ratio | LOEO range | within (oct) | spread max/min | beats 64/128? |
|---|---|---|---|---|---|---|---|---|
| --ema_advantage_normalization (in use) (reference) | 90.51 | **0.500** | 0.500-0.500 | 0.50 | 0.50-0.50 | 0.000 | 2.00x | no |
| Q magnitude (no flag) | 578.9 | **0.886** | 0.781-1.024 | 0.57 | 0.37-0.73 | 0.785 | 4.18x | no |
| --q_loss_normalization | 2.165 | **1.258** | 1.154-1.369 | 1.13 | 0.53-1.82 | 1.255 | 11.79x | no |
| critic-ensemble sd (no flag) | 0.614 | **1.363** | 1.218-1.513 | 1.61 | 0.70-2.89 | 1.319 | 17.01x | no |
| no normalizer | 2.014 | **1.754** | 1.709-1.818 | 0.25 | 0.13-0.64 | 1.435 | 40.28x | no |

## Per-env detail

`S` is the divisor in use, `Z = eta_pres/S` the prescribed effective step, `N'` the candidate divisor, `g_req = Z N'` the global eta that env would need. Read the `g_req` columns for constancy, not size.

| quantity | Ant | HalfCheetah | Hopper | Humanoid | Swimmer | Walker2d |
|---|---|---|---|---|---|---|
| **eta the row picks** | **64** | **128** | **64** | **64** | **128** | **128** |
| `S` (divisor in use) | 46.32 | 117.13 | 59.28 | 95.30 | 4.73 | 56.78 |
| `Z` = prescribed step | 1.382 | 1.093 | 1.08 | 0.6716 | 27.05 | 2.254 |
| `N'`: --ema_advantage_normalization (in use) | 46.32 | 117.1 | 59.28 | 95.3 | 4.731 | 56.78 |
| -> `g_req` | **64** | **128** | **64** | **64** | **128** | **128** |
| `N'`: no normalizer | 1 | 1 | 1 | 1 | 1 | 1 |
| -> `g_req` | **1.382** | **1.093** | **1.08** | **0.6716** | **27.05** | **2.254** |
| `N'`: --q_loss_normalization | 3.392 | 3.342 | 1.071 | 3.369 | 0.01633 | 2.31 |
| -> `g_req` | **4.687** | **3.652** | **1.156** | **2.263** | **0.4417** | **5.209** |
| `N'`: Q magnitude (no flag) | 604.5 | 1160 | 301.4 | 451.9 | 11.95 | 493.6 |
| -> `g_req` | **835.3** | **1268** | **325.4** | **303.5** | **323.3** | **1113** |
| `N'`: critic-ensemble sd (no flag) | 0.8419 | 0.7313 | 0.3822 | 1.468 | 0.003372 | 0.6885 |
| -> `g_req` | **1.163** | **0.7991** | **0.4126** | **0.9862** | **0.09124** | **1.552** |

## `--q_loss_normalization` eta, restated as an equivalent current eta

Both schemes apply a step `eta / N`, so running `--q_loss_normalization` at `eta_q` gives the same step on the raw Q as `eta_S = eta_q * S / sqrt(EMA(Q_loss))` would give under the divisor now in use. The row wants `eta_S` to come out at the **target** row.

| env | target `eta_S` | `S` | `sqrt(EMA(Q_loss))` | `S/sqrt(EMA(Q_loss))` | eta_q=1 | eta_q=2 | eta_q=4 | eta_q=8 | `eta_q` to hit target |
|---|---|---|---|---|---|---|---|---|---|
| Ant | **64** | 46.32 | 3.392 | 13.7 | 13.7 | 27.3 | 54.6 | 109.2 | **4.69** |
| HalfCheetah | **128** | 117.13 | 3.342 | 35.0 | 35.0 | 70.1 | 140.2 | 280.4 | **3.65** |
| Hopper | **64** | 59.28 | 1.071 | 55.4 | 55.4 | 110.7 | 221.4 | 442.9 | **1.16** |
| Humanoid | **64** | 95.30 | 3.369 | 28.3 | 28.3 | 56.6 | 113.1 | 226.3 | **2.26** |
| Swimmer | **128** | 4.73 | 0.01633 | 289.8 | 289.8 | 579.6 | 1159.2 | 2318.5 | **0.44** |
| Walker2d | **128** | 56.78 | 2.31 | 24.6 | 24.6 | 49.1 | 98.3 | 196.6 | **5.21** |
| **RMSE vs target (octaves)** | - | - | - | - | 1.68 | 1.26 | 1.54 | 2.27 | - |

The last column is the `eta_q` each env would need on its own; a normalizer works only if that column is flat, and it spans 0.44 to 5.21. No single `eta_q` in the grid lands near the targets: Hopper and Swimmer pull one way (55 and 290 per unit eta_q) while Ant and Walker2d pull the other (14 and 25).


## Caveats

- **First order.** Every `N'` is measured on trajectories that divided by `S`. A run that had actually divided by `N'` would have grown a different Q scale, so this says what these runs exhibited, not what such a run would converge to.
- **`--batch_advantage_normalization` and `--ema_within_advantage_normalization` cannot be evaluated here at all.** Both divide by a WITHIN-state spread over the K denoised actions, and this row ran `--num_denoised_actions 1`, so that variance does not exist in these runs (`ddof=1` on one sample) and nothing proportional to it was logged. `Critic/E(Var({Q_i})_env)` is the sd across ensemble MEMBERS at one action, a different quantity, and it underflows to 0 on Swimmer.
- `--kl_budget` sets `beta = sqrt(2 delta / M)`. If `M` is the pooled `Var(Q)` it is algebraically the same rule as the divisor already in use, so it inherits the same residual; only a within-state `M` would differ, and that is the K>=2 case above.
- `--reward_scale` cannot substitute for eta at all here: it scales Q and hence `S` by the same factor, which cancels exactly in `eta / S`.
- Trap for anyone rerunning this: `config.yaml` writes `beta` BEFORE the per-slot `--hp_pack_inline` override, so the eta=128 slots record a stale `beta=64`. The logged `Global_EMAs/beta` is the runtime value and equals `eta` (128) on every one of those runs, as `T=0` requires. This analysis keys off `eta`, never the config's `beta`.


---

# Could a normalizer have replaced the per-env eta of row 2?

Row 2 of `adaptive_oracle.md`: base config `buffer_size=400000 gamma=0.99 rollout_alpha=1`, with **eta 64 for Ant/Hopper/Humanoid and 128 for HalfCheetah/Swimmer/Walker2d** (oracle 0.9904 vs 0.9693 fixed).

These runs have `T=0`, so `alpha=1` and `beta=eta`, and the energy is `U = E_theta - (eta / N) Q`: the effective mirror-descent step on the raw Q is **`eta / N`**, `N` = guidance-Q divisor. They ran `--ema_advantage_normalization`, so `N = S =` `Critic/adv_norm_running_std`. The step the row prescribes is therefore `Z = eta_pres / S`, and swapping in divisor `N'` with ONE global eta `g` reproduces it per env iff

```
g_req(env) = Z(env) * N'(env)   is constant across envs
```

Aggregation: geometric mean over logged steps (all of training) then over seeds (54 runs: 6 seeds/env at eta=64 incl. the sweep-233 pool, 3 at eta=128).

`RMSE` is the root-mean-square of `log2(g_req / eta*)` -- the per-env error in eta, **in octaves (doublings)**, if you shipped the single best `eta*`. **The benchmark to beat is the status quo, whose `g_req` IS 64/128, giving exactly 0.500 octaves.** A normalizer only earns its place by scoring BELOW that.

The miss splits in two. The row asks for exactly one thing -- the eta-64 envs at half the eta of the eta-128 envs -- so `group ratio` = geomean(`g_req`) over the eta-64 envs divided by that over the eta-128 envs measures whether the divisor delivers it: **1.00 absorbs the whole split, 0.50 absorbs none of it** (the status quo, by construction). `within` is the extra per-env scatter the divisor introduces on top, which is pure cost.

With only three envs per group one env can manufacture the whole group signal, so `group ratio` is also quoted as a **leave-one-env-out range**; a range straddling 0.50 means nothing robust is absorbed.

| divisor `N'` | best single eta* | RMSE (octaves) | boot 5-95% | group ratio | LOEO range | within (oct) | spread max/min | beats 64/128? |
|---|---|---|---|---|---|---|---|---|
| --ema_advantage_normalization (in use) (reference) | 90.51 | **0.500** | 0.500-0.500 | 0.50 | 0.50-0.50 | 0.000 | 2.00x | no |
| Q magnitude (no flag) | 343.8 | **0.915** | 0.846-0.998 | 0.54 | 0.32-0.73 | 0.799 | 5.09x | no |
| --q_loss_normalization | 2.724 | **0.986** | 0.941-1.029 | 1.06 | 0.56-1.79 | 0.985 | 10.06x | no |
| critic-ensemble sd (no flag) | 0.8044 | **1.123** | 1.068-1.182 | 1.33 | 0.69-2.43 | 1.104 | 12.30x | no |
| no normalizer | 1.816 | **1.653** | 1.631-1.679 | 0.25 | 0.13-0.59 | 1.309 | 30.78x | no |

## Per-env detail

`S` is the divisor in use, `Z = eta_pres/S` the prescribed effective step, `N'` the candidate divisor, `g_req = Z N'` the global eta that env would need. Read the `g_req` columns for constancy, not size.

| quantity | Ant | HalfCheetah | Hopper | Humanoid | Swimmer | Walker2d |
|---|---|---|---|---|---|---|
| **eta the row picks** | **64** | **128** | **64** | **64** | **128** | **128** |
| `S` (divisor in use) | 63.28 | 119.35 | 60.18 | 94.09 | 6.11 | 58.68 |
| `Z` = prescribed step | 1.011 | 1.072 | 1.063 | 0.6802 | 20.94 | 2.181 |
| `N'`: --ema_advantage_normalization (in use) | 63.28 | 119.4 | 60.18 | 94.09 | 6.114 | 58.68 |
| -> `g_req` | **64** | **128** | **64** | **64** | **128** | **128** |
| `N'`: no normalizer | 1 | 1 | 1 | 1 | 1 | 1 |
| -> `g_req` | **1.011** | **1.072** | **1.063** | **0.6802** | **20.94** | **2.181** |
| `N'`: --q_loss_normalization | 3.379 | 3.036 | 2.228 | 3.995 | 0.03597 | 3.474 |
| -> `g_req` | **3.418** | **3.256** | **2.369** | **2.718** | **0.753** | **7.578** |
| `N'`: Q magnitude (no flag) | 339 | 799.8 | 258.9 | 250.7 | 8.042 | 325.3 |
| -> `g_req` | **342.9** | **857.8** | **275.4** | **170.6** | **168.4** | **709.5** |
| `N'`: critic-ensemble sd (no flag) | 0.9817 | 0.7217 | 0.558 | 1.989 | 0.009023 | 1.065 |
| -> `g_req` | **0.9929** | **0.774** | **0.5935** | **1.353** | **0.1889** | **2.323** |

## `--q_loss_normalization` eta, restated as an equivalent current eta

Both schemes apply a step `eta / N`, so running `--q_loss_normalization` at `eta_q` gives the same step on the raw Q as `eta_S = eta_q * S / sqrt(EMA(Q_loss))` would give under the divisor now in use. The row wants `eta_S` to come out at the **target** row.

| env | target `eta_S` | `S` | `sqrt(EMA(Q_loss))` | `S/sqrt(EMA(Q_loss))` | eta_q=1 | eta_q=2 | eta_q=4 | eta_q=8 | `eta_q` to hit target |
|---|---|---|---|---|---|---|---|---|---|
| Ant | **64** | 63.28 | 3.379 | 18.7 | 18.7 | 37.4 | 74.9 | 149.8 | **3.42** |
| HalfCheetah | **128** | 119.35 | 3.036 | 39.3 | 39.3 | 78.6 | 157.2 | 314.5 | **3.26** |
| Hopper | **64** | 60.18 | 2.228 | 27.0 | 27.0 | 54.0 | 108.0 | 216.1 | **2.37** |
| Humanoid | **64** | 94.09 | 3.995 | 23.5 | 23.5 | 47.1 | 94.2 | 188.4 | **2.72** |
| Swimmer | **128** | 6.11 | 0.03597 | 170.0 | 170.0 | 340.0 | 679.9 | 1359.8 | **0.75** |
| Walker2d | **128** | 58.68 | 3.474 | 16.9 | 16.9 | 33.8 | 67.6 | 135.1 | **7.58** |
| **RMSE vs target (octaves)** | - | - | - | - | 1.75 | 1.08 | 1.13 | 1.84 | - |

The last column is the `eta_q` each env would need on its own; a normalizer works only if that column is flat, and it spans 0.75 to 7.58. No single `eta_q` in the grid lands near the targets: Hopper and Swimmer pull one way (27 and 170 per unit eta_q) while Ant and Walker2d pull the other (19 and 17).


## Caveats

- **First order.** Every `N'` is measured on trajectories that divided by `S`. A run that had actually divided by `N'` would have grown a different Q scale, so this says what these runs exhibited, not what such a run would converge to.
- **`--batch_advantage_normalization` and `--ema_within_advantage_normalization` cannot be evaluated here at all.** Both divide by a WITHIN-state spread over the K denoised actions, and this row ran `--num_denoised_actions 1`, so that variance does not exist in these runs (`ddof=1` on one sample) and nothing proportional to it was logged. `Critic/E(Var({Q_i})_env)` is the sd across ensemble MEMBERS at one action, a different quantity, and it underflows to 0 on Swimmer.
- `--kl_budget` sets `beta = sqrt(2 delta / M)`. If `M` is the pooled `Var(Q)` it is algebraically the same rule as the divisor already in use, so it inherits the same residual; only a within-state `M` would differ, and that is the K>=2 case above.
- `--reward_scale` cannot substitute for eta at all here: it scales Q and hence `S` by the same factor, which cancels exactly in `eta / S`.
- Trap for anyone rerunning this: `config.yaml` writes `beta` BEFORE the per-slot `--hp_pack_inline` override, so the eta=128 slots record a stale `beta=64`. The logged `Global_EMAs/beta` is the runtime value and equals `eta` (128) on every one of those runs, as `T=0` requires. This analysis keys off `eta`, never the config's `beta`.
