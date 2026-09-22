# Sweep 239+240+241 baseline-normalized eval scores

window 800k-1000k env steps; 25/43 configs finished all runs; 20000 seed bootstrap replicates.

denominator per env = max(DPMD, DIPO, SAC): Ant 5280 (DPMD), HalfCheetah 12848 (DPMD), Hopper 3407 (DIPO), Humanoid 5129 (DIPO), Swimmer 147 (DPMD), Walker2d 4617 (DIPO)

| sweep | config | Ant | HalfCheetah | Hopper | Humanoid | Swimmer | Walker2d | min | score | boot 5-95% | best% | seeds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 241 | buffer_size=250000 eta=64 lr_policy=0.0003 lr_q=0.00015 policy_noise_samples_per_target=1 rollout_alpha=1 | 1.244 | 0.986 | 0.895 | 0.934 | 0.989 | 1.000 | 0.895 | **0.9665** | 0.951-0.976 | 63.5 | 3 |
| 241 | buffer_size=250000 eta=100 lr_policy=0.0003 lr_q=0.00015 policy_noise_samples_per_target=1 rollout_alpha=1 | 1.164 | 0.984 | 0.948 | 0.835 | 1.011 | 1.011 | 0.835 | **0.9593** | 0.939-0.972 | 23.2 | 3 |
| 239+240 | buffer_size=400000 eta=64 lr_policy=0.0003 lr_q=0.0003 policy_noise_samples_per_target=1 rollout_alpha=1 | 1.172 | 1.018 | 0.898 | 0.878 | 0.956 | 0.904 | 0.878 | **0.9380** | 0.913-0.960 | 2.3 | 6 (pooled) |
| 240 | buffer_size=400000 eta=64 lr_policy=0.001 lr_q=0.0003 policy_noise_samples_per_target=1 rollout_alpha=1 | 1.100 | 1.017 | 0.969 | 0.796 | 0.950 | 0.919 | 0.796 | **0.9362** | 0.897-0.963 | 3.6 | 3 |
| 239 | buffer_size=400000 eta=64 lr_policy=0.0006 lr_q=0.0003 policy_noise_samples_per_target=1 rollout_alpha=0.9 | 1.067 | 1.063 | 0.952 | 0.728 | 0.937 | 0.960 | 0.728 | **0.9244** | 0.874-0.952 | 0.5 | 3 |
| 241 | buffer_size=200000 eta=64 lr_policy=0.0003 lr_q=0.00015 policy_noise_samples_per_target=1 rollout_alpha=1 | 1.216 | 1.012 | 0.739 | 0.931 | 0.997 | 0.889 | 0.739 | **0.9208** | 0.861-0.965 | 4.7 | 3 |
| 240 | buffer_size=400000 eta=64 lr_policy=0.0001 lr_q=0.0003 policy_noise_samples_per_target=4 rollout_alpha=1 | 1.170 | 1.009 | 0.949 | 0.732 | 0.873 | 0.977 | 0.732 | **0.9166** | 0.885-0.939 | 0.1 | 3 |
| 239 | buffer_size=400000 eta=64 lr_policy=0.0006 lr_q=0.0003 policy_noise_samples_per_target=1 rollout_alpha=1 | 1.202 | 1.005 | 0.959 | 0.839 | 0.951 | 0.743 | 0.743 | **0.9103** | 0.898-0.922 | 0.0 | 3 |
| 239 | buffer_size=400000 eta=64 lr_policy=0.0003 lr_q=0.00015 policy_noise_samples_per_target=1 rollout_alpha=1 | 1.203 | 0.932 | 0.709 | 0.976 | 0.944 | 0.892 | 0.709 | **0.9031** | 0.824-0.957 | 2.1 | 3 |
| 239 | buffer_size=400000 eta=64 lr_policy=0.0003 lr_q=0.00015 policy_noise_samples_per_target=1 rollout_alpha=0.9 | 1.196 | 0.963 | 0.889 | 0.789 | 0.956 | 0.795 | 0.789 | **0.8948** | 0.875-0.913 | 0.0 | 3 |
| 239 | buffer_size=400000 eta=64 lr_policy=0.0006 lr_q=0.00015 policy_noise_samples_per_target=1 rollout_alpha=0.9 | 1.102 | 0.952 | 0.659 | 0.987 | 0.943 | 0.877 | 0.659 | **0.8945** | 0.864-0.921 | 0.0 | 3 |
| 239 | buffer_size=400000 eta=64 lr_policy=0.0006 lr_q=0.00015 policy_noise_samples_per_target=1 rollout_alpha=1 | 1.163 | 0.957 | 0.647 | 1.017 | 0.960 | 0.824 | 0.647 | **0.8879** | 0.844-0.922 | 0.0 | 3 |
| 241 | buffer_size=200000 eta=128 lr_policy=0.0003 lr_q=0.00015 policy_noise_samples_per_target=1 rollout_alpha=1 | 1.174 | 1.083 | 0.618 | 0.729 | 1.007 | 0.988 | 0.618 | **0.8738** | 0.832-0.902 | 0.0 | 3 |
| 239 | buffer_size=400000 eta=64 lr_policy=0.0003 lr_q=0.0003 policy_noise_samples_per_target=1 rollout_alpha=0.9 | 1.185 | 0.997 | 0.847 | 0.679 | 0.942 | 0.803 | 0.679 | **0.8702** | 0.803-0.919 | 0.0 | 3 |
| 240 | buffer_size=400000 eta=64 lr_policy=0.0001 lr_q=0.0003 policy_noise_samples_per_target=1 rollout_alpha=1 | 1.105 | 0.861 | 0.978 | 0.836 | 0.784 | 0.772 | 0.772 | **0.8675** | 0.792-0.916 | 0.0 | 3 |
| 241 | buffer_size=250000 eta=128 lr_policy=0.0003 lr_q=0.00015 policy_noise_samples_per_target=1 rollout_alpha=1 | 1.168 | 0.901 | 0.747 | 0.665 | 0.985 | 0.953 | 0.665 | **0.8654** | 0.817-0.906 | 0.0 | 3 |
| 241 | buffer_size=200000 eta=100 lr_policy=0.0003 lr_q=0.00015 policy_noise_samples_per_target=1 rollout_alpha=1 | 1.269 | 1.030 | 0.554 | 0.757 | 0.905 | 0.998 | 0.554 | **0.8505** | 0.808-0.885 | 0.0 | 3 |
| 240 | buffer_size=400000 eta=64 lr_policy=0.0003 lr_q=0.0003 policy_noise_samples_per_target=4 rollout_alpha=1 | 1.183 | 1.037 | 0.776 | 0.623 | 0.939 | 0.831 | 0.623 | **0.8501** | 0.805-0.889 | 0.0 | 3 |
| 240 | buffer_size=400000 eta=64 lr_policy=0.0001 lr_q=0.001 policy_noise_samples_per_target=1 rollout_alpha=1 | 1.092 | 1.014 | 0.860 | 0.697 | 0.830 | 0.612 | 0.612 | **0.8202** | 0.755-0.871 | 0.0 | 3 |
| 240 | buffer_size=400000 eta=64 lr_policy=0.001 lr_q=0.0003 policy_noise_samples_per_target=4 rollout_alpha=1 | 0.782 | 0.910 | 0.692 | 0.833 | 0.961 | 0.753 | 0.692 | **0.8170** | 0.775-0.855 | 0.0 | 3 |
| 240 | buffer_size=400000 eta=64 lr_policy=0.0001 lr_q=0.001 policy_noise_samples_per_target=4 rollout_alpha=1 | 0.817 | 0.913 | 0.812 | 0.751 | 0.962 | 0.533 | 0.533 | **0.7845** | 0.720-0.838 | 0.0 | 3 |
| 240 | buffer_size=400000 eta=64 lr_policy=0.001 lr_q=0.001 policy_noise_samples_per_target=4 rollout_alpha=1 | 0.792 | 0.968 | 0.755 | 0.611 | 0.992 | 0.625 | 0.611 | **0.7766** | 0.702-0.836 | 0.0 | 3 |
| 240 | buffer_size=400000 eta=64 lr_policy=0.0003 lr_q=0.001 policy_noise_samples_per_target=1 rollout_alpha=1 | 0.626 | 0.980 | 0.844 | 0.647 | 0.982 | 0.548 | 0.548 | **0.7514** | 0.673-0.802 | 0.0 | 3 |
| 240 | buffer_size=400000 eta=64 lr_policy=0.001 lr_q=0.001 policy_noise_samples_per_target=1 rollout_alpha=1 | 0.979 | 1.004 | 0.507 | 0.424 | 1.013 | 0.415 | 0.415 | **0.6660** | 0.565-0.730 | 0.0 | 3 |
| 240 | buffer_size=400000 eta=64 lr_policy=0.0003 lr_q=0.001 policy_noise_samples_per_target=4 rollout_alpha=1 | 0.949 | 0.938 | 0.733 | 0.121 | 0.937 | 0.382 | 0.121 | **0.5521** | 0.433-0.621 | 0.0 | 3 |

**18 config(s) are not in this table**: their runs are unfinished in the window, and a mean over part of a seed set is not comparable to a mean over all of it.

Levels that therefore appear NOWHERE above, so no comparison below can see them:

- `buffer_size`: 300000
- `policy_noise_samples_per_target`: 16, 64

<details><summary>the unfinished configs</summary>

- 240: buffer_size=400000 eta=64 lr_policy=0.0001 lr_q=0.0003 policy_noise_samples_per_target=16 rollout_alpha=1
- 240: buffer_size=400000 eta=64 lr_policy=0.0001 lr_q=0.0003 policy_noise_samples_per_target=64 rollout_alpha=1
- 240: buffer_size=400000 eta=64 lr_policy=0.0001 lr_q=0.001 policy_noise_samples_per_target=16 rollout_alpha=1
- 240: buffer_size=400000 eta=64 lr_policy=0.0001 lr_q=0.001 policy_noise_samples_per_target=64 rollout_alpha=1
- 240: buffer_size=400000 eta=64 lr_policy=0.0003 lr_q=0.0003 policy_noise_samples_per_target=16 rollout_alpha=1
- 240: buffer_size=400000 eta=64 lr_policy=0.0003 lr_q=0.0003 policy_noise_samples_per_target=64 rollout_alpha=1
- 240: buffer_size=400000 eta=64 lr_policy=0.0003 lr_q=0.001 policy_noise_samples_per_target=16 rollout_alpha=1
- 240: buffer_size=400000 eta=64 lr_policy=0.0003 lr_q=0.001 policy_noise_samples_per_target=64 rollout_alpha=1
- 240: buffer_size=400000 eta=64 lr_policy=0.001 lr_q=0.0003 policy_noise_samples_per_target=16 rollout_alpha=1
- 240: buffer_size=400000 eta=64 lr_policy=0.001 lr_q=0.0003 policy_noise_samples_per_target=64 rollout_alpha=1
- 240: buffer_size=400000 eta=64 lr_policy=0.001 lr_q=0.001 policy_noise_samples_per_target=16 rollout_alpha=1
- 240: buffer_size=400000 eta=64 lr_policy=0.001 lr_q=0.001 policy_noise_samples_per_target=64 rollout_alpha=1
- 241: buffer_size=300000 eta=100 lr_policy=0.0003 lr_q=0.00015 policy_noise_samples_per_target=1 rollout_alpha=1
- 241: buffer_size=300000 eta=128 lr_policy=0.0003 lr_q=0.00015 policy_noise_samples_per_target=1 rollout_alpha=1
- 241: buffer_size=300000 eta=64 lr_policy=0.0003 lr_q=0.00015 policy_noise_samples_per_target=1 rollout_alpha=1
- 241: buffer_size=400000 eta=100 lr_policy=0.0003 lr_q=0.00015 policy_noise_samples_per_target=1 rollout_alpha=1
- 241: buffer_size=400000 eta=128 lr_policy=0.0003 lr_q=0.00015 policy_noise_samples_per_target=1 rollout_alpha=1
- 241: buffer_size=400000 eta=64 lr_policy=0.0003 lr_q=0.00015 policy_noise_samples_per_target=1 rollout_alpha=1

</details>

**The `sweep` column is itself a confound.** These sweeps also differ on knobs neither of them swept, so a row's rank mixes its own axes with these:

- `delay_target_policy_update`: sweep 239 = 1, sweep 240 = 1, sweep 241 = 2
- `delay_target_q_update`: sweep 239 = 1, sweep 240 = 1, sweep 241 = 2
- `eval_every`: sweep 239 = 50000, sweep 240 = 50000, sweep 241 = 25000
- `policy_polyak_tau`: sweep 239 = 0.0025, sweep 240 = 0.0025, sweep 241 = 0.005
- `q_polyak_tau`: sweep 239 = 0.0025, sweep 240 = 0.0025, sweep 241 = 0.005
- `use_target_networks`: sweep 239 = True, sweep 240 = True, sweep 241 = False

`score` = min(frac) when every env's frac > 1 (marked `*`), else geomean(min(frac, 1)).
`best%` = share of bootstrap replicates in which that config had the highest score.
`seeds` marked `(pooled)` merge a config-identical run set from another sweep; see `seed_manifest.csv` for the exact seeds behind every cell.