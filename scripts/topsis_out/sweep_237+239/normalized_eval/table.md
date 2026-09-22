# Sweep 237+239 baseline-normalized eval scores

window 800k-1000k env steps; 32/32 configs finished all runs; 20000 seed bootstrap replicates.

denominator per env = max(DPMD, DIPO, SAC): Ant 5280 (DPMD), HalfCheetah 12848 (DPMD), Hopper 3407 (DIPO), Humanoid 5129 (DIPO), Swimmer 147 (DPMD), Walker2d 4617 (DIPO)

| sweep | config | Ant | HalfCheetah | Hopper | Humanoid | Swimmer | Walker2d | min | score | boot 5-95% | best% | seeds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 237 | buffer_size=200000 eta=64 gamma=0.99 rollout_alpha=0.9 | 0.995 | 1.029 | 0.896 | 1.028 | 0.980 | 1.011 | 0.896 | **0.9777** | 0.902-0.996 | 39.7 | 3 |
| 237 | buffer_size=400000 eta=64 gamma=0.99 rollout_alpha=1 | 1.241 | 0.952 | 0.968 | 1.009 | 0.981 | 0.918 | 0.918 | **0.9693** | 0.954-0.983 | 46.9 | 6 (pooled) |
| 239 | lr_policy=0.0003 lr_q=0.0003 rollout_alpha=1 | 1.127 | 1.008 | 0.931 | 0.915 | 0.949 | 0.921 | 0.915 | **0.9519** | 0.924-0.970 | 4.8 | 3 |
| 237 | buffer_size=1000000 eta=128 gamma=0.99 rollout_alpha=0.9 | 1.158 | 0.957 | 0.885 | 0.877 | 0.978 | 0.968 | 0.877 | **0.9432** | 0.920-0.960 | 0.6 | 3 |
| 237 | buffer_size=400000 eta=64 gamma=0.99 rollout_alpha=0.9 | 1.245 | 0.951 | 0.822 | 0.959 | 0.979 | 0.943 | 0.822 | **0.9405** | 0.904-0.970 | 3.8 | 3 |
| 237 | buffer_size=200000 eta=64 gamma=0.995 rollout_alpha=0.9 | 1.015 | 0.874 | 0.859 | 0.928 | 1.386 | 0.972 | 0.859 | **0.9371** | 0.843-0.964 | 2.2 | 3 |
| 237 | buffer_size=400000 eta=64 gamma=0.995 rollout_alpha=1 | 1.145 | 0.857 | 0.835 | 0.998 | 1.218 | 0.908 | 0.835 | **0.9304** | 0.825-0.950 | 0.1 | 3 |
| 237 | buffer_size=400000 eta=128 gamma=0.99 rollout_alpha=0.9 | 1.031 | 0.992 | 0.899 | 0.801 | 0.978 | 0.925 | 0.801 | **0.9296** | 0.892-0.952 | 0.2 | 3 |
| 237 | buffer_size=200000 eta=128 gamma=0.995 rollout_alpha=1 | 0.919 | 0.910 | 0.978 | 0.869 | 1.180 | 0.891 | 0.869 | **0.9267** | 0.819-0.944 | 0.0 | 3 |
| 239 | lr_policy=0.0006 lr_q=0.0003 rollout_alpha=0.9 | 1.067 | 1.063 | 0.952 | 0.728 | 0.937 | 0.960 | 0.728 | **0.9244** | 0.872-0.952 | 0.1 | 3 |
| 237 | buffer_size=200000 eta=64 gamma=0.99 rollout_alpha=1 | 1.214 | 1.000 | 0.696 | 0.887 | 0.999 | 1.040 | 0.696 | **0.9226** | 0.901-0.938 | 0.0 | 3 |
| 237 | buffer_size=1000000 eta=64 gamma=0.995 rollout_alpha=1 | 0.978 | 0.777 | 0.911 | 0.990 | 2.232 | 0.862 | 0.777 | **0.9160** | 0.904-0.927 | 0.0 | 3 |
| 239 | lr_policy=0.0006 lr_q=0.0003 rollout_alpha=1 | 1.202 | 1.005 | 0.959 | 0.839 | 0.951 | 0.743 | 0.743 | **0.9103** | 0.898-0.923 | 0.0 | 3 |
| 237 | buffer_size=200000 eta=128 gamma=0.995 rollout_alpha=0.9 | 0.875 | 0.984 | 0.951 | 0.814 | 0.905 | 0.942 | 0.814 | **0.9102** | 0.768-0.951 | 0.4 | 3 |
| 237 | buffer_size=400000 eta=128 gamma=0.99 rollout_alpha=1 | 1.212 | 0.978 | 0.653 | 0.860 | 0.997 | 1.042 | 0.653 | **0.9047** | 0.886-0.920 | 0.0 | 3 |
| 237 | buffer_size=1000000 eta=128 gamma=0.99 rollout_alpha=1 | 1.096 | 0.942 | 0.698 | 0.916 | 0.979 | 0.928 | 0.698 | **0.9045** | 0.839-0.952 | 0.1 | 3 |
| 239 | lr_policy=0.0003 lr_q=0.00015 rollout_alpha=1 | 1.203 | 0.932 | 0.709 | 0.976 | 0.944 | 0.892 | 0.709 | **0.9031** | 0.827-0.958 | 0.8 | 3 |
| 237 | buffer_size=400000 eta=128 gamma=0.995 rollout_alpha=1 | 0.997 | 0.925 | 0.875 | 0.823 | 1.446 | 0.817 | 0.817 | **0.9030** | 0.879-0.914 | 0.0 | 3 |
| 239 | lr_policy=0.0003 lr_q=0.00015 rollout_alpha=0.9 | 1.196 | 0.963 | 0.889 | 0.789 | 0.956 | 0.795 | 0.789 | **0.8948** | 0.875-0.913 | 0.0 | 3 |
| 239 | lr_policy=0.0006 lr_q=0.00015 rollout_alpha=0.9 | 1.102 | 0.952 | 0.659 | 0.987 | 0.943 | 0.877 | 0.659 | **0.8945** | 0.864-0.921 | 0.0 | 3 |
| 237 | buffer_size=200000 eta=64 gamma=0.995 rollout_alpha=1 | 1.091 | 0.938 | 0.853 | 0.998 | 0.759 | 0.827 | 0.759 | **0.8914** | 0.687-0.951 | 0.3 | 3 |
| 239 | lr_policy=0.0006 lr_q=0.00015 rollout_alpha=1 | 1.163 | 0.957 | 0.647 | 1.017 | 0.960 | 0.824 | 0.647 | **0.8879** | 0.844-0.922 | 0.0 | 3 |
| 237 | buffer_size=400000 eta=64 gamma=0.995 rollout_alpha=0.9 | 0.793 | 0.940 | 0.848 | 0.956 | 1.197 | 0.804 | 0.793 | **0.8866** | 0.756-0.915 | 0.0 | 3 |
| 237 | buffer_size=400000 eta=128 gamma=0.995 rollout_alpha=0.9 | 0.899 | 0.932 | 0.808 | 0.841 | 0.986 | 0.860 | 0.808 | **0.8858** | 0.785-0.924 | 0.0 | 3 |
| 237 | buffer_size=1000000 eta=64 gamma=0.99 rollout_alpha=1 | 0.985 | 0.840 | 0.911 | 0.952 | 0.951 | 0.697 | 0.697 | **0.8835** | 0.782-0.942 | 0.0 | 3 |
| 237 | buffer_size=1000000 eta=128 gamma=0.995 rollout_alpha=1 | 0.790 | 0.889 | 0.913 | 0.985 | 1.365 | 0.718 | 0.718 | **0.8764** | 0.785-0.910 | 0.0 | 3 |
| 239 | lr_policy=0.0003 lr_q=0.0003 rollout_alpha=0.9 | 1.185 | 0.997 | 0.847 | 0.679 | 0.942 | 0.803 | 0.679 | **0.8702** | 0.802-0.919 | 0.0 | 3 |
| 237 | buffer_size=1000000 eta=128 gamma=0.995 rollout_alpha=0.9 | 0.964 | 0.891 | 0.790 | 0.960 | 2.400 | 0.646 | 0.646 | **0.8656** | 0.842-0.887 | 0.0 | 3 |
| 237 | buffer_size=200000 eta=128 gamma=0.99 rollout_alpha=1 | 1.167 | 0.968 | 0.718 | 0.779 | 0.882 | 0.802 | 0.718 | **0.8521** | 0.791-0.902 | 0.0 | 3 |
| 237 | buffer_size=1000000 eta=64 gamma=0.995 rollout_alpha=0.9 | 0.830 | 0.687 | 0.922 | 0.988 | 2.433 | 0.720 | 0.687 | **0.8486** | 0.788-0.898 | 0.0 | 3 |
| 237 | buffer_size=200000 eta=128 gamma=0.99 rollout_alpha=0.9 | 1.159 | 0.869 | 0.660 | 0.621 | 1.016 | 0.976 | 0.621 | **0.8387** | 0.807-0.866 | 0.0 | 3 |
| 237 | buffer_size=1000000 eta=64 gamma=0.99 rollout_alpha=0.9 | 0.487 | 0.773 | 0.773 | 0.873 | 0.973 | 0.814 | 0.487 | **0.7655** | 0.588-0.865 | 0.0 | 3 |

**The `sweep` column is itself a confound.** These sweeps also differ on knobs neither of them swept, so a row's rank mixes its own axes with these:

- `delay_target_policy_update`: sweep 237 = 2, sweep 239 = 1
- `delay_target_q_update`: sweep 237 = 2, sweep 239 = 1
- `policy_polyak_tau`: sweep 237 = 0.005, sweep 239 = 0.0025
- `q_polyak_tau`: sweep 237 = 0.005, sweep 239 = 0.0025
- `use_target_networks`: sweep 237 = (absent), sweep 239 = True
- `use_target_policy_training`: sweep 237 = False, sweep 239 = (absent)
- `use_target_q_sampling_training`: sweep 237 = False, sweep 239 = (absent)

`score` = min(frac) when every env's frac > 1 (marked `*`), else geomean(min(frac, 1)).
`best%` = share of bootstrap replicates in which that config had the highest score.
`seeds` marked `(pooled)` merge a config-identical run set from another sweep; see `seed_manifest.csv` for the exact seeds behind every cell.