# Sweep 237+239+240+241 baseline-normalized eval scores

window 800k-1000k env steps; 53/65 configs finished all runs; 20000 seed bootstrap replicates.

denominator per env = max(DPMD, DIPO, SAC): Ant 5280 (DPMD), HalfCheetah 12848 (DPMD), Hopper 3407 (DIPO), Humanoid 5129 (DIPO), Swimmer 147 (DPMD), Walker2d 4617 (DIPO)

| sweep | config | Ant | HalfCheetah | Hopper | Humanoid | Swimmer | Walker2d | min | score | boot 5-95% | best% | seeds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 237 | buffer_size=200000 eta=64 gamma=0.99 lr_policy=0.0003 lr_q=0.00015 policy_noise_samples_per_target=1 rollout_alpha=0.9 | 0.995 | 1.029 | 0.896 | 1.028 | 0.980 | 1.011 | 0.896 | **0.9777** | 0.902-0.996 | 43.3 | 3 |
| 241 | buffer_size=250000 eta=64 gamma=0.99 lr_policy=0.0003 lr_q=0.00015 policy_noise_samples_per_target=1 rollout_alpha=1 | 1.244 | 0.986 | 0.895 | 0.934 | 0.989 | 1.000 | 0.895 | **0.9665** | 0.951-0.976 | 26.2 | 3 |
| 241 | buffer_size=250000 eta=100 gamma=0.99 lr_policy=0.0003 lr_q=0.00015 policy_noise_samples_per_target=1 rollout_alpha=1 | 1.164 | 0.984 | 0.948 | 0.835 | 1.011 | 1.011 | 0.835 | **0.9593** | 0.940-0.971 | 8.9 | 3 |
| 237+241 | buffer_size=400000 eta=64 gamma=0.99 lr_policy=0.0003 lr_q=0.00015 policy_noise_samples_per_target=1 rollout_alpha=1 | 1.206 | 0.948 | 0.873 | 0.999 | 0.985 | 0.909 | 0.873 | **0.9512** | 0.929-0.968 | 4.2 | 9 (pooled) |
| 237 | buffer_size=1000000 eta=128 gamma=0.99 lr_policy=0.0003 lr_q=0.00015 policy_noise_samples_per_target=1 rollout_alpha=0.9 | 1.158 | 0.957 | 0.885 | 0.877 | 0.978 | 0.968 | 0.877 | **0.9432** | 0.920-0.960 | 0.6 | 3 |
| 237 | buffer_size=400000 eta=64 gamma=0.99 lr_policy=0.0003 lr_q=0.00015 policy_noise_samples_per_target=1 rollout_alpha=0.9 | 1.245 | 0.951 | 0.822 | 0.959 | 0.979 | 0.943 | 0.822 | **0.9405** | 0.904-0.970 | 4.4 | 3 |
| 239+240 | buffer_size=400000 eta=64 gamma=0.99 lr_policy=0.0003 lr_q=0.0003 policy_noise_samples_per_target=1 rollout_alpha=1 | 1.172 | 1.018 | 0.898 | 0.878 | 0.956 | 0.904 | 0.878 | **0.9380** | 0.913-0.960 | 0.8 | 6 (pooled) |
| 237 | buffer_size=200000 eta=64 gamma=0.995 lr_policy=0.0003 lr_q=0.00015 policy_noise_samples_per_target=1 rollout_alpha=0.9 | 1.015 | 0.874 | 0.859 | 0.928 | 1.386 | 0.972 | 0.859 | **0.9371** | 0.843-0.964 | 2.3 | 3 |
| 240 | buffer_size=400000 eta=64 gamma=0.99 lr_policy=0.001 lr_q=0.0003 policy_noise_samples_per_target=1 rollout_alpha=1 | 1.100 | 1.017 | 0.969 | 0.796 | 0.950 | 0.919 | 0.796 | **0.9362** | 0.896-0.963 | 1.8 | 3 |
| 241 | buffer_size=300000 eta=100 gamma=0.99 lr_policy=0.0003 lr_q=0.00015 policy_noise_samples_per_target=1 rollout_alpha=1 | 1.261 | 0.984 | 0.820 | 0.845 | 0.974 | 1.003 | 0.820 | **0.9339** | 0.892-0.965 | 2.0 | 3 |
| 241 | buffer_size=400000 eta=100 gamma=0.99 lr_policy=0.0003 lr_q=0.00015 policy_noise_samples_per_target=1 rollout_alpha=1 | 1.181 | 0.999 | 0.811 | 0.852 | 0.986 | 0.971 | 0.811 | **0.9334** | 0.887-0.962 | 1.2 | 3 |
| 237 | buffer_size=400000 eta=64 gamma=0.995 lr_policy=0.0003 lr_q=0.00015 policy_noise_samples_per_target=1 rollout_alpha=1 | 1.145 | 0.857 | 0.835 | 0.998 | 1.218 | 0.908 | 0.835 | **0.9304** | 0.825-0.950 | 0.1 | 3 |
| 237 | buffer_size=400000 eta=128 gamma=0.99 lr_policy=0.0003 lr_q=0.00015 policy_noise_samples_per_target=1 rollout_alpha=0.9 | 1.031 | 0.992 | 0.899 | 0.801 | 0.978 | 0.925 | 0.801 | **0.9296** | 0.892-0.952 | 0.1 | 3 |
| 237 | buffer_size=200000 eta=128 gamma=0.995 lr_policy=0.0003 lr_q=0.00015 policy_noise_samples_per_target=1 rollout_alpha=1 | 0.919 | 0.910 | 0.978 | 0.869 | 1.180 | 0.891 | 0.869 | **0.9267** | 0.819-0.944 | 0.0 | 3 |
| 239 | buffer_size=400000 eta=64 gamma=0.99 lr_policy=0.0006 lr_q=0.0003 policy_noise_samples_per_target=1 rollout_alpha=0.9 | 1.067 | 1.063 | 0.952 | 0.728 | 0.937 | 0.960 | 0.728 | **0.9244** | 0.873-0.952 | 0.1 | 3 |
| 241 | buffer_size=300000 eta=128 gamma=0.99 lr_policy=0.0003 lr_q=0.00015 policy_noise_samples_per_target=1 rollout_alpha=1 | 1.181 | 1.010 | 0.800 | 0.797 | 0.999 | 0.972 | 0.797 | **0.9231** | 0.904-0.936 | 0.0 | 3 |
| 237 | buffer_size=200000 eta=64 gamma=0.99 lr_policy=0.0003 lr_q=0.00015 policy_noise_samples_per_target=1 rollout_alpha=1 | 1.214 | 1.000 | 0.696 | 0.887 | 0.999 | 1.040 | 0.696 | **0.9226** | 0.901-0.938 | 0.0 | 3 |
| 241 | buffer_size=200000 eta=64 gamma=0.99 lr_policy=0.0003 lr_q=0.00015 policy_noise_samples_per_target=1 rollout_alpha=1 | 1.216 | 1.012 | 0.739 | 0.931 | 0.997 | 0.889 | 0.739 | **0.9208** | 0.862-0.965 | 2.1 | 3 |
| 241 | buffer_size=300000 eta=64 gamma=0.99 lr_policy=0.0003 lr_q=0.00015 policy_noise_samples_per_target=1 rollout_alpha=1 | 1.204 | 0.883 | 0.743 | 0.983 | 1.006 | 0.942 | 0.743 | **0.9203** | 0.888-0.947 | 0.1 | 3 |
| 240 | buffer_size=400000 eta=64 gamma=0.99 lr_policy=0.0001 lr_q=0.0003 policy_noise_samples_per_target=4 rollout_alpha=1 | 1.170 | 1.009 | 0.949 | 0.732 | 0.873 | 0.977 | 0.732 | **0.9166** | 0.885-0.938 | 0.0 | 3 |
| 237 | buffer_size=1000000 eta=64 gamma=0.995 lr_policy=0.0003 lr_q=0.00015 policy_noise_samples_per_target=1 rollout_alpha=1 | 0.978 | 0.777 | 0.911 | 0.990 | 2.232 | 0.862 | 0.777 | **0.9160** | 0.904-0.927 | 0.0 | 3 |
| 237+241 | buffer_size=400000 eta=128 gamma=0.99 lr_policy=0.0003 lr_q=0.00015 policy_noise_samples_per_target=1 rollout_alpha=1 | 1.173 | 0.980 | 0.700 | 0.854 | 0.990 | 0.994 | 0.700 | **0.9123** | 0.895-0.924 | 0.0 | 6 (pooled) |
| 239 | buffer_size=400000 eta=64 gamma=0.99 lr_policy=0.0006 lr_q=0.0003 policy_noise_samples_per_target=1 rollout_alpha=1 | 1.202 | 1.005 | 0.959 | 0.839 | 0.951 | 0.743 | 0.743 | **0.9103** | 0.899-0.922 | 0.0 | 3 |
| 237 | buffer_size=200000 eta=128 gamma=0.995 lr_policy=0.0003 lr_q=0.00015 policy_noise_samples_per_target=1 rollout_alpha=0.9 | 0.875 | 0.984 | 0.951 | 0.814 | 0.905 | 0.942 | 0.814 | **0.9102** | 0.768-0.951 | 0.4 | 3 |
| 237 | buffer_size=1000000 eta=128 gamma=0.99 lr_policy=0.0003 lr_q=0.00015 policy_noise_samples_per_target=1 rollout_alpha=1 | 1.096 | 0.942 | 0.698 | 0.916 | 0.979 | 0.928 | 0.698 | **0.9045** | 0.839-0.952 | 0.1 | 3 |
| 239 | buffer_size=400000 eta=64 gamma=0.99 lr_policy=0.0003 lr_q=0.00015 policy_noise_samples_per_target=1 rollout_alpha=1 | 1.203 | 0.932 | 0.709 | 0.976 | 0.944 | 0.892 | 0.709 | **0.9031** | 0.826-0.957 | 0.8 | 3 |
| 237 | buffer_size=400000 eta=128 gamma=0.995 lr_policy=0.0003 lr_q=0.00015 policy_noise_samples_per_target=1 rollout_alpha=1 | 0.997 | 0.925 | 0.875 | 0.823 | 1.446 | 0.817 | 0.817 | **0.9030** | 0.879-0.914 | 0.0 | 3 |
| 239 | buffer_size=400000 eta=64 gamma=0.99 lr_policy=0.0003 lr_q=0.00015 policy_noise_samples_per_target=1 rollout_alpha=0.9 | 1.196 | 0.963 | 0.889 | 0.789 | 0.956 | 0.795 | 0.789 | **0.8948** | 0.875-0.913 | 0.0 | 3 |
| 239 | buffer_size=400000 eta=64 gamma=0.99 lr_policy=0.0006 lr_q=0.00015 policy_noise_samples_per_target=1 rollout_alpha=0.9 | 1.102 | 0.952 | 0.659 | 0.987 | 0.943 | 0.877 | 0.659 | **0.8945** | 0.864-0.921 | 0.0 | 3 |
| 237 | buffer_size=200000 eta=64 gamma=0.995 lr_policy=0.0003 lr_q=0.00015 policy_noise_samples_per_target=1 rollout_alpha=1 | 1.091 | 0.938 | 0.853 | 0.998 | 0.759 | 0.827 | 0.759 | **0.8914** | 0.687-0.951 | 0.3 | 3 |
| 239 | buffer_size=400000 eta=64 gamma=0.99 lr_policy=0.0006 lr_q=0.00015 policy_noise_samples_per_target=1 rollout_alpha=1 | 1.163 | 0.957 | 0.647 | 1.017 | 0.960 | 0.824 | 0.647 | **0.8879** | 0.844-0.922 | 0.0 | 3 |
| 237 | buffer_size=400000 eta=64 gamma=0.995 lr_policy=0.0003 lr_q=0.00015 policy_noise_samples_per_target=1 rollout_alpha=0.9 | 0.793 | 0.940 | 0.848 | 0.956 | 1.197 | 0.804 | 0.793 | **0.8866** | 0.756-0.915 | 0.0 | 3 |
| 237 | buffer_size=400000 eta=128 gamma=0.995 lr_policy=0.0003 lr_q=0.00015 policy_noise_samples_per_target=1 rollout_alpha=0.9 | 0.899 | 0.932 | 0.808 | 0.841 | 0.986 | 0.860 | 0.808 | **0.8858** | 0.785-0.924 | 0.0 | 3 |
| 237 | buffer_size=1000000 eta=64 gamma=0.99 lr_policy=0.0003 lr_q=0.00015 policy_noise_samples_per_target=1 rollout_alpha=1 | 0.985 | 0.840 | 0.911 | 0.952 | 0.951 | 0.697 | 0.697 | **0.8835** | 0.782-0.942 | 0.0 | 3 |
| 237 | buffer_size=1000000 eta=128 gamma=0.995 lr_policy=0.0003 lr_q=0.00015 policy_noise_samples_per_target=1 rollout_alpha=1 | 0.790 | 0.889 | 0.913 | 0.985 | 1.365 | 0.718 | 0.718 | **0.8764** | 0.785-0.910 | 0.0 | 3 |
| 241 | buffer_size=200000 eta=128 gamma=0.99 lr_policy=0.0003 lr_q=0.00015 policy_noise_samples_per_target=1 rollout_alpha=1 | 1.174 | 1.083 | 0.618 | 0.729 | 1.007 | 0.988 | 0.618 | **0.8738** | 0.832-0.902 | 0.0 | 3 |
| 239 | buffer_size=400000 eta=64 gamma=0.99 lr_policy=0.0003 lr_q=0.0003 policy_noise_samples_per_target=1 rollout_alpha=0.9 | 1.185 | 0.997 | 0.847 | 0.679 | 0.942 | 0.803 | 0.679 | **0.8702** | 0.801-0.919 | 0.0 | 3 |
| 240 | buffer_size=400000 eta=64 gamma=0.99 lr_policy=0.0001 lr_q=0.0003 policy_noise_samples_per_target=1 rollout_alpha=1 | 1.105 | 0.861 | 0.978 | 0.836 | 0.784 | 0.772 | 0.772 | **0.8675** | 0.794-0.916 | 0.0 | 3 |
| 237 | buffer_size=1000000 eta=128 gamma=0.995 lr_policy=0.0003 lr_q=0.00015 policy_noise_samples_per_target=1 rollout_alpha=0.9 | 0.964 | 0.891 | 0.790 | 0.960 | 2.400 | 0.646 | 0.646 | **0.8656** | 0.842-0.887 | 0.0 | 3 |
| 241 | buffer_size=250000 eta=128 gamma=0.99 lr_policy=0.0003 lr_q=0.00015 policy_noise_samples_per_target=1 rollout_alpha=1 | 1.168 | 0.901 | 0.747 | 0.665 | 0.985 | 0.953 | 0.665 | **0.8654** | 0.815-0.906 | 0.0 | 3 |
| 237 | buffer_size=200000 eta=128 gamma=0.99 lr_policy=0.0003 lr_q=0.00015 policy_noise_samples_per_target=1 rollout_alpha=1 | 1.167 | 0.968 | 0.718 | 0.779 | 0.882 | 0.802 | 0.718 | **0.8521** | 0.791-0.902 | 0.0 | 3 |
| 241 | buffer_size=200000 eta=100 gamma=0.99 lr_policy=0.0003 lr_q=0.00015 policy_noise_samples_per_target=1 rollout_alpha=1 | 1.269 | 1.030 | 0.554 | 0.757 | 0.905 | 0.998 | 0.554 | **0.8505** | 0.809-0.885 | 0.0 | 3 |
| 240 | buffer_size=400000 eta=64 gamma=0.99 lr_policy=0.0003 lr_q=0.0003 policy_noise_samples_per_target=4 rollout_alpha=1 | 1.183 | 1.037 | 0.776 | 0.623 | 0.939 | 0.831 | 0.623 | **0.8501** | 0.805-0.888 | 0.0 | 3 |
| 237 | buffer_size=1000000 eta=64 gamma=0.995 lr_policy=0.0003 lr_q=0.00015 policy_noise_samples_per_target=1 rollout_alpha=0.9 | 0.830 | 0.687 | 0.922 | 0.988 | 2.433 | 0.720 | 0.687 | **0.8486** | 0.788-0.898 | 0.0 | 3 |
| 237 | buffer_size=200000 eta=128 gamma=0.99 lr_policy=0.0003 lr_q=0.00015 policy_noise_samples_per_target=1 rollout_alpha=0.9 | 1.159 | 0.869 | 0.660 | 0.621 | 1.016 | 0.976 | 0.621 | **0.8387** | 0.807-0.866 | 0.0 | 3 |
| 240 | buffer_size=400000 eta=64 gamma=0.99 lr_policy=0.0001 lr_q=0.001 policy_noise_samples_per_target=1 rollout_alpha=1 | 1.092 | 1.014 | 0.860 | 0.697 | 0.830 | 0.612 | 0.612 | **0.8202** | 0.756-0.871 | 0.0 | 3 |
| 240 | buffer_size=400000 eta=64 gamma=0.99 lr_policy=0.001 lr_q=0.0003 policy_noise_samples_per_target=4 rollout_alpha=1 | 0.782 | 0.910 | 0.692 | 0.833 | 0.961 | 0.753 | 0.692 | **0.8170** | 0.774-0.855 | 0.0 | 3 |
| 240 | buffer_size=400000 eta=64 gamma=0.99 lr_policy=0.0001 lr_q=0.001 policy_noise_samples_per_target=4 rollout_alpha=1 | 0.817 | 0.913 | 0.812 | 0.751 | 0.962 | 0.533 | 0.533 | **0.7845** | 0.720-0.837 | 0.0 | 3 |
| 240 | buffer_size=400000 eta=64 gamma=0.99 lr_policy=0.001 lr_q=0.001 policy_noise_samples_per_target=4 rollout_alpha=1 | 0.792 | 0.968 | 0.755 | 0.611 | 0.992 | 0.625 | 0.611 | **0.7766** | 0.701-0.836 | 0.0 | 3 |
| 237 | buffer_size=1000000 eta=64 gamma=0.99 lr_policy=0.0003 lr_q=0.00015 policy_noise_samples_per_target=1 rollout_alpha=0.9 | 0.487 | 0.773 | 0.773 | 0.873 | 0.973 | 0.814 | 0.487 | **0.7655** | 0.588-0.865 | 0.0 | 3 |
| 240 | buffer_size=400000 eta=64 gamma=0.99 lr_policy=0.0003 lr_q=0.001 policy_noise_samples_per_target=1 rollout_alpha=1 | 0.626 | 0.980 | 0.844 | 0.647 | 0.982 | 0.548 | 0.548 | **0.7514** | 0.674-0.802 | 0.0 | 3 |
| 240 | buffer_size=400000 eta=64 gamma=0.99 lr_policy=0.001 lr_q=0.001 policy_noise_samples_per_target=1 rollout_alpha=1 | 0.979 | 1.004 | 0.507 | 0.424 | 1.013 | 0.415 | 0.415 | **0.6660** | 0.566-0.731 | 0.0 | 3 |
| 240 | buffer_size=400000 eta=64 gamma=0.99 lr_policy=0.0003 lr_q=0.001 policy_noise_samples_per_target=4 rollout_alpha=1 | 0.949 | 0.938 | 0.733 | 0.121 | 0.937 | 0.382 | 0.121 | **0.5521** | 0.433-0.620 | 0.0 | 3 |

Eval cadence differs between these sweeps (sweep 237: every 50k, sweep 239: every 50k, sweep 240: every 50k, sweep 241: every 25k), so a seed's window mean averages 5 or 9 eval points depending on its sweep. Both estimate the same window; the finer one is less noisy. Pooled cells can mix the two.

**2 pair(s) of configs are algorithmically identical but were NOT pooled**, so the same config appears on two rows, one per sweep:

- `sweep237_T=0_buffer_size=200000_eta=128_gamma=0.99_mcmc_proposal_type=euler_maruyama_rollout_alpha=1` vs `sweep241_T=0_buffer_size=200000_eta=128_mcmc_proposal_type=euler_maruyama`: they share seeds (Hopper-v3: [15, 16, 17]), and a seed fixes the init and env stream, so these are the same draw twice, not two draws.
- `sweep237_T=0_buffer_size=200000_eta=64_gamma=0.99_mcmc_proposal_type=euler_maruyama_rollout_alpha=1` vs `sweep241_T=0_buffer_size=200000_eta=64_mcmc_proposal_type=euler_maruyama`: they share seeds (Walker2d-v3: [0, 1, 2]), and a seed fixes the init and env stream, so these are the same draw twice, not two draws.

**12 config(s) are not in this table**: their runs are unfinished in the window, and a mean over part of a seed set is not comparable to a mean over all of it.

Levels that therefore appear NOWHERE above, so no comparison below can see them:

- `policy_noise_samples_per_target`: 16, 64

<details><summary>the unfinished configs</summary>

- 240: buffer_size=400000 eta=64 gamma=0.99 lr_policy=0.0001 lr_q=0.0003 policy_noise_samples_per_target=16 rollout_alpha=1
- 240: buffer_size=400000 eta=64 gamma=0.99 lr_policy=0.0001 lr_q=0.0003 policy_noise_samples_per_target=64 rollout_alpha=1
- 240: buffer_size=400000 eta=64 gamma=0.99 lr_policy=0.0001 lr_q=0.001 policy_noise_samples_per_target=16 rollout_alpha=1
- 240: buffer_size=400000 eta=64 gamma=0.99 lr_policy=0.0001 lr_q=0.001 policy_noise_samples_per_target=64 rollout_alpha=1
- 240: buffer_size=400000 eta=64 gamma=0.99 lr_policy=0.0003 lr_q=0.0003 policy_noise_samples_per_target=16 rollout_alpha=1
- 240: buffer_size=400000 eta=64 gamma=0.99 lr_policy=0.0003 lr_q=0.0003 policy_noise_samples_per_target=64 rollout_alpha=1
- 240: buffer_size=400000 eta=64 gamma=0.99 lr_policy=0.0003 lr_q=0.001 policy_noise_samples_per_target=16 rollout_alpha=1
- 240: buffer_size=400000 eta=64 gamma=0.99 lr_policy=0.0003 lr_q=0.001 policy_noise_samples_per_target=64 rollout_alpha=1
- 240: buffer_size=400000 eta=64 gamma=0.99 lr_policy=0.001 lr_q=0.0003 policy_noise_samples_per_target=16 rollout_alpha=1
- 240: buffer_size=400000 eta=64 gamma=0.99 lr_policy=0.001 lr_q=0.0003 policy_noise_samples_per_target=64 rollout_alpha=1
- 240: buffer_size=400000 eta=64 gamma=0.99 lr_policy=0.001 lr_q=0.001 policy_noise_samples_per_target=16 rollout_alpha=1
- 240: buffer_size=400000 eta=64 gamma=0.99 lr_policy=0.001 lr_q=0.001 policy_noise_samples_per_target=64 rollout_alpha=1

</details>

**The `sweep` column is itself a confound.** These sweeps also differ on knobs neither of them swept, so a row's rank mixes its own axes with these:

- `delay_target_policy_update`: sweep 237 = 2, sweep 239 = 1, sweep 240 = 1, sweep 241 = 2
- `delay_target_q_update`: sweep 237 = 2, sweep 239 = 1, sweep 240 = 1, sweep 241 = 2
- `policy_polyak_tau`: sweep 237 = 0.005, sweep 239 = 0.0025, sweep 240 = 0.0025, sweep 241 = 0.005
- `q_polyak_tau`: sweep 237 = 0.005, sweep 239 = 0.0025, sweep 240 = 0.0025, sweep 241 = 0.005
- `use_target_networks`: sweep 237 = False, sweep 239 = True, sweep 240 = True, sweep 241 = False

`score` = min(frac) when every env's frac > 1 (marked `*`), else geomean(min(frac, 1)).
`best%` = share of bootstrap replicates in which that config had the highest score.
`seeds` marked `(pooled)` merge a config-identical run set from another sweep; see `seed_manifest.csv` for the exact seeds behind every cell.