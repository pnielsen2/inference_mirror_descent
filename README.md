# Efficient Online Reinforcement Learning for Diffusion Policies

This is the official implementation of

## Installation

```bash
# Create environment
conda create -n inf-md python=3.10 numpy tqdm tensorboardX matplotlib scikit-learn black snakeviz ipykernel setproctitle numba
conda activate inf-md

# One of: Install jax WITH CUDA 
pip install --upgrade "jax[cuda12]==0.4.27" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install package
pip install -r requirements.txt
pip install -e .
```



## Run
```bash
# Run one experiment
XLA_FLAGS='--xla_gpu_deterministic_ops=true' CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=.1 python scripts/train_mujoco.py --alg sdac --seed 100
```

baseline (Plain DPMD):
```bash
python scripts/train_mujoco.py --alg dpmd --env HalfCheetah-v4 --beta_schedule_type cosine --beta_schedule_scale 1 --dpmd_long_lr_schedule 
```
which achieves an average of 11267.6 (SE: 153) over 5 runs.

There appears to be a bug in the code relating to q-normalization, but fixing it by adding the flag --fix_q_norm_bug: 

```bash
python scripts/train_mujoco.py --alg dpmd --env HalfCheetah-v4 --beta_schedule_type cosine --beta_schedule_scale 1 --dpmd_long_lr_schedule --fix_q_norm_bug
```

doesn't seem to help, achieving a mean of 10263 (SE: 281.2) over 5 runs.

```bash
python scripts/train_mujoco.py --alg dpmd --env HalfCheetah-v4 --beta_schedule_type cosine --beta_schedule_scale 1 --fix_q_norm_bug
```

same with this, which gets 10146.4 (SE: 244.46) over 5 runs.

This project attempts to introduce three things: 
1. inference time tilt of policies. For this modification alone, the best performing command so far is
```bash
python scripts/train_mujoco.py --alg dpmd --env HalfCheetah-v4 --dpmd_constant_weight --tfg_lambda 16.0 --num_particles 1 --mala_steps 2 --q_critic_agg mean --beta_schedule_type cosine --beta_schedule_scale 1 --dpmd_no_entropy_tuning --buffer_size 200000
```
which achieves approx 10442 (SE: 233.44) (5 runs in progress):

We also find that denoising many particles and behavior cloning the best/soft best under the Q function dominates performance:

```bash
python scripts/train_mujoco.py --alg dpmd --env HalfCheetah-v4 --dpmd_constant_weight --num_particles 128 --q_critic_agg mean --particle_selection_lambda 64 --beta_schedule_type cosine --beta_schedule_scale 1
```

This command achieves an average of 10941 (SE: 183.78) without any Q-guidance at inference time or Q-based reweighting during training. This is purely the result of behavior cloning the best/soft best over many denoised action particles under the Q function.


2. Use of transition models and reward models to guide tilts, rather than Q

There are two methods employed in this project: a deterministic dynamics model and a diffusion based dynamics model. For the deterministic model, the best performance is achieved by this command:

```bash
python scripts/train_mujoco.py --alg dpmd_mb_pc --env HalfCheetah-v4 --tfg_lambda 2 --mala_steps 1 --pc_use_value --update_per_iteration 4 --lr_dyn 1e-4 --lr_value 3e-3 --buffer_size 200000 --pc_deterministic_dyn --lr_policy 1e-4 --use_validation --validation_ratio 0.0 --diffusion_steps 20 --beta_schedule_type cosine --beta_schedule_scale 1
```

with an average episode return of around 8962 (SE: 481.1) over 5 runs.

For the diffusion based model, the best performance is achieved by this command:

```bash
python scripts/train_mujoco.py --alg dpmd_mb_pc --env HalfCheetah-v4 --tfg_lambda 2 --mala_steps 1 --pc_use_value --sprime_num_particles 32 --update_per_iteration 4 --lr_dyn 3e-3 --lr_value 3e-3 --sprime_refresh_steps 6 --bprop_refresh_steps 6 --buffer_size 200000 --beta_schedule_type cosine --beta_schedule_scale 1
```
with an episode return of 6315 (SE: 462.06) over 5 runs.


3. Planning - denoising a sequence of actions and tilting based on the quality of the entire sequence

The best command here is 

```bash
python scripts/train_mujoco.py --alg dpmd_mb_pc --env HalfCheetah-v4 --tfg_lambda 2 --mala_steps 1 --pc_use_value --sprime_num_particles 32 --update_per_iteration 4 --lr_dyn 1e-4 --lr_value 3e-3 --buffer_size 200000 --pc_deterministic_dyn --lr_policy 1e-4 --use_validation --validation_ratio 0.0 --pc_H_plan 2 --pc_joint_seq --beta_schedule_type cosine --beta_schedule_scale 1
```

Which achieves an average episode return of 7173 (SE: 228.43) over 5 runs.

## Visualize results
```python
from relax.utils.inspect_results import load_results, plot_mean

env_name = 'Ant-v4'

patterns_dict = {
        'sdac': r'sdac.*' # regex expression of saved folders
    }

for key, value in patterns_dict.items():
    print(key)
    _ = load_results(value, env_name, show_df=False)

plot_mean(patterns_dict, env_name)
```

## Ackwonledgement
We developed this repo based on [DACER](https://github.com/happy-yan/DACER-Diffusion-with-Online-RL.git). We thank the authors of DACER for providing high-quality code base.

## Bibtex
If you used this repo in your paper, please considering 
giving us a star 🌟 and citing our related paper.

```bibtex
@article{ma2025soft,
  title={Efficient Online Reinforcement Learning for Diffusion Policy},
  author={Ma, Haitong and Chen, Tianyi and Wang, Kai and Li, Na and Dai, Bo},
  journal={arXiv preprint arXiv:2502.00361},
  year={2025}
}
```


