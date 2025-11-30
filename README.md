# Efficient Online Reinforcement Learning for Diffusion Policies

This is the official implementation of

## Installation

```bash
# Create environment
conda create -n relax python=3.9 numpy tqdm tensorboardX matplotlib scikit-learn black snakeviz ipykernel setproctitle numba
conda activate relax

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

This project attempts to introduce three things: 
1. inference time tilt of policies. For this modificaiton alone, the best performing command so far is
```bash
python scripts/train_mujoco.py --alg dpmd --env HalfCheetah-v4 --dpmd_constant_weight -tfg --tfg_lambda 128.0 --num_particles 32 --mala_steps 2
```
which achieves an average of 10425 (SE: 371.966) over 5 runs compared to plain DPMD:
```bash
python scripts/train_mujoco.py --alg dpmd --env HalfCheetah-v4 --num_particles 32
```
which achieves an average of 9748 (SE: 199.7) over 5 runs.

There appears to be a bug in the code relating to q-normalization, but fixing it by adding the flag --fix_q_norm_bug: 

```bash
python scripts/train_mujoco.py --alg dpmd --env HalfCheetah-v4 --num_particles 32 --fix_q_norm_bug
```

doesn't seem to help, achieving a mean of 8995 (SE: 246) over 5 runs.

We also find that denoising many particles and behavior cloning the best/soft best under the Q function dominates performance:

```bash
python scripts/train_mujoco.py --alg dpmd --env HalfCheetah-v4 --dpmd_constant_weight --num_particles 128 --q_critic_agg mean --particle_selection_lambda 64
```

This command achieves and average of 10429 (SE: 157.16) without any Q-guidance at inference time or Q-based reweighting during training. This is purely the result of behavior cloning the best/soft best over many denoised action particles under the Q function.

2. Use of transition models and reward models to guide tilts, rather than Q
3. Planning - denoising a sequence of actions and tilting based on the quality of the entire sequence



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


