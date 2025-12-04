from typing import Callable, Tuple
import functools

import jax
import jax.numpy as jnp


PolicyEnergyFn = Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]
DynamicsEnergyFn = Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]
RewardFn = Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]
ValueFn = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]


def get_alpha_sigma(t_scalar: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Interpret t_scalar as the VP cumulative alpha_bar.

    In the Relax PCMD port we construct levels_jax["alpha_bar"] directly from
    GaussianDiffusion.beta_schedule().alphas_cumprod, so the sampler receives
    alpha_bar values in (0, 1]. Using them here ensures that refresh and
    Tweedie-style re-noising use the same (alpha_bar, 1-alpha_bar) pair as
    the training corruption kernel q(x_t | x_0).
    """
    alpha = t_scalar
    sigma = jnp.sqrt(jnp.maximum(jnp.float32(1.0) - alpha, jnp.float32(0.0)))
    return alpha, sigma


def _log_gauss_isotropic(x: jnp.ndarray, mean: jnp.ndarray, sd: jnp.ndarray) -> jnp.ndarray:
    sd = jnp.maximum(sd, jnp.float32(1e-12))
    z = (x - mean) / sd
    D = jnp.asarray(x.shape[-1], dtype=jnp.float32)
    return -jnp.float32(0.5) * (D * jnp.log(jnp.float32(2.0) * jnp.pi) + 2.0 * jnp.log(sd) + jnp.sum(z * z, axis=-1))


def _state_score(dynamics_energy: DynamicsEnergyFn, s0: jnp.ndarray, a: jnp.ndarray, s: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
    return -jax.grad(lambda st: dynamics_energy(s0, a, st, t))(s)


def _action_score(policy_energy: PolicyEnergyFn, s0: jnp.ndarray, a: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
    return -jax.grad(lambda av: policy_energy(s0, av, t))(a)


def refresh_sps_same_t_once(
    key: jax.Array,
    dynamics_energy: DynamicsEnergyFn,
    s0: jnp.ndarray,
    a: jnp.ndarray,
    sps: jnp.ndarray,
    t: jnp.ndarray,
    use_ula_refresh: bool,
    cs: float,
) -> Tuple[jax.Array, jnp.ndarray]:
    alpha, sigma = get_alpha_sigma(t)
    sqrt_oma = jnp.sqrt(jnp.maximum(jnp.float32(1.0) - alpha, jnp.float32(0.0)))

    M = sps.shape[0]
    key, kN = jax.random.split(key)
    z = jax.random.normal(kN, sps.shape, dtype=sps.dtype)

    def one(sp, z_):
        sc = _state_score(dynamics_energy, s0, a, sp, t)
        if use_ula_refresh:
            eta_s = jnp.float32(cs) * (jnp.float32(1.0) - alpha)
            return sp + eta_s * sc + jnp.sqrt(jnp.float32(2.0) * jnp.maximum(eta_s, jnp.float32(1e-12))) * z_
        else:
            return sp + (jnp.float32(1.0) - alpha) * sc + sqrt_oma * z_

    s_new = jax.vmap(one)(sps, z)
    return key, s_new


def refresh_sps_same_t_L(
    key: jax.Array,
    dynamics_energy: DynamicsEnergyFn,
    s0: jnp.ndarray,
    a: jnp.ndarray,
    sps: jnp.ndarray,
    t: jnp.ndarray,
    L: int,
    use_ula_refresh: bool,
    cs: float,
) -> Tuple[jax.Array, jnp.ndarray]:
    def body(carry, _):
        k, cur = carry
        k, k_step = jax.random.split(k)
        k_out, s_out = refresh_sps_same_t_once(k_step, dynamics_energy, s0, a, cur, t, use_ula_refresh, cs)
        return (k_out, s_out), None

    (key_out, sps_out), _ = jax.lax.scan(body, (key, sps), jnp.arange(L))
    return key_out, sps_out


def _accept_sprime(
    key: jax.Array,
    dynamics_energy: DynamicsEnergyFn,
    s0: jnp.ndarray,
    a: jnp.ndarray,
    sps: jnp.ndarray,
    t: jnp.ndarray,
    eta_s: jnp.ndarray,
    mode: str = "none",
) -> Tuple[jax.Array, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    if mode == "none":
        return key, sps, jnp.float32(0.0), jnp.float32(0.0), jnp.float32(0.0)

    def log_pi_s(sp):
        return -dynamics_energy(s0, a, sp, t)

    def grad_log_pi_s(sp):
        return jax.grad(lambda s: log_pi_s(s))(sp)

    key, k_prop, k_u = jax.random.split(key, 3)
    z = jax.random.normal(k_prop, shape=sps.shape, dtype=sps.dtype)
    sd = jnp.sqrt(jnp.float32(2.0) * jnp.maximum(eta_s, jnp.float32(1e-12)))

    if mode == "barker":
        s_prop = sps + sd * z
        logpi_curr = jax.vmap(log_pi_s)(sps)
        logpi_prop = jax.vmap(log_pi_s)(s_prop)
        logits = logpi_prop - logpi_curr
        u = jax.random.uniform(k_u, shape=logits.shape, dtype=logits.dtype)
        acc = jnp.log(u) < -jax.nn.softplus(-logits)
        s_out = jnp.where(acc, s_prop, sps)
        st_ct = jnp.float32(sps.shape[0] * 2)
        rw_ct = jnp.float32(sps.shape[0] * 0)
        acc_rate = jnp.mean(acc.astype(jnp.float32))
        return key, s_out, st_ct, rw_ct, acc_rate

    elif mode == "mala":
        gl_curr = jax.vmap(grad_log_pi_s)(sps)
        mean_f = sps + jnp.maximum(eta_s, jnp.float32(1e-12)) * gl_curr
        s_prop = mean_f + sd * z
        gl_prop = jax.vmap(grad_log_pi_s)(s_prop)
        mean_r = s_prop + jnp.maximum(eta_s, jnp.float32(1e-12)) * gl_prop

        def log_g(x, m, sd_):
            sd_ = jnp.maximum(sd_, jnp.float32(1e-12))
            z_ = (x - m) / sd_
            D = jnp.asarray(x.shape[-1], dtype=jnp.float32)
            return -jnp.float32(0.5) * (D * jnp.log(jnp.float32(2.0) * jnp.pi) + 2.0 * jnp.log(sd_) + jnp.sum(z_ * z_, axis=-1))

        logq_f = log_g(s_prop, mean_f, sd)
        logq_r = log_g(sps, mean_r, sd)
        logpi_curr = jax.vmap(log_pi_s)(sps)
        logpi_prop = jax.vmap(log_pi_s)(s_prop)
        log_alpha = (logpi_prop - logpi_curr) + (logq_r - logq_f)
        key, k_u2 = jax.random.split(key)
        u = jax.random.uniform(k_u2, shape=log_alpha.shape, dtype=log_alpha.dtype)
        acc = jnp.log(u) < jnp.minimum(jnp.float32(0.0), log_alpha)
        s_out = jnp.where(acc, s_prop, sps)
        st_ct = jnp.float32(sps.shape[0] * 2)
        rw_ct = jnp.float32(sps.shape[0] * 0)
        acc_rate = jnp.mean(acc.astype(jnp.float32))
        return key, s_out, st_ct, rw_ct, acc_rate

    else:
        return key, sps, jnp.float32(0.0), jnp.float32(0.0), jnp.float32(0.0)


@functools.partial(
    jax.jit,
    static_argnames=(
        "H",
        "points_per_seed",
        "refresh_L",
        "action_steps_per_level",
        "use_ula_refresh",
        "bprop_refresh",
        "accept_sprime",
        "use_crn",
        "level_offset",
        "policy_energy",
        "dynamics_energy",
        "reward_fn",
        "value_fn",
    ),
)

def jax_run_pc_mala_fac_rr_seq(
    key: jax.Array,
    levels_jax,
    H: int,
    points_per_seed: int,
    refresh_L: int,
    action_steps_per_level: int,
    use_ula_refresh: bool,
    cs: float,
    cs_accept: float,
    bprop_refresh: bool,
    accept_sprime: str,
    adapt_s_accept: bool,
    s_accept_target: float,
    s_accept_lr: float,
    use_crn: bool,
    level_offset: int,
    gamma: float,
    s0: jnp.ndarray,
    a_init: jnp.ndarray,
    policy_energy: PolicyEnergyFn,
    dynamics_energy: DynamicsEnergyFn,
    reward_fn: RewardFn,
    value_fn: ValueFn,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    K = levels_jax["K"]
    idx = jnp.int32(jnp.maximum(0, K - 1 - int(level_offset)))
    alpha = levels_jax["alpha_bar"][idx]
    beta_k = levels_jax["beta_t"][idx]
    eta_a_base = levels_jax["eta_a"][idx]

    s_dim = s0.shape[0]
    a_dim = a_init.shape[-1]

    key, kS = jax.random.split(key)
    sps = jax.random.normal(kS, shape=(H, points_per_seed, s_dim), dtype=jnp.float32)

    a_seq = a_init
    esjd_sum = jnp.float32(0.0)
    acc_sum = jnp.float32(0.0)
    state_evals = jnp.float32(0.0)
    reward_evals = jnp.float32(0.0)
    energy_evals = jnp.float32(0.0)
    action_evals = jnp.float32(0.0)
    log_eta_scale0 = jnp.float32(0.0)

    def pre_refresh_all(key_in, a_seq_in, sps_in):
        def one_step(carry, k_idx):
            key_loc, sps_loc = carry
            a_k = a_seq_in[k_idx]
            key_loc, sps_loc = refresh_sps_same_t_L(
                key_loc, dynamics_energy, s0, a_k, sps_loc, alpha, refresh_L, use_ula_refresh, cs
            )
            return (key_loc, sps_loc), None

        def scan_step(carry, idx_k):
            key_loc, sps_all = carry
            sps_k = sps_all[idx_k]
            (key_loc, sps_k_new), _ = one_step((key_loc, sps_k), idx_k)
            sps_all = sps_all.at[idx_k].set(sps_k_new)
            return (key_loc, sps_all), None

        (key_out, sps_out), _ = jax.lax.scan(scan_step, (key_in, sps_in), jnp.arange(H))
        return key_out, sps_out

    def accept_sprime_all(key_in, a_seq_in, sps_in):
        eta_s_base = jnp.float32(cs_accept) * (jnp.float32(1.0) - alpha)

        def step(carry, idx_k):
            key_loc, sps_all, st_ct, rw_ct, acc_sum = carry
            a_k = a_seq_in[idx_k]
            key_loc, sps_k, st_add, rw_add, acc_rate = _accept_sprime(
                key_loc, dynamics_energy, s0, a_k, sps_all[idx_k], alpha, eta_s_base, accept_sprime
            )
            sps_all = sps_all.at[idx_k].set(sps_k)
            return (key_loc, sps_all, st_ct + st_add, rw_ct + rw_add, acc_sum + acc_rate), None

        (key_out, sps_out, st_tot, rw_tot, acc_sum), _ = jax.lax.scan(
            step,
            (key_in, sps_in, jnp.float32(0.0), jnp.float32(0.0), jnp.float32(0.0)),
            jnp.arange(H),
        )
        acc_mean = acc_sum / jnp.maximum(jnp.float32(H), jnp.float32(1.0))
        return key_out, sps_out, st_tot, rw_tot, acc_mean

    key, sps = pre_refresh_all(key, a_seq, sps)
    state_evals += jnp.float32(H * points_per_seed * refresh_L)

    def mala_block(step_carry, _):
        key, a_seq, sps, esjd_sum, state_evals, reward_evals, energy_evals, action_evals, acc_sum, log_eta_scale = step_carry
        key, k_a_noise, k_crn_base, k_rev_base = jax.random.split(key, 4)
        eta_a = jnp.clip(jnp.exp(log_eta_scale) * eta_a_base, jnp.float32(1e-8), jnp.float32(0.5))

        def drift_single(idx_k):
            a_k = a_seq[idx_k]
            sps_k = sps[idx_k]
            disc = jnp.float32(gamma) ** jnp.asarray(idx_k, dtype=jnp.float32)
            sa_k = _action_score(policy_energy, s0, a_k, alpha)
            if bprop_refresh:
                def Qhat(a_in):
                    k_use = k_crn_base if use_crn else jax.random.fold_in(k_crn_base, jnp.int32(1000) + idx_k)
                    _, sps_ref = refresh_sps_same_t_L(k_use, dynamics_energy, s0, a_in, sps_k, alpha, refresh_L, use_ula_refresh, cs)
                    vals = jax.vmap(lambda sp: disc * reward_fn(s0, a_in, sp, alpha) + (disc * jnp.float32(gamma)) * value_fn(sp, alpha))(sps_ref)
                    return jnp.mean(vals)

                _, gQ_k = jax.jvp(Qhat, (a_k,), (jnp.ones_like(a_k),))
                den_ct = jnp.float32(points_per_seed * refresh_L)
                rw_ct = jnp.float32(points_per_seed)
            else:
                def f_val_and_grad(a_in, sp):
                    return jax.value_and_grad(lambda av: disc * reward_fn(s0, av, sp, alpha) + (disc * jnp.float32(gamma)) * value_fn(sp, alpha))(a_in)

                f_vals, gQ_vec = jax.vmap(lambda sp: f_val_and_grad(a_k, sp))(sps_k)
                gQ_k = jnp.mean(gQ_vec, axis=0)
                den_ct = jnp.float32(points_per_seed)
                rw_ct = jnp.float32(points_per_seed)
            drift_k = sa_k + jnp.float32(beta_k) * gQ_k
            return drift_k, rw_ct, den_ct

        drifts, rw_cts, den_cts = jax.vmap(drift_single)(jnp.arange(H))
        reward_evals = reward_evals + jnp.sum(rw_cts)
        energy_evals = energy_evals + jnp.sum(den_cts)

        z = jax.random.normal(k_a_noise, shape=a_seq.shape, dtype=a_seq.dtype)
        mean_prop = a_seq + eta_a * drifts
        sd_prop = jnp.sqrt(jnp.float32(2.0) * jnp.maximum(eta_a, jnp.float32(1e-12)))
        a_prop = mean_prop + sd_prop * z

        def reverse_single(idx_k, key_in):
            key_in, k_step = jax.random.split(key_in)
            a_kp = a_prop[idx_k]
            sps_k = sps[idx_k]
            k_rr = k_step if use_crn else jax.random.fold_in(k_step, jnp.int32(2000) + idx_k)
            key_in, sps_rev_k = refresh_sps_same_t_once(k_rr, dynamics_energy, s0, a_kp, sps_k, alpha, use_ula_refresh, cs)
            sa_p_k = _action_score(policy_energy, s0, a_kp, alpha)
            if bprop_refresh:
                def Qhat_p(a_in):
                    k_use = k_step if use_crn else jax.random.fold_in(k_step, jnp.int32(3000) + idx_k)
                    _, sps_ref = refresh_sps_same_t_L(k_use, dynamics_energy, s0, a_in, sps_k, alpha, refresh_L, use_ula_refresh, cs)
                    vals = jax.vmap(lambda sp: reward_fn(s0, a_in, sp, alpha) + jnp.float32(gamma) * value_fn(sp, alpha))(sps_ref)
                    return jnp.mean(vals)

                _, gQ_p_k = jax.jvp(Qhat_p, (a_kp,), (jnp.ones_like(a_kp),))
                rw_ct2 = jnp.float32(points_per_seed)
                den_ct2 = jnp.float32(points_per_seed * refresh_L)
            else:
                def f_vg(a_in, sp):
                    return jax.value_and_grad(
                        lambda av: reward_fn(s0, av, sp, alpha) + jnp.float32(gamma) * value_fn(sp, alpha)
                    )(a_in)

                # Gradients w.r.t. action for each state sample
                f_vals_p, gQ_vec_p = jax.vmap(lambda sp: f_vg(a_kp, sp))(sps_rev_k)
                # Use simple average over action gradients; avoid mixing in state-score baseline,
                # which would have incompatible dimensionality when state_dim != act_dim.
                gQ_p_k = jnp.mean(gQ_vec_p, axis=0)
                rw_ct2 = jnp.float32(points_per_seed)
                den_ct2 = jnp.float32(points_per_seed)
            drift_prop_k = sa_p_k + jnp.float32(beta_k) * gQ_p_k
            return key_in, sps_rev_k, drift_prop_k, rw_ct2, den_ct2

        def rev_scan(carry, idx_k):
            key_loc = carry
            key_loc, sps_rev_k, drift_prop_k, rw_ct2, den_ct2 = reverse_single(idx_k, key_loc)
            return key_loc, (sps_rev_k, drift_prop_k, rw_ct2, den_ct2)

        key, rev_pack = jax.lax.scan(rev_scan, key, jnp.arange(H))
        sps_rev_all, drift_prop_all, rw_ct2_all, den_ct2_all = rev_pack
        reward_evals = reward_evals + jnp.sum(rw_ct2_all)
        energy_evals = energy_evals + jnp.sum(den_ct2_all)

        logq_f = jnp.sum(_log_gauss_isotropic(a_prop, mean_prop, sd_prop))
        mean_rev = a_prop + eta_a * drift_prop_all
        logq_r = jnp.sum(_log_gauss_isotropic(a_seq, mean_rev, sd_prop))

        s_fwd = sps[:, 0, :]
        s_rev = sps_rev_all[:, 0, :]

        def logpi_step(a_k, s_k, idx_k):
            disc = jnp.float32(gamma) ** jnp.asarray(idx_k, dtype=jnp.float32)
            base = -dynamics_energy(s0, a_k, s_k, alpha)
            tilt = jnp.float32(beta_k) * (disc * reward_fn(s0, a_k, s_k, alpha) + (disc * jnp.float32(gamma)) * value_fn(s_k, alpha))
            return base + tilt

        idxs = jnp.arange(H)
        logpi_curr = jnp.sum(jax.vmap(logpi_step)(a_seq, s_fwd, idxs))
        logpi_prop = jnp.sum(jax.vmap(logpi_step)(a_prop, s_rev, idxs))
        state_evals = state_evals + jnp.float32(2.0 * H)
        reward_evals = reward_evals + jnp.float32(2.0 * H)

        log_alpha = (logpi_prop - logpi_curr) + (logq_r - logq_f)
        key, k_u = jax.random.split(key)
        u = jax.random.uniform(k_u, ())
        accept = jnp.log(u) < jnp.minimum(jnp.float32(0.0), log_alpha)

        a_new = jnp.where(accept, a_prop, a_seq)
        esjd_sum = esjd_sum + jnp.sum((a_new - a_seq) * (a_new - a_seq)) * accept.astype(jnp.float32)
        acc_sum = acc_sum + accept.astype(jnp.float32)

        sps_accept = jnp.where(accept, sps.at[:, 0, :].set(s_rev), sps)
        key, sps_new = pre_refresh_all(key, a_new, sps_accept)
        state_evals = state_evals + jnp.float32(H * points_per_seed * refresh_L)

        target = jnp.float32(0.574)
        adapt_rate = jnp.float32(0.05)
        log_eta_scale = log_eta_scale + adapt_rate * (accept.astype(jnp.float32) - target)

        return (
            key,
            a_new,
            sps_new,
            esjd_sum,
            state_evals,
            reward_evals,
            energy_evals,
            action_evals,
            acc_sum,
            log_eta_scale,
        ), None

    (key, a_seq, sps, esjd_sum, state_evals, reward_evals, energy_evals, action_evals, acc_sum, _), _ = jax.lax.scan(
        mala_block,
        (key, a_seq, sps, esjd_sum, state_evals, reward_evals, energy_evals, action_evals, acc_sum, log_eta_scale0),
        jnp.arange(action_steps_per_level),
    )

    return (
        a_seq,
        sps,
        state_evals,
        reward_evals,
        energy_evals,
        jnp.maximum(esjd_sum, jnp.float32(0.0)),
        jnp.maximum(acc_sum, jnp.float32(0.0)),
    )
