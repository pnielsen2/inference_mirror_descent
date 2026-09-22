"""Cost of a wrong advantage-normalization scalar, in KL.

Tilted policy with normalizer sigma_hat:   p_hat(a|o) ∝ pi(a|o) exp( (beta/sigma_hat) Q(o,a) )
Ideal (correct normalizer sigma_Q(o)):     p_*  (a|o) ∝ pi(a|o) exp( (beta/sigma_Q  ) Q(o,a) )

Both are members of the SAME natural exponential family in eta = beta/sigma_hat,
base measure pi, sufficient statistic Q, log-partition psi(eta)=log E_pi[e^{eta Q}].
So the KL is exactly a Bregman divergence of the CGF:

  KL(p_eta1 || p_eta2) = psi(eta2) - psi(eta1) - (eta2-eta1) psi'(eta1)

Claims:
 1. that Bregman identity, on an arbitrary non-Gaussian Q;
 2. Gaussian Q  =>  psi is quadratic  =>  KL = (beta^2/2) (sigma_Q/sigma_hat - 1)^2 EXACTLY;
 3. the cost is ASYMMETRIC in log(sigma_Q/sigma_hat): over-tilting (sigma_hat too
    small) blows up like e^{2u}, under-tilting saturates at beta^2/2 = KL(pi||p_*);
 4. the global sigma_hat minimising E_o[KL] is the CONTRAHARMONIC mean
    E[sigma_Q^2]/E[sigma_Q], with residual floor (beta^2/2) CV^2/(1+CV^2);
 5. the RMS  sqrt(E[sigma_Q^2])  (what --batch_advantage_normalization uses) is
    too small by sqrt(1+CV^2) -> systematically over-tilts;
 6. finite K: E[s_K^2]=sigma^2 (Bessel) but E[s_K]=c4(K) sigma, so the two
    accumulators need the c4 correction, and both are fine at K=2.
"""
import numpy as np
from scipy.special import gammaln
from scipy import optimize

rng = np.random.default_rng(0)


def c4(K):
    nu = K - 1
    return np.sqrt(2.0 / nu) * np.exp(gammaln(K / 2) - gammaln(nu / 2))


def banner(t):
    print("\n" + "=" * 86 + f"\n{t}\n" + "=" * 86)


# ---------------------------------------------------------------- 1. Bregman
banner("1. KL between two tilts = Bregman divergence of the CGF (arbitrary Q)")
a = np.linspace(-6, 6, 20001)
pi = np.exp(-0.5 * a**2)
pi /= pi.sum()
Q = 0.7 * a + 0.4 * np.tanh(2 * a) - 0.15 * a**2          # deliberately non-Gaussian


def tilt(eta):
    w = pi * np.exp(eta * Q)
    return w / w.sum()


def psi(eta):
    return np.log(np.sum(pi * np.exp(eta * Q)))


def kl(p, q):
    m = p > 1e-300
    return np.sum(p[m] * np.log(p[m] / q[m]))


print(f"{'eta1':>7} {'eta2':>7} {'KL direct':>12} {'Bregman':>12} {'rel err':>10}")
for e1, e2 in [(0.3, 0.9), (1.2, 0.5), (-0.4, 0.8), (2.0, 1.9)]:
    p1, p2 = tilt(e1), tilt(e2)
    dpsi = np.sum(p1 * Q)                                   # psi'(eta1) = E_{p1}[Q]
    breg = psi(e2) - psi(e1) - (e2 - e1) * dpsi
    d = kl(p1, p2)
    print(f"{e1:>7.2f} {e2:>7.2f} {d:>12.6f} {breg:>12.6f} {abs(breg/d-1):>10.2e}")
print("""  Exact for any Q. Second order: KL ~ (1/2) Var_{p1}(Q) (eta2-eta1)^2.""")

# ------------------------------------------------------- 2+3. Gaussian, exact
banner("2+3. Gaussian Q: KL = (beta^2/2)(sigma_Q/sigma_hat - 1)^2, exactly & asymmetric")
# pi = N(0,1), Q(a) = g*a  =>  Var_pi(Q) = g^2 = sigma_Q^2, tilt_eta = N(eta g, 1)
g = 1.7
sigma_Q = g
print(f"{'beta':>5} {'sigma_hat':>10} {'u=log(s/sh)':>12} {'KL exact':>11} "
      f"{'(b^2/2)(r-1)^2':>15} {'err':>9}")
for beta in (0.5, 1.0):
    for sh in (0.5 * g, 0.8 * g, g, 1.25 * g, 2.0 * g, 1e6 * g):
        eta_s, eta_h = beta / sigma_Q, beta / sh
        # KL(N(eta_h g,1) || N(eta_s g,1)) = 0.5 (g(eta_h-eta_s))^2
        exact = 0.5 * (g * (eta_h - eta_s)) ** 2
        form = 0.5 * beta**2 * (sigma_Q / sh - 1) ** 2
        print(f"{beta:>5} {sh:>10.3f} {np.log(sigma_Q/sh):>12.3f} {exact:>11.6f} "
              f"{form:>15.6f} {abs(form-exact):>9.1e}")
print(f"""  Note the last row: sigma_hat -> infinity gives KL -> beta^2/2 = {0.5*1.0**2:.3f},
  which is exactly KL(pi || p_*). UNDER-tilting is bounded -- the worst you can do
  is fall back to the untilted policy. OVER-tilting is unbounded:""")
print(f"  {'u':>7} {'cost (r-1)^2':>14}")
for u in (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0):
    print(f"  {u:>7.1f} {(np.exp(u)-1)**2:>14.4f}")
print("""  So the loss geometry is NOT the symmetric cosh of the s_hat problem: it is
  (e^u - 1)^2, flat to the left and exponential to the right. Err large.""")

# ------------------------------------------------- 4+5. optimal global scalar
banner("4+5. Optimal single scalar = CONTRAHARMONIC mean E[sig^2]/E[sig]")
print(f"{'CV':>6} | {'argmin (MC)':>12} {'E[s^2]/E[s]':>12} | "
      f"{'floor/(b^2/2)':>14} {'CV^2/(1+CV^2)':>14} | {'RMS':>8} {'CH/RMS':>7} "
      f"{'sqrt(1+CV^2)':>13}")
for cv in (0.2, 0.5, 1.0):
    k = 1.0 / cv**2                       # gamma shape -> CV = 1/sqrt(k)
    sig = rng.gamma(k, 1.0 / k, 400000)   # mean 1, coefficient of variation cv
    risk = lambda sh: np.mean((sig / sh - 1) ** 2)
    opt = optimize.minimize_scalar(risk, bounds=(0.05, 20), method="bounded").x
    ch = (sig**2).mean() / sig.mean()
    rms = np.sqrt((sig**2).mean())
    print(f"{cv:>6} | {opt:>12.5f} {ch:>12.5f} | {risk(ch):>14.5f} "
          f"{cv**2/(1+cv**2):>14.5f} | {rms:>8.4f} {ch/rms:>7.4f} "
          f"{np.sqrt(1+cv**2):>13.4f}")
print("""  The contraharmonic mean is the exact minimiser, the floor is CV^2/(1+CV^2)
  times beta^2/2, and the RMS -- which is what mean_s Var_K(Q) then sqrt gives --
  is smaller by exactly sqrt(1+CV^2). Since sigma_hat too small = over-tilt, the
  current estimator errs on the EXPENSIVE side. Excess KL from using RMS:""")
for cv in (0.2, 0.5, 1.0):
    k = 1.0 / cv**2
    sig = rng.gamma(k, 1.0 / k, 400000)
    ch, rms = (sig**2).mean() / sig.mean(), np.sqrt((sig**2).mean())
    am = sig.mean()
    r = lambda sh: np.mean((sig / sh - 1) ** 2)
    print(f"    CV={cv}:  E[KL]/(b^2/2)  CH={r(ch):.4f}  RMS={r(rms):.4f} "
          f"(+{100*(r(rms)/r(ch)-1):.1f}%)  AM={r(am):.4f} "
          f"(+{100*(r(am)/r(ch)-1):.1f}%)")

# ------------------------------------------------------------- 6. finite K
banner("6. Finite-K accumulators: EMA(s_K) needs /c4(K), EMA(s_K^2) is unbiased")
print(f"{'K':>3} {'c4(K)':>8} | {'E[s_K]/sig':>11} {'E[s_K^2]/sig^2':>15} | "
      f"{'CH est (naive)':>15} {'CH est (c4-corr)':>17} {'true CH':>9}")
cv = 0.6
k = 1.0 / cv**2
S = rng.gamma(k, 1.0 / k, 200000)                 # per-state sigma_Q, mean 1
true_ch = (S**2).mean() / S.mean()
for K in (2, 3, 5, 10):
    x = rng.normal(0.0, 1.0, (K, S.size)) * S     # K draws of Q per state
    sK = x.std(axis=0, ddof=1)
    naive = (sK**2).mean() / sK.mean()
    corr = c4(K) * (sK**2).mean() / sK.mean()
    print(f"{K:>3} {c4(K):>8.4f} | {(sK/S).mean():>11.4f} "
          f"{((sK**2)/S**2).mean():>15.4f} | {naive:>15.4f} {corr:>17.4f} "
          f"{true_ch:>9.4f}")
print("""  Both moments are available at K=2 (no negative moment to diverge, unlike the
  s_hat problem). Skipping c4 inflates sigma_hat by 1/c4 = 1.25 at K=2 -- that is
  a 25% UNDER-tilt, the cheap direction, but free to fix.""")
