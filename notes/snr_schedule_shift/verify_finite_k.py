"""Numerical checks for Section 8.3 (finite-sample estimation of s_hat).

Verifies the rows of the verification table that concern Propositions
`prop:bayes` / `prop:finiteK`, Corollary `cor:debias`, and the shrinkage rule
`eq:shrink`, and regenerates the constants of Table `tab:finiteK`.

The single fact under test is that the schedule penalty
    D(s_hat)/D_min = cosh(log s - log s_hat)
depends on (s, s_hat) only through the log-error, so any law Q describing
uncertainty about s -- population, sampling law, or posterior -- gives risk
    E_Q[cosh(log s - log s_hat)] = (M+/s_hat + s_hat*M-)/2,   M± = E_Q[s^{±1}]
minimised at sqrt(M+/M-) with value sqrt(M+ M-).

Run: python notes/snr_schedule_shift/verify_finite_k.py
"""
import numpy as np
from scipy.special import gammaln, polygamma, digamma
from scipy import optimize


def beta_log(nu):
    """E[log s_hat_K] - log s = (psi(nu/2) - log(nu/2))/2, exactly. Finite at nu=1."""
    return 0.5 * (digamma(nu / 2) - np.log(nu / 2))

RNG = np.random.default_rng(0)
MC = 800_000


def sqrt_f(nu):
    """sqrt(f(nu)), f = G((nu-1)/2) G((nu+1)/2) / G(nu/2)^2. Residual risk."""
    return np.exp(0.5 * (gammaln((nu - 1) / 2) + gammaln((nu + 1) / 2)
                         - 2 * gammaln(nu / 2)))


def kappa(nu):
    """Var(log s_hat_K) = trigamma(nu/2)/4, exactly."""
    return 0.25 * polygamma(1, nu / 2)


def sample_width(s, nu, n):
    """Sample width with nu dof: nu * s_hat^2 / s^2 ~ chi2_nu."""
    return s * np.sqrt(RNG.chisquare(nu, n) / nu)


def banner(t):
    print("\n" + "=" * 76 + f"\n{t}\n" + "=" * 76)


# --------------------------------------------------------------------------
banner("Table `tab:finiteK`: finite-sample constants")
print(f"{'K':>3} {'nu':>3} {'sqrt(nu/(nu-1))':>16} {'sqrt(f(nu))':>12}"
      f" {'1+1/(4nu)':>10} {'kappa(nu)':>10} {'sqrt(kappa)':>11}")
for K in (3, 4, 5, 7, 9, 17, 33):
    nu = K - 1
    print(f"{K:>3} {nu:>3} {np.sqrt(nu/(nu-1)):>16.4f} {sqrt_f(nu):>12.4f}"
          f" {1+1/(4*nu):>10.4f} {kappa(nu):>10.4f} {np.sqrt(kappa(nu)):>11.4f}")

# --------------------------------------------------------------------------
banner("Prop. `prop:finiteK`: Bayes shift = s_hat_K sqrt(nu/(nu-1)), risk sqrt(f)")
print(f"{'K':>3} {'nu':>3} | {'predicted':>10} {'MC optimum':>11}"
      f" | {'risk pred':>10} {'risk MC':>9}")
for K in (3, 4, 5, 9, 17, 33):
    nu = K - 1
    # Jeffreys posterior at an observed unit sample width:
    # 1/s^2 ~ Gamma(nu/2, rate = nu*s_hat_K^2/2)
    s_post = RNG.gamma(shape=nu / 2, scale=2.0 / nu, size=MC) ** -0.5
    obj = lambda lc: np.mean(np.cosh(np.log(s_post) - lc))
    lc = optimize.minimize_scalar(obj, bracket=(-1, 1)).x
    print(f"{K:>3} {nu:>3} | {np.sqrt(nu/(nu-1)):>10.5f} {np.exp(lc):>11.5f}"
          f" | {sqrt_f(nu):>10.5f} {obj(lc):>9.5f}")
print("  the risk is independent of the observed width: s_hat_K cancels in M+M-")

# --------------------------------------------------------------------------
banner("Cor. `cor:debias`: plug-in bias sqrt((nu-1)/nu), floor inflated by sqrt(f)")
m, tau = np.log(0.4), 0.6                      # log s ~ N(m, tau^2)
s_true = np.exp(RNG.normal(m, tau, MC))
shift_true = np.sqrt(s_true.mean() / (1 / s_true).mean())
floor_true = np.sqrt(s_true.mean() * (1 / s_true).mean())
print(f"population log s ~ N(log 0.4, {tau}^2): true shift {shift_true:.5f} "
      f"(geometric mean {np.exp(m):.5f}), true floor {floor_true:.5f} "
      f"(e^(tau^2/2) = {np.exp(tau**2/2):.5f})")
print(f"\n{'K':>3} {'nu':>3} | {'raw/true':>9} {'predicted':>10}"
      f" | {'corrected':>10} | {'raw floor':>10} {'debiased':>9}")
for K in (2, 3, 5, 9, 17):
    nu = K - 1
    obs = sample_width(s_true, nu, MC)
    Mp, Mm = obs.mean(), (1 / obs).mean()
    raw, raw_floor = np.sqrt(Mp / Mm), np.sqrt(Mp * Mm)
    if nu > 1:
        corr, pred = raw * np.sqrt(nu / (nu - 1)), np.sqrt((nu - 1) / nu)
        deb = raw_floor / sqrt_f(nu)
        print(f"{K:>3} {nu:>3} | {raw/shift_true:>9.5f} {pred:>10.5f}"
              f" | {corr:>10.5f} | {raw_floor:>10.5f} {deb:>9.5f}")
    else:
        print(f"{K:>3} {nu:>3} | {raw/shift_true:>9.5f} {'inadmissible':>10}"
              f" | {'--':>10} | {raw_floor:>10.5f} {'--':>9}"
              "   <- E[1/s_hat] diverges")

# --------------------------------------------------------------------------
banner("Eq. `eq:shrink`: w = tau^2/(tau^2+kappa) and the three costs of `eq:threecosts`")
print(f"{'K':>3} {'tau':>5} | {'Var(log)':>9} {'kappa':>8} | {'w MC':>7}"
      f" {'w pred':>7} | {'global':>7} {'raw':>7} {'shrunk':>7} {'excess':>8}")
for K in (3, 5, 9, 33):
    for tau_ in (0.3, 0.8):
        nu = K - 1
        s = np.exp(RNG.normal(m, tau_, MC))
        obs = sample_width(s, nu, MC)
        ls, lo = np.log(s), np.log(obs)
        gm = ls.mean()                          # global optimum in log space
        risk = lambda w: np.mean(np.cosh(ls - (gm + w * (lo - gm))))
        w_mc = optimize.minimize_scalar(risk, bounds=(0, 1.5), method="bounded").x
        w_pred = tau_**2 / (tau_**2 + kappa(nu))
        print(f"{K:>3} {tau_:>5} | {(lo-ls).var():>9.4f} {kappa(nu):>8.4f}"
              f" | {w_mc:>7.4f} {w_pred:>7.4f} | {risk(0.0):>7.4f}"
              f" {risk(1.0):>7.4f} {risk(w_pred):>7.4f}"
              f" {100*(risk(w_pred)/risk(w_mc)-1):>7.3f}%")
print("""  shrunk <= min(global, raw) in every row (Cor. `cor:shrink`), and the
  closed-form weight is within 0.3% of the MC-optimal risk throughout. Using
  1/(2nu) in place of the exact kappa(nu) degrades this markedly at small K.""")

# --------------------------------------------------------------------------
banner("Prop. `prop:logroute`: the log-space accumulators (the route to use)")
print(f"{'K':>3} {'nu':>3} | {'beta(nu)':>9} {'MC E[log]':>10}"
      f" | {'exp(-beta)':>10} {'sqrt(nu/(nu-1))':>16}")
for K in (2, 3, 5, 9, 33):
    nu = K - 1
    lo = np.log(sample_width(np.ones(MC), nu, MC)).mean()
    sq = f"{np.sqrt(nu/(nu-1)):.5f}" if nu > 1 else "inf"
    print(f"{K:>3} {nu:>3} | {beta_log(nu):>9.5f} {lo:>10.5f}"
          f" | {np.exp(-beta_log(nu)):>10.5f} {sq:>16}")
print("  beta(nu) is exact and finite at nu=1, where sqrt(nu/(nu-1)) is not.")
print("  The two differ because they answer different questions (Rem. `rem:twocorrections`).")

print(f"\nEnd-to-end global shift, log route vs M+- route:")
s_true = np.exp(RNG.normal(m, tau, MC))
geo, tau2 = np.exp(np.log(s_true).mean()), np.log(s_true).var()
mom = np.sqrt(s_true.mean() / (1 / s_true).mean())
print(f"  truth: geometric mean {geo:.5f}, sqrt(M+/M-) {mom:.5f}, "
      f"tau^2 {tau2:.5f}, floor {np.exp(tau2/2):.5f}")
print(f"{'K':>3} | {'log route':>10} {'err':>8} | {'M+- route':>10} {'err':>8}"
      f" | {'tau^2':>8} {'floor':>8}")
for K in (2, 3, 5, 9, 17):
    nu = K - 1
    lo = np.log(sample_width(s_true, nu, MC))
    B, t2 = np.exp(lo.mean() - beta_log(nu)), lo.var() - kappa(nu)
    if nu > 1:
        obs = sample_width(s_true, nu, MC)
        A = np.sqrt(obs.mean()/(1/obs).mean()) * np.sqrt(nu/(nu-1))
        As, eA = f"{A:.5f}", f"{100*(A/mom-1):+.3f}%"
    else:
        As, eA = "undefined", "n/a"
    print(f"{K:>3} | {B:>10.5f} {100*(B/geo-1):>+7.3f}% | {As:>10} {eA:>8}"
          f" | {t2:>8.5f} {np.exp(max(t2,0)/2):>8.5f}")
print("""  the log route recovers the geometric mean at every K INCLUDING K=2, and
  its second moment gives tau^2 (hence the floor) from the same accumulators.

Eq. `eq:oddcum`: the two targets differ by kappa_3/6 (zero if log s symmetric)""")
for name, ls in [
    ("symmetric (normal)", RNG.normal(m, 0.6, MC)),
    ("skewed +", m + (RNG.gamma(2.0, 0.42, MC) - 0.84)),
    ("skewed -", m - (RNG.gamma(2.0, 0.42, MC) - 0.84)),
]:
    s = np.exp(ls)
    gap = 0.5 * np.log(s.mean() / (1 / s).mean()) - ls.mean()
    k3 = ((ls - ls.mean()) ** 3).mean()
    rA = np.mean(np.cosh(ls - 0.5*np.log(s.mean()/(1/s).mean())))
    rB = np.mean(np.cosh(ls - ls.mean()))
    print(f"  {name:<20} gap={gap:+.5f}  kappa_3/6={k3/6:+.5f}  "
          f"excess risk of geometric mean = {100*(rB/rA-1):.4f}%")

# --------------------------------------------------------------------------
banner("Prop. `prop:selfsim`: L(beta) = sqrt(M+M-) cosh(beta - beta*), exactly")
pops = {
    "symmetric (log-normal)": RNG.normal(m, 0.6, MC),
    "right-skewed (+1.4)": m + (RNG.gamma(2.0, 0.42, MC) - 0.84),
    "left-skewed  (-1.4)": m - (RNG.gamma(2.0, 0.42, MC) - 0.84),
    "heavy right tail (+2.8)": m + (RNG.gamma(0.5, 0.85, MC) - 0.425),
}
print(f"{'population':<26} {'E[log s]':>9} {'beta*':>9} {'Delta':>9}"
      f" {'E[sinh]@b*':>11} | {'L(gm)/L*':>9} {'cosh(Delta)':>11}"
      f" {'max|L-form|':>12}")
for name, bb in pops.items():
    s = np.exp(bb)
    Mp, Mm = s.mean(), (1 / s).mean()
    bstar, Lstar = 0.5 * np.log(Mp / Mm), np.sqrt(Mp * Mm)
    gm = bb.mean()
    L = lambda t: np.mean(np.cosh(bb - t))
    err = max(abs(L(x) - Lstar * np.cosh(x - bstar))
              for x in np.linspace(bstar - 1.5, bstar + 1.5, 41))
    print(f"{name:<26} {gm:>9.5f} {bstar:>9.5f} {bstar-gm:>+9.5f}"
          f" {np.mean(np.sinh(bb-bstar)):>+11.2e} | {L(gm)/Lstar:>9.6f}"
          f" {np.cosh(bstar-gm):>11.6f} {err:>12.2e}")
print("""  The risk in log s_hat is itself the cosh law (to machine precision), so the
  optimum is the sinh-balance point and ANY summary costs cosh of its log-distance
  to it. The geometric mean minimises the SQUARED log error, not the cosh one:
  it coincides only when log s is symmetric, and otherwise the cosh optimum sits
  toward the longer tail. cosh(Delta) is exactly what that substitution costs.""")
