"""the ancestral step must reproduce the EXACT reverse transition under the model.

for the per-mode OU forward with stationary variance v = sigma2/theta the reverse
transition is N(a_d x_c, v (1 - a_d^2)), a_d = exp(-theta dc). the coded step used the
x0-known variance v_post (DDPM's beta-tilde), which is too small by up to 2x for slow
modes; the posterior-variance term restores exactness. also: the default floor is the
schedule's training floor, not a hardcoded 0.02.
"""
import numpy as np
import pytest
import torch

from src.diffusion.forward_fcps import FCPSSchedule, derived_floor, derived_steps, undenoised_fraction
from src.diffusion.prior import KroneckerPrior
from src.diffusion.reverse import reverse_sample_ancestral


def _toy_prior(thetas, H=3, sigma2=1.0):
    N = len(thetas)
    theta = np.outer(np.asarray(thetas, float), np.ones(H))
    return KroneckerPrior(dataset="toy", gamma=0.8, V_S=np.eye(N), lam_S=theta[:, 0] - 0.8,
                          V_T=np.eye(H), mu_T=np.ones(H), theta=theta, delta_h=np.ones(H),
                          R_H=np.eye(H), sigma2_scale=sigma2)


@pytest.mark.parametrize("theta", [0.8, 16.0, 200.0, 5000.0])
@pytest.mark.parametrize("c,dc", [(1.0, 0.5), (1.0, 0.05), (0.1, 0.05), (0.005, 0.001)])
def test_corrected_step_variance_is_exact(theta, c, dc):
    s2, v = 1.0, 1.0 / theta
    cp = c - dc
    a_d = np.exp(-theta * dc); s_d = s2 * (1 - a_d ** 2) / theta
    a_p = np.exp(-theta * cp); s_p = s2 * (1 - a_p ** 2) / theta
    a_c = np.exp(-theta * c);  s_c = s2 * (1 - a_c ** 2) / theta
    v_post = s_d * s_p / (a_d * a_d * s_p + s_d)
    sig_p2 = 1.0 / (a_c * a_c / s_c + 1.0 / v)
    corrected = v_post + (v_post * a_p / s_p) ** 2 * sig_p2
    exact = v * (1 - a_d ** 2)
    assert abs(corrected / exact - 1.0) < 1e-6


def test_corrected_sampler_preserves_stationary_variance():
    """model exact, denoiser Bayes-optimal: the spread that comes out must equal the
    ceiling 1 - f(c_lo) at 50 steps. the uncorrected step falls well short."""
    torch.manual_seed(0)
    prior = _toy_prior([0.8, 4.0, 20.0, 100.0], H=2)
    sch = FCPSSchedule(c_min=0.1, c_0=7.0, alpha=4.0, T=1.0)
    theta = torch.as_tensor(prior.theta, dtype=torch.float32).unsqueeze(0)
    v = prior.sigma2_scale / theta

    def bayes(R_t, t, Y):                       # identity basis -> per-mode shrinkage
        c = torch.as_tensor(sch.c_bar(t.numpy()), dtype=torch.float32).view(-1, 1, 1)
        a = torch.exp(-theta * c); s = torch.clamp(prior.sigma2_scale * (1 - torch.exp(-2 * theta * c)) / theta, min=1e-20)
        return a * v / (a * a * v + s) * R_t

    B, Y = 4000, torch.zeros(4000, 4, 2)
    ceiling = 1.0 - undenoised_fraction(prior, sch.c_lo)
    out_c, _ = reverse_sample_ancestral(prior, sch, bayes, Y, n_steps=50, corrected=True)
    out_u, _ = reverse_sample_ancestral(prior, sch, bayes, Y, n_steps=50, corrected=False)
    ratio = lambda o: float((o.pow(2).mean(0).sum() / v.squeeze(0).sum()).item())
    assert abs(ratio(out_c) - ceiling) < 0.03, (ratio(out_c), ceiling)
    assert ratio(out_u) < ceiling - 0.05, (ratio(out_u), ceiling)


def test_default_floor_is_the_training_floor():
    sch = FCPSSchedule(c_lo_override=3e-4)
    assert sch.c_lo == 3e-4
    assert FCPSSchedule().c_lo == 0.04 ** 2 / 2


def test_derived_floor_equalises_fidelity_across_arms():
    iso = _toy_prior([0.8, 0.8, 0.8])
    stiff = _toy_prior([0.8, 50.0, 2000.0])
    f_iso, f_stiff = derived_floor(iso), derived_floor(stiff)
    assert f_stiff < 0.5 * f_iso                 # the stiff arm needs a lower floor
    assert abs(undenoised_fraction(iso, f_iso) - 1.3e-3) < 1e-4
    assert abs(undenoised_fraction(stiff, f_stiff) - 1.3e-3) < 1e-4
    sch_s = FCPSSchedule(c_lo_override=f_stiff); sch_i = FCPSSchedule(c_lo_override=f_iso)
    assert derived_steps(stiff, sch_s) > derived_steps(iso, sch_i)


def test_common_mode_gauss_matches_oracle_and_changes_only_that_mode():
    """with the prior's own Bayes denoiser, common_mode='gauss' must reproduce the learned
    path exactly (the override equals what the denoiser returns); with a denoiser that is
    wrong on the common mode only, the two paths differ."""
    import numpy as np, torch
    from src.diffusion.reverse import reverse_sample_ancestral, _rotate_to_eig_torch, _rotate_from_eig_torch
    from src.diffusion.forward_fcps import FCPSSchedule
    from src.diffusion.prior import KroneckerPrior
    N, H = 6, 3
    lam = np.array([0.0, 0.3, 0.8, 1.2, 1.6, 1.9]); V_S = np.linalg.qr(np.random.default_rng(0).standard_normal((N, N)))[0]
    V_S[:, 0] = 1 / np.sqrt(N); V_S = np.linalg.qr(V_S)[0] * np.sign(V_S[0, 0])     # first column = constant mode
    mu_T = np.array([2.0, 1.0, 0.5]); V_T = np.eye(H)
    theta = ((lam + 0.2)[:, None] / mu_T[None, :]); theta = theta * (0.8 / theta.min())
    prior = KroneckerPrior(dataset="toy", gamma=0.2, V_S=V_S, lam_S=lam, V_T=V_T, mu_T=mu_T, theta=theta,
                           delta_h=mu_T, R_H=np.eye(H), sigma2_scale=1.0)
    sch = FCPSSchedule(c_0=7.0, c_lo_override=1e-3)
    th = torch.as_tensor(theta, dtype=torch.float32); VS = torch.as_tensor(V_S, dtype=torch.float32); VT = torch.as_tensor(V_T, dtype=torch.float32)
    def bayes(R_t, t, Y):                                   # E[x0 | x_c] under the prior, per mode
        c = torch.as_tensor(sch.c_bar(t.numpy()), dtype=torch.float32).view(-1, 1, 1)
        a = torch.exp(-th * c); s = (1 - torch.exp(-2 * th * c)) / th; v = 1 / th
        C = _rotate_to_eig_torch(R_t, VS, VT)
        return _rotate_from_eig_torch((1 / (a * a / s + 1 / v)) * a / s * C, VS, VT)
    def bad_common(R_t, t, Y):                              # bayes, but the constant mode is zeroed out
        C = _rotate_to_eig_torch(bayes(R_t, t, Y), VS, VT).clone(); C[:, 0, :] = 0
        return _rotate_from_eig_torch(C, VS, VT)
    Y = torch.zeros(64, N, H)
    a, _ = reverse_sample_ancestral(prior, sch, bayes, Y, n_steps=40, generator=torch.Generator().manual_seed(1))
    b, _ = reverse_sample_ancestral(prior, sch, bayes, Y, n_steps=40, generator=torch.Generator().manual_seed(1), common_mode="gauss")
    assert torch.allclose(a, b, atol=1e-5)
    c, _ = reverse_sample_ancestral(prior, sch, bad_common, Y, n_steps=40, generator=torch.Generator().manual_seed(1))
    d, _ = reverse_sample_ancestral(prior, sch, bad_common, Y, n_steps=40, generator=torch.Generator().manual_seed(1), common_mode="gauss")
    assert torch.allclose(d, b, atol=1e-5)                  # the gauss rule repairs the broken mode exactly
    assert not torch.allclose(c, b, atol=1e-3)              # and the broken denoiser alone does not
