"""guards for the GTCNN product-graph spectra (prior._product_spectrum).

These are pure-function tests on the combination rules, so they need no residual data
and run everywhere. The expensive end-to-end checks (forward reaches the stationary law,
eigenbasis correctness) live in scripts/verify_product_graph.py, which gates submission.

What can silently break here:
  - a combination rule swapped between forms (cartesian <-> strong is a one-character slip)
  - the [0,2] rescale dropped, which would make gamma mean something different per form
    and per dataset, quietly un-controlling the whole campaign
  - the spatial/temporal axes transposed
"""
import numpy as np
import pytest

from src.diffusion.prior import PRODUCT_FORMS, _product_spectrum

# deliberately asymmetric so a transposed axis cannot pass
LAM_S = np.array([0.0, 0.5, 2.0])
LAM_T = np.array([0.0, 1.0])


def _raw(form, a, b, s=None):
    """the combination rule BEFORE the [0,2] rescale, written out independently."""
    if form == "prod_cartesian":
        return a[:, None] + b[None, :]
    if form == "prod_strong":
        return (1 + a[:, None]) * (1 + b[None, :]) - 1
    if form == "prod_tensor":
        return a[:, None] * b[None, :]
    s00, s01, s10, s11 = s
    lam = s00 + s01 * a[:, None] + s10 * b[None, :] + s11 * a[:, None] * b[None, :]
    return lam - lam.min()


@pytest.mark.parametrize("form", PRODUCT_FORMS)
def test_spectrum_is_rescaled_to_0_2(form):
    """every form must land in exactly [0,2] -- this is what makes ONE gamma give the
    same theta_min, stiffness and GAD mixing number in every cell."""
    s = (0.1, 1.0, 1.0, 0.0) if form == "prod_param" else None
    lam = _product_spectrum(form, LAM_S, LAM_T, s)
    assert lam.shape == (LAM_S.size, LAM_T.size)
    assert lam.min() == pytest.approx(0.0, abs=1e-12)
    assert lam.max() == pytest.approx(2.0, rel=1e-12)


@pytest.mark.parametrize("form", PRODUCT_FORMS)
def test_matches_the_independent_rule_up_to_the_rescale(form):
    """the shipped spectrum must be the stated GTCNN rule, rescaled -- nothing else."""
    s = (0.1, 1.0, 0.3, 0.7) if form == "prod_param" else None
    got = _product_spectrum(form, LAM_S, LAM_T, s)
    want = _raw(form, LAM_S, LAM_T, s)
    want = 2.0 * want / want.max()
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)


def test_tensor_collapses_N_plus_H_minus_1_modes():
    """the Kronecker product graph pins every mode with lam_S=0 OR lam_T=0 to the bottom.
    that is N+H-1 of the NH modes at theta_min -- the prior cannot tell them apart, which
    is exactly why this arm is a negative control and not a candidate."""
    lam = _product_spectrum("prod_tensor", LAM_S, LAM_T)
    n_zero = int((lam < 1e-12).sum())
    assert n_zero == LAM_S.size + LAM_T.size - 1


def test_cartesian_and_strong_keep_a_single_bottom_mode():
    """additive/strong couplings vanish only when BOTH factors do -- one bottom mode."""
    for form in ("prod_cartesian", "prod_strong"):
        lam = _product_spectrum(form, LAM_S, LAM_T)
        assert int((lam < 1e-12).sum()) == 1, form


def test_parametric_contains_the_fixed_forms():
    """GTCNN eq.10 strictly generalises the three fixed products, so the right
    coefficients must reproduce them exactly after the shared rescale."""
    for form, s in (("prod_cartesian", (0.0, 1.0, 1.0, 0.0)),
                    ("prod_tensor", (0.0, 0.0, 0.0, 1.0)),
                    ("prod_strong", (0.0, 1.0, 1.0, 1.0))):
        np.testing.assert_allclose(
            _product_spectrum("prod_param", LAM_S, LAM_T, s),
            _product_spectrum(form, LAM_S, LAM_T),
            rtol=1e-12, atol=1e-12, err_msg=form)


def test_parametric_requires_coefficients():
    with pytest.raises(ValueError, match="fitted s_ij"):
        _product_spectrum("prod_param", LAM_S, LAM_T, None)


def test_unknown_form_rejected():
    with pytest.raises(ValueError, match="unknown product form"):
        _product_spectrum("prod_nonsense", LAM_S, LAM_T)
