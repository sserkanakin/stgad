"""empirical_crps must be the FAIR estimator: unbiased for a calibrated forecast at
every K, and NOT rewarding a narrower ensemble at K=4 (the k used for selection)."""
import math

import torch

from src.eval.crps import empirical_crps

TRUE = 1.0 / math.sqrt(math.pi)          # CRPS of N(0,1) predicting a N(0,1) draw


def test_unbiased_at_small_k():
    torch.manual_seed(0)
    y = torch.randn(20000, 1, 1)
    for K in (4, 8, 100):
        s = torch.randn(K, 20000, 1, 1)
        est = empirical_crps(s, y).mean().item()
        assert abs(est / TRUE - 1.0) < 0.02, (K, est)   # the 1/K^2 form is +25% at K=4


def test_does_not_reward_underdispersion():
    torch.manual_seed(1)
    y = torch.randn(20000, 1, 1)
    calibrated = torch.randn(4, 20000, 1, 1)
    narrow = torch.randn(4, 20000, 1, 1) * 0.87
    assert empirical_crps(calibrated, y).mean() < empirical_crps(narrow, y).mean()


def test_empirical_crps_any_trailing_shape():
    """the joint metrics call it on (K, B, H) totals; the result must equal the (K, B, N, H)
    computation with a singleton axis, and must NOT scale with K or lose the spread term."""
    g = torch.Generator().manual_seed(0)
    K, B, H = 16, 5, 12
    S = torch.randn(K, B, H, generator=g); y = torch.randn(B, H, generator=g)
    ref = empirical_crps(S.unsqueeze(2), y.unsqueeze(1)).squeeze(1)
    out = empirical_crps(S, y)
    assert out.shape == (B, H)
    assert torch.allclose(out, ref, atol=1e-6)
    # closed form for K samples: E|X-y| - 0.5 E|X-X'| computed pairwise
    t1 = (S - y).abs().mean(0)
    t2 = 0.5 * (S.unsqueeze(0) - S.unsqueeze(1)).abs().sum((0, 1)) / (K * (K - 1))
    assert torch.allclose(out, t1 - t2, atol=1e-5)
