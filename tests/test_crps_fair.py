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
