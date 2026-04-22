# Denoiser network: graph-MLP with AdaLN-Zero conditioning on diffusion time and STID features.
# Implemented in Phase 5. Stub only.
import torch.nn as nn


class Denoiser(nn.Module):
    """Score / noise prediction network conditioned on tau and backbone features.

    Architecture: 4 blocks of AdaLN-Zero → ChebConv(K=3) → AdaLN-Zero → MLP(GELU).
    Target parameter budget: ≤ 1M. See Phase 5 of CLAUDE_CODE_TASK.md.
    """

    def forward(self, x_noisy, tau, history_features):
        """Args:
            x_noisy: (batch, H, N, 1) noisy residual tensor.
            tau: (batch,) diffusion time.
            history_features: (batch, d) pooled STID penultimate features.

        Returns:
            Noise prediction of same shape as ``x_noisy``.
        """
        raise NotImplementedError
