"""exponential moving average of model weights (the EDM/GenCast convention:
the EMA weights ARE the model that gets evaluated and shipped; the raw weights
only exist to compute gradients). decay 0.999 ~ averaging over the last ~1000
optimizer steps, which irons out the per-batch noise that dominates late
diffusion training."""
from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn as nn


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = float(decay)
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(self.decay).add_(v.detach().float(), alpha=1.0 - self.decay)
            else:                                   # ints/buffers: track verbatim
                self.shadow[k] = v.detach().clone()

    def state_dict(self, like: nn.Module) -> dict:
        """shadow weights cast back to the module's dtypes (loadable as-is)."""
        ref = like.state_dict()
        return {k: self.shadow[k].to(ref[k].dtype) for k in ref}

    @contextmanager
    def applied(self, model: nn.Module):
        """temporarily swap the EMA weights into the module (for validation)."""
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.state_dict(model))
        try:
            yield
        finally:
            model.load_state_dict(backup)

    def load(self, shadow: dict) -> None:
        self.shadow = {k: v.clone().float() for k, v in shadow.items()}
