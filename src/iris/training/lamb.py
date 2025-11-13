"""Layer-wise Adaptive Moments (Lamb) optimizer implementation."""

from __future__ import annotations

from typing import Iterable, Optional

import torch
from torch.optim import Optimizer


class Lamb(Optimizer):
    r"""
    Implements the Lamb optimizer introduced in:
        "Large Batch Optimization for Deep Learning: Training BERT in 76 minutes"
        (You, et al. 2019).

    This implementation closely follows the reference algorithm and supports
    decoupled weight decay together with trust ratio scaling.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        clamp_trust_ratio: Optional[tuple[float, float]] = (0.0, 10.0),
    ) -> None:
        if lr <= 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            clamp_trust_ratio=clamp_trust_ratio,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            clamp_trust = group["clamp_trust_ratio"]

            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad
                if grad.is_sparse:
                    raise RuntimeError("Lamb does not support sparse gradients.")

                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param)
                    state["exp_avg_sq"] = torch.zeros_like(param)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                state["step"] += 1
                step = state["step"]

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step

                step_size = group["lr"]
                update = exp_avg / bias_correction1
                denom = (exp_avg_sq / bias_correction2).sqrt().add_(eps)
                update = update / denom

                if weight_decay != 0.0:
                    update = update + weight_decay * param

                weight_norm = torch.norm(param)
                update_norm = torch.norm(update)
                if weight_norm == 0.0 or update_norm == 0.0:
                    trust_ratio = 1.0
                else:
                    trust_ratio = weight_norm / update_norm

                if clamp_trust is not None:
                    trust_ratio = float(
                        torch.clamp(
                            torch.tensor(trust_ratio, device=param.device),
                            clamp_trust[0],
                            clamp_trust[1],
                        ).item()
                    )

                param.add_(update, alpha=-step_size * trust_ratio)

        return loss


