"""
@file noisy_optimizer.py
@brief Noisy AdamW optimizer for SGD noise regularization experiments

@details
Adds isotropic Gaussian noise to all parameters after each AdamW update step.
This approximates Langevin dynamics and acts as a regularizer by preventing
the optimizer from settling in sharp minima.

Noise is applied as:
    theta <- theta + N(0, noise_std^2)

Classes:
- NoisyAdamW - AdamW with additive Gaussian weight noise
"""

import torch
from torch.optim import AdamW


class NoisyAdamW(AdamW):
    """
    @brief AdamW optimizer with additive Gaussian noise injected into weights after each step.
    @details Inherits all AdamW behaviour. The noise_std hyperparameter controls the noise
             magnitude and is tuned via Ray Tune alongside lr.
    """

    def __init__(self, params, lr: float = 1e-3, noise_std: float = 1e-4,
                 weight_decay: float = 0.0, amsgrad: bool = False, **kwargs) -> None:
        """
        @brief Initialise NoisyAdamW
        @param params: model parameters (same as AdamW)
        @param lr: learning rate
        @param noise_std: standard deviation of the Gaussian noise added to weights each step
        @param weight_decay: L2 penalty coefficient (kept at 0 for noise-only experiments)
        @param amsgrad: whether to use AMSGrad variant
        """
        self.noise_std = noise_std
        super().__init__(params, lr=lr, weight_decay=weight_decay, amsgrad=amsgrad, **kwargs)

    def step(self, closure=None):
        """
        @brief Performs a single optimisation step followed by Gaussian noise injection
        @param closure: optional closure (same as AdamW)
        @return: loss value if closure is provided, else None
        """
        loss = super().step(closure)
        with torch.no_grad():
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        p.data.add_(torch.randn_like(p.data) * self.noise_std)
        return loss
