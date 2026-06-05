"""
@file discrete_optimizer.py
@brief SignSGD optimizer for gradient discretization regularization experiments

@details
Implements SignSGD, a form of gradient discretization where only the sign of each
gradient element is used for the weight update (Bernstein et al., 2018).
This implicitly regularizes by treating all gradient magnitudes as equal,
reducing sensitivity to gradient scale and acting as a form of implicit regularization.

Update rule:
    theta <- theta - lr * sign(grad)

Classes:
- SignSGD - Sign Stochastic Gradient Descent optimizer
"""

import torch
from torch.optim import Optimizer


class SignSGD(Optimizer):
    """
    @brief SignSGD optimizer: uses only the sign of gradients for weight updates.
    @details Discretizes gradient information to {-1, 0, +1}, which prevents large
             gradient magnitudes from dominating updates and acts as a natural
             gradient clipping / regularization mechanism.
    """

    def __init__(self, params, lr: float = 1e-3) -> None:
        """
        @brief Initialise SignSGD
        @param params: model parameters
        @param lr: learning rate (step size per sign update)
        """
        if lr <= 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {'lr': lr}
        super().__init__(params, defaults)

    def step(self, closure=None) -> float | None:
        """
        @brief Performs a single SignSGD update step
        @param closure: optional closure for loss re-evaluation
        @return: loss value if closure is provided, else None
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                p.data.add_(torch.sign(p.grad.data), alpha=-lr)

        return loss
