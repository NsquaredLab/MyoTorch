"""Utility layers for neural network models."""

from __future__ import annotations

import torch
from torch import nn


class WeightedSum(nn.Module):
    """Learnable weighted sum of two tensors.

    Computes alpha * x + (1 - alpha) * y where alpha is a learnable parameter.

    Parameters
    ----------
    alpha : float, optional
        Initial weight for the first input. Default is 0.5.

    """

    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=True)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.alpha * x + (1 - self.alpha) * y


class CircularPad(nn.Module):
    """Circular padding layer used in Sîmpetru et al. [1]_.

    Applies fixed circular padding to 4D input tensors along dimensions 2 and 3.

    References
    ----------
    .. [1] Sîmpetru, R.C., Osswald, M., Braun, D.I., Oliveira, D.S., Cakici, A.L., Del Vecchio, A., 2022.
           Accurate Continuous Prediction of 14 Degrees of Freedom of the Hand from Myoelectrical Signals
           through Convolutive Deep Learning, in: 2022 44th Annual International Conference of the IEEE
           Engineering in Medicine & Biology Society (EMBC), pp. 702-706.
           https://doi.org/10.1109/EMBC48229.2022.9870937

    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([torch.narrow(x, 2, 3, 2), x, torch.narrow(x, 2, 0, 2)], dim=2)
        x = torch.cat([torch.narrow(x, 3, 48, 16), x, torch.narrow(x, 3, 0, 16)], dim=3)
        return x
