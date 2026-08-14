"""Custom activation functions for neural networks."""

from __future__ import annotations

import torch
from torch import nn


class PSerf(nn.Module):
    """PSerf activation function from Biswas et al.

    Parameters
    ----------
    gamma : float, optional
        The gamma parameter, by default 1.0.
    sigma : float, optional
        The sigma parameter, by default 1.25.
    stabilisation_term : float, optional
        The stabilisation term, by default 1e-12.

    References
    ----------
    Biswas, K., Kumar, S., Banerjee, S., Pandey, A.K., 2021.
    ErfAct and PSerf: Non-monotonic smooth trainable Activation Functions. arXiv:2109.04386 [cs].

    """

    def __init__(
        self,
        gamma: float = 1.0,
        sigma: float = 1.25,
        stabilisation_term: float = 1e-12,
    ):
        super().__init__()

        self.gamma = nn.Parameter(torch.tensor(gamma), requires_grad=True)
        self.sigma = nn.Parameter(torch.tensor(sigma), requires_grad=True)

        self.stabilisation_term = torch.tensor(stabilisation_term)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            x * torch.erf(self.gamma * torch.log(1 + torch.exp(self.sigma * x)))
            + self.stabilisation_term
        )


class SAU(nn.Module):
    """SAU activation function from Biswas et al.

    Parameters
    ----------
    alpha : float, optional
        The alpha parameter, by default 0.15.
    n : int, optional
        The n parameter, by default 20000.

    References
    ----------
    Biswas, K., Kumar, S., Banerjee, S., Pandey, A.K., 2021.
    SAU: Smooth activation function using convolution with approximate identities. arXiv:2109.13210 [cs].

    """

    def __init__(self, alpha: float = 0.15, n: int = 20000):
        super().__init__()

        self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=True)
        # Register constants as buffers so they move with the module to different devices
        self.register_buffer("n", torch.tensor(n, dtype=torch.float32))
        self.register_buffer("sqrt_2_over_pi", torch.sqrt(torch.tensor(2.0 / torch.pi)))
        self.register_buffer("sqrt_2", torch.sqrt(torch.tensor(2.0)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_squared = self.n * self.n
        return (
            self.sqrt_2_over_pi * torch.exp(-(n_squared * x * x) / 2) / (2 * self.n)
            + (1 + self.alpha) / 2 * x
            + (1 - self.alpha) / 2 * x * torch.erf(self.n * x / self.sqrt_2)
        )


class SMU(nn.Module):
    """SMU activation function from Biswas et al.

    Parameters
    ----------
    alpha : float, optional
        The alpha parameter, by default 0.01.
    mu : float, optional
        The mu parameter, by default 2.5.

    References
    ----------
    Biswas, K., Kumar, S., Banerjee, S., Pandey, A.K., 2022.
    SMU: smooth activation function for deep networks using smoothing maximum technique. arXiv:2111.04682 [cs].

    Notes
    -----
    This version also make alpha trainable.

    """

    def __init__(self, alpha: float = 0.01, mu: float = 2.5):
        super().__init__()

        self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=True)
        self.mu = nn.Parameter(torch.tensor(mu), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            (1 + self.alpha) * x
            + (1 - self.alpha) * x * torch.erf(self.mu * (1 - self.alpha) * x)
        ) / 2
