"""Loss functions for neural network training."""

from __future__ import annotations

import torch
from torch import nn


class EuclideanDistance(nn.Module):
    """Euclidean distance loss for 3D joint positions.

    Computes the mean Euclidean distance between predicted and ground truth
    3D joint positions. Expects input tensors to be reshaped to (batch, joints, xyz).

    Parameters
    ----------
    n_joints : int
        Number of joints in the skeleton. Default is 20.
    n_dims : int
        Number of dimensions per joint (typically 3 for x, y, z). Default is 3.

    Examples
    --------
    >>> loss_fn = EuclideanDistance(n_joints=20)
    >>> pred = torch.randn(32, 60)  # batch_size=32, 20 joints * 3 dims
    >>> target = torch.randn(32, 60)
    >>> loss = loss_fn(pred, target)

    """

    def __init__(self, n_joints: int = 20, n_dims: int = 3):
        super().__init__()
        self.n_joints = n_joints
        self.n_dims = n_dims

    def forward(
        self, prediction: torch.Tensor, ground_truth: torch.Tensor
    ) -> torch.Tensor:
        """Compute the mean Euclidean distance loss.

        Parameters
        ----------
        prediction : torch.Tensor
            Predicted joint positions, shape (batch, n_joints * n_dims).
        ground_truth : torch.Tensor
            Ground truth joint positions, shape (batch, n_joints * n_dims).

        Returns
        -------
        torch.Tensor
            Scalar loss value.

        """
        pred_reshaped = prediction.reshape(-1, self.n_joints, self.n_dims)
        gt_reshaped = ground_truth.reshape(-1, self.n_joints, self.n_dims)

        # Compute per-joint Euclidean distances and average
        distances = torch.sqrt(
            torch.sum(torch.square(pred_reshaped - gt_reshaped), dim=-1)
        )
        return distances.mean()


# Backward compatibility alias (deprecated spelling)
EuclidianDistance = EuclideanDistance
