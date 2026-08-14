"""GPU-accelerated normalization transforms using PyTorch.

All transforms work with named tensors and run on any device.

Example:
-------
>>> import torch
>>> from myotorch.transforms.tensor import ZScore, MinMax, InstanceNorm
>>>
>>> x = torch.randn(32, 64, 200, device='cuda', names=('batch', 'channel', 'time'))
>>>
>>> # Z-score normalize per sample
>>> zscore = ZScore(dim='time')
>>> y = zscore(x)  # mean=0, std=1 along time axis

"""

from __future__ import annotations

import torch

from myotorch.transforms.base import TensorTransform, get_dim_index


class ZScore(TensorTransform):
    """Z-score normalization (mean=0, std=1) along a dimension.

    Parameters
    ----------
    dim : str
        Dimension to normalize over.
    eps : float
        Small value to avoid division by zero.
    keepdim : bool
        Whether to keep the dimension in mean/std computation.

    Examples
    --------
    >>> x = torch.randn(64, 2048, device='cuda', names=('channel', 'time'))
    >>> zscore = ZScore(dim='time')
    >>> y = zscore(x)  # Normalized to mean=0, std=1 per channel

    """

    def __init__(
        self,
        dim: str = "time",
        eps: float = 1e-8,
        keepdim: bool = True,
        **kwargs,
    ):
        super().__init__(dim=dim, **kwargs)
        self.eps = eps
        self.keepdim = keepdim

    def _apply(self, x: torch.Tensor) -> torch.Tensor:
        dim_idx = get_dim_index(x, self.dim)
        names = x.names

        x = x.rename(None)

        mean = x.mean(dim=dim_idx, keepdim=self.keepdim)
        std = x.std(dim=dim_idx, keepdim=self.keepdim)

        result = (x - mean) / (std + self.eps)

        if names[0] is not None:
            result = result.rename(*names)

        return result


class MinMax(TensorTransform):
    """Min-max normalization to [0, 1] range along a dimension.

    Parameters
    ----------
    dim : str
        Dimension to normalize over.
    eps : float
        Small value to avoid division by zero.
    range : tuple[float, float]
        Target range (default: (0, 1)).

    Examples
    --------
    >>> x = torch.randn(64, 2048, names=('channel', 'time'))
    >>> minmax = MinMax(dim='time')
    >>> y = minmax(x)  # Values in [0, 1]

    """

    def __init__(
        self,
        dim: str = "time",
        eps: float = 1e-8,
        range: tuple[float, float] = (0.0, 1.0),
        **kwargs,
    ):
        super().__init__(dim=dim, **kwargs)
        self.eps = eps
        self.range = range

    def _apply(self, x: torch.Tensor) -> torch.Tensor:
        dim_idx = get_dim_index(x, self.dim)
        names = x.names

        x = x.rename(None)

        x_min = x.min(dim=dim_idx, keepdim=True).values
        x_max = x.max(dim=dim_idx, keepdim=True).values

        # Normalize to [0, 1]
        result = (x - x_min) / (x_max - x_min + self.eps)

        # Scale to target range
        low, high = self.range
        if low != 0.0 or high != 1.0:
            result = result * (high - low) + low

        if names[0] is not None:
            result = result.rename(*names)

        return result


class Normalize(TensorTransform):
    """L-p normalization along a dimension.

    Parameters
    ----------
    p : float
        Norm type (1=L1, 2=L2/Euclidean, inf=max).
    dim : str
        Dimension to normalize over.
    eps : float
        Small value to avoid division by zero.

    Examples
    --------
    >>> x = torch.randn(64, 2048, names=('channel', 'time'))
    >>> norm = Normalize(p=2, dim='channel')
    >>> y = norm(x)  # L2 normalized along channels

    """

    def __init__(
        self,
        p: float = 2.0,
        dim: str = "channel",
        eps: float = 1e-8,
        **kwargs,
    ):
        super().__init__(dim=dim, **kwargs)
        self.p = p
        self.eps = eps

    def _apply(self, x: torch.Tensor) -> torch.Tensor:
        dim_idx = get_dim_index(x, self.dim)
        names = x.names

        x = x.rename(None)
        result = torch.nn.functional.normalize(x, p=self.p, dim=dim_idx, eps=self.eps)

        if names[0] is not None:
            result = result.rename(*names)

        return result


class InstanceNorm(TensorTransform):
    """Instance normalization (normalize each sample independently).

    Normalizes over channel and time dimensions for each sample.
    Commonly used in style transfer and generative models.

    Parameters
    ----------
    eps : float
        Small value for numerical stability.
    affine : bool
        Whether to use learnable parameters (requires registration).

    Examples
    --------
    >>> x = torch.randn(32, 64, 200, names=('batch', 'channel', 'time'))
    >>> inorm = InstanceNorm()
    >>> y = inorm(x)  # Each sample normalized independently

    """

    def __init__(self, eps: float = 1e-5, **kwargs):
        super().__init__(dim="time", **kwargs)
        self.eps = eps

    def _apply(self, x: torch.Tensor) -> torch.Tensor:
        names = x.names
        x = x.rename(None)

        # InstanceNorm expects (N, C, L) format
        if x.ndim == 2:
            x = x.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        result = torch.nn.functional.instance_norm(x, eps=self.eps)

        if squeeze:
            result = result.squeeze(0)

        if names[0] is not None:
            result = result.rename(*names)

        return result


class LayerNorm(TensorTransform):
    """Layer normalization along specified dimensions.

    Parameters
    ----------
    normalized_shape : tuple[int, ...]
        Shape of the dimensions to normalize over.
    eps : float
        Small value for numerical stability.

    Examples
    --------
    >>> x = torch.randn(32, 64, 200, names=('batch', 'channel', 'time'))
    >>> lnorm = LayerNorm(normalized_shape=(64, 200))
    >>> y = lnorm(x)  # Normalized over channel and time

    """

    def __init__(
        self,
        normalized_shape: tuple[int, ...] | int,
        eps: float = 1e-5,
        **kwargs,
    ):
        super().__init__(dim="time", **kwargs)
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps

    def _apply(self, x: torch.Tensor) -> torch.Tensor:
        names = x.names
        x = x.rename(None)

        result = torch.nn.functional.layer_norm(
            x,
            self.normalized_shape,
            eps=self.eps,
        )

        if names[0] is not None:
            result = result.rename(*names)

        return result


class BatchNorm(TensorTransform):
    """Batch normalization (normalize over batch dimension).

    Note: This is a stateless version for inference. For training with
    running statistics, use torch.nn.BatchNorm1d.

    Parameters
    ----------
    eps : float
        Small value for numerical stability.

    Examples
    --------
    >>> x = torch.randn(32, 64, 200, names=('batch', 'channel', 'time'))
    >>> bnorm = BatchNorm()
    >>> y = bnorm(x)  # Normalized over batch dimension

    """

    def __init__(self, eps: float = 1e-5, **kwargs):
        super().__init__(dim="batch", **kwargs)
        self.eps = eps

    def _apply(self, x: torch.Tensor) -> torch.Tensor:
        names = x.names
        x = x.rename(None)

        if x.ndim == 2:
            # Add batch dimension
            x = x.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        # Compute batch statistics
        mean = x.mean(dim=0, keepdim=True)
        var = x.var(dim=0, keepdim=True, unbiased=False)

        result = (x - mean) / torch.sqrt(var + self.eps)

        if squeeze:
            result = result.squeeze(0)

        if names[0] is not None:
            result = result.rename(*names)

        return result


class ClampRange(TensorTransform):
    """Clamp values to a specified range.

    Parameters
    ----------
    min_val : float | None
        Minimum value.
    max_val : float | None
        Maximum value.

    Examples
    --------
    >>> x = torch.randn(64, 2048, names=('channel', 'time'))
    >>> clamp = ClampRange(min_val=-3, max_val=3)
    >>> y = clamp(x)  # Values clamped to [-3, 3]

    """

    def __init__(
        self,
        min_val: float | None = None,
        max_val: float | None = None,
        **kwargs,
    ):
        super().__init__(dim="time", **kwargs)
        self.min_val = min_val
        self.max_val = max_val

    def _apply(self, x: torch.Tensor) -> torch.Tensor:
        names = x.names
        x = x.rename(None)

        result = torch.clamp(x, min=self.min_val, max=self.max_val)

        if names[0] is not None:
            result = result.rename(*names)

        return result


class Standardize(TensorTransform):
    """Standardize using pre-computed mean and std.

    Useful when you have statistics from the training set.

    Parameters
    ----------
    mean : float | torch.Tensor
        Mean value(s) to subtract.
    std : float | torch.Tensor
        Standard deviation(s) to divide by.
    eps : float
        Small value to avoid division by zero.

    Examples
    --------
    >>> # Compute stats on training data
    >>> train_mean = train_data.mean()
    >>> train_std = train_data.std()
    >>>
    >>> # Apply to test data
    >>> standardize = Standardize(mean=train_mean, std=train_std)
    >>> test_normalized = standardize(test_data)

    """

    def __init__(
        self,
        mean: float | torch.Tensor,
        std: float | torch.Tensor,
        eps: float = 1e-8,
        **kwargs,
    ):
        super().__init__(dim="time", **kwargs)
        self.mean = mean
        self.std = std
        self.eps = eps

    def _apply(self, x: torch.Tensor) -> torch.Tensor:
        names = x.names
        x = x.rename(None)

        # Convert to tensor on same device if needed
        mean = self.mean
        std = self.std
        if isinstance(mean, (int, float)):
            mean = torch.tensor(mean, device=x.device, dtype=x.dtype)
        else:
            mean = mean.to(device=x.device, dtype=x.dtype)
        if isinstance(std, (int, float)):
            std = torch.tensor(std, device=x.device, dtype=x.dtype)
        else:
            std = std.to(device=x.device, dtype=x.dtype)

        result = (x - mean) / (std + self.eps)

        if names[0] is not None:
            result = result.rename(*names)

        return result
