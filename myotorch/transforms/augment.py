"""GPU-accelerated augmentation transforms using PyTorch.

All augmentations work with named tensors and run on GPU.
They are stochastic and respect torch.random state.

Example:
-------
>>> import torch
>>> from myotorch.transforms.tensor import GaussianNoise, MagnitudeWarp, TimeWarp
>>>
>>> x = torch.randn(32, 64, 200, device='cuda', names=('batch', 'channel', 'time'))
>>>
>>> # Augmentation pipeline
>>> augment = Pipeline([
...     GaussianNoise(std=0.1),
...     MagnitudeWarp(sigma=0.2),
... ])
>>> y = augment(x)  # Augmented on GPU

"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from myotorch.transforms.base import TensorTransform, get_dim_index


class GaussianNoise(TensorTransform):
    """Add Gaussian noise to the signal.

    Parameters
    ----------
    std : float
        Standard deviation of the noise.
    p : float
        Probability of applying the augmentation.

    Examples
    --------
    >>> x = torch.randn(64, 2048, device='cuda', names=('channel', 'time'))
    >>> noise = GaussianNoise(std=0.1)
    >>> y = noise(x)

    """

    def __init__(self, std: float = 0.1, p: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.std = std
        self.p = p

    def _apply(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return x

        names = x.names
        x = x.rename(None)

        noise = torch.randn_like(x) * self.std
        result = x + noise

        if names[0] is not None:
            result = result.rename(*names)

        return result


class MagnitudeWarp(TensorTransform):
    """Warp magnitude using smooth random curves.

    Creates smooth random scaling factors that vary over time.

    Parameters
    ----------
    sigma : float
        Standard deviation for the warping curves.
    n_knots : int
        Number of control points for the spline.
    p : float
        Probability of applying the augmentation.
    dim : str
        Dimension to warp along.

    Examples
    --------
    >>> x = torch.randn(64, 2048, device='cuda', names=('channel', 'time'))
    >>> warp = MagnitudeWarp(sigma=0.2, n_knots=4)
    >>> y = warp(x)

    """

    def __init__(
        self,
        sigma: float = 0.2,
        n_knots: int = 4,
        p: float = 1.0,
        dim: str = "time",
        **kwargs,
    ):
        super().__init__(dim=dim, **kwargs)
        self.sigma = sigma
        self.n_knots = n_knots
        self.p = p

    def _apply(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return x

        dim_idx = get_dim_index(x, self.dim)
        names = x.names
        n_samples = x.shape[dim_idx]

        x = x.rename(None)

        # Generate smooth warping curve
        # Create random knots
        knots = (
            torch.randn(self.n_knots, device=x.device, dtype=x.dtype) * self.sigma + 1.0
        )

        # Interpolate to full length
        warp = F.interpolate(
            knots.view(1, 1, -1),
            size=n_samples,
            mode="linear",
            align_corners=True,
        ).squeeze()

        # Expand warp to match x dimensions
        shape = [1] * x.ndim
        shape[dim_idx] = n_samples
        warp = warp.view(*shape)

        result = x * warp

        if names[0] is not None:
            result = result.rename(*names)

        return result


class TimeWarp(TensorTransform):
    """Warp time axis with smooth random curves.

    Creates smooth random time shifts using cubic interpolation.

    Parameters
    ----------
    sigma : float
        Standard deviation for the warping curves.
    n_knots : int
        Number of control points.
    p : float
        Probability of applying the augmentation.
    dim : str
        Time dimension to warp.

    Examples
    --------
    >>> x = torch.randn(64, 2048, device='cuda', names=('channel', 'time'))
    >>> warp = TimeWarp(sigma=0.2, n_knots=4)
    >>> y = warp(x)

    """

    def __init__(
        self,
        sigma: float = 0.2,
        n_knots: int = 4,
        p: float = 1.0,
        dim: str = "time",
        **kwargs,
    ):
        super().__init__(dim=dim, **kwargs)
        self.sigma = sigma
        self.n_knots = n_knots
        self.p = p

    def _apply(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return x

        dim_idx = get_dim_index(x, self.dim)
        names = x.names
        n_samples = x.shape[dim_idx]

        x = x.rename(None)

        # Generate warping indices
        # Create random cumulative distortions
        distortions = torch.randn(self.n_knots + 2, device=x.device, dtype=x.dtype)
        distortions[0] = 0
        distortions[-1] = 0
        distortions = distortions.cumsum(0) * self.sigma

        # Interpolate to get warped indices
        orig_indices = torch.linspace(
            0, n_samples - 1, self.n_knots + 2, device=x.device
        )
        warped_indices = orig_indices + distortions * (n_samples / self.n_knots)
        warped_indices = torch.clamp(warped_indices, 0, n_samples - 1)

        # Interpolate warping function to full length
        warp_func = F.interpolate(
            warped_indices.view(1, 1, -1),
            size=n_samples,
            mode="linear",
            align_corners=True,
        ).squeeze()

        # Apply warping using grid_sample (need to reshape for grid_sample)
        # Move time dimension to last position
        if dim_idx != x.ndim - 1:
            perm = list(range(x.ndim))
            perm[dim_idx], perm[-1] = perm[-1], perm[dim_idx]
            x = x.permute(*perm)
            moved = True
        else:
            moved = False

        # Reshape for grid_sample: need (N, C, H, W) or (N, C, D, H, W)
        original_shape = x.shape
        x_flat = x.reshape(-1, 1, 1, n_samples)

        # Create sampling grid
        # grid_sample expects coordinates in [-1, 1]
        grid = warp_func / (n_samples - 1) * 2 - 1
        grid = grid.view(1, 1, 1, -1).expand(x_flat.shape[0], 1, 1, -1)

        # Apply warping
        warped = F.grid_sample(
            x_flat,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

        result = warped.reshape(original_shape)

        # Restore dimension order
        if moved:
            result = result.permute(*perm)

        if names[0] is not None:
            result = result.rename(*names)

        return result


class Dropout(TensorTransform):
    """Randomly zero out elements.

    Parameters
    ----------
    p : float
        Probability of zeroing each element.
    dim : str
        If specified, drops entire slices along this dimension.
        If None, drops individual elements.

    Examples
    --------
    >>> x = torch.randn(64, 2048, device='cuda', names=('channel', 'time'))
    >>> # Element-wise dropout
    >>> dropout = Dropout(p=0.1)
    >>> # Channel dropout (drop entire channels)
    >>> channel_dropout = Dropout(p=0.1, dim='channel')

    """

    def __init__(self, p: float = 0.1, dim: str | None = None, **kwargs):
        super().__init__(**kwargs)
        self.p = p
        self.drop_dim = dim

    def _apply(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training_mode:
            return x

        names = x.names
        x = x.rename(None)

        if self.drop_dim is None:
            # Element-wise dropout
            result = F.dropout(x, p=self.p, training=True)
        else:
            # Structured dropout along dimension
            dim_idx = get_dim_index(x, self.drop_dim) if names[0] is not None else -2

            # Create mask
            mask_shape = [1] * x.ndim
            mask_shape[dim_idx] = x.shape[dim_idx]
            mask = (torch.rand(mask_shape, device=x.device) > self.p).float()

            result = x * mask

        if names[0] is not None:
            result = result.rename(*names)

        return result

    @property
    def training_mode(self) -> bool:
        """Check if in training mode (dropout only during training)."""
        return getattr(self, "_training", True)

    def train(self):
        """Set to training mode."""
        self._training = True
        return self

    def eval(self):
        """Set to evaluation mode."""
        self._training = False
        return self


class ChannelShuffle(TensorTransform):
    """Randomly shuffle channel order.

    Parameters
    ----------
    p : float
        Probability of applying shuffle.

    Examples
    --------
    >>> x = torch.randn(64, 2048, device='cuda', names=('channel', 'time'))
    >>> shuffle = ChannelShuffle(p=0.5)
    >>> y = shuffle(x)

    """

    def __init__(self, p: float = 0.5, **kwargs):
        super().__init__(dim="channel", **kwargs)
        self.p = p

    def _apply(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return x

        dim_idx = get_dim_index(x, self.dim)
        names = x.names

        x = x.rename(None)

        # Generate random permutation
        n_channels = x.shape[dim_idx]
        perm = torch.randperm(n_channels, device=x.device)

        result = x.index_select(dim_idx, perm)

        if names[0] is not None:
            result = result.rename(*names)

        return result


class TimeShift(TensorTransform):
    """Randomly shift signal in time.

    Parameters
    ----------
    max_shift : int | float
        Maximum shift amount. If float, interpreted as fraction of length.
    p : float
        Probability of applying shift.
    fill : str
        How to fill shifted regions: 'zero', 'wrap', 'edge'.

    Examples
    --------
    >>> x = torch.randn(64, 2048, device='cuda', names=('channel', 'time'))
    >>> shift = TimeShift(max_shift=100, p=0.5)
    >>> y = shift(x)

    """

    def __init__(
        self,
        max_shift: float = 100,
        p: float = 0.5,
        fill: str = "zero",
        **kwargs,
    ):
        super().__init__(dim="time", **kwargs)
        self.max_shift = max_shift
        self.p = p
        self.fill = fill

    def _apply(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return x

        dim_idx = get_dim_index(x, self.dim)
        names = x.names
        n_samples = x.shape[dim_idx]

        x = x.rename(None)

        # Compute shift amount
        if isinstance(self.max_shift, float) and self.max_shift < 1:
            max_shift = int(n_samples * self.max_shift)
        else:
            max_shift = int(self.max_shift)

        shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()

        if shift == 0:
            if names[0] is not None:
                x = x.rename(*names)
            return x

        # Apply shift
        result = torch.roll(x, shifts=shift, dims=dim_idx)

        # Handle fill mode
        if self.fill == "zero":
            # Zero out the wrapped region
            if shift > 0:
                idx = [slice(None)] * x.ndim
                idx[dim_idx] = slice(0, shift)
                result[tuple(idx)] = 0
            else:
                idx = [slice(None)] * x.ndim
                idx[dim_idx] = slice(shift, None)
                result[tuple(idx)] = 0
        elif self.fill == "edge":
            # Use edge values for the wrapped region
            if shift > 0:
                idx = [slice(None)] * x.ndim
                idx[dim_idx] = slice(0, shift)
                edge_idx = [slice(None)] * x.ndim
                edge_idx[dim_idx] = shift
                result[tuple(idx)] = x[tuple(edge_idx)].unsqueeze(dim_idx)
            else:
                idx = [slice(None)] * x.ndim
                idx[dim_idx] = slice(shift, None)
                edge_idx = [slice(None)] * x.ndim
                edge_idx[dim_idx] = shift - 1
                result[tuple(idx)] = x[tuple(edge_idx)].unsqueeze(dim_idx)
        # 'wrap' is default behavior of roll

        if names[0] is not None:
            result = result.rename(*names)

        return result


class Scale(TensorTransform):
    """Random amplitude scaling.

    Parameters
    ----------
    scale_range : tuple[float, float]
        Range of scale factors (min, max).
    p : float
        Probability of applying scaling.

    Examples
    --------
    >>> x = torch.randn(64, 2048, device='cuda', names=('channel', 'time'))
    >>> scale = Scale(scale_range=(0.8, 1.2))
    >>> y = scale(x)

    """

    def __init__(
        self,
        scale_range: tuple[float, float] = (0.8, 1.2),
        p: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.scale_range = scale_range
        self.p = p

    def _apply(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return x

        names = x.names
        x = x.rename(None)

        scale = torch.empty(1, device=x.device, dtype=x.dtype).uniform_(
            *self.scale_range
        )
        result = x * scale

        if names[0] is not None:
            result = result.rename(*names)

        return result


class Cutout(TensorTransform):
    """Randomly zero out contiguous regions.

    Parameters
    ----------
    n_holes : int
        Number of regions to cut out.
    length : int | float
        Length of each cutout. If float < 1, fraction of total length.
    p : float
        Probability of applying cutout.
    dim : str
        Dimension to cut along.

    Examples
    --------
    >>> x = torch.randn(64, 2048, device='cuda', names=('channel', 'time'))
    >>> cutout = Cutout(n_holes=3, length=50)
    >>> y = cutout(x)

    """

    def __init__(
        self,
        n_holes: int = 1,
        length: float = 50,
        p: float = 0.5,
        dim: str = "time",
        **kwargs,
    ):
        super().__init__(dim=dim, **kwargs)
        self.n_holes = n_holes
        self.length = length
        self.p = p

    def _apply(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return x

        dim_idx = get_dim_index(x, self.dim)
        names = x.names
        n_samples = x.shape[dim_idx]

        x = x.rename(None)
        result = x.clone()

        # Compute hole length
        if isinstance(self.length, float) and self.length < 1:
            hole_length = int(n_samples * self.length)
        else:
            hole_length = int(self.length)

        for _ in range(self.n_holes):
            # Random position
            start = torch.randint(0, n_samples - hole_length + 1, (1,)).item()

            # Zero out region
            idx = [slice(None)] * x.ndim
            idx[dim_idx] = slice(start, start + hole_length)
            result[tuple(idx)] = 0

        if names[0] is not None:
            result = result.rename(*names)

        return result
