"""Base transform classes using PyTorch named tensors.

Named tensors provide dimension-aware operations on GPU:
- Operations use dimension names: x.mean('time') instead of x.mean(-1)
- Automatic dimension alignment and broadcasting
- Full GPU acceleration via PyTorch

Design:
- TensorTransform: Base class for GPU-accelerated transforms
- Works with named tensors (experimental PyTorch feature)
- Composable with torchvision.transforms.Compose

Example:
-------
>>> import torch
>>> from myotorch.transforms.tensor import RMS, ZScore, Pipeline
>>>
>>> # Create named tensor
>>> x = torch.randn(64, 2048, names=('channel', 'time'))
>>>
>>> # Apply transforms on GPU
>>> pipeline = Pipeline([
...     ZScore(dim='time'),
...     RMS(window_size=200, dim='time'),
... ])
>>> y = pipeline(x.cuda())  # Runs on GPU

"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import torch

logger = logging.getLogger("myotorch.transforms.tensor")


def named_tensor(
    data: torch.Tensor,
    names: tuple[str, ...] | None = None,
) -> torch.Tensor:
    """Create or rename a tensor with dimension names.

    Parameters
    ----------
    data : torch.Tensor
        Input tensor.
    names : tuple[str, ...] | None
        Dimension names. If None, auto-generates based on ndim.

    Returns
    -------
    torch.Tensor
        Named tensor.

    Examples
    --------
    >>> x = torch.randn(64, 2048)
    >>> x = named_tensor(x, ('channel', 'time'))
    >>> x.mean('time').shape
    torch.Size([64])

    """
    if names is None:
        # Default naming convention
        if data.ndim == 1:
            names = ("time",)
        elif data.ndim == 2:
            names = ("channel", "time")
        elif data.ndim == 3:
            names = ("batch", "channel", "time")
        elif data.ndim == 4:
            names = ("batch", "representation", "channel", "time")
        else:
            names = tuple(f"dim_{i}" for i in range(data.ndim))

    return data.rename(*names)


def emg_tensor(
    data: torch.Tensor | Any,
    fs: float = 2048.0,
    grid_layouts: list | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create an EMG tensor with named dimensions and metadata.

    This is the entry point for GPU-accelerated EMG processing.

    Parameters
    ----------
    data : array-like
        EMG data. Shape should be (channels, time) or (batch, channels, time).
    fs : float
        Sampling frequency in Hz.
    grid_layouts : list[np.ndarray] | None
        Grid layouts for spatial transforms.
    device : torch.device | str | None
        Device to place tensor on ('cuda', 'cpu', etc.).
    dtype : torch.dtype
        Data type for the tensor.

    Returns
    -------
    torch.Tensor
        Named tensor on the specified device.

    Note
    ----
    Metadata (fs, grid_layouts) is stored as tensor attributes.
    PyTorch doesn't have built-in metadata like xarray, so we store
    these as custom attributes that transforms can access.

    Examples
    --------
    >>> import torch
    >>> from myotorch.transforms.base import emg_tensor
    >>>
    >>> # Create EMG tensor on GPU
    >>> emg = emg_tensor(data, fs=2048, device='cuda')
    >>> emg.names  # ('channel', 'time')
    >>> emg.fs  # 2048.0

    """
    # Convert to tensor if needed
    if not isinstance(data, torch.Tensor):
        data = torch.as_tensor(data, dtype=dtype)
    else:
        data = data.to(dtype=dtype)

    # Move to device
    if device is not None:
        data = data.to(device=device)

    # Add dimension names
    if data.ndim == 2:
        names = ("channel", "time")
    elif data.ndim == 3:
        names = ("batch", "channel", "time")
    else:
        names = tuple(f"dim_{i}" for i in range(data.ndim))

    data = data.rename(*names)

    # Store metadata as attributes
    # Note: These persist through most operations but may be lost on some
    data.fs = fs
    data.sampling_frequency = fs
    if grid_layouts is not None:
        data.grid_layouts = grid_layouts

    return data


class TensorTransform(ABC):
    """Base class for GPU-accelerated transforms using named tensors.

    Transforms operate on PyTorch tensors with named dimensions.
    They work on any device (CPU, CUDA, MPS) and are fully differentiable.

    Parameters
    ----------
    dim : str
        Dimension to operate on (e.g., 'time', 'channel').

    Examples
    --------
    >>> class MyTransform(TensorTransform):
    ...     def _apply(self, x: torch.Tensor) -> torch.Tensor:
    ...         return x.mean(self.dim, keepdim=True)
    ...
    >>> transform = MyTransform(dim='time')
    >>> x = torch.randn(64, 2048, names=('channel', 'time'))
    >>> y = transform(x)

    """

    def __init__(self, dim: str = "time", *, name: str | None = None):
        self.dim = dim
        self._name = name

    @property
    def name(self) -> str:
        """Transform name for debugging."""
        return self._name or self.__class__.__name__

    @property
    def params(self) -> dict[str, Any]:
        """Get transform parameters for debugging/serialization."""
        return {
            k: v
            for k, v in self.__dict__.items()
            if not k.startswith("_") and not callable(v)
        }

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply transform to input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, optionally with named dimensions.

        Returns
        -------
        torch.Tensor
            Transformed tensor.

        """
        # Validate input
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(x).__name__}")

        # Check if dimension exists (if tensor has names)
        if x.names[0] is not None and self.dim not in x.names:
            raise ValueError(
                f"Dimension '{self.dim}' not found in tensor with names {x.names}",
            )

        try:
            result = self._apply(x)

            # Preserve metadata
            if hasattr(x, "fs"):
                result.fs = x.fs
                result.sampling_frequency = x.fs
            if hasattr(x, "grid_layouts"):
                result.grid_layouts = x.grid_layouts

            logger.debug(f"{self.name}: {x.shape} -> {result.shape}")
            return result

        except Exception as e:
            raise TensorTransformError(
                transform=self,
                input_shape=x.shape,
                input_names=x.names,
                original_error=e,
            ) from e

    @abstractmethod
    def _apply(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the transform. Implement in subclasses."""
        raise NotImplementedError

    def __repr__(self) -> str:
        params_str = ", ".join(f"{k}={v!r}" for k, v in self.params.items())
        return f"{self.__class__.__name__}({params_str})"


class TensorTransformError(Exception):
    """Error during transform with debugging context."""

    def __init__(
        self,
        transform: TensorTransform,
        input_shape: tuple[int, ...],
        input_names: tuple[str | None, ...],
        original_error: Exception,
    ):
        self.transform = transform
        self.input_shape = input_shape
        self.input_names = input_names
        self.original_error = original_error

        message = (
            f"\n\nTransform failed: {transform.name}\n"
            f"\tInput shape: {input_shape}\n"
            f"\tInput names: {input_names}\n"
            f"\tParameters: {transform.params}\n"
            f"\tError: {type(original_error).__name__}: {original_error}\n"
        )
        super().__init__(message)


# Utility functions for working with named tensors


def get_dim_index(x: torch.Tensor, dim: str) -> int:
    """Get axis index for a named dimension.

    Parameters
    ----------
    x : torch.Tensor
        Tensor with named dimensions.
    dim : str
        Dimension name.

    Returns
    -------
    int
        Axis index.

    """
    if x.names[0] is None:
        # Unnamed tensor - use convention
        dim_map = {"batch": 0, "representation": 1, "channel": -2, "time": -1}
        return dim_map.get(dim, -1)

    try:
        return x.names.index(dim)
    except ValueError:
        raise ValueError(f"Dimension '{dim}' not in {x.names}")


def align_tensors(*tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Align multiple named tensors for broadcasting.

    Parameters
    ----------
    *tensors : torch.Tensor
        Named tensors to align.

    Returns
    -------
    tuple[torch.Tensor, ...]
        Aligned tensors.

    """
    return torch.align_tensors(*tensors)
