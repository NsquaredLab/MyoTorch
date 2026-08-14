"""GPU-accelerated generic transforms using PyTorch.

Array manipulation transforms that work with named tensors.

Example:
-------
>>> import torch
>>> from myotorch.transforms.tensor import Reshape, Index, Flatten
>>>
>>> x = torch.randn(64, 2048, names=('channel', 'time'))
>>> y = Reshape((8, 8, 2048), names=('row', 'col', 'time'))(x)

"""

from __future__ import annotations

from collections.abc import Callable

import torch

from myotorch.transforms.base import TensorTransform, get_dim_index


class Reshape(TensorTransform):
    """Reshape tensor with new dimension names.

    Parameters
    ----------
    shape : tuple[int, ...]
        New shape (-1 allowed for one dimension).
    names : tuple[str, ...] | None
        New dimension names.

    Examples
    --------
    >>> x = torch.randn(64, 2048, names=('channel', 'time'))
    >>> reshape = Reshape((8, 8, 2048), names=('row', 'col', 'time'))
    >>> y = reshape(x)  # Shape: (8, 8, 2048)

    """

    def __init__(
        self,
        shape: tuple[int, ...],
        names: tuple[str, ...] | None = None,
        **kwargs,
    ):
        super().__init__(dim="time", **kwargs)
        self.shape = shape
        self.names = names

    def _apply(self, x: torch.Tensor) -> torch.Tensor:
        x = x.rename(None)
        result = x.reshape(self.shape)

        if self.names is not None:
            result = result.rename(*self.names)

        return result


class Index(TensorTransform):
    """Index/slice along a dimension.

    Parameters
    ----------
    indices : int | slice | list[int]
        Indices to select.
    dim : str
        Dimension to index.

    Examples
    --------
    >>> x = torch.randn(64, 2048, names=('channel', 'time'))
    >>> # Select first 10 channels
    >>> index = Index(slice(0, 10), dim='channel')
    >>> y = index(x)  # Shape: (10, 2048)

    """

    def __init__(
        self,
        indices: int | slice | list[int],
        dim: str = "time",
        **kwargs,
    ):
        super().__init__(dim=dim, **kwargs)
        self.indices = indices

    def _apply(self, x: torch.Tensor) -> torch.Tensor:
        dim_idx = get_dim_index(x, self.dim)
        names = x.names

        x = x.rename(None)

        # Build index tuple
        index = [slice(None)] * x.ndim
        index[dim_idx] = self.indices

        result = x[tuple(index)]

        if names[0] is not None and result.ndim == len(names):
            result = result.rename(*names)
        elif names[0] is not None:
            # Dimension was squeezed
            new_names = list(names)
            if isinstance(self.indices, int):
                new_names.pop(dim_idx)
            result = result.rename(*new_names)

        return result


class Flatten(TensorTransform):
    """Flatten dimensions of a tensor.

    Parameters
    ----------
    start_dim : int
        First dimension to flatten.
    end_dim : int
        Last dimension to flatten.

    Examples
    --------
    >>> x = torch.randn(8, 8, 2048, names=('row', 'col', 'time'))
    >>> flatten = Flatten(start_dim=0, end_dim=1)
    >>> y = flatten(x)  # Shape: (64, 2048)

    """

    def __init__(
        self,
        start_dim: int = 0,
        end_dim: int = -1,
        **kwargs,
    ):
        super().__init__(dim="time", **kwargs)
        self.start_dim = start_dim
        self.end_dim = end_dim

    def _apply(self, x: torch.Tensor) -> torch.Tensor:
        x = x.rename(None)
        result = torch.flatten(x, start_dim=self.start_dim, end_dim=self.end_dim)
        return result


class Squeeze(TensorTransform):
    """Remove dimensions of size 1.

    Parameters
    ----------
    dim : int | None
        Specific dimension to squeeze, or None for all.

    """

    def __init__(self, dim: int | None = None, **kwargs):
        super().__init__(**kwargs)
        self.squeeze_dim = dim

    def _apply(self, x: torch.Tensor) -> torch.Tensor:
        x = x.rename(None)
        if self.squeeze_dim is not None:
            return x.squeeze(self.squeeze_dim)
        return x.squeeze()


class Unsqueeze(TensorTransform):
    """Add a dimension of size 1.

    Parameters
    ----------
    dim : int
        Position to insert new dimension.
    name : str | None
        Name for the new dimension.

    """

    def __init__(self, dim: int, name: str | None = None, **kwargs):
        super().__init__(**kwargs)
        self.unsqueeze_dim = dim
        self.new_name = name

    def _apply(self, x: torch.Tensor) -> torch.Tensor:
        names = list(x.names) if x.names[0] is not None else None
        x = x.rename(None)

        result = x.unsqueeze(self.unsqueeze_dim)

        if names is not None and self.new_name is not None:
            names.insert(self.unsqueeze_dim, self.new_name)
            result = result.rename(*names)

        return result


class Transpose(TensorTransform):
    """Transpose/permute dimensions.

    Parameters
    ----------
    dims : tuple[int, ...] | tuple[str, ...]
        New dimension order (by index or name).

    Examples
    --------
    >>> x = torch.randn(64, 2048, names=('channel', 'time'))
    >>> transpose = Transpose(('time', 'channel'))
    >>> y = transpose(x)  # Shape: (2048, 64)

    """

    def __init__(self, dims: tuple, **kwargs):
        super().__init__(**kwargs)
        self.dims = dims

    def _apply(self, x: torch.Tensor) -> torch.Tensor:
        names = x.names

        # Convert string dims to indices
        if all(isinstance(d, str) for d in self.dims):
            if names[0] is None:
                raise ValueError("Cannot use string dims on unnamed tensor")
            perm = [names.index(d) for d in self.dims]
        else:
            perm = list(self.dims)

        x = x.rename(None)
        result = x.permute(*perm)

        if names[0] is not None:
            new_names = [names[i] for i in perm]
            result = result.rename(*new_names)

        return result


class Mean(TensorTransform):
    """Compute mean along a dimension.

    Parameters
    ----------
    dim : str
        Dimension to reduce.
    keepdim : bool
        Whether to keep the reduced dimension.

    """

    def __init__(self, dim: str = "time", keepdim: bool = False, **kwargs):
        super().__init__(dim=dim, **kwargs)
        self.keepdim = keepdim

    def _apply(self, x: torch.Tensor) -> torch.Tensor:
        dim_idx = get_dim_index(x, self.dim)
        names = x.names

        x = x.rename(None)
        result = x.mean(dim=dim_idx, keepdim=self.keepdim)

        if names[0] is not None:
            if self.keepdim:
                result = result.rename(*names)
            else:
                new_names = [n for i, n in enumerate(names) if i != dim_idx]
                if new_names:
                    result = result.rename(*new_names)

        return result


class Sum(TensorTransform):
    """Compute sum along a dimension.

    Parameters
    ----------
    dim : str
        Dimension to reduce.
    keepdim : bool
        Whether to keep the reduced dimension.

    """

    def __init__(self, dim: str = "time", keepdim: bool = False, **kwargs):
        super().__init__(dim=dim, **kwargs)
        self.keepdim = keepdim

    def _apply(self, x: torch.Tensor) -> torch.Tensor:
        dim_idx = get_dim_index(x, self.dim)
        names = x.names

        x = x.rename(None)
        result = x.sum(dim=dim_idx, keepdim=self.keepdim)

        if names[0] is not None:
            if self.keepdim:
                result = result.rename(*names)
            else:
                new_names = [n for i, n in enumerate(names) if i != dim_idx]
                if new_names:
                    result = result.rename(*new_names)

        return result


class Stack(TensorTransform):
    """Stack multiple tensors along a new dimension.

    This is a container transform that holds multiple paths.

    Parameters
    ----------
    transforms : dict[str, Callable]
        Named transforms to apply and stack.
    dim : str
        Name for the new stacking dimension.

    Examples
    --------
    >>> from myotorch.transforms.tensor import RMS, MAV, Stack
    >>> stack = Stack({
    ...     'rms': RMS(window_size=200),
    ...     'mav': MAV(window_size=200),
    ... }, dim='feature')
    >>> y = stack(x)  # Shape: (2, channel, time_windows)

    """

    def __init__(
        self,
        transforms: dict[str, Callable],
        dim: str = "representation",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.transforms = transforms
        self.stack_dim = dim

    def _apply(self, x: torch.Tensor) -> torch.Tensor:
        results = []
        first_names = None
        for name, transform in self.transforms.items():
            result = transform(x)
            if first_names is None:
                first_names = result.names
            results.append(result.rename(None))

        stacked = torch.stack(results, dim=0)

        # Add dimension names
        if first_names is not None and first_names[0] is not None:
            names = (self.stack_dim,) + tuple(first_names)
        else:
            names = (self.stack_dim,) + tuple(
                f"dim_{i}" for i in range(results[0].ndim)
            )

        return stacked.rename(*names)


class Concat(TensorTransform):
    """Concatenate multiple tensors along an existing dimension.

    Parameters
    ----------
    transforms : dict[str, Callable]
        Named transforms to apply and concatenate.
    dim : str
        Dimension to concatenate along.

    Examples
    --------
    >>> from myotorch.transforms.tensor import RMS, MAV, Concat
    >>> concat = Concat({
    ...     'rms': RMS(window_size=200),
    ...     'mav': MAV(window_size=200),
    ... }, dim='channel')
    >>> y = concat(x)  # Concatenated along channel dimension

    """

    def __init__(
        self,
        transforms: dict[str, Callable],
        dim: str = "channel",
        **kwargs,
    ):
        super().__init__(dim=dim, **kwargs)
        self.transforms = transforms

    def _apply(self, x: torch.Tensor) -> torch.Tensor:
        results = []
        first_names = None
        for name, transform in self.transforms.items():
            result = transform(x)
            if first_names is None:
                first_names = result.names
            results.append(result.rename(None))

        # Get dim index from first result's original names
        if first_names is not None and first_names[0] is not None:
            dim_idx = first_names.index(self.dim)
        else:
            dim_idx = -2

        concatenated = torch.cat(results, dim=dim_idx)

        if first_names is not None and first_names[0] is not None:
            concatenated = concatenated.rename(*first_names)

        return concatenated


class Lambda(TensorTransform):
    """Apply a custom function.

    Parameters
    ----------
    func : Callable
        Function to apply.

    Examples
    --------
    >>> transform = Lambda(lambda x: x ** 2)
    >>> y = transform(x)

    """

    def __init__(self, func: Callable, **kwargs):
        super().__init__(**kwargs)
        self.func = func

    def _apply(self, x: torch.Tensor) -> torch.Tensor:
        return self.func(x)


class Identity(TensorTransform):
    """Identity transform (returns input unchanged)."""

    def _apply(self, x: torch.Tensor) -> torch.Tensor:
        return x


class Repeat(TensorTransform):
    """Repeat tensor along a dimension.

    Parameters
    ----------
    repeats : int
        Number of repetitions.
    dim : str
        Dimension to repeat along.

    """

    def __init__(self, repeats: int, dim: str = "channel", **kwargs):
        super().__init__(dim=dim, **kwargs)
        self.repeats = repeats

    def _apply(self, x: torch.Tensor) -> torch.Tensor:
        dim_idx = get_dim_index(x, self.dim)
        names = x.names

        x = x.rename(None)

        # Build repeat tuple
        repeat_tuple = [1] * x.ndim
        repeat_tuple[dim_idx] = self.repeats

        result = x.repeat(*repeat_tuple)

        if names[0] is not None:
            result = result.rename(*names)

        return result


class Pad(TensorTransform):
    """Pad tensor along a dimension.

    Parameters
    ----------
    padding : tuple[int, int]
        Padding (before, after) along the dimension.
    dim : str
        Dimension to pad.
    mode : str
        Padding mode: 'constant', 'reflect', 'replicate', 'circular'.
    value : float
        Fill value for constant padding.

    """

    def __init__(
        self,
        padding: tuple[int, int],
        dim: str = "time",
        mode: str = "constant",
        value: float = 0.0,
        **kwargs,
    ):
        super().__init__(dim=dim, **kwargs)
        self.padding = padding
        self.mode = mode
        self.value = value

    def _apply(self, x: torch.Tensor) -> torch.Tensor:
        dim_idx = get_dim_index(x, self.dim)
        names = x.names

        x = x.rename(None)

        # F.pad expects padding in reverse order: (last_dim, ..., first_dim)
        # Each dim needs (left, right) padding
        ndim = x.ndim
        pad_list = [0] * (2 * ndim)
        # dim_idx from the end
        idx_from_end = ndim - 1 - dim_idx
        pad_list[2 * idx_from_end] = self.padding[0]  # left
        pad_list[2 * idx_from_end + 1] = self.padding[1]  # right

        result = torch.nn.functional.pad(x, pad_list, mode=self.mode, value=self.value)

        if names[0] is not None:
            result = result.rename(*names)

        return result
