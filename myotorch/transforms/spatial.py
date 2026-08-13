"""GPU-accelerated spatial transforms for grid-based EMG.

Spatial filters operate on electrode grids using 2D convolutions.
All transforms work with named tensors on any device.

Example:
-------
>>> import torch
>>> from myotorch.transforms import NDD, LSD, Pipeline
>>>
>>> # Create EMG tensor with grid info
>>> emg = myotorch.emg_tensor(data, grid_layouts=[grid1, grid2])
>>>
>>> # Apply spatial filter
>>> ndd = NDD(grids="all")
>>> filtered = ndd(emg)

"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from myotorch.transforms.base import TensorTransform

# Standard spatial filter kernels
SPATIAL_KERNELS = {
    # Normal Double Differential (Laplacian)
    "NDD": torch.tensor(
        [
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0],
        ],
        dtype=torch.float32,
    )
    / 4,
    # Longitudinal Single Differential (vertical)
    "LSD": torch.tensor(
        [
            [0, -1, 0],
            [0, 1, 0],
            [0, 0, 0],
        ],
        dtype=torch.float32,
    ),
    # Transverse Single Differential (horizontal)
    "TSD": torch.tensor(
        [
            [0, 0, 0],
            [-1, 1, 0],
            [0, 0, 0],
        ],
        dtype=torch.float32,
    ),
    # Inverse Binomial 2nd order
    "IB2": torch.tensor(
        [
            [1, -2, 1],
            [-2, 4, -2],
            [1, -2, 1],
        ],
        dtype=torch.float32,
    )
    / 4,
}


def _channels_to_grid(
    data: torch.Tensor,
    grid_layout: np.ndarray,
) -> torch.Tensor:
    """Reshape channel data to grid layout.

    Parameters
    ----------
    data : torch.Tensor
        Data with shape (..., n_channels, time).
    grid_layout : np.ndarray
        2D array mapping (row, col) to channel index. -1 for gaps.

    Returns
    -------
    torch.Tensor
        Data with shape (..., rows, cols, time). Gaps filled with 0.

    """
    rows, cols = grid_layout.shape
    time_dim = data.shape[-1]
    batch_shape = data.shape[:-2]

    # Create output tensor
    out = torch.zeros(
        *batch_shape,
        rows,
        cols,
        time_dim,
        dtype=data.dtype,
        device=data.device,
    )

    # Fill in valid electrodes using relative indexing
    ch_idx = 0
    for r in range(rows):
        for c in range(cols):
            if grid_layout[r, c] >= 0:
                out[..., r, c, :] = data[..., ch_idx, :]
                ch_idx += 1

    return out


def _grid_to_channels(
    data: torch.Tensor,
    grid_layout: np.ndarray,
) -> torch.Tensor:
    """Reshape grid data back to channels.

    Parameters
    ----------
    data : torch.Tensor
        Data with shape (..., rows, cols, time).
    grid_layout : np.ndarray
        2D array mapping (row, col) to channel index. -1 for gaps.

    Returns
    -------
    torch.Tensor
        Data with shape (..., n_channels, time).

    """
    rows, cols = grid_layout.shape
    time_dim = data.shape[-1]
    batch_shape = data.shape[:-3]

    # Count valid channels
    n_channels = np.sum(grid_layout >= 0)

    # Create output
    out = torch.zeros(
        *batch_shape,
        n_channels,
        time_dim,
        dtype=data.dtype,
        device=data.device,
    )

    # Extract valid electrodes
    ch_idx = 0
    for r in range(rows):
        for c in range(cols):
            if grid_layout[r, c] >= 0:
                out[..., ch_idx, :] = data[..., r, c, :]
                ch_idx += 1

    return out


class SpatialFilter(TensorTransform):
    """Apply spatial filtering using grid layouts.

    Spatial filters use 2D convolution on electrode grids. Grid layouts
    must be stored as a tensor attribute (via myotorch.emg_tensor).

    Parameters
    ----------
    kernel : str | torch.Tensor
        Filter kernel. Either a name ("NDD", "LSD", "TSD", "IB2") or
        a custom 2D tensor.
    grids : str | list[int]
        Which grids to filter. "all" for all grids, or list of indices.
    dim : str
        Channel dimension name.

    Examples
    --------
    >>> import myotorch
    >>> emg = myotorch.emg_tensor(data, grid_layouts=[grid1, grid2])
    >>> ndd = SpatialFilter("NDD", grids="all")
    >>> filtered = ndd(emg)

    """

    def __init__(
        self,
        kernel: str | torch.Tensor = "NDD",
        grids: str | list[int] = "all",
        dim: str = "channel",
        **kwargs,
    ):
        super().__init__(dim=dim, **kwargs)

        if isinstance(kernel, str):
            if kernel not in SPATIAL_KERNELS:
                raise ValueError(
                    f"Unknown kernel '{kernel}'. "
                    f"Available: {list(SPATIAL_KERNELS.keys())}",
                )
            self.kernel = SPATIAL_KERNELS[kernel]
            self.kernel_name = kernel
        else:
            self.kernel = kernel
            self.kernel_name = "custom"

        self.grids = grids

    def _apply(self, x: torch.Tensor) -> torch.Tensor:
        # Get grid layouts from tensor attributes
        if not hasattr(x, "grid_layouts"):
            raise ValueError(
                "Tensor missing 'grid_layouts' attribute. "
                "Create with myotorch.emg_tensor(data, grid_layouts=[...]):\n\n"
                "\timport myotorch\n"
                "\temg = myotorch.emg_tensor(data, grid_layouts=[grid1, grid2])\n",
            )

        grid_layouts = x.grid_layouts
        names = x.names

        # Determine which grids to process
        if self.grids == "all":
            grid_indices = list(range(len(grid_layouts)))
        else:
            grid_indices = self.grids

        x = x.rename(None)

        # Move kernel to same device
        kernel = self.kernel.to(device=x.device, dtype=x.dtype)

        # Process each grid
        results = []
        channel_offset = 0

        for grid_idx, grid_layout in enumerate(grid_layouts):
            n_channels = np.sum(grid_layout >= 0)

            # Extract this grid's channels
            grid_data = x[..., channel_offset : channel_offset + n_channels, :]

            if grid_idx in grid_indices:
                # Reshape to grid
                grid_shaped = _channels_to_grid(grid_data, grid_layout)

                # Apply 2D convolution
                # conv2d expects (N, C, H, W) - we have (..., H, W, T)
                # Reshape: move time to batch, grid to spatial
                original_shape = grid_shaped.shape
                time_dim = original_shape[-1]
                batch_shape = original_shape[:-3]

                # Flatten batch dims and time into batch
                grid_shaped = grid_shaped.reshape(-1, *original_shape[-3:])
                # (batch*time_batch, rows, cols, time) -> need (N, C, H, W)
                # Actually we want to convolve over (rows, cols) for each time point

                # Reshape to (batch, time, rows, cols) then to (batch*time, 1, rows, cols)
                grid_shaped = grid_shaped.permute(0, 3, 1, 2)  # (B, T, R, C)
                B, T, R, C = grid_shaped.shape
                grid_shaped = grid_shaped.reshape(B * T, 1, R, C)

                # Prepare kernel for conv2d: (out_channels, in_channels, H, W)
                kernel_4d = kernel.unsqueeze(0).unsqueeze(0)

                # Apply convolution with zero padding
                filtered = F.conv2d(grid_shaped, kernel_4d, padding=1)

                # Reshape back
                filtered = filtered.reshape(B, T, R, C)
                filtered = filtered.permute(0, 2, 3, 1)  # (B, R, C, T)
                filtered = filtered.reshape(*batch_shape, R, C, time_dim)

                # Back to channels
                filtered_channels = _grid_to_channels(filtered, grid_layout)
                results.append(filtered_channels)
            else:
                # Keep unfiltered
                results.append(grid_data)

            channel_offset += n_channels

        # Concatenate all grids
        result = torch.cat(results, dim=-2)

        if names[0] is not None:
            result = result.rename(*names)

        return result


class NDD(SpatialFilter):
    """Normal Double Differential (Laplacian) filter.

    Enhances localized activity by subtracting the average of 4 neighbors.

    Parameters
    ----------
    grids : str | list[int]
        Which grids to filter. "all" or list of indices.

    """

    def __init__(self, grids: str | list[int] = "all", **kwargs):
        super().__init__(kernel="NDD", grids=grids, **kwargs)


class LSD(SpatialFilter):
    """Longitudinal Single Differential filter.

    Computes vertical (along muscle fiber) differences.

    Parameters
    ----------
    grids : str | list[int]
        Which grids to filter. "all" or list of indices.

    """

    def __init__(self, grids: str | list[int] = "all", **kwargs):
        super().__init__(kernel="LSD", grids=grids, **kwargs)


class TSD(SpatialFilter):
    """Transverse Single Differential filter.

    Computes horizontal (across muscle fiber) differences.

    Parameters
    ----------
    grids : str | list[int]
        Which grids to filter. "all" or list of indices.

    """

    def __init__(self, grids: str | list[int] = "all", **kwargs):
        super().__init__(kernel="TSD", grids=grids, **kwargs)


class IB2(SpatialFilter):
    """Inverse Binomial 2nd order filter.

    2D high-pass filter using binomial weighting.

    Parameters
    ----------
    grids : str | list[int]
        Which grids to filter. "all" or list of indices.

    """

    def __init__(self, grids: str | list[int] = "all", **kwargs):
        super().__init__(kernel="IB2", grids=grids, **kwargs)
