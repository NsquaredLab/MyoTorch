"""Supervised learning dataset.

This module provides SupervisedDataset, which extends WindowedDataset
to implement the supervised learning paradigm where inputs are mapped
to targets (e.g., EMG signals → kinematics).

Example:
-------
>>> from myotorch.datasets import SupervisedDataset
>>> from myotorch.transforms import Compose, ZScore, RMS
>>>
>>> ds = SupervisedDataset(
...     "data.zip",
...     split="training",
...     inputs=["emg"],
...     targets=["kinematics"],
...     transform=Compose([ZScore(), RMS(200)]),
...     target_transform=Mean(dim="time"),
...     device="cuda",
... )
>>> inputs, targets = ds[0]
>>> inputs["emg"].shape  # (channels, time)
>>> targets["kinematics"].shape  # (joints,)

"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch

from myotorch.datasets.base import WindowedDataset


class SupervisedDataset(WindowedDataset):
    """Dataset for supervised learning with inputs and targets.

    Extends WindowedDataset to split modalities into inputs and targets,
    with separate transforms for each.

    Parameters
    ----------
    zarr_path : Path | str
        Path to the Zarr dataset.
    split : str
        Dataset split ('training', 'validation', 'testing').
    inputs : Sequence[str]
        Modality names to use as model inputs.
    targets : Sequence[str]
        Modality names to use as model targets.
    transform : Callable | None
        Transform to apply to input data (only when device is set).
    target_transform : Callable | None
        Transform to apply to target data (only when device is set).
    window_size : int
        Number of samples per window.
    window_stride : int | None
        Stride between windows. If None, uses random positions.
    n_windows : int | None
        Number of windows per epoch. Required if window_stride is None.
    seed : int | None
        Random seed for reproducible window positions.
    device : torch.device | str | None
        Output device ('cpu', 'cuda', or None for numpy).
    dtype : torch.dtype
        Data type for tensors.
    cache_in_ram : bool
        Cache entire split in RAM.

    Examples
    --------
    >>> # Supervised learning: EMG → kinematics
    >>> ds = SupervisedDataset(
    ...     "data.zip",
    ...     inputs=["emg"],
    ...     targets=["kinematics"],
    ...     window_size=200,
    ...     n_windows=10000,
    ...     device="cuda",
    ... )
    >>> inputs, targets = ds[0]
    >>> inputs["emg"].device  # cuda:0

    """

    def __init__(
        self,
        zarr_path: Path | str,
        split: str = "training",
        inputs: Sequence[str] = ("emg",),
        targets: Sequence[str] = ("kinematics",),
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        window_size: int = 200,
        window_stride: int | None = None,
        n_windows: int | None = None,
        seed: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
        cache_in_ram: bool = True,
    ):
        # Combine inputs and targets for base class
        all_modalities = list(set(inputs) | set(targets))

        super().__init__(
            zarr_path=zarr_path,
            split=split,
            modalities=all_modalities,
            window_size=window_size,
            window_stride=window_stride,
            n_windows=n_windows,
            seed=seed,
            device=device,
            dtype=dtype,
            cache_in_ram=cache_in_ram,
        )

        self.inputs = list(inputs)
        self.targets = list(targets)
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(
        self,
        idx: int,
    ) -> tuple[
        dict[str, torch.Tensor | np.ndarray], dict[str, torch.Tensor | np.ndarray]
    ]:
        """Load windows and split into inputs/targets.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        tuple[dict, dict]
            (inputs, targets) where each is a dict mapping modality names to data.

        """
        # Get all modalities from base class
        all_data = super().__getitem__(idx)

        # Split into inputs
        inputs_dict = {}
        for mod in self.inputs:
            if mod in all_data:
                data = all_data[mod]
                # Apply transform only to tensors (when device is set)
                if self.transform is not None and isinstance(data, torch.Tensor):
                    data = self.transform(data)
                inputs_dict[mod] = data

        # Split into targets
        targets_dict = {}
        for mod in self.targets:
            if mod in all_data:
                data = all_data[mod]
                # Apply target transform only to tensors
                if self.target_transform is not None and isinstance(data, torch.Tensor):
                    data = self.target_transform(data)
                targets_dict[mod] = data

        return inputs_dict, targets_dict
