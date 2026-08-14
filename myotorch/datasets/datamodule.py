"""PyTorch Lightning DataModule for MyoTorch datasets.

This module provides the DataModule class that integrates datasets
with PyTorch Lightning's training loop.

Example:
-------
>>> from myotorch.datasets import DataModule
>>> from myotorch.transforms import Compose, ZScore, RMS
>>>
>>> dm = DataModule(
...     "data.zip",
...     inputs=["emg"],
...     targets=["kinematics"],
...     window_size=200,
...     n_windows_per_epoch=10000,
...     device="cuda",
...     train_transform=Compose([ZScore(), RMS(50)]),
... )
>>> dm.setup("fit")
>>> train_loader = dm.train_dataloader()

"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader

from myotorch.datasets.paradigms import SupervisedDataset


def _stack_modalities(
    samples: list[dict[str, torch.Tensor | np.ndarray]],
) -> dict[str, torch.Tensor | np.ndarray]:
    """Stack samples for each modality, stripping named tensor names."""
    first = next(iter(samples[0].values()))
    is_numpy = isinstance(first, np.ndarray)

    result = {}
    for key in samples[0]:
        items = [s[key] for s in samples]
        if is_numpy:
            result[key] = np.stack(items)
        else:
            # Strip named tensor names - models don't support them
            items_unnamed = [
                t.rename(None) if t.names[0] is not None else t for t in items
            ]
            result[key] = torch.stack(items_unnamed)
    return result


def collate_supervised(
    batch: list[tuple[dict, dict]],
) -> tuple[
    dict[str, torch.Tensor] | torch.Tensor, dict[str, torch.Tensor] | torch.Tensor
]:
    """Collate function for supervised datasets.

    Handles both numpy arrays and tensors.
    Strips named tensor names since most models don't support them.

    Parameters
    ----------
    batch : list[tuple[dict, dict]]
        List of (inputs, targets) tuples from dataset.

    Returns
    -------
    tuple
        Batched (inputs, targets). If single modality, returns tensor directly.

    """
    inputs_list = [b[0] for b in batch]
    targets_list = [b[1] for b in batch]

    inputs = _stack_modalities(inputs_list)
    targets = _stack_modalities(targets_list)

    # Return directly if single input/target
    if len(inputs) == 1 and len(targets) == 1:
        return next(iter(inputs.values())), next(iter(targets.values()))

    return inputs, targets


class DataModule(L.LightningDataModule):
    """Lightning DataModule for supervised learning.

    Wraps SupervisedDataset instances for train/val/test splits
    and provides DataLoaders.

    Parameters
    ----------
    data_path : Path | str
        Path to the Zarr dataset.
    inputs : Sequence[str]
        Modality names to use as model inputs.
    targets : Sequence[str]
        Modality names to use as model targets.
    batch_size : int
        Batch size for all dataloaders.
    window_size : int
        Window size in samples.
    window_stride : int | None
        Window stride for validation/test.
    n_windows_per_epoch : int | None
        Number of random windows per training epoch.
    num_workers : int
        Number of dataloader workers.
    train_transform : Callable | None
        Transform for training inputs.
    val_transform : Callable | None
        Transform for validation inputs.
    test_transform : Callable | None
        Transform for test inputs.
    target_transform : Callable | None
        Transform for targets.
    pin_memory : bool
        Pin memory for faster GPU transfer.
    persistent_workers : bool
        Keep workers alive between epochs.
    device : torch.device | str | None
        Output device ('cpu', 'cuda', or None for numpy).
    dtype : torch.dtype
        Data type for tensors.
    cache_in_ram : bool
        Cache entire split in RAM.

    Examples
    --------
    >>> dm = DataModule(
    ...     "data.zip",
    ...     inputs=["emg"],
    ...     targets=["kinematics"],
    ...     window_size=200,
    ...     n_windows_per_epoch=10000,
    ...     device="cuda",
    ... )
    >>> dm.setup("fit")
    >>> for inputs, targets in dm.train_dataloader():
    ...     # inputs: Tensor of shape (batch, channels, time)
    ...     # targets: Tensor of shape (batch, joints)
    ...     pass

    """

    def __init__(
        self,
        data_path: Path | str,
        inputs: Sequence[str] = ("emg",),
        targets: Sequence[str] = ("kinematics",),
        batch_size: int = 32,
        window_size: int = 200,
        window_stride: int | None = None,
        n_windows_per_epoch: int | None = None,
        num_workers: int = 4,
        train_transform: Callable | None = None,
        val_transform: Callable | None = None,
        test_transform: Callable | None = None,
        target_transform: Callable | None = None,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
        cache_in_ram: bool = True,
    ):
        super().__init__()
        self.data_path = Path(data_path)
        self.inputs = list(inputs)
        self.targets = list(targets)
        self.batch_size = batch_size
        self.window_size = window_size
        self.window_stride = window_stride
        self.n_windows_per_epoch = n_windows_per_epoch
        self.num_workers = num_workers
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform or val_transform
        self.target_transform = target_transform
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.device = device
        self.dtype = dtype
        self.cache_in_ram = cache_in_ram

        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.data_path}")

        if n_windows_per_epoch is None and window_stride is None:
            raise ValueError("Need n_windows_per_epoch or window_stride")

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: str | None = None) -> None:
        """Setup datasets for each stage."""
        if stage == "fit" or stage is None:
            self.train_dataset = SupervisedDataset(
                self.data_path,
                split="training",
                inputs=self.inputs,
                targets=self.targets,
                window_size=self.window_size,
                n_windows=self.n_windows_per_epoch,
                transform=self.train_transform,
                target_transform=self.target_transform,
                device=self.device,
                dtype=self.dtype,
                cache_in_ram=self.cache_in_ram,
            )
            self.val_dataset = SupervisedDataset(
                self.data_path,
                split="validation",
                inputs=self.inputs,
                targets=self.targets,
                window_size=self.window_size,
                window_stride=self.window_stride or self.window_size,
                transform=self.val_transform,
                target_transform=self.target_transform,
                device=self.device,
                dtype=self.dtype,
                cache_in_ram=self.cache_in_ram,
            )

            # Pre-load cache in main process before workers are spawned
            # (avoids zarr ZipStore concurrency issues in multiprocessing)
            if self.cache_in_ram and self.num_workers > 0:
                _ = self.train_dataset[0]
                _ = self.val_dataset[0]

        if stage == "test" or stage is None:
            self.test_dataset = SupervisedDataset(
                self.data_path,
                split="testing",
                inputs=self.inputs,
                targets=self.targets,
                window_size=self.window_size,
                window_stride=self.window_stride or self.window_size,
                transform=self.test_transform,
                target_transform=self.target_transform,
                device=self.device,
                dtype=self.dtype,
                cache_in_ram=self.cache_in_ram,
            )

            # Pre-load cache in main process before workers are spawned
            if self.cache_in_ram and self.num_workers > 0:
                _ = self.test_dataset[0]

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory and self.device is None,
            persistent_workers=self.persistent_workers,
            collate_fn=collate_supervised,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory and self.device is None,
            persistent_workers=self.persistent_workers,
            collate_fn=collate_supervised,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory and self.device is None,
            persistent_workers=self.persistent_workers,
            collate_fn=collate_supervised,
        )
