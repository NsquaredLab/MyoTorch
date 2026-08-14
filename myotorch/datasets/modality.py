"""Modality configuration for dataset creation.

This module provides the Modality dataclass for configuring data sources
when creating datasets with DatasetCreator.

Example:
-------
>>> from myotorch.datasets import Modality
>>> from myotorch.transforms import Compose, Flatten, Index
>>>
>>> emg = Modality(
...     path="emg.pkl",
...     dims=("channel", "time"),
... )
>>>
>>> # With preprocessing transform
>>> kinematics = Modality(
...     path="kinematics.pkl",
...     dims=("dof", "time"),
...     transform=Compose([
...         Flatten(0, 1),  # (21, 3, time) -> (63, time)
...         Index(slice(3, None), dim="channel"),  # Remove wrist
...     ]),
... )

"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import torch


@dataclass
class Modality:
    """Configuration for a data modality.

    A modality represents a single data stream (EMG, kinematics, EEG, etc.)
    with its dimensionality and optional preprocessing.

    Parameters
    ----------
    data : np.ndarray | dict[str, np.ndarray] | None
        Data array or dict of arrays per task.
    path : Path | str | None
        Path to pickle file containing data.
    dims : tuple[str, ...]
        Dimension names (last must be 'time').
    attrs : dict
        Optional attributes to store with the data.
    transform : Callable | None
        Transform to apply after loading (receives tensor, returns tensor).

    Examples
    --------
    >>> emg = Modality(
    ...     path="emg.pkl",
    ...     dims=("channel", "time"),
    ... )
    >>> # With preprocessing transform
    >>> from myotorch.transforms import Compose, Flatten, Index
    >>> kinematics = Modality(
    ...     path="kinematics.pkl",
    ...     dims=("dof", "time"),
    ...     transform=Compose([
    ...         Flatten(0, 1),  # (21, 3, time) -> (63, time)
    ...         Index(slice(3, None), dim="channel"),  # Remove wrist -> (60, time)
    ...     ]),
    ... )

    """

    data: np.ndarray | dict[str, np.ndarray] | None = None
    path: Path | str | None = None
    dims: tuple[str, ...] = ("channel", "time")
    attrs: dict = field(default_factory=dict)
    transform: Any = None

    def __post_init__(self):
        if self.path is not None:
            self.path = Path(self.path)
        if self.dims[-1] != "time":
            raise ValueError(f"Last dimension must be 'time', got {self.dims}")
        if self.data is None and (self.path is None or not self.path.exists()):
            raise ValueError("Must provide data or valid path")

    def load(self) -> dict[str, np.ndarray]:
        """Load data from path or return data dict, applying transform if set.

        Returns
        -------
        dict[str, np.ndarray]
            Dict mapping task names to data arrays.

        """
        if self.data is not None:
            if isinstance(self.data, np.ndarray):
                data = {"default": self.data}
            else:
                data = self.data
        else:
            with open(self.path, "rb") as f:
                data = pickle.load(f)

        # Apply transform (converts to tensor, applies transform, back to numpy)
        if self.transform is not None:
            # Lazy import torch only when transform is used
            import torch

            transformed = {}
            for task, arr in data.items():
                tensor = torch.from_numpy(arr.astype(np.float32))
                result = self.transform(tensor)
                # Strip named tensor names before converting to numpy
                if isinstance(result, torch.Tensor) and result.names[0] is not None:
                    result = result.rename(None)
                transformed[task] = result.numpy()
            data = transformed

        return data
