"""Dataset utilities for MyoTorch.

Architecture
------------
This module uses a layered architecture for flexibility:

**Base Layer (Infrastructure)**
    WindowedDataset : Handles zarr I/O, windowing, caching, device management.
    Returns all modalities as a dict.

**Paradigm Layer (Learning Paradigms)**
    SupervisedDataset : Supervised learning (inputs -> targets).
    Returns (inputs_dict, targets_dict) tuple.

    Future: ContrastiveDataset, MaskedDataset, etc.

**Integration Layer**
    DataModule : Lightning DataModule for training integration.

**Storage Layer**
    DatasetCreator : Creates zarr datasets from multi-modal data.
    Modality : Configuration for a data modality.

**Presets**
    Pre-configured transforms for published papers (EMBC 2022, etc.).

Example:
-------
>>> from myotorch.datasets import DatasetCreator, DataModule, Modality
>>> from myotorch.datasets.presets import embc_train_transform
>>>
>>> # Create dataset
>>> creator = DatasetCreator(
...     modalities={
...         "emg": Modality(path="emg.pkl", dims=("channel", "time")),
...         "kinematics": Modality(path="kin.pkl", dims=("joint", "time")),
...     },
...     sampling_frequency=2048.0,
...     save_path="data.zip",
... )
>>> creator.create()
>>>
>>> # Load for training
>>> dm = DataModule(
...     "data.zip",
...     inputs=["emg"],
...     targets=["kinematics"],
...     window_size=200,
...     n_windows_per_epoch=10000,
...     train_transform=embc_train_transform(),
...     device="cuda",
... )
>>> dm.setup("fit")
>>> inputs, targets = next(iter(dm.train_dataloader()))

"""

from __future__ import annotations

from typing import TYPE_CHECKING

# Light imports - these don't pull in torch/lightning
from myotorch.datasets.creator import DatasetCreator
from myotorch.datasets.modality import Modality
from myotorch.datasets.utils import DatasetFormatter, DataSplitter

if TYPE_CHECKING:
    # For type checkers only - not imported at runtime
    from myotorch.datasets.base import WindowedDataset
    from myotorch.datasets.datamodule import DataModule, collate_supervised
    from myotorch.datasets.paradigms import SupervisedDataset
    from myotorch.datasets.presets import (
        EMBCConfig,
        embc_eval_transform,
        embc_kinematics_transform,
        embc_target_transform,
        embc_train_transform,
    )

__all__ = [
    # Base
    "WindowedDataset",
    # Paradigms
    "SupervisedDataset",
    # Integration
    "DataModule",
    "collate_supervised",
    # Storage
    "DatasetCreator",
    "Modality",
    # Utilities
    "DataSplitter",
    "DatasetFormatter",
    # Presets
    "EMBCConfig",
    "embc_kinematics_transform",
    "embc_train_transform",
    "embc_eval_transform",
    "embc_target_transform",
]

# Lazy imports for heavy modules (torch, lightning)
# These are only loaded when actually accessed
_LAZY_IMPORTS = {
    # Base layer (imports torch)
    "WindowedDataset": ("myotorch.datasets.base", "WindowedDataset"),
    # Paradigms (imports torch)
    "SupervisedDataset": ("myotorch.datasets.paradigms", "SupervisedDataset"),
    # Integration (imports lightning)
    "DataModule": ("myotorch.datasets.datamodule", "DataModule"),
    "collate_supervised": ("myotorch.datasets.datamodule", "collate_supervised"),
    # Presets (imports transforms which may import torch)
    "EMBCConfig": ("myotorch.datasets.presets", "EMBCConfig"),
    "embc_train_transform": ("myotorch.datasets.presets", "embc_train_transform"),
    "embc_eval_transform": ("myotorch.datasets.presets", "embc_eval_transform"),
    "embc_target_transform": ("myotorch.datasets.presets", "embc_target_transform"),
    "embc_kinematics_transform": (
        "myotorch.datasets.presets",
        "embc_kinematics_transform",
    ),
}


def __getattr__(name: str):
    """Lazy import for heavy dependencies."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        import importlib

        module = importlib.import_module(module_path)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
