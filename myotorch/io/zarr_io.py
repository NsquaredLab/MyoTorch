"""Zarr I/O abstraction layer for version-agnostic Zarr operations.

This module provides a unified interface for Zarr operations that works with
both Zarr 2 and Zarr 3, making migrations easier.

Uses zarrs (Rust-based codec pipeline) for faster I/O when available.

Migration guide: https://zarr.readthedocs.io/en/latest/user-guide/v3_migration/
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import zarr

# Configure zarr to use the Rust-based zarrs codec pipeline for faster I/O
try:
    import zarrs  # noqa: F401 - import needed to register the codec pipeline

    zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})
    ZARRS_AVAILABLE = True
except ImportError:
    ZARRS_AVAILABLE = False


def get_zarr_version() -> int:
    """Get the major version of the installed Zarr package."""
    version_str = zarr.__version__
    major = int(version_str.split(".")[0])
    return major


ZARR_VERSION = get_zarr_version()


class ZarrIO:
    """Zarr I/O abstraction layer for version-agnostic operations.

    This class centralizes all Zarr operations to make version migrations easier.
    Automatically detects Zarr version and uses appropriate API.

    Parameters
    ----------
    path : Path | str
        Path to the Zarr store.
    mode : str
        File mode ('r', 'w', 'a', 'r+').
    zarr_format : int
        Zarr format version (2 or 3). Default is 3 for Zarr 3.x, 2 for Zarr 2.x.
    silence_warnings : bool
        Whether to silence Zarr-related warnings.

    Examples
    --------
    >>> zio = ZarrIO("data.zarr", mode="w")
    >>> zio.create_group("training")
    >>> zio.add_array("training", "emg", data)
    >>> zio.close()

    """

    def __init__(
        self,
        path: Path | str,
        mode: str = "r",
        zarr_format: int | None = None,
        silence_warnings: bool = True,
    ):
        self.path = Path(path)
        self.mode = mode
        self.silence_warnings = silence_warnings

        # Default format based on installed version
        if zarr_format is None:
            self.zarr_format = 3 if ZARR_VERSION >= 3 else 2
        else:
            self.zarr_format = zarr_format

        if silence_warnings:
            self._silence_zarr_warnings()

        self._root = self._open_store()

    def _silence_zarr_warnings(self) -> None:
        """Silence Zarr-related warnings."""
        warnings.filterwarnings("ignore", category=UserWarning, module="zarr.codecs")
        warnings.filterwarnings("ignore", category=UserWarning, module="zarr.core")
        warnings.filterwarnings("ignore", category=UserWarning, module="zarr")
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="zarr")

    def _open_store(self) -> zarr.Group:
        """Open the Zarr store with version-appropriate settings."""
        if self.mode == "w":
            self.path.mkdir(parents=True, exist_ok=True)

        if ZARR_VERSION >= 3:
            # Zarr 3.x API
            return zarr.open_group(
                store=str(self.path),
                mode=self.mode,
                zarr_format=self.zarr_format,
            )
        # Zarr 2.x API
        return zarr.open(
            str(self.path),
            mode=self.mode,
            zarr_version=self.zarr_format,
        )

    @property
    def root(self) -> zarr.Group:
        """Get the root group of the Zarr store."""
        return self._root

    def create_group(self, name: str) -> zarr.Group:
        """Create a group in the Zarr store.

        Parameters
        ----------
        name : str
            Name of the group to create.

        Returns
        -------
        zarr.Group
            The created group.

        """
        if name in self._root:
            return self._root[name]
        return self._root.create_group(name)

    def get_group(self, name: str) -> zarr.Group:
        """Get an existing group from the Zarr store.

        Parameters
        ----------
        name : str
            Name of the group.

        Returns
        -------
        zarr.Group
            The requested group.

        Raises
        ------
        KeyError
            If the group does not exist.

        """
        if name not in self._root:
            raise KeyError(f"Group '{name}' not found in Zarr store")
        return self._root[name]

    def has_group(self, name: str) -> bool:
        """Check if a group exists in the Zarr store.

        Parameters
        ----------
        name : str
            Name of the group.

        Returns
        -------
        bool
            True if the group exists, False otherwise.

        """
        return name in self._root

    def group_keys(self, group: zarr.Group | None = None) -> list[str]:
        """Get the keys of subgroups in a group.

        Parameters
        ----------
        group : zarr.Group | None
            Group to inspect. If None, uses root.

        Returns
        -------
        list[str]
            List of subgroup names.

        """
        target = group if group is not None else self._root
        if ZARR_VERSION >= 3:
            # Zarr 3: Use members() and filter by Group type
            return [
                name for name, item in target.members() if isinstance(item, zarr.Group)
            ]
        # Zarr 2: Use group_keys()
        return list(target.group_keys())

    def array_keys(self, group: zarr.Group | None = None) -> list[str]:
        """Get the keys of arrays in a group.

        Parameters
        ----------
        group : zarr.Group | None
            Group to inspect. If None, uses root.

        Returns
        -------
        list[str]
            List of array names.

        """
        target = group if group is not None else self._root
        if ZARR_VERSION >= 3:
            # Zarr 3: Use members() and filter by Array type
            return [
                name for name, item in target.members() if isinstance(item, zarr.Array)
            ]
        # Zarr 2: Use array_keys()
        return list(target.array_keys())

    def add_array(
        self,
        group: zarr.Group | str,
        name: str,
        data: np.ndarray,
        chunks: tuple[int, ...] | None = None,
    ) -> None:
        """Add an array to a group, creating or appending as needed.

        This method handles the differences between Zarr 2 and 3 for
        array creation and appending.

        Parameters
        ----------
        group : zarr.Group | str
            Target group or path to group.
        name : str
            Name of the array.
        data : np.ndarray
            Data to add.
        chunks : tuple[int, ...] | None
            Chunk shape. If None, uses (1, *data.shape[1:]).

        """
        if data is None or (isinstance(data, np.ndarray) and data.size == 0):
            return

        if not isinstance(data, np.ndarray):
            data = np.array(data)

        # Resolve group if string path provided
        if isinstance(group, str):
            group = self.get_group(group)

        if chunks is None:
            chunks = (1, *data.shape[1:])

        if name in group:
            self._append_to_array(group, name, data)
        else:
            self._create_array(group, name, data, chunks)

    def _create_array(
        self,
        group: zarr.Group,
        name: str,
        data: np.ndarray,
        chunks: tuple[int, ...],
    ) -> None:
        """Create a new array in the group."""
        if data.size == 0:
            return

        if ZARR_VERSION >= 3:
            # Zarr 3: Use create_array() - can't pass both data and shape
            group.create_array(
                name=name,
                data=data,
                chunks=chunks,
            )
        else:
            # Zarr 2: Use create_dataset()
            group.create_dataset(
                name=name,
                data=data,
                shape=data.shape,
                chunks=chunks,
            )

    def _append_to_array(
        self,
        group: zarr.Group,
        name: str,
        data: np.ndarray,
    ) -> None:
        """Append data to an existing array.

        Handles Zarr 2/3 API differences for appending.
        Zarr 3 removed .append() method, so we use resize + setitem.
        """
        if data.size == 0:
            return

        arr = group[name]
        current_shape = arr.shape
        new_shape = list(current_shape)
        new_shape[0] += data.shape[0]

        if ZARR_VERSION >= 3:
            # Zarr 3: resize takes a tuple
            arr.resize(tuple(new_shape))
        else:
            # Zarr 2: resize can take *args or tuple
            try:
                arr.resize(tuple(new_shape))
            except TypeError:
                # Fallback for older Zarr 2 versions
                arr.resize(*new_shape)

        # Insert the new data
        arr[current_shape[0] :] = data

    def get_array(self, path: str) -> zarr.Array:
        """Get an array by path (e.g., 'training/emg/raw').

        Parameters
        ----------
        path : str
            Path to the array.

        Returns
        -------
        zarr.Array
            The requested array.

        """
        return self._root[path]

    def __getitem__(self, key: str) -> Any:
        """Access items in the root group."""
        return self._root[key]

    def __contains__(self, key: str) -> bool:
        """Check if key exists in root group."""
        return key in self._root

    def close(self) -> None:
        """Close the Zarr store (no-op for directory stores)."""
        # Directory stores don't need explicit closing

    def __enter__(self) -> ZarrIO:
        return self

    def __exit__(self, *args) -> None:
        self.close()


class ZarrDataset:
    """High-level interface for reading EMG datasets stored in Zarr format.

    Provides iteration and random access to dataset splits with proper
    handling of the MyoTorch Zarr structure.

    Parameters
    ----------
    path : Path | str
        Path to the Zarr dataset.
    split : str
        Dataset split to use ('training', 'validation', 'testing').

    Examples
    --------
    >>> dataset = ZarrDataset("data.zarr", split="training")
    >>> for emg, target in dataset:
    ...     process(emg, target)

    """

    def __init__(self, path: Path | str, split: str = "training"):
        self.path = Path(path)
        self.split = split
        self._zio = ZarrIO(path, mode="r")

        self._validate_structure()
        self._load_metadata()

    def _validate_structure(self) -> None:
        """Validate the Zarr dataset has expected structure."""
        if not self._zio.has_group(self.split):
            raise ValueError(f"Split '{self.split}' not found in dataset")

        split_group = self._zio.get_group(self.split)
        if "emg" not in split_group:
            raise ValueError(f"EMG data not found in split '{self.split}'")

    def _load_metadata(self) -> None:
        """Load dataset metadata."""
        split_group = self._zio.get_group(self.split)

        # Get EMG array keys
        self._emg_keys = self._zio.array_keys(split_group["emg"])
        if not self._emg_keys:
            raise ValueError("No EMG arrays found in dataset")

        # Get target group name (ground_truth or similar)
        self._target_group = None
        for key in self._zio.group_keys(split_group):
            if key not in ("emg", "label", "class", "one_hot_class"):
                self._target_group = key
                break

        if self._target_group:
            self._target_keys = self._zio.array_keys(split_group[self._target_group])
        else:
            self._target_keys = []

        # Get dataset length
        first_emg = split_group["emg"][self._emg_keys[0]]
        self._length = first_emg.shape[0]

    @property
    def emg_keys(self) -> list[str]:
        """Get the EMG representation keys."""
        return self._emg_keys

    @property
    def target_keys(self) -> list[str]:
        """Get the target representation keys."""
        return self._target_keys

    def __len__(self) -> int:
        return self._length

    def __getitem__(
        self, idx: int
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Get a single sample by index.

        Returns
        -------
        tuple[dict, dict]
            (emg_data, target_data) dictionaries keyed by representation name.

        """
        split_group = self._zio.get_group(self.split)

        emg_data = {key: split_group["emg"][key][idx] for key in self._emg_keys}

        target_data = {}
        if self._target_group:
            target_data = {
                key: split_group[self._target_group][key][idx]
                for key in self._target_keys
            }

        return emg_data, target_data

    def __iter__(self) -> Iterator[tuple[dict[str, np.ndarray], dict[str, np.ndarray]]]:
        """Iterate over all samples in the dataset."""
        for i in range(len(self)):
            yield self[i]

    def get_emg_array(self, key: str | None = None) -> zarr.Array:
        """Get direct access to an EMG array for efficient streaming.

        Parameters
        ----------
        key : str | None
            EMG representation key. If None, uses the first available.

        Returns
        -------
        zarr.Array
            The EMG array.

        """
        if key is None:
            key = self._emg_keys[0]
        return self._zio[f"{self.split}/emg/{key}"]

    def get_target_array(self, key: str | None = None) -> zarr.Array | None:
        """Get direct access to a target array for efficient streaming.

        Parameters
        ----------
        key : str | None
            Target representation key. If None, uses the first available.

        Returns
        -------
        zarr.Array | None
            The target array, or None if no targets exist.

        """
        if not self._target_group or not self._target_keys:
            return None
        if key is None:
            key = self._target_keys[0]
        return self._zio[f"{self.split}/{self._target_group}/{key}"]
