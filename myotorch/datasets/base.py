"""Base dataset for windowed multi-modal data loading.

This module provides the paradigm-agnostic infrastructure for loading
windowed data from zarr stores. It handles:
- Zarr I/O with optional GPU Direct Storage (GDS)
- Window sampling (random or deterministic)
- RAM caching for performance
- Device management (CPU/GPU)
- Multiprocessing support

The WindowedDataset class returns all modalities as a dict, without
making assumptions about the learning paradigm (supervised, contrastive, etc.).
Paradigm-specific datasets should subclass WindowedDataset.

Example:
-------
>>> from myotorch.datasets.base import WindowedDataset
>>>
>>> # Load all modalities from zarr
>>> ds = WindowedDataset(
...     "data.zip",
...     split="training",
...     modalities=["emg", "kinematics"],
...     window_size=200,
...     n_windows=10000,
...     device="cuda",
... )
>>> data = ds[0]  # dict[str, Tensor] with 'emg' and 'kinematics'

"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from numbers import Integral
from pathlib import Path

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset
from zarr.storage import ZipStore

from myotorch.datasets.utils.splitter import splice_positions

# Suppress named tensor experimental warning
warnings.filterwarnings("ignore", category=UserWarning, message=".*Named tensors.*")


class WindowedDataset(Dataset):
    """Base dataset that loads windows from zarr for any modality.

    This is the infrastructure layer - it handles loading, windowing, caching,
    and device management. It returns ALL requested modalities as a dict.

    Subclasses implement paradigm-specific logic (e.g., SupervisedDataset
    splits into inputs/targets, ContrastiveDataset creates augmented views).

    Parameters
    ----------
    zarr_path : Path | str
        Path to the Zarr dataset.
    split : str
        Dataset split ('training', 'validation', 'testing').
    modalities : Sequence[str] | None
        Modality names to load. If None, loads all available modalities.
    window_size : int
        Number of samples per window.
    window_stride : int | None
        Stride between windows. If None, uses random positions.
    n_windows : int | None
        Number of windows per epoch. Required if window_stride is None.
    seed : int | None
        Random seed for reproducible window positions.
    device : torch.device | str | None
        Output device:
        - None: return numpy arrays
        - "cpu": return tensors on CPU
        - "cuda": return tensors on GPU (uses kvikio GDS if available)
    dtype : torch.dtype
        Data type for tensors. Default: torch.float32.
    cache_in_ram : bool
        Cache entire split in RAM for faster access. Default: True.
    context : int
        Real samples preceding each window. Context modalities load
        [position-context, position+window_size); other modalities load only
        the window. Filter first, then crop with
        ``Index(slice(-window_size, None), dim="time")``. Default: 0.
    context_modalities : Sequence[str] | None
        Modalities given history. Defaults to all non-target modalities.
        Supervised datasets exclude targets even if they are also inputs.
    respect_splices : bool | None
        Treat split joins as recording boundaries. None (automatic) enables
        this when context > 0. With context=0, omitting this option preserves
        the original window positions exactly; pass True to opt in. False
        explicitly disables splice handling. Every continuous piece starts
        its stride grid at ``piece_start + context``.

    Examples
    --------
    >>> # Return numpy arrays
    >>> ds = WindowedDataset("data.zip", modalities=["emg"], device=None)
    >>> data = ds[0]
    >>> type(data["emg"])  # numpy.ndarray
    >>>
    >>> # Return tensors on GPU with named dimensions
    >>> ds = WindowedDataset("data.zip", modalities=["emg"], device="cuda")
    >>> data["emg"].device  # cuda:0
    >>> data["emg"].names   # ('channel', 'time')

    """

    def __init__(
        self,
        zarr_path: Path | str,
        split: str = "training",
        modalities: Sequence[str] | None = None,
        window_size: int = 200,
        window_stride: int | None = None,
        n_windows: int | None = None,
        seed: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
        cache_in_ram: bool = True,
        context: int = 0,
        context_modalities: Sequence[str] | None = None,
        respect_splices: bool | None = None,
    ):
        if not isinstance(context, Integral) or context < 0:
            raise ValueError("context must be a non-negative integer")
        self.zarr_path = Path(zarr_path)
        self.split = split
        self.window_size = window_size
        self.window_stride = window_stride
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self.device = torch.device(device) if device else None
        self.cache_in_ram = cache_in_ram
        self.dtype = dtype
        self.context = int(context)
        self.respect_splices = context > 0 if respect_splices is None else respect_splices

        # Validate path
        if not self.zarr_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.zarr_path}")

        # Reject directory-based .zarr (no longer supported)
        if self.zarr_path.is_dir() or self.zarr_path.suffix.lower() == ".zarr":
            raise ValueError(
                f"Directory-based .zarr is no longer supported: {self.zarr_path}\n"
                f"Please recreate the dataset using DatasetCreator with a .zip path."
            )

        if window_stride is None and n_windows is None:
            raise ValueError("Must specify n_windows when window_stride is None")

        # Open zarr store from zip file
        self._zip_store = ZipStore(self.zarr_path, mode="r")
        self._store = zarr.open(self._zip_store, mode="r")

        # Get metadata from standard zarr attrs
        self._available_modalities = self._store.attrs.get("modalities", [])
        self._tasks = self._store.attrs.get("tasks", [])
        self._dims_info = self._store.attrs.get("dims", {})

        # Get split group
        if split not in self._store:
            raise FileNotFoundError(f"Split '{split}' not found in {self.zarr_path}")
        self._split_group = self._store[split]

        # Determine modalities to load
        if modalities is None:
            self.modalities = list(self._available_modalities)
        else:
            self.modalities = list(modalities)
            # Validate modalities exist
            missing = set(self.modalities) - set(self._available_modalities)
            if missing:
                raise ValueError(
                    f"Requested modalities {missing} not in dataset. "
                    f"Available: {self._available_modalities}",
                )

        targets = set(getattr(self, "targets", ()))
        if context_modalities is None:
            self.context_modalities = [mod for mod in self.modalities if mod not in targets]
        else:
            self.context_modalities = list(context_modalities)
            missing = set(self.context_modalities) - set(self.modalities)
            if missing:
                raise ValueError(f"Unknown context modalities: {missing}")
            if targets.intersection(self.context_modalities):
                raise ValueError("Context modalities must not include target modalities")

        # RAM cache is loaded lazily on first data access
        self._ram_cache = None
        self._cache_loaded = False

        # Build task lists for each modality (nested structure: split/modality/task)
        self._modality_tasks: dict[str, list[str]] = {mod: [] for mod in self.modalities}

        for mod in self.modalities:
            if mod in self._split_group:
                mod_group = self._split_group[mod]
                self._modality_tasks[mod] = sorted(mod_group.keys())

        # For compatibility, also store full paths
        self._modality_vars: dict[str, list[str]] = {
            mod: [f"{mod}/{task}" for task in tasks]
            for mod, tasks in self._modality_tasks.items()
        }

        # Get recording lengths from first modality (nested structure)
        first_mod = self.modalities[0]
        self._recording_lengths = []
        self._recording_tasks = []  # Store task names directly

        mod_group = self._split_group[first_mod]
        for task in self._modality_tasks[first_mod]:
            arr = mod_group[task]
            length = arr.shape[-1]  # Time is last dimension
            self._recording_lengths.append(length)
            self._recording_tasks.append(task)

        self._total_length = sum(self._recording_lengths)
        self._splices = self._get_splices() if self.respect_splices else {}
        self._setup_recording_ranges()

        # Compute number of windows
        if window_stride is not None:
            self._n_windows = sum(
                (end - start) // window_stride + 1
                for _, start, end in self._valid_ranges
            )
            self._random_mode = False
        else:
            self._n_windows = n_windows
            self._random_mode = True

    def __getstate__(self):
        """Prepare state for pickling (used by multiprocessing workers)."""
        state = self.__dict__.copy()
        state["_store"] = None
        state["_split_group"] = None
        state["_rng"] = None
        return state

    def __setstate__(self, state):
        """Restore state after unpickling (in worker processes)."""
        try:
            self.__dict__.update(state)
            self._rng = np.random.default_rng(self.seed)

            try:
                zarr.config.reset()
            except Exception:
                pass

            # If cache already loaded, no need to reopen store
            if self._ram_cache is not None and self._cache_loaded:
                return

            # Reopen store for lazy cache loading or direct reads
            self._zip_store = ZipStore(self.zarr_path, mode="r")
            self._store = zarr.open(self._zip_store, mode="r")
            self._split_group = self._store[self.split]
        except Exception as e:
            import sys

            print(f"ERROR in __setstate__: {e}", file=sys.stderr)
            raise

    def get_sample_shape(self, modality: str) -> tuple[int, ...]:
        """Get the shape of a sample for a given modality (without time dimension).

        Parameters
        ----------
        modality : str
            Modality name.

        Returns
        -------
        tuple[int, ...]
            Shape without time dimension.

        """
        task_list = self._modality_tasks.get(modality)
        if not task_list:
            raise ValueError(f"Modality '{modality}' not found")

        first_task = task_list[0]
        arr = self._split_group[modality][first_task]
        return arr.shape[:-1]

    def _get_splices(self) -> dict[str, list[int]]:
        """Read boundaries, or reconstruct older centered-extraction splits."""
        recorded = self._split_group.attrs.get("splices")
        if recorded is not None:
            return recorded

        attrs = self._store.attrs
        test_ratio = attrs.get("test_ratio", 0.2)
        val_ratio = attrs.get("val_ratio", 0.2)
        if "test_ratio" not in attrs or "val_ratio" not in attrs:
            warnings.warn(
                "Dataset has no splice metadata and missing split ratios; "
                f"assuming test_ratio={test_ratio}, val_ratio={val_ratio} "
                "for centered-extraction splice reconstruction.",
                UserWarning,
                stacklevel=3,
            )

        first_mod = self.modalities[0]
        positions = {}
        for task in self._recording_tasks:
            # Sum stored split lengths: inverting ratios loses odd-sample rounding.
            n_samples = 0
            for split in ("training", "testing", "validation"):
                if split in self._store and first_mod in self._store[split]:
                    group = self._store[split][first_mod]
                    if task in group:
                        n_samples += group[task].shape[-1]
            positions[task] = splice_positions(n_samples, test_ratio, val_ratio)[self.split]
        return positions

    def _setup_recording_ranges(self) -> None:
        """Setup valid sampling ranges for each recording."""
        self._valid_ranges = []
        cumsum = 0

        for rec_idx, length in enumerate(self._recording_lengths):
            task = self._recording_tasks[rec_idx]
            splices = sorted(set(self._splices.get(task, [])))
            if any(not isinstance(p, Integral) or not 0 < p < length for p in splices):
                raise ValueError(f"Invalid splice positions for {task}: {splices}")
            boundaries = [0, *splices, length]
            for start, end in zip(boundaries, boundaries[1:]):
                if end - start >= self.window_size + self.context:
                    valid_start = cumsum + start + self.context
                    valid_end = cumsum + end - self.window_size
                    self._valid_ranges.append((rec_idx, valid_start, valid_end))
            cumsum += length

        if not self._valid_ranges:
            raise ValueError(
                f"No continuous pieces long enough for window_size={self.window_size} "
                f"and context={self.context}",
            )

        self._total_valid = sum(end - start + 1 for _, start, end in self._valid_ranges)

    def _global_to_local(self, global_pos: int) -> tuple[int, int]:
        """Convert global position to (recording_idx, local_position)."""
        cumsum = 0
        for rec_idx, length in enumerate(self._recording_lengths):
            if global_pos < cumsum + length:
                return rec_idx, global_pos - cumsum
            cumsum += length
        raise ValueError(f"Position {global_pos} out of range")

    def _sample_random_position(self) -> tuple[int, int]:
        """Sample a random valid window position."""
        pos = self._rng.integers(0, self._total_valid)

        cumsum = 0
        for rec_idx, start, end in self._valid_ranges:
            range_size = end - start + 1
            if pos < cumsum + range_size:
                global_pos = start + (pos - cumsum)
                return self._global_to_local(global_pos)
            cumsum += range_size

        raise RuntimeError(
            f"Failed to map random position {pos} to valid range "
            f"(total_valid={self._total_valid}, n_ranges={len(self._valid_ranges)})",
        )

    def _get_deterministic_position(self, idx: int) -> tuple[int, int]:
        """Get deterministic window position for given index."""
        cumsum = 0
        for rec_idx, start, end in self._valid_ranges:
            valid_positions = (end - start) // self.window_stride + 1
            if idx < cumsum + valid_positions:
                local_idx = idx - cumsum
                return self._global_to_local(start + local_idx * self.window_stride)
            cumsum += valid_positions

        raise ValueError(f"Index {idx} out of range")

    def _get_task_for_recording(self, rec_idx: int) -> str:
        """Get the task name for a recording index."""
        return self._recording_tasks[rec_idx]

    # Default dimension names by modality (fallback when not in metadata)
    _DEFAULT_DIMS: dict[str, tuple[str, ...]] = {
        "emg": ("channel", "time"),
        "kinematics": ("joint", "time"),
        "eeg": ("electrode", "time"),
    }

    def _get_dim_names(self, modality: str) -> tuple[str, ...]:
        """Get dimension names for a modality from metadata."""
        if modality in self._dims_info:
            return tuple(self._dims_info[modality])
        return self._DEFAULT_DIMS.get(modality, ("channel", "time"))

    def _ensure_cache_loaded(self) -> None:
        """Load data into RAM cache if caching is enabled and not yet loaded."""
        if not self.cache_in_ram or self._cache_loaded:
            return

        zarr.config.set({"async.concurrency": 32})
        self._ram_cache = {}

        # Load nested structure: modality -> task -> array
        for mod in self.modalities:
            if mod not in self._split_group:
                continue
            mod_group = self._split_group[mod]
            for task in mod_group.keys():
                cache_key = f"{mod}/{task}"
                self._ram_cache[cache_key] = np.asarray(mod_group[task][:])

        self._cache_loaded = True

    def _to_tensor(self, data) -> torch.Tensor:
        """Convert data to tensor on target device."""
        tensor = torch.from_numpy(np.ascontiguousarray(data))
        if self.device is not None:
            return tensor.to(device=self.device, dtype=self.dtype)
        return tensor.to(dtype=self.dtype)

    def _load_window(
        self, var_path: str, local_pos: int, modality: str
    ) -> torch.Tensor | np.ndarray:
        """Load a window for a variable and convert to tensor.

        Parameters
        ----------
        var_path : str
            Variable path in zarr (e.g., "emg/task1").
        local_pos : int
            Starting position within the recording.
        modality : str
            Modality name for dimension info.

        Returns
        -------
        torch.Tensor | np.ndarray
            Window data as tensor (if device set) or numpy array.

        """
        if self._ram_cache is not None:
            arr = self._ram_cache[var_path]
        else:
            # Navigate nested structure: modality/task
            mod, task = var_path.split("/", 1)
            arr = self._split_group[mod][task]

        # Validate window fits within recording
        end_pos = local_pos + self.window_size
        start_pos = local_pos - self.context if modality in self.context_modalities else local_pos
        if start_pos < 0 or end_pos > arr.shape[-1]:
            raise ValueError(
                f"Window [{start_pos}:{end_pos}] exceeds recording length {arr.shape[-1]} "
                f"for variable {var_path}",
            )

        data = arr[..., start_pos:end_pos]

        if self.device is None:
            return np.ascontiguousarray(data)
        tensor = self._to_tensor(data)
        names = self._get_dim_names(modality)
        tensor = tensor.rename(*names)
        return tensor

    def __len__(self) -> int:
        return self._n_windows

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | np.ndarray]:
        """Load windows for all modalities.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        dict[str, torch.Tensor | np.ndarray]
            Dict mapping modality names to data windows.

        """
        # Lazy load cache on first access
        self._ensure_cache_loaded()

        # Get window position
        if self._random_mode:
            rec_idx, local_pos = self._sample_random_position()
        else:
            rec_idx, local_pos = self._get_deterministic_position(idx)

        task = self._get_task_for_recording(rec_idx)

        # Extract windows for all modalities (nested structure: modality/task)
        data = {}
        for mod in self.modalities:
            cache_key = f"{mod}/{task}"
            if self._ram_cache is not None and cache_key in self._ram_cache:
                data[mod] = self._load_window(cache_key, local_pos, mod)
            elif mod in self._split_group and task in self._split_group[mod]:
                data[mod] = self._load_window(cache_key, local_pos, mod)

        return data

    def reseed(self, seed: int | None = None) -> None:
        """Reseed the random number generator."""
        self._rng = np.random.default_rng(seed)
