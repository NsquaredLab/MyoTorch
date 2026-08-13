"""Base class for all data types."""

from __future__ import annotations

import copy
import os
import pickle
from abc import abstractmethod
from typing import Any

import numpy as np

from myotorch.datatypes.types import (
    DeletedRepresentation,
    InputRepresentationName,
    LastRepresentationName,
    Representation,
)


class _Data:
    """Base class for all data types.

    This class provides common functionality for handling different types of data,
    including maintaining original and processed representations.

    Parameters
    ----------
    raw_data : np.ndarray
        The raw data to store.
    sampling_frequency : float
        The sampling frequency of the data.

    Attributes
    ----------
    sampling_frequency : float
        The sampling frequency of the data.
    _last_processing_step : str
        The last processing step applied to the data.
    _data : dict[str, np.ndarray | DeletedRepresentation]
        Dictionary of all data. The keys are the names of the representations and the values are
        either numpy arrays or DeletedRepresentation objects (for representations that have been
        deleted to save memory).

    Raises
    ------
    ValueError
        If the sampling frequency is less than or equal to 0.

    Notes
    -----
    Memory Management:
        When representations are deleted with delete_data(), they are replaced with
        DeletedRepresentation objects that store essential metadata (shape, dtype)
        but don't consume memory for the actual data. The chunking status is determined from
        the shape when needed.

    Examples
    --------
    This is an abstract base class and should not be instantiated directly.
    Instead, use one of the concrete subclasses like EMGData or KinematicsData:

    >>> import numpy as np
    >>> from myotorch.datatypes import EMGData
    >>>
    >>> # Create sample data
    >>> data = np.random.randn(16, 1000)
    >>> emg = EMGData(data, 2000)  # 2000 Hz sampling rate
    >>>
    >>> # Access attributes from the base _Data class
    >>> print(f"Sampling frequency: {emg.sampling_frequency} Hz")
    >>> print(f"Is input data chunked: {emg.is_chunked['Input']}")

    """

    def __init__(
        self,
        raw_data: np.ndarray,
        sampling_frequency: float,
        nr_of_dimensions_when_unchunked: int,
    ):
        self.sampling_frequency: float = sampling_frequency

        self.nr_of_dimensions_when_unchunked: int = nr_of_dimensions_when_unchunked

        if self.sampling_frequency <= 0:
            raise ValueError("The sampling frequency should be greater than 0.")

        self._data: dict[str, np.ndarray | DeletedRepresentation] = {
            InputRepresentationName: raw_data,
        }

        self.__last_processing_step: str = InputRepresentationName

    @property
    def is_chunked(self) -> dict[str, bool]:
        """Returns whether the data is chunked or not.

        Returns
        -------
        dict[str, bool]
            A dictionary where the keys are the representations and the values are whether the data is chunked or not.

        """
        # Create cache if it doesn't exist or if _data might have changed
        if not hasattr(self, "_chunked_cache") or len(self._chunked_cache) != len(
            self._data,
        ):
            self._chunked_cache = {
                key: self._check_if_chunked(value) for key, value in self._data.items()
            }

        return self._chunked_cache

    def _check_if_chunked(self, data: np.ndarray | DeletedRepresentation) -> bool:
        """Checks if the data is chunked or not.

        Parameters
        ----------
        data : np.ndarray | DeletedRepresentation
            The data to check.

        Returns
        -------
        bool
            Whether the data is chunked or not.

        """
        return len(data.shape) == self.nr_of_dimensions_when_unchunked

    @property
    def input_data(self) -> np.ndarray:
        """Returns the input data."""
        return self._data[InputRepresentationName]

    @input_data.setter
    def input_data(self, value: np.ndarray):
        raise RuntimeError("This property is read-only.")

    @property
    def processed_representations(self) -> dict[str, np.ndarray]:
        """Returns the processed representations of the data."""
        return self._data

    @processed_representations.setter
    def processed_representations(self, value: dict[str, Representation]):
        raise RuntimeError("This property is read-only.")

    @property
    def _last_processing_step(self) -> str:
        """Returns the last processing step applied to the data.

        Returns
        -------
        str
            The last processing step applied to the data.

        """
        if self.__last_processing_step is None:
            raise ValueError("No processing steps have been applied.")

        return self.__last_processing_step

    @_last_processing_step.setter
    def _last_processing_step(self, value: str):
        """Sets the last processing step applied to the data.

        Parameters
        ----------
        value : str
            The last processing step applied to the data.

        """
        self.__last_processing_step = value

    @abstractmethod
    def plot(self, *_: Any, **__: Any):
        """Plots the data."""
        raise NotImplementedError(
            "This method should be implemented in the child class.",
        )

    def __repr__(self) -> str:
        # Get input data shape directly from _data dictionary to avoid copying
        input_shape = self._data[InputRepresentationName].shape

        # Build a structured string representation
        lines = []
        lines.append(f"{self.__class__.__name__}")
        lines.append(f"Sampling frequency: {self.sampling_frequency} Hz")
        lines.append(f"(0) Input {input_shape}")

        # Show other representations if they exist
        other_reps = [k for k in self._data.keys() if k != InputRepresentationName]
        if other_reps:
            lines.append("")
            lines.append("Representations:")

            for idx, rep_name in enumerate(other_reps, 1):
                rep_data = self._data[rep_name]
                # Both np.ndarray and DeletedRepresentation have .shape attribute
                lines.append(f"({idx}) {rep_name} {rep_data.shape}")

        # Join all parts with newlines
        return "\n".join(lines)

    def __str__(self) -> str:
        return (
            "--\n"
            + self.__repr__()
            .replace("; ", "\n")
            .replace("Filter(s): ", "\nFilter(s):\n")
            + "\n--"
        )

    def __getitem__(self, key: str) -> np.ndarray:
        if key == InputRepresentationName:
            # Use array.view() for more efficient copying when possible
            data = self.input_data
            return data.view() if data.flags.writeable else data.copy()

        if key == LastRepresentationName:
            return self[self._last_processing_step]

        if key not in self._data:
            raise KeyError(f'The representation "{key}" does not exist.')

        data_to_return = self._data[key]

        if isinstance(data_to_return, DeletedRepresentation):
            raise RuntimeError(
                f'The representation "{key}" was deleted and cannot be automatically '
                f"recomputed. Use the new Transform API for preprocessing.",
            )

        # Use view when possible for more efficient memory usage
        return (
            data_to_return.view()
            if data_to_return.flags.writeable
            else data_to_return.copy()
        )

    def __setitem__(self, key: str, value: np.ndarray) -> None:
        raise RuntimeError(
            "Direct assignment is not supported. Use the Transform API for preprocessing.",
        )

    def delete_data(self, representation_to_delete: str):
        """Delete data from a representation while keeping its metadata.

        This replaces the actual numpy array with a DeletedRepresentation object
        that contains metadata about the array, saving memory while allowing
        regeneration when needed.

        Parameters
        ----------
        representation_to_delete : str
            The representation to delete the data from.

        """
        if representation_to_delete == InputRepresentationName:
            return
        if representation_to_delete == LastRepresentationName:
            self.delete_data(self._last_processing_step)
            return

        if representation_to_delete not in self._data:
            raise KeyError(
                f'The representation "{representation_to_delete}" does not exist.',
            )

        data = self._data[representation_to_delete]
        if isinstance(data, np.ndarray):
            self._data[representation_to_delete] = DeletedRepresentation(
                shape=data.shape,
                dtype=data.dtype,
            )

    def __copy__(self) -> _Data:
        """Create a shallow copy of the instance.

        Returns
        -------
        _Data
            A shallow copy of the instance.

        """
        # Create a new instance with the basic initialization
        new_instance = self.__class__(
            self._data[InputRepresentationName].copy(),
            self.sampling_frequency,
        )

        # Deep copy the data dictionary to preserve all representations
        new_instance._data = copy.deepcopy(self._data)

        # Copy the last processing step
        new_instance._Data__last_processing_step = self._Data__last_processing_step

        # Copy any additional instance attributes from subclasses
        # (excluding private attributes, methods, and already-copied attributes)
        skip_attrs = {
            "_data",
            "_Data__last_processing_step",
            "sampling_frequency",
            "nr_of_dimensions_when_unchunked",
            "_chunked_cache",
        }
        for name, value in vars(self).items():
            if name not in skip_attrs:
                setattr(new_instance, name, copy.copy(value))

        return new_instance

    def save(self, filename: str):
        """Save the data to a file.

        Parameters
        ----------
        filename : str
            The name of the file to save the data to.

        """
        # Make sure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)

        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename: str) -> _Data:
        """Load data from a file.

        Parameters
        ----------
        filename : str
            The name of the file to load the data from.

        Returns
        -------
        _Data
            The loaded data.

        """
        with open(filename, "rb") as f:
            return pickle.load(f)

    def memory_usage(self) -> dict[str, tuple[str, int]]:
        """Calculate memory usage of each representation.

        Returns
        -------
        dict[str, tuple[str, int]]
            Dictionary with representation names as keys and tuples containing
            shape as string and memory usage in bytes as values.

        """
        memory_usage = {}
        for key, value in self._data.items():
            if isinstance(value, np.ndarray):
                memory_usage[key] = (str(value.shape), value.nbytes)
            elif isinstance(value, DeletedRepresentation):
                memory_usage[key] = (
                    str(value.shape),
                    0,  # DeletedRepresentation objects use negligible memory
                )

        return memory_usage
