"""Type definitions and constants for datatypes module."""

from __future__ import annotations

from typing import Final, NamedTuple, TypedDict

import numpy as np


class DeletedRepresentation(NamedTuple):
    """Class to hold metadata about deleted representations.

    This stores the shape and dtype of the deleted array. Making it compatible with the numpy array interface.

    Attributes
    ----------
    shape : tuple
        The shape of the deleted array
    dtype : np.dtype
        The data type of the deleted array

    """

    shape: tuple
    dtype: np.dtype

    def __str__(self) -> str:
        """String representation of the deleted data."""
        return str(self.shape)


class Representation(TypedDict):
    data: np.ndarray


InputRepresentationName: Final[str] = "Input"
LastRepresentationName: Final[str] = "Last"
