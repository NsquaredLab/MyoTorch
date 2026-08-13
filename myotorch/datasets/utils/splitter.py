"""Data splitting utilities for dataset creation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

import numpy as np

T = TypeVar("T", bound=np.ndarray)


@dataclass
class SplitResult:
    """Result of a data split operation.

    Attributes
    ----------
    training : np.ndarray
        Training data.
    testing : np.ndarray | None
        Testing data (None if no test split).
    validation : np.ndarray | None
        Validation data (None if no validation split).

    """

    training: np.ndarray
    testing: np.ndarray | None = None
    validation: np.ndarray | None = None

    @property
    def sizes(self) -> tuple[int, int, int]:
        """Get sizes of each split."""
        return (
            self.training.shape[0],
            self.testing.shape[0] if self.testing is not None else 0,
            self.validation.shape[0] if self.validation is not None else 0,
        )


class DataSplitter:
    """Handles splitting data into training, testing, and validation sets.

    The splitting strategy extracts data from the middle of the array,
    which is useful for time-series data where you want to avoid
    temporal leakage at the boundaries.

    Parameters
    ----------
    test_ratio : float
        Ratio of data for testing (0.0 to 1.0).
    val_ratio : float
        Ratio of data for validation (0.0 to 1.0).

    Examples
    --------
    >>> splitter = DataSplitter(test_ratio=0.2, val_ratio=0.2)
    >>> result = splitter.split(data)
    >>> print(result.sizes)
    (800, 100, 100)

    """

    def __init__(self, test_ratio: float = 0.2, val_ratio: float = 0.2):
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self._validate_ratios()

    def _validate_ratios(self) -> None:
        """Validate split ratios."""
        if not 0.0 <= self.test_ratio <= 1.0:
            raise ValueError(
                f"test_ratio must be between 0 and 1, got {self.test_ratio}"
            )
        if not 0.0 <= self.val_ratio <= 1.0:
            raise ValueError(f"val_ratio must be between 0 and 1, got {self.val_ratio}")
        if self.test_ratio + self.val_ratio > 1.0:
            raise ValueError(
                f"test_ratio + val_ratio must be <= 1.0, got {self.test_ratio + self.val_ratio}",
            )

    def split(self, data: np.ndarray) -> SplitResult:
        """Split data into training, testing, and validation sets.

        The split is performed by extracting from the middle of the data:
        1. First, test data is extracted from the center
        2. Then, validation data is extracted from the center of test data

        Parameters
        ----------
        data : np.ndarray
            Data to split. First dimension is assumed to be samples.

        Returns
        -------
        SplitResult
            Named tuple with training, testing, and validation arrays.

        """
        if self.test_ratio == 0:
            return SplitResult(training=data, testing=None, validation=None)

        # Split out test data from the middle
        training, testing = self._split_middle(data, self.test_ratio)

        # Split validation from test data if needed
        if self.val_ratio > 0 and testing is not None:
            testing, validation = self._split_middle(testing, self.val_ratio)
        else:
            validation = None

        return SplitResult(training=training, testing=testing, validation=validation)

    def _split_middle(
        self,
        data: np.ndarray,
        ratio: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Split data by extracting a portion from the middle.

        Parameters
        ----------
        data : np.ndarray
            Data to split.
        ratio : float
            Ratio of data to extract from the middle.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (remaining_data, extracted_middle_data)

        """
        n_samples = data.shape[0]
        split_amount = int(n_samples * ratio / 2)
        middle_index = n_samples // 2

        # Create mask for training data (excludes middle portion)
        mask = np.ones(n_samples, dtype=bool)
        mask[middle_index - split_amount : middle_index + split_amount] = False

        return data[mask], data[~mask]

    def split_dict(
        self,
        data_dict: dict[str, np.ndarray],
    ) -> dict[str, SplitResult]:
        """Split multiple arrays with the same split indices.

        Parameters
        ----------
        data_dict : dict[str, np.ndarray]
            Dictionary of arrays to split.

        Returns
        -------
        dict[str, SplitResult]
            Dictionary mapping keys to their split results.

        """
        return {key: self.split(arr) for key, arr in data_dict.items()}
