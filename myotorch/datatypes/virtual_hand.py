"""Virtual hand kinematics data type for MyoGestic."""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

from myotorch.datatypes.base import _Data


class VirtualHandKinematics(_Data):
    """Class for storing virtual hand kinematics data from MyoGestic [1]_.

    Parameters
    ----------
    input_data : np.ndarray
        The raw kinematics data for a virtual hand. The shape of the array should be (9, n_samples)
        or (n_chunks, 9, n_samples).

        .. important:: The class will only accept 2D or 3D arrays.
        There is no way to check if you actually have it in (n_chunks, n_samples) or (n_chunks, 9, n_samples) format.
        Please make sure to provide the correct shape of the data.

    sampling_frequency : float
        The sampling frequency of the kinematics data.

    Attributes
    ----------
    input_data : np.ndarray
        The raw kinematics data for a virtual hand. The shape of the array should be (9, n_samples)
        or (n_chunks, 9, n_samples).
        The 9 typically represents the degrees of freedom: wrist flexion/extension,
        wrist pronation/supination, wrist deviation, and the flexion of all 5 fingers.
    sampling_frequency : float
        The sampling frequency of the kinematics data.
    processed_data : dict[str, np.ndarray]
        A dictionary where the keys are the names of filters applied to the kinematics data and
        the values are the processed kinematics data.

    Examples
    --------
    >>> import numpy as np
    >>> from myotorch.datatypes import VirtualHandKinematics
    >>>
    >>> # Create sample virtual hand kinematics data (9 DOFs, 1000 samples)
    >>> joint_data = np.random.randn(9, 1000)
    >>>
    >>> # Create a VirtualHandKinematics object with 100 Hz sampling rate
    >>> kinematics = VirtualHandKinematics(joint_data, 100)
    >>>
    >>> # Access the raw data
    >>> raw_data = kinematics.input_data
    >>> print(f"Data shape: {raw_data.shape}")

    References
    ----------
    .. [1] MyoGestic: https://github.com/NsquaredLab/MyoGestic

    """

    def __init__(self, input_data: np.ndarray, sampling_frequency: float):
        if input_data.ndim not in (2, 3):
            raise ValueError(
                "The shape of the raw kinematics data should be (9, n_samples) "
                "or (n_chunks, 9, n_samples).",
            )
        super().__init__(
            input_data,
            sampling_frequency,
            nr_of_dimensions_when_unchunked=3,
        )

    def plot(
        self,
        representation: str,
        nr_of_fingers: int = 5,
        visualize_wrist: bool = True,
    ):
        """Plots the virtual hand kinematics data.

        Parameters
        ----------
        representation : str
            The representation to plot.
            The representation should be a 2D tensor with shape (9, n_samples)
            or a 3D tensor with shape (n_chunks, 9, n_samples).
        nr_of_fingers : int, optional
            The number of fingers to plot. Default is 5.
        visualize_wrist : bool, optional
            Whether to visualize wrist movements. Default is True.

        Raises
        ------
        KeyError
            If the representation does not exist.

        """
        if representation not in self._data:
            raise KeyError(f'The representation "{representation}" does not exist.')

        data = self[representation]
        is_chunked = self.is_chunked[representation]

        if is_chunked:
            # Use only the first chunk for visualization
            data = data[0]

        # Check if we have the expected number of DOFs
        if data.shape[0] != 9:
            raise ValueError(f"Expected 9 degrees of freedom, but got {data.shape[0]}")

        fig = plt.figure(figsize=(12, 8))

        # Create a separate plot for each DOF
        wrist_ax = fig.add_subplot(2, 1, 1)
        fingers_ax = fig.add_subplot(2, 1, 2)

        # Plot wrist DOFs (first 3 channels)
        if visualize_wrist:
            wrist_ax.set_title("Wrist Kinematics")
            wrist_ax.plot(data[0], label="Wrist Flexion/Extension")
            wrist_ax.plot(data[1], label="Wrist Pronation/Supination")
            wrist_ax.plot(data[2], label="Wrist Deviation")
            wrist_ax.legend()
            wrist_ax.set_xlabel("Time (samples)")
            wrist_ax.set_ylabel("Normalized Position")
            wrist_ax.grid(True)

        # Plot finger DOFs (remaining channels)
        fingers_ax.set_title("Finger Kinematics")
        finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
        for i in range(min(nr_of_fingers, 5)):
            fingers_ax.plot(data[i + 3], label=finger_names[i])

        fingers_ax.legend()
        fingers_ax.set_xlabel("Time (samples)")
        fingers_ax.set_ylabel("Normalized Flexion")
        fingers_ax.grid(True)

        plt.tight_layout()
        plt.show()
