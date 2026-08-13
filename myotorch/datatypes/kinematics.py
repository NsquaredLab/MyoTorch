"""Kinematics data type for joint position tracking."""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider

from myotorch.datatypes.base import _Data


class KinematicsData(_Data):
    """Class for storing kinematics data.

    Parameters
    ----------
    input_data : np.ndarray
        The raw kinematics data. The shape of the array should be (n_joints, 3, n_samples)
        or (n_chunks, n_joints, 3, n_samples).

        .. important:: The class will only accept 3D or 4D arrays.
        There is no way to check if you actually have it in (n_chunks, n_joints, 3, n_samples) format.
        Please make sure to provide the correct shape of the data.

    sampling_frequency : float
        The sampling frequency of the kinematics data.

    Attributes
    ----------
    input_data : np.ndarray
        The raw kinematics data. The shape of the array should be (n_joints, 3, n_samples)
        or (n_chunks, n_joints, 3, n_samples).
        The 3 represents the x, y, and z coordinates of the joints.
    sampling_frequency : float
        The sampling frequency of the kinematics data.
    processed_data : dict[str, np.ndarray]
        A dictionary where the keys are the names of filters applied to the kinematics data and
        the values are the processed kinematics data.

    Examples
    --------
    >>> import numpy as np
    >>> from myotorch.datatypes import KinematicsData
    >>>
    >>> # Create sample kinematics data (16 joints, 3 coordinates, 1000 samples)
    >>> # Each joint has x, y, z coordinates
    >>> joint_data = np.random.randn(16, 3, 1000)
    >>>
    >>> # Create a KinematicsData object with 100 Hz sampling rate
    >>> kinematics = KinematicsData(joint_data, 100)
    >>>
    >>> # Access the raw data
    >>> raw_data = kinematics.input_data
    >>> print(f"Data shape: {raw_data.shape}")
    Data shape: (16, 3, 1000)

    """

    def __init__(self, input_data: np.ndarray, sampling_frequency: float):
        if input_data.ndim not in (3, 4):
            raise ValueError(
                "The shape of the raw kinematics data should be (n_joints, 3, n_samples) "
                "or (n_chunks, n_joints, 3, n_samples).",
            )
        super().__init__(
            input_data,
            sampling_frequency,
            nr_of_dimensions_when_unchunked=4,
        )

    def plot(
        self,
        representation: str,
        nr_of_fingers: int,
        wrist_included: bool = True,
    ):
        """Plots the data.

        Parameters
        ----------
        representation : str
            The representation to plot.
            .. important :: The representation should be a 3D tensor with shape (n_joints, 3, n_samples).
        nr_of_fingers : int
            The number of fingers to plot.
        wrist_included : bool, optional
            Whether the wrist is included in the representation. The default is True.
            .. note :: The wrist is always the first joint in the representation.

        Raises
        ------
        KeyError
            If the representation does not exist.

        Examples
        --------
        >>> import numpy as np
        >>> from myotorch.datatypes import KinematicsData
        >>>
        >>> # Create sample kinematics data for a hand with 5 fingers
        >>> # 16 joints: 1 wrist + 3 joints for each of the 5 fingers
        >>> joint_data = np.random.randn(16, 3, 100)
        >>> kinematics = KinematicsData(joint_data, 100)
        >>>
        >>> # Plot the kinematics data
        >>> kinematics.plot('Input', nr_of_fingers=5)
        >>>
        >>> # Plot without wrist
        >>> kinematics.plot('Input', nr_of_fingers=5, wrist_included=False)

        """
        if representation not in self._data:
            raise KeyError(f'The representation "{representation}" does not exist.')

        kinematics = self[representation]

        if not wrist_included:
            kinematics = np.concatenate(
                [np.zeros((1, 3, kinematics.shape[2])), kinematics],
                axis=0,
            )

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # get biggest axis range
        max_range = (
            np.array(
                [
                    kinematics[:, 0].max() - kinematics[:, 0].min(),
                    kinematics[:, 1].max() - kinematics[:, 1].min(),
                    kinematics[:, 2].max() - kinematics[:, 2].min(),
                ],
            ).max()
            / 2.0
        )

        # set axis limits
        ax.set_xlim(
            kinematics[:, 0].mean() - max_range,
            kinematics[:, 0].mean() + max_range,
        )
        ax.set_ylim(
            kinematics[:, 1].mean() - max_range,
            kinematics[:, 1].mean() + max_range,
        )
        ax.set_zlim(
            kinematics[:, 2].mean() - max_range,
            kinematics[:, 2].mean() + max_range,
        )

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        # create joint and finger plots
        (joints_plot,) = ax.plot(*kinematics[..., 0].T, "o", color="black")

        finger_plots = []
        for finger in range(nr_of_fingers):
            finger_plots.append(
                ax.plot(
                    *kinematics[
                        [0] + list(reversed(range(1 + finger * 4, 5 + finger * 4))),
                        :,
                        0,
                    ].T,
                    color="blue",
                ),
            )

        samp = plt.axes([0.25, 0.02, 0.65, 0.03])
        sample_slider = Slider(
            samp,
            label="Sample (a. u.)",
            valmin=0,
            valmax=kinematics.shape[2] - 1,
            valstep=1,
            valinit=0,
        )

        def update(val):
            kinematics_new_sample = kinematics[..., int(val)]

            joints_plot._verts3d = tuple(kinematics_new_sample.T)

            for finger in range(nr_of_fingers):
                finger_plots[finger][0]._verts3d = tuple(
                    kinematics_new_sample[
                        [0] + list(reversed(range(1 + finger * 4, 5 + finger * 4))),
                        :,
                    ].T,
                )

            fig.canvas.draw_idle()

        sample_slider.on_changed(update)
        plt.tight_layout()
        plt.show()
