import unittest
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock

from myotorch.datatypes import VirtualHandKinematics


class TestVirtualHandKinematics(unittest.TestCase):
    """Test class for VirtualHandKinematics functionality that's not covered by the _Data tests."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample 2D data (9 DOFs, 100 samples)
        # The 9 DOFs represent: wrist flexion/extension, wrist pronation/supination,
        # wrist deviation, and the flexion of all 5 fingers
        self.vhk_data_2d = np.random.randn(9, 100)

        # Create sample 3D data (5 chunks, 9 DOFs, 100 samples)
        self.vhk_data_3d = np.random.randn(5, 9, 100)

        # Create sample VirtualHandKinematics objects with 2D and 3D data
        self.vhk_2d = VirtualHandKinematics(self.vhk_data_2d, 100.0)
        self.vhk_3d = VirtualHandKinematics(self.vhk_data_3d, 100.0)

    def test_initialization(self):
        """Test initialization with different data formats."""
        # Test initialization with 2D data
        vhk_2d = VirtualHandKinematics(self.vhk_data_2d, 100.0)
        self.assertEqual(vhk_2d.sampling_frequency, 100.0)
        np.testing.assert_array_equal(vhk_2d.input_data, self.vhk_data_2d)

        # Test initialization with 3D data
        vhk_3d = VirtualHandKinematics(self.vhk_data_3d, 100.0)
        self.assertEqual(vhk_3d.sampling_frequency, 100.0)
        np.testing.assert_array_equal(vhk_3d.input_data, self.vhk_data_3d)

        # Test initialization with invalid data dimensions
        with self.assertRaises(ValueError):
            VirtualHandKinematics(
                np.random.randn(100), 100.0
            )  # 1D array, should be at least 2D

        with self.assertRaises(ValueError):
            VirtualHandKinematics(
                np.random.randn(5, 9, 100, 2), 100.0
            )  # 4D array, should be at most 3D

    def test_check_if_chunked(self):
        """Test the _check_if_chunked method specific to VirtualHandKinematics."""
        # 2D data should not be chunked
        self.assertFalse(self.vhk_2d.is_chunked["Input"])

        # 3D data should be chunked
        self.assertTrue(self.vhk_3d.is_chunked["Input"])

    def test_plot(self):
        """Test the plot method arguments and error cases."""
        # Mock the plt.figure and plt.show functions to avoid actually plotting
        with (
            patch("matplotlib.pyplot.figure") as mock_figure,
            patch("matplotlib.pyplot.show") as mock_show,
        ):
            # Create mock figure and axes
            mock_fig = MagicMock()
            mock_figure.return_value = mock_fig
            mock_ax1 = MagicMock()
            mock_ax2 = MagicMock()
            mock_fig.add_subplot.side_effect = [mock_ax1, mock_ax2]

            # Test with default parameters
            self.vhk_2d.plot("Input")
            mock_show.assert_called_once()
            mock_show.reset_mock()

            # Check that both subplots were created
            mock_fig.add_subplot.assert_any_call(2, 1, 1)  # First call for wrist plot
            mock_fig.add_subplot.assert_any_call(
                2, 1, 2
            )  # Second call for fingers plot

            # Test with custom parameters
            mock_fig.reset_mock()
            mock_ax1.reset_mock()
            mock_ax2.reset_mock()
            mock_fig.add_subplot.side_effect = [mock_ax1, mock_ax2]

            self.vhk_2d.plot("Input", nr_of_fingers=3, visualize_wrist=False)
            mock_show.assert_called_once()

            # Check that wrist data wasn't plotted when visualize_wrist=False
            mock_ax1.plot.assert_not_called()

            # Test with non-existent representation
            with self.assertRaises(KeyError):
                self.vhk_2d.plot("NonExistentRepresentation")

    def test_plot_value_error(self):
        """Test that plot raises ValueError for data with wrong DOF count."""
        # Create VirtualHandKinematics object
        vhk = VirtualHandKinematics(self.vhk_data_2d, 100.0)

        # Create a representation with wrong DOF count (use 7 instead of 9)
        wrong_dof_data = np.random.randn(7, 100)

        # Mock __getitem__ to return the wrong DOF data for a specific representation
        original_getitem = vhk.__getitem__

        def mock_getitem(key):
            if key == "WrongDOF":
                return wrong_dof_data
            return original_getitem(key)

        vhk.__getitem__ = mock_getitem

        # Add the representation to _data
        vhk._data["WrongDOF"] = wrong_dof_data

        # Test that ValueError is raised for data with wrong DOF count
        with self.assertRaises(ValueError):
            with patch("matplotlib.pyplot.figure"), patch("matplotlib.pyplot.show"):
                vhk.plot("WrongDOF")


if __name__ == "__main__":
    unittest.main()
