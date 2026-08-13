import unittest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from unittest.mock import patch

from myotorch.datatypes import EMGData, create_grid_layout


class TestEMGData(unittest.TestCase):
    """Test class for EMGData functionality that's not covered by the _Data tests."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample 2D data (16 channels, 100 samples)
        self.emg_data_2d = np.random.randn(16, 100)

        # Create sample 3D data (5 chunks, 16 channels, 100 samples)
        self.emg_data_3d = np.random.randn(5, 16, 100)

        # Create a sample 4x4 grid layout
        self.grid_layout = create_grid_layout(4, 4, fill_pattern="row")

        # Create a sample EMG object with 2D data
        self.emg_2d = EMGData(self.emg_data_2d, 1000.0)

        # Create a sample EMG object with 3D data
        self.emg_3d = EMGData(self.emg_data_3d, 1000.0)

        # Create a sample EMG object with grid layout
        self.emg_with_grid = EMGData(
            self.emg_data_2d, 1000.0, grid_layouts=[self.grid_layout]
        )

    def test_initialization(self):
        """Test initialization with different data formats and grid layouts."""
        # Test initialization with 2D data
        emg_2d = EMGData(self.emg_data_2d, 1000.0)
        self.assertEqual(emg_2d.sampling_frequency, 1000.0)
        np.testing.assert_array_equal(emg_2d.input_data, self.emg_data_2d)
        self.assertIsNone(emg_2d.grid_layouts)

        # Test initialization with 3D data
        emg_3d = EMGData(self.emg_data_3d, 1000.0)
        self.assertEqual(emg_3d.sampling_frequency, 1000.0)
        np.testing.assert_array_equal(emg_3d.input_data, self.emg_data_3d)
        self.assertIsNone(emg_3d.grid_layouts)

        # Test initialization with grid layout
        emg_with_grid = EMGData(
            self.emg_data_2d, 1000.0, grid_layouts=[self.grid_layout]
        )
        self.assertEqual(emg_with_grid.sampling_frequency, 1000.0)
        np.testing.assert_array_equal(emg_with_grid.input_data, self.emg_data_2d)
        self.assertEqual(len(emg_with_grid.grid_layouts), 1)
        np.testing.assert_array_equal(emg_with_grid.grid_layouts[0], self.grid_layout)

        # Test initialization with invalid data dimensions
        with self.assertRaises(ValueError):
            EMGData(np.random.randn(16, 100, 100, 100), 1000.0)  # 4D array

        with self.assertRaises(ValueError):
            EMGData(np.random.randn(16), 1000.0)  # 1D array

    def test_grid_layout_validation(self):
        """Test grid layout validation during initialization."""
        # Test with non-numpy array grid layout
        with self.assertRaises(ValueError):
            EMGData(self.emg_data_2d, 1000.0, grid_layouts=[[1, 2], [3, 4]])

        # Test with duplicate electrode indices
        duplicate_grid = np.array([[0, 1], [1, 2]])  # Duplicate index 1
        with self.assertRaises(ValueError):
            EMGData(self.emg_data_2d, 1000.0, grid_layouts=[duplicate_grid])

        # Note: Out-of-bounds electrode index validation is not yet implemented.
        # This is documented in examples/02_filters/1_emg_grid_layouts.py.

        # Test with multiple grid layouts
        grid1 = create_grid_layout(2, 4, 8, fill_pattern="row")
        grid2 = create_grid_layout(2, 4, 8, fill_pattern="row")
        # Shift indices for second grid
        grid2[grid2 >= 0] += 8

        # This should work - 16 channels total, each grid having 8
        multi_grid_emg = EMGData(self.emg_data_2d, 1000.0, grid_layouts=[grid1, grid2])
        self.assertEqual(len(multi_grid_emg.grid_layouts), 2)

    def test_check_if_chunked(self):
        """Test the _check_if_chunked method specific to EMGData."""
        # 2D data should not be chunked
        self.assertFalse(self.emg_2d.is_chunked["Input"])

        # 3D data should be chunked
        self.assertTrue(self.emg_3d.is_chunked["Input"])

    def test_get_grid_dimensions(self):
        """Test the _get_grid_dimensions method."""
        # Without grid layouts, should return empty list
        self.assertEqual(self.emg_2d._get_grid_dimensions(), [])

        # With grid layout, should return dimensions
        dimensions = self.emg_with_grid._get_grid_dimensions()
        self.assertEqual(len(dimensions), 1)
        # For a 4x4 grid with all positions filled, expect (4, 4, 16)
        self.assertEqual(dimensions[0], (4, 4, 16))

        # Test with a grid that has missing electrodes
        grid_with_missing = create_grid_layout(
            4, 4, fill_pattern="row", missing_indices=[(0, 0), (3, 3)]
        )
        emg_with_missing = EMGData(
            self.emg_data_2d, 1000.0, grid_layouts=[grid_with_missing]
        )
        dimensions = emg_with_missing._get_grid_dimensions()
        # Should have 14 electrodes (16 - 2 missing)
        self.assertEqual(dimensions[0], (4, 4, 14))

    @patch("matplotlib.pyplot.show")
    def test_plot(self, mock_show):
        """Test the plot method with different configurations."""
        # Test plotting 2D data
        self.emg_2d.plot("Input")
        mock_show.assert_called_once()
        mock_show.reset_mock()

        # Test plotting 3D data
        self.emg_3d.plot("Input")
        mock_show.assert_called_once()
        mock_show.reset_mock()

        # Test plotting with grid layout
        self.emg_with_grid.plot("Input", use_grid_layouts=True)
        mock_show.assert_called_once()
        mock_show.reset_mock()

        # Test plotting with manual grid configuration
        self.emg_2d.plot("Input", nr_of_grids=2, nr_of_electrodes_per_grid=8)
        mock_show.assert_called_once()
        mock_show.reset_mock()

        # Test plotting with custom scaling factor
        self.emg_2d.plot("Input", scaling_factor=10.0)
        mock_show.assert_called_once()
        mock_show.reset_mock()

        # Test plotting with multiple scaling factors
        self.emg_2d.plot(
            "Input",
            nr_of_grids=2,
            nr_of_electrodes_per_grid=8,
            scaling_factor=[10.0, 20.0],
        )
        mock_show.assert_called_once()
        mock_show.reset_mock()

        # Test that assertion fails when scaling factors don't match grid count
        with self.assertRaises(AssertionError):
            self.emg_2d.plot("Input", nr_of_grids=2, scaling_factor=[10.0])

    @patch("matplotlib.pyplot.show")
    def test_plot_grid_layout(self, mock_show):
        """Test the plot_grid_layout method."""
        # Test with grid layout
        self.emg_with_grid.plot_grid_layout(0, show_indices=True)
        mock_show.assert_called_once()
        mock_show.reset_mock()

        # Test with grid layout but without indices
        self.emg_with_grid.plot_grid_layout(0, show_indices=False)
        mock_show.assert_called_once()
        mock_show.reset_mock()

        # Test with no grid layout - should raise error
        with self.assertRaises(ValueError):
            self.emg_2d.plot_grid_layout(0)

        # Test with invalid grid index
        with self.assertRaises(ValueError):
            self.emg_with_grid.plot_grid_layout(1)  # Only has one grid (index 0)


if __name__ == "__main__":
    unittest.main()
