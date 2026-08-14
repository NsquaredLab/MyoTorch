import numpy as np
import pytest

from myotorch.datasets.filters.generic import _get_windows_with_shift


class TestGetWindowsWithShift:
    def test_basic_shape_1d(self):
        """Test _get_windows_with_shift with 1D input array."""
        # Create a 1D array [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        input_array = np.arange(10)
        window_size = 3
        shift = 1

        # Expected: 8 windows of size 3
        # Shape should be (8, 3)
        output = _get_windows_with_shift(input_array, window_size, shift)

        # Check if output shape is correct (n_windows, window_size)
        assert output.shape == (8, 3)

        # Check if the content is as expected
        expected_windows = np.array(
            [
                [0, 1, 2],
                [1, 2, 3],
                [2, 3, 4],
                [3, 4, 5],
                [4, 5, 6],
                [5, 6, 7],
                [6, 7, 8],
                [7, 8, 9],
            ]
        )
        assert np.array_equal(output, expected_windows)

    def test_basic_shape_2d(self):
        """Test _get_windows_with_shift with 2D input array."""
        # Create a 2D array of shape (2, 10)
        input_array = np.array(
            [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]]
        )
        window_size = 3
        shift = 2

        # Expected: 4 windows of size 3
        # Shape should be (4, 2, 3) - (n_windows, channels, window_size)
        output = _get_windows_with_shift(input_array, window_size, shift)

        # Check if output shape is correct (n_windows, channels, window_size)
        assert output.shape == (4, 2, 3)

        # Check first and last windows
        assert np.array_equal(output[0, 0, :], np.array([0, 1, 2]))
        assert np.array_equal(output[0, 1, :], np.array([10, 11, 12]))
        assert np.array_equal(output[3, 0, :], np.array([6, 7, 8]))
        assert np.array_equal(output[3, 1, :], np.array([16, 17, 18]))

    def test_basic_shape_3d(self):
        """Test _get_windows_with_shift with 3D input array."""
        # Create a 3D array of shape (2, 3, 10)
        input_array = np.zeros((2, 3, 10))
        for i in range(2):
            for j in range(3):
                input_array[i, j, :] = np.arange(10) + 10 * (j + 3 * i)

        window_size = 4
        shift = 2

        # Expected: 4 windows of size 4
        # Shape should be (4, 2, 3, 4) - (n_windows, dim1, dim2, window_size)
        output = _get_windows_with_shift(input_array, window_size, shift)

        # Check if output shape is correct
        assert output.shape == (4, 2, 3, 4)

        # Test a specific window
        for i in range(2):
            for j in range(3):
                val_offset = 10 * (j + 3 * i)
                assert np.array_equal(
                    output[1, i, j, :], np.array([2, 3, 4, 5]) + val_offset
                )

    def test_different_shifts(self):
        """Test _get_windows_with_shift with different shift values."""
        input_array = np.arange(20)
        window_size = 5

        # Test with shift=1
        output_shift1 = _get_windows_with_shift(input_array, window_size, shift=1)
        assert output_shift1.shape == (16, 5)  # (n_windows, window_size)

        # Test with shift=5 (non-overlapping)
        output_shift5 = _get_windows_with_shift(input_array, window_size, shift=5)
        assert output_shift5.shape == (4, 5)
        assert np.array_equal(output_shift5[0], np.array([0, 1, 2, 3, 4]))
        assert np.array_equal(output_shift5[1], np.array([5, 6, 7, 8, 9]))

        # Test with shift > window_size (gaps between windows)
        output_shift7 = _get_windows_with_shift(input_array, window_size, shift=7)
        assert output_shift7.shape == (3, 5)
        assert np.array_equal(output_shift7[0], np.array([0, 1, 2, 3, 4]))
        assert np.array_equal(output_shift7[1], np.array([7, 8, 9, 10, 11]))

    def test_edge_cases(self):
        """Test _get_windows_with_shift with edge cases."""
        # Test with window_size equal to array length
        input_array = np.arange(10)
        window_size = 10
        shift = 1
        output = _get_windows_with_shift(input_array, window_size, shift)
        assert output.shape == (1, 10)
        assert np.array_equal(output[0], input_array)

        # Test with shift equal to window_size
        input_array = np.arange(20)
        window_size = 5
        shift = 5
        output = _get_windows_with_shift(input_array, window_size, shift)
        assert output.shape == (4, 5)

        # Test with high dimensional array
        input_array = np.zeros((2, 3, 4, 10))
        window_size = 3
        shift = 2
        output = _get_windows_with_shift(input_array, window_size, shift)
        assert output.shape == (
            4,
            2,
            3,
            4,
            3,
        )  # (n_windows, dim1, dim2, dim3, window_size)

    def test_readability(self):
        """Test that the windows are read-only."""
        input_array = np.arange(10)
        window_size = 3
        shift = 1
        output = _get_windows_with_shift(input_array, window_size, shift)

        with pytest.raises(ValueError):
            output[0, 0] = 99  # This should raise an error as output is read-only

    def test_memory_efficiency(self):
        """Test that _get_windows_with_shift uses views and not copies."""
        input_array = np.arange(10)
        window_size = 3
        shift = 1
        output = _get_windows_with_shift(input_array, window_size, shift)

        # Change the input array
        input_array[0] = 99

        # If output is a view, changing input_array should affect output
        assert output[0, 0] == 99
