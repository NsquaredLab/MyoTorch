import numpy as np
import pytest

from myotorch.datasets.filters.emg_augmentations import MagnitudeWarping


def generate_emg_data(shape=(8, 1000)):
    """Generate synthetic EMG-like data for testing."""
    # Use a fixed seed for reproducible tests
    np.random.seed(42)
    return (
        np.random.randn(*shape).astype(np.float64) * 100
    )  # Using float64 to match expected behavior


class TestMagnitudeWarpingAugmentation:
    @pytest.mark.loop(3)  # Reduced iterations for faster tests
    @pytest.mark.parametrize(
        "nr_of_point_for_spline,gaussian_mean,gaussian_std",
        [
            (4, 1.0, 0.2),
            (6, 1.0, 0.35),  # Default values
            (10, 1.05, 0.3),
        ],
    )
    def test_magnitude_warping(
        self, nr_of_point_for_spline, gaussian_mean, gaussian_std
    ):
        """Test MagnitudeWarping augmentation with different parameters."""
        # Generate test data
        original_data = generate_emg_data()
        original_shape = original_data.shape

        # Number of grids should be a divisor of the number of channels
        nr_of_grids = original_shape[0] // 2  # Using half the number of channels

        # Create the filter
        filter_obj = MagnitudeWarping(
            nr_of_point_for_spline=nr_of_point_for_spline,
            gaussian_mean=gaussian_mean,
            gaussian_std=gaussian_std,
            nr_of_grids=nr_of_grids,
            input_is_chunked=False,
        )

        # Apply the filter to a copy of the original data
        augmented_data = filter_obj(original_data.copy())

        # Check that the shape is preserved
        assert augmented_data.shape == original_shape

        # Check that the data has been modified (warping applied)
        mad = np.mean(np.abs(original_data - augmented_data))
        assert mad > 0.1, (
            f"Magnitude warping didn't modify data sufficiently. MAD: {mad}"
        )

        # Check that the data is still float64
        assert augmented_data.dtype == np.float64

        # Verify the magnitude warping didn't completely destroy the signal
        # The mean absolute error should not exceed a certain threshold
        assert mad < 200, f"Magnitude warping changed data too drastically. MAD: {mad}"

    def test_magnitude_warping_edge_cases(self):
        """Test MagnitudeWarping augmentation with edge cases."""
        # Test with minimal warping (mean=1.0, very small std)
        data = generate_emg_data()
        minimal_filter = MagnitudeWarping(
            nr_of_point_for_spline=6,
            gaussian_mean=1.0,
            gaussian_std=0.01,  # Very small std
            nr_of_grids=2,
            input_is_chunked=False,
        )
        minimal_result = minimal_filter(data.copy())

        # Should be modified but changes should be small
        mad_small = np.mean(np.abs(data - minimal_result))
        assert mad_small > 0.01, (
            f"Even minimal warping should modify data. MAD: {mad_small}"
        )

        # Test with larger warping (larger std)
        large_filter = MagnitudeWarping(
            nr_of_point_for_spline=6,
            gaussian_mean=1.0,
            gaussian_std=0.8,  # Larger std
            nr_of_grids=2,
            input_is_chunked=False,
        )
        large_result = large_filter(data.copy())

        # Should be modified more significantly
        mad_large = np.mean(np.abs(data - large_result))
        assert mad_large > 1.0, (
            f"Larger warping should modify data more. MAD: {mad_large}"
        )

        # Compare the effects - larger std should cause more modification
        assert mad_large > mad_small, "Larger std should cause more significant changes"

        # Test without specifying nr_of_grids
        with pytest.raises(ValueError, match="nr_of_grids must be specified"):
            invalid_filter = MagnitudeWarping(
                input_is_chunked=False,
            )

    def test_magnitude_warping_different_shapes(self):
        """Test MagnitudeWarping augmentation with different data dimensions."""
        # Test with standard 2D EMG array - only this case should be supported
        data_2d = generate_emg_data(shape=(8, 1000))
        filter_2d = MagnitudeWarping(
            nr_of_grids=4,
            input_is_chunked=False,
        )
        result_2d = filter_2d(data_2d.copy())
        assert result_2d.shape == data_2d.shape

        # Test with small number of samples
        data_small = generate_emg_data(shape=(8, 20))
        filter_small = MagnitudeWarping(
            nr_of_grids=4,
            input_is_chunked=False,
        )
        result_small = filter_small(data_small.copy())
        assert result_small.shape == data_small.shape

        # Test with different grid configurations
        data_multi = generate_emg_data(shape=(12, 1000))

        # Test with nr_of_grids as a divisor of number of channels
        filter_divisor = MagnitudeWarping(
            nr_of_grids=3,  # 12 / 3 = 4 channels per grid
            input_is_chunked=False,
        )
        result_divisor = filter_divisor(data_multi.copy())
        assert result_divisor.shape == data_multi.shape

    def test_invalid_grid_configuration(self):
        """Test MagnitudeWarping with invalid grid configuration."""
        # Create data where channels aren't divisible by nr_of_grids
        data_odd = generate_emg_data(shape=(7, 1000))  # 7 channels

        # Test with nr_of_grids that doesn't divide evenly
        filter_non_divisor = MagnitudeWarping(
            nr_of_grids=3,  # 7 is not divisible by 3
            input_is_chunked=False,
        )

        # Implementation might handle this without error but give incorrect results
        # We should verify the filter's behavior in this case
        with pytest.raises(ValueError):
            filter_non_divisor(data_odd.copy())

    def test_filter_parameters(self):
        """Test the filter parameters are correctly stored and used."""
        filter_obj = MagnitudeWarping(
            nr_of_point_for_spline=8,
            gaussian_mean=0.9,
            gaussian_std=0.4,
            nr_of_grids=4,
            input_is_chunked=False,
            name="MyCustomWarping",
        )

        # Check parameters are set correctly
        assert filter_obj.nr_of_point_for_spline == 8
        assert filter_obj.gaussian_mean == 0.9
        assert filter_obj.gaussian_std == 0.4
        assert filter_obj.nr_of_grids == 4
        assert filter_obj.input_is_chunked is False
        assert filter_obj.name == "MyCustomWarping"

        # Apply the filter
        test_data = generate_emg_data(shape=(8, 200))
        result = filter_obj(test_data.copy())

        # Ensure data was modified
        mad = np.mean(np.abs(test_data - result))
        assert mad > 0.1, f"Filter didn't modify data sufficiently. MAD: {mad}"
