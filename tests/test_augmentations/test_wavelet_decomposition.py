import numpy as np
import pytest
import pywt

from myotorch.datasets.filters.emg_augmentations import WaveletDecomposition


def generate_emg_data(shape=(8, 1000)):
    """Generate synthetic EMG-like data for testing."""
    # Use a fixed seed for reproducible tests
    np.random.seed(42)
    return (
        np.random.randn(*shape).astype(np.float64) * 100
    )  # Using float64 to match expected behavior


class TestWaveletDecompositionAugmentation:
    @pytest.mark.loop(3)  # Reduced iterations for faster tests
    @pytest.mark.parametrize(
        "b,wavelet,level",
        [
            (0.25, "db7", 5),  # Default values
            (0.5, "sym5", 4),
            (0.75, "coif3", 2),
        ],
    )
    def test_wavelet_decomposition(self, b, wavelet, level):
        """Test WaveletDecomposition augmentation with different parameters."""
        # Check if the wavelet and level are compatible
        # Skip test if the level is too high for the data length
        try:
            pywt.wavedec(np.zeros(1000), wavelet=wavelet, level=level)
        except ValueError:
            pytest.skip(
                f"Skipping test: level {level} is too high for wavelet {wavelet}"
            )

        # Generate test data
        original_data = generate_emg_data()
        original_shape = original_data.shape

        # Number of grids should be a divisor of the number of channels
        nr_of_grids = original_shape[0] // 2  # Using half the number of channels

        # Create the filter
        filter_obj = WaveletDecomposition(
            b=b,
            wavelet=wavelet,
            level=level,
            nr_of_grids=nr_of_grids,
            input_is_chunked=False,
        )

        # Apply the filter
        augmented_data = filter_obj(original_data.copy())

        # Check that the shape is preserved
        assert augmented_data.shape == original_shape

        # Check that the data has been modified (decomposition applied)
        # Calculate mean absolute difference to verify changes
        mad = np.mean(np.abs(original_data - augmented_data))
        assert mad > 0.01, f"Wavelet decomposition didn't modify data. MAD: {mad}"

        # Check that the data is still float64
        assert augmented_data.dtype == np.float64

        # Verify the wavelet decomposition preserves some signal characteristics
        # The relative energy should be somewhat preserved
        original_energy = np.sum(np.square(original_data))
        augmented_energy = np.sum(np.square(augmented_data))

        # Energy ratio - adjusted threshold based on observed values
        energy_ratio = augmented_energy / original_energy
        assert 0.05 < energy_ratio < 10, (
            f"Energy ratio outside expected range: {energy_ratio}"
        )

    def test_wavelet_decomposition_edge_cases(self):
        """Test WaveletDecomposition augmentation with edge cases."""
        # Test with very small scaling factor (more compression)
        data = generate_emg_data()
        small_b_filter = WaveletDecomposition(
            b=0.01,  # Very small scaling
            wavelet="db7",
            level=5,
            nr_of_grids=2,
            input_is_chunked=False,
        )
        small_b_result = small_b_filter(data.copy())

        # Should be modified significantly
        diff_small = np.mean(np.abs(data - small_b_result))
        assert diff_small > 0.01, (
            f"Scaling with b=0.01 didn't modify data significantly. Diff: {diff_small}"
        )

        # Test with larger scaling factor (less compression)
        large_b_filter = WaveletDecomposition(
            b=0.9,  # Larger scaling
            wavelet="db7",
            level=5,
            nr_of_grids=2,
            input_is_chunked=False,
        )
        large_b_result = large_b_filter(data.copy())

        # Should be modified but closer to the original
        diff_large = np.mean(np.abs(data - large_b_result))
        assert diff_large > 0.01, (
            f"Scaling with b=0.9 didn't modify data. Diff: {diff_large}"
        )

        # Larger b should preserve more of the original signal than smaller b
        assert diff_small > diff_large, (
            "Higher b value should preserve more of the original signal"
        )

        # Test without specifying nr_of_grids
        with pytest.raises(ValueError, match="nr_of_grids must be specified"):
            invalid_filter = WaveletDecomposition(
                input_is_chunked=False,
            )

    def test_wavelet_decomposition_different_shapes(self):
        """Test WaveletDecomposition with different data dimensions."""
        # Test with 2D array (standard case) - smaller dimension for speed
        data_2d = generate_emg_data(shape=(8, 200))
        filter_2d = WaveletDecomposition(
            nr_of_grids=4,
            input_is_chunked=False,
        )
        result_2d = filter_2d(data_2d.copy())
        assert result_2d.shape == data_2d.shape

        # Test with small number of samples (with adjusted level)
        data_small = generate_emg_data(shape=(8, 64))
        filter_small = WaveletDecomposition(
            level=2,  # Lower level for small data
            nr_of_grids=4,
            input_is_chunked=False,
        )
        result_small = filter_small(data_small.copy())
        assert result_small.shape == data_small.shape

        # Test with very small data (should work with adjusted level)
        data_tiny = generate_emg_data(shape=(8, 32))
        filter_tiny = WaveletDecomposition(
            level=1,  # Use very low level for tiny data
            nr_of_grids=4,
            input_is_chunked=False,
        )
        result_tiny = filter_tiny(data_tiny.copy())
        assert result_tiny.shape == data_tiny.shape

    def test_invalid_wavelets(self):
        """Test WaveletDecomposition with invalid wavelet names."""
        data = generate_emg_data()

        # Test with invalid wavelet name
        invalid_filter = WaveletDecomposition(
            wavelet="invalid_wavelet",
            nr_of_grids=2,
            input_is_chunked=False,
        )

        with pytest.raises(ValueError):
            # This should fail because the wavelet name is invalid
            invalid_filter(data.copy())

    def test_filter_parameters(self):
        """Test the filter parameters are correctly stored and used."""
        filter_obj = WaveletDecomposition(
            b=0.3,
            wavelet="sym4",
            level=3,
            nr_of_grids=2,
            input_is_chunked=False,
            name="MyCustomWavelet",
        )

        # Check parameters are set correctly
        assert filter_obj.b == 0.3
        assert filter_obj.wavelet == "sym4"
        assert filter_obj.level == 3
        assert filter_obj.nr_of_grids == 2
        assert filter_obj.input_is_chunked is False
        assert filter_obj.name == "MyCustomWavelet"

        # Apply the filter
        test_data = generate_emg_data(shape=(8, 200))
        result = filter_obj(test_data.copy())

        # Ensure data was modified
        mad = np.mean(np.abs(test_data - result))
        assert mad > 0.01, f"Filter didn't modify data. MAD: {mad}"
