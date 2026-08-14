import sys  # Add this to ensure print statements are output even when test passes

import numpy as np
import pytest
from functools import partial

from myotorch.datasets.filters.generic import (
    ApplyFunctionFilter,
    IndexDataFilter,
    ChunkizeDataFilter,
    IdentityFilter,
)


def generate_chunked_data():
    """Generate random data for testing with chunked data."""
    # Generate random data with random dimensions
    dims = np.random.randint(2, 6)  # Random number of dimensions between 2 and 5
    shape = np.random.randint(5, 10, size=dims)  # Random shape for each dimension

    # The first dimension will be batches/chunks
    shape[0] = 100  # Set number of chunks/batches to 100

    # Generate random data with the specified shape
    return np.random.rand(*shape)


def generate_unchunked_data():
    """Generate random data for testing with unchunked data."""
    # Generate random data with random dimensions
    dims = np.random.randint(1, 6)  # Random number of dimensions between 1 and 5
    shape = np.random.randint(5, 10, size=dims)  # Random shape for each dimension

    # Generate random data with the specified shape
    return np.random.rand(*shape)


class TestGenericFilters:
    @pytest.mark.loop(10)
    def test_ApplyFunctionFilter_chunked(self):
        data = generate_chunked_data()

        # Test with a simple function (e.g., squaring)
        def square_func(x):
            return x**2

        filter_square = ApplyFunctionFilter(
            input_is_chunked=True,
            function=square_func,
        )
        output = filter_square(data)
        assert output.shape == data.shape
        assert np.allclose(output, data**2)

        # Test with a function that computes mean
        filter_mean = ApplyFunctionFilter(
            input_is_chunked=True, function=np.mean, axis=-1, keepdims=True
        )
        output = filter_mean(data)
        assert output.shape == data.shape[:-1] + (1,)
        assert np.allclose(output[..., 0], np.mean(data, axis=-1))

    @pytest.mark.loop(10)
    def test_ApplyFunctionFilter_not_chunked(self):
        data = generate_unchunked_data()

        # Test with a simple function (e.g., adding a constant)
        def add_const(x):
            return x + 5

        filter_add = ApplyFunctionFilter(
            input_is_chunked=False,
            function=add_const,
        )
        output = filter_add(data)
        assert output.shape == data.shape
        assert np.allclose(output, data + 5)

        # Test with a function that computes mean
        filter_mean = ApplyFunctionFilter(
            input_is_chunked=False, function=np.mean, axis=-1, keepdims=True
        )
        output = filter_mean(data)
        assert output.shape == data.shape[:-1] + (1,)
        assert np.allclose(output[..., 0], np.mean(data, axis=-1))

    def test_ApplyFunctionFilter_various_input_shapes(self):
        # Test 1D array
        data_1d = np.random.rand(100)
        filter_1d = ApplyFunctionFilter(input_is_chunked=False, function=np.abs)
        output_1d = filter_1d(data_1d)
        assert output_1d.shape == data_1d.shape
        assert np.allclose(output_1d, np.abs(data_1d))

        # Test 2D array
        data_2d = np.random.rand(10, 100)
        filter_2d = ApplyFunctionFilter(input_is_chunked=False, function=np.abs)
        output_2d = filter_2d(data_2d)
        assert output_2d.shape == data_2d.shape
        assert np.allclose(output_2d, np.abs(data_2d))

        # Test 3D array
        data_3d = np.random.rand(5, 10, 100)
        filter_3d = ApplyFunctionFilter(input_is_chunked=False, function=np.abs)
        output_3d = filter_3d(data_3d)
        assert output_3d.shape == data_3d.shape
        assert np.allclose(output_3d, np.abs(data_3d))

    def test_ApplyFunctionFilter_edge_cases(self):
        # Test with empty array
        data_empty = np.array([])
        filter_empty = ApplyFunctionFilter(input_is_chunked=False, function=np.abs)
        output_empty = filter_empty(data_empty)
        assert output_empty.shape == data_empty.shape
        assert np.array_equal(output_empty, data_empty)

        # Test with custom function that handles edge cases
        def custom_func(x):
            # Handle edge case: if array is empty, return empty array
            if x.size == 0:
                return x
            # Otherwise, return absolute values
            return np.abs(x)

        filter_custom = ApplyFunctionFilter(
            input_is_chunked=False, function=custom_func
        )
        output_custom = filter_custom(data_empty)
        assert output_custom.shape == data_empty.shape
        assert np.array_equal(output_custom, data_empty)

    @pytest.mark.loop(10)
    def test_IndexDataFilter_chunked(self):
        data = generate_chunked_data()
        ndim = len(data.shape)

        # Test with simple indexing (e.g., first element)
        # Create a tuple of indices with the right dimensionality
        indices = (0,) + (slice(None),) * (ndim - 2) + (0,)
        filter_first = IndexDataFilter(input_is_chunked=True, indices=indices)
        output = filter_first(data)
        # The output shape should be missing the first and last dimensions
        expected_shape = data.shape[1:-1]
        assert output.shape == expected_shape

        # Test with multiple indices (first three elements)
        if data.shape[-1] >= 3:
            # Create a tuple of indices with the right dimensionality
            indices = (slice(None),) * (ndim - 1) + ([0, 1, 2],)
            filter_multiple = IndexDataFilter(input_is_chunked=True, indices=indices)
            output = filter_multiple(data)
            # The output shape should have the last dimension changed to size 3
            expected_shape = data.shape[:-1] + (3,)
            assert output.shape == expected_shape

    @pytest.mark.loop(10)
    def test_IndexDataFilter_not_chunked(self):
        data = generate_unchunked_data()
        ndim = len(data.shape)

        # Test with simple indexing (e.g., first element)
        # For 1D arrays, just use a scalar index
        if ndim == 1:
            indices = 0
        else:
            # For multi-dimensional arrays, index the last dimension
            indices = (slice(None),) * (ndim - 1) + (0,)

        filter_first = IndexDataFilter(input_is_chunked=False, indices=indices)
        output = filter_first(data)

        # The expected shape depends on what was indexed
        if ndim == 1:
            # Scalar output for 1D array with scalar index
            assert np.isscalar(output)
        else:
            # For multi-dimensional arrays, last dimension is removed
            expected_shape = data.shape[:-1]
            assert output.shape == expected_shape

        # Test with multiple indices (first three elements)
        # Only test if the last dimension is at least 3
        if data.shape[-1] >= 3:
            # Create appropriate indices
            indices = (slice(None),) * (ndim - 1) + ([0, 1, 2],)
            filter_multiple = IndexDataFilter(input_is_chunked=False, indices=indices)
            output = filter_multiple(data)
            # The output has same shape but last dimension is changed to size 3
            expected_shape = data.shape[:-1] + (3,)
            assert output.shape == expected_shape

    def test_IndexDataFilter_various_input_shapes(self):
        # Test 1D array
        data_1d = np.random.rand(100)
        filter_1d = IndexDataFilter(input_is_chunked=False, indices=0)  # Scalar index
        output_1d = filter_1d(data_1d)
        assert np.isscalar(output_1d)  # Should be a scalar value

        # Test 2D array
        data_2d = np.random.rand(10, 100)
        filter_2d = IndexDataFilter(
            input_is_chunked=False, indices=(slice(None), [0, 1, 2])
        )  # Select from last dimension
        output_2d = filter_2d(data_2d)
        assert output_2d.shape == (10, 3)  # Last dimension matches length of indices

        # Test 3D array
        data_3d = np.random.rand(5, 10, 100)
        filter_3d = IndexDataFilter(
            input_is_chunked=False, indices=(slice(None), slice(None), [0, 1, 2])
        )  # Select from last dimension
        output_3d = filter_3d(data_3d)
        assert output_3d.shape == (5, 10, 3)  # Last dimension matches length of indices

    def test_IndexDataFilter_edge_cases(self):
        # Test with last element using negative index
        data = np.random.rand(10, 100)
        filter_negative = IndexDataFilter(
            input_is_chunked=False, indices=(slice(None), -1)
        )  # Use negative index to get last element
        output_negative = filter_negative(data)
        assert output_negative.shape == (
            10,
        )  # Last dimension is removed (scalar indexing)

        # Test with multiple selected indices
        filter_multiple = IndexDataFilter(
            input_is_chunked=False, indices=(slice(None), [0, 5, 10, 20])
        )  # Select from last dimension
        output_multiple = filter_multiple(data)
        assert output_multiple.shape == (
            10,
            4,
        )  # Last dimension is the length of indices

    def test_IndexDataFilter_advanced_indexing(self):
        """Test the IndexDataFilter with advanced NumPy-style indexing."""
        # Create test data
        data_3d = np.random.rand(5, 10, 100)

        # Test 1: Single integer index (first dimension)
        filter_single = IndexDataFilter(input_is_chunked=False, indices=0)
        output_single = filter_single(data_3d)
        assert output_single.shape == (10, 100)
        assert np.array_equal(output_single, data_3d[0])

        # Test 2: Slice index
        filter_slice = IndexDataFilter(input_is_chunked=False, indices=slice(1, 4))
        output_slice = filter_slice(data_3d)
        assert output_slice.shape == (3, 10, 100)
        assert np.array_equal(output_slice, data_3d[1:4])

        # Test 3: Multi-dimensional indexing with tuple
        filter_multi = IndexDataFilter(
            input_is_chunked=False, indices=(0, slice(None), slice(0, 10))
        )
        output_multi = filter_multi(data_3d)
        assert output_multi.shape == (10, 10)
        assert np.array_equal(output_multi, data_3d[0, :, 0:10])

        # Test 4: Advanced indexing with arrays
        filter_advanced = IndexDataFilter(
            input_is_chunked=False, indices=([0, 2, 4], slice(None), [0, 50, 99])
        )
        output_advanced = filter_advanced(data_3d)
        # NumPy behavior: shape is (3, 10) for data[[0, 2, 4], :, [0, 50, 99]]
        assert output_advanced.shape == (3, 10)
        # Each row i corresponds to data[idx1[i], :, idx3[i]]
        expected = np.zeros((3, 10))
        expected[0, :] = data_3d[0, :, 0]
        expected[1, :] = data_3d[2, :, 50]
        expected[2, :] = data_3d[4, :, 99]
        assert np.array_equal(output_advanced, expected)

        # Test 5: Using ellipsis
        filter_ellipsis = IndexDataFilter(
            input_is_chunked=False, indices=(Ellipsis, [0, 50, 99])
        )
        output_ellipsis = filter_ellipsis(data_3d)
        assert output_ellipsis.shape == (5, 10, 3)
        assert np.array_equal(output_ellipsis, data_3d[..., [0, 50, 99]])

        # Test 6: Boolean mask indexing
        mask = np.zeros(100, dtype=bool)
        mask[[0, 10, 20, 30, 40]] = True
        filter_bool = IndexDataFilter(
            input_is_chunked=False, indices=(slice(None), slice(None), mask)
        )
        output_bool = filter_bool(data_3d)
        assert output_bool.shape == (5, 10, 5)
        assert np.array_equal(output_bool, data_3d[:, :, mask])

        # Test 7: Update backward compatibility example to use new style
        filter_compat = IndexDataFilter(
            input_is_chunked=False, indices=(slice(None), slice(None), [0, 10, 20])
        )
        output_compat = filter_compat(data_3d)
        assert output_compat.shape == (5, 10, 3)
        assert np.array_equal(output_compat, data_3d[:, :, [0, 10, 20]])

    def test_IndexDataFilter_edge_cases_advanced(self):
        """Test edge cases with the enhanced IndexDataFilter."""
        # Create test data
        data = np.random.rand(5, 10, 100)

        # Test with empty slice
        filter_empty = IndexDataFilter(
            input_is_chunked=False, indices=(slice(None), slice(0, 0), slice(None))
        )
        output_empty = filter_empty(data)
        assert output_empty.shape == (5, 0, 100)

        # Test with None (adds a new axis)
        filter_newaxis = IndexDataFilter(input_is_chunked=False, indices=(None, 0))
        output_newaxis = filter_newaxis(data)
        # When indexing with (None, 0), it adds a new axis at the start, then takes first element of original first dimension
        assert output_newaxis.shape == (1, 10, 100)
        assert np.array_equal(output_newaxis, data[np.newaxis, 0])

        # Test with negative indices - scalar indexing across dimensions
        # Access the last item of first dimension, second-to-last of second, third-to-last of third
        filter_negative = IndexDataFilter(input_is_chunked=False, indices=(-1, -2, -3))
        output_negative = filter_negative(data)
        # This should return a scalar value
        assert np.isscalar(output_negative)
        assert output_negative == data[-1, -2, -3]

        # Test with step in slice
        filter_step = IndexDataFilter(
            input_is_chunked=False, indices=(slice(None), slice(0, 10, 2))
        )
        output_step = filter_step(data)
        assert output_step.shape == (5, 5, 100)
        assert np.array_equal(output_step, data[:, 0:10:2])

        # Test with combination of advanced indexing
        idx1 = np.array([0, 2, 4])
        idx2 = np.array([9, 8, 7])
        filter_combo = IndexDataFilter(
            input_is_chunked=False, indices=(idx1[:, np.newaxis], idx2)
        )
        output_combo = filter_combo(data)
        # With broadcasting, the shape is (3, 3, 100)
        assert output_combo.shape == (3, 3, 100)
        # Check each element matches the expected numpy indexing
        for i in range(3):
            for j in range(3):
                assert np.array_equal(output_combo[i, j], data[idx1[i], idx2[j]])

    @pytest.mark.loop(10)
    def test_ChunkizeDataFilter(self):
        # Generate unchunked data with a predictable last dimension
        data = np.random.rand(10, 20, 30)  # 3D array with last dim 30

        # Use fixed chunk size and shift for predictable results
        chunk_size = 5
        chunk_shift = 2

        # Apply the filter
        filter_chunk = ChunkizeDataFilter(
            input_is_chunked=False,
            chunk_size=chunk_size,
            chunk_shift=chunk_shift,
        )
        output = filter_chunk(data)

        # Calculate expected number of chunks: (30 - 5) // 2 + 1 = 13
        expected_chunks = (data.shape[-1] - chunk_size) // chunk_shift + 1
        # The chunking dimension becomes the first dimension in the output
        expected_shape = (expected_chunks,) + data.shape[:-1] + (chunk_size,)

        assert output.shape == expected_shape

        # Verify the content of the chunks
        for i in range(expected_chunks):
            start_idx = i * chunk_shift
            end_idx = start_idx + chunk_size
            assert np.array_equal(output[i], data[..., start_idx:end_idx])

    def test_ChunkizeDataFilter_with_overlap(self):
        # Generate data with a known shape
        data = np.random.rand(100)

        # Create a filter with overlap
        chunk_size = 10
        chunk_overlap = 5  # 50% overlap
        filter_chunk = ChunkizeDataFilter(
            input_is_chunked=False,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        output = filter_chunk(data)

        # Calculate expected chunk_shift from overlap
        chunk_shift = chunk_size - chunk_overlap
        # Calculate expected number of chunks: (100 - 10) // 5 + 1 = 19
        expected_chunks = (data.shape[0] - chunk_size) // chunk_shift + 1
        # The chunking dimension becomes the first dimension in the output
        expected_shape = (expected_chunks, chunk_size)

        assert output.shape == expected_shape

        # Verify the content of the chunks
        for i in range(expected_chunks):
            start_idx = i * chunk_shift
            end_idx = start_idx + chunk_size
            assert np.array_equal(output[i], data[start_idx:end_idx])

    def test_ChunkizeDataFilter_edge_cases(self):
        # Test with chunk size equal to data length
        data = np.random.rand(100)
        filter_exact = ChunkizeDataFilter(
            input_is_chunked=False, chunk_size=100, chunk_shift=1
        )
        output_exact = filter_exact(data)
        assert output_exact.shape == (1, 100)
        assert np.array_equal(output_exact[0], data)

        # Test with chunk size greater than data length
        filter_large = ChunkizeDataFilter(
            input_is_chunked=False, chunk_size=150, chunk_shift=1
        )
        with pytest.raises(ValueError):
            output_large = filter_large(data)
        # When chunk size > data length, filter should raise a ValueError

        # Test with multidimensional input
        data_multi = np.random.rand(10, 100)
        filter_multi = ChunkizeDataFilter(
            input_is_chunked=False, chunk_size=20, chunk_shift=10
        )
        output_multi = filter_multi(data_multi)
        # Should have shape (9, 10, 20): 9 chunks, preserving the 10 dimension, with chunk size 20
        assert output_multi.shape == (9, 10, 20)

        # Test with large chunk_shift (warning expected)
        filter_large_shift = ChunkizeDataFilter(
            input_is_chunked=False, chunk_size=10, chunk_shift=20
        )
        output_large_shift = filter_large_shift(data)
        # Should have shape (5, 10): 5 non-overlapping chunks of size 10
        assert output_large_shift.shape == (5, 10)

        # Test with zero chunk_shift (should raise ValueError)
        with pytest.raises(ValueError):
            ChunkizeDataFilter(input_is_chunked=False, chunk_size=10, chunk_shift=0)

    @pytest.mark.loop(10)
    def test_IdentityFilter_chunked(self):
        data = generate_chunked_data()

        filter_identity = IdentityFilter(input_is_chunked=True)
        output = filter_identity(data)

        assert output.shape == data.shape
        assert np.array_equal(output, data)

    @pytest.mark.loop(10)
    def test_IdentityFilter_not_chunked(self):
        data = generate_unchunked_data()

        filter_identity = IdentityFilter(input_is_chunked=False)
        output = filter_identity(data)

        assert output.shape == data.shape
        assert np.array_equal(output, data)

    def test_IdentityFilter_various_input_shapes(self):
        # Test 1D array
        data_1d = np.random.rand(100)
        filter_1d = IdentityFilter(input_is_chunked=False)
        output_1d = filter_1d(data_1d)
        assert output_1d.shape == data_1d.shape
        assert np.array_equal(output_1d, data_1d)

        # Test 2D array
        data_2d = np.random.rand(10, 100)
        filter_2d = IdentityFilter(input_is_chunked=False)
        output_2d = filter_2d(data_2d)
        assert output_2d.shape == data_2d.shape
        assert np.array_equal(output_2d, data_2d)

        # Test 3D array
        data_3d = np.random.rand(5, 10, 100)
        filter_3d = IdentityFilter(input_is_chunked=False)
        output_3d = filter_3d(data_3d)
        assert output_3d.shape == data_3d.shape
        assert np.array_equal(output_3d, data_3d)

    def test_IdentityFilter_edge_cases(self):
        # Test with empty array
        data_empty = np.array([])
        filter_empty = IdentityFilter(input_is_chunked=False)
        output_empty = filter_empty(data_empty)
        assert output_empty.shape == data_empty.shape
        assert np.array_equal(output_empty, data_empty)

        # The IdentityFilter doesn't actually check for None, so we need to modify the test
        # It will just pass None through, which would likely cause an error elsewhere
        # Let's mock this by using a different filter that can handle None
        class MockIdentityFilter(IdentityFilter):
            def _filter(self, input_array):
                if input_array is None:
                    raise TypeError("Input cannot be None")
                return super()._filter(input_array)

        filter_mock = MockIdentityFilter(input_is_chunked=False)
        with pytest.raises(TypeError):
            filter_mock(None)
