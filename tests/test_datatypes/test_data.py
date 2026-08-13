import os
import tempfile
import unittest

import numpy as np

from myotorch.datatypes import (
    _Data,
    DeletedRepresentation,
    InputRepresentationName,
    LastRepresentationName,
)


class TestData(_Data):
    """Concrete implementation of _Data for testing."""

    def __init__(self, raw_data, sampling_frequency):
        super().__init__(
            raw_data, sampling_frequency, nr_of_dimensions_when_unchunked=2
        )

    def _check_if_chunked(self, data):
        """Checks if the data is chunked or not.

        For testing purposes, we'll say data is chunked if the shape has at least 2 dimensions
        and the first dimension is greater than 1.
        """
        if isinstance(data, DeletedRepresentation):
            shape = data.shape
        else:
            shape = data.shape

        return len(shape) >= 2 and shape[0] > 1

    def plot(self, representation=None):
        """Dummy plot method for testing."""
        pass


class TestDataClass(unittest.TestCase):
    """Test suite for the _Data base class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a simple data array (2 channels, 100 samples)
        self.raw_data = np.random.randn(2, 100)
        self.sampling_frequency = 1000.0  # 1 kHz
        self.data_obj = TestData(self.raw_data, self.sampling_frequency)

    def test_initialization(self):
        """Test that the object initializes correctly."""
        self.assertEqual(self.data_obj.sampling_frequency, self.sampling_frequency)
        np.testing.assert_array_equal(self.data_obj.input_data, self.raw_data)
        self.assertEqual(self.data_obj._last_processing_step, InputRepresentationName)

        # Test initialization with invalid sampling frequency
        with self.assertRaises(ValueError):
            TestData(self.raw_data, 0)

        with self.assertRaises(ValueError):
            TestData(self.raw_data, -100)

    def test_properties(self):
        """Test properties and their setters."""
        # Test read-only properties
        with self.assertRaises(RuntimeError):
            self.data_obj.input_data = np.zeros((2, 100))

        with self.assertRaises(RuntimeError):
            self.data_obj.processed_representations = {}

        # Test is_chunked property
        chunked_status = self.data_obj.is_chunked
        self.assertIsInstance(chunked_status, dict)
        self.assertTrue(
            chunked_status[InputRepresentationName]
        )  # Our test data is chunked

    def test_getitem(self):
        """Test item access."""
        # Check normal access to input data
        np.testing.assert_array_equal(
            self.data_obj[InputRepresentationName], self.raw_data
        )

        # Test access with nonexistent key
        with self.assertRaises(KeyError):
            self.data_obj["nonexistent"]

        # Test __setitem__ (which should raise an error)
        with self.assertRaises(RuntimeError):
            self.data_obj["new_key"] = np.zeros((2, 100))

    def test_delete_data(self):
        """Test delete_data method."""
        # Add some data manually for testing
        test_data = np.random.randn(2, 50)
        self.data_obj._data["test_rep"] = test_data
        self.data_obj._last_processing_step = "test_rep"

        # Test delete_data
        self.data_obj.delete_data("test_rep")
        self.assertIsInstance(self.data_obj._data["test_rep"], DeletedRepresentation)

        # Test deleting InputRepresentationName (should be a no-op)
        self.data_obj.delete_data(InputRepresentationName)
        self.assertIn(InputRepresentationName, self.data_obj._data)
        self.assertIsInstance(self.data_obj._data[InputRepresentationName], np.ndarray)

        # Test deleting LastRepresentationName
        self.data_obj._data["another_rep"] = np.random.randn(2, 30)
        self.data_obj._last_processing_step = "another_rep"
        self.data_obj.delete_data(LastRepresentationName)
        self.assertIsInstance(self.data_obj._data["another_rep"], DeletedRepresentation)

        # Test deleting nonexistent representation
        with self.assertRaises(KeyError):
            self.data_obj.delete_data("nonexistent")

    def test_deleted_representation_access(self):
        """Test that accessing deleted representation raises error."""
        # Add and delete a representation
        self.data_obj._data["deleted_rep"] = np.random.randn(2, 50)
        self.data_obj.delete_data("deleted_rep")

        # Accessing should raise RuntimeError since recomputation is not supported
        with self.assertRaises(RuntimeError):
            _ = self.data_obj["deleted_rep"]

    def test_save_and_load(self):
        """Test saving and loading the data."""
        # Create a temporary file for testing
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_filename = temp_file.name

        try:
            # Save the data
            self.data_obj.save(temp_filename)

            # Load the data
            loaded_data = TestData.load(temp_filename)

            # Check that the loaded data has the same attributes
            self.assertEqual(
                loaded_data.sampling_frequency, self.data_obj.sampling_frequency
            )
            np.testing.assert_array_equal(
                loaded_data.input_data, self.data_obj.input_data
            )

        finally:
            # Clean up the temporary file
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)

    def test_memory_usage(self):
        """Test the memory_usage method."""
        # Create an initial memory usage snapshot
        initial_usage = self.data_obj.memory_usage()
        self.assertIn(InputRepresentationName, initial_usage)
        self.assertEqual(
            initial_usage[InputRepresentationName][0], str(self.raw_data.shape)
        )
        self.assertEqual(
            initial_usage[InputRepresentationName][1], self.raw_data.nbytes
        )

        # Add a representation
        test_data = np.random.randn(2, 50)
        self.data_obj._data["memory_test"] = test_data

        # Check updated memory usage
        updated_usage = self.data_obj.memory_usage()
        self.assertIn("memory_test", updated_usage)
        self.assertEqual(updated_usage["memory_test"][0], str(test_data.shape))
        self.assertEqual(updated_usage["memory_test"][1], test_data.nbytes)

        # Delete the data and check that memory usage shows 0 bytes
        self.data_obj.delete_data("memory_test")
        after_delete_usage = self.data_obj.memory_usage()
        self.assertIn("memory_test", after_delete_usage)
        self.assertEqual(after_delete_usage["memory_test"][1], 0)

    def test_str_and_repr(self):
        """Test the string representation methods."""
        # Create a basic object and check its representation
        simple_data = TestData(np.zeros((2, 10)), 1000.0)

        # Test __repr__
        repr_str = repr(simple_data)
        self.assertIn("TestData", repr_str)
        self.assertIn("Sampling frequency: 1000.0 Hz", repr_str)
        self.assertIn("Input (2, 10)", repr_str)

        # Test __str__
        str_output = str(simple_data)
        self.assertIn("TestData", str_output)
        self.assertIn("Sampling frequency: 1000.0 Hz", str_output)
        self.assertIn("Input (2, 10)", str_output)

        # Add a representation and check that it appears
        simple_data._data["test_rep"] = np.zeros((2, 10))
        updated_repr = repr(simple_data)
        self.assertIn("test_rep", updated_repr)
        self.assertIn("Representations:", updated_repr)

    def test_is_chunked_cache(self):
        """Test that is_chunked uses caching correctly."""
        # Access is_chunked twice - should use cache
        status1 = self.data_obj.is_chunked
        status2 = self.data_obj.is_chunked
        self.assertEqual(status1, status2)

        # Add new data - cache should be invalidated on next access
        self.data_obj._data["new_rep"] = np.random.randn(2, 50)
        status3 = self.data_obj.is_chunked
        self.assertIn("new_rep", status3)


if __name__ == "__main__":
    unittest.main()
