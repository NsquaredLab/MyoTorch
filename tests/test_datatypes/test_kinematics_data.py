import unittest
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import patch

from myotorch.datatypes import KinematicsData


class TestKinematicsData(unittest.TestCase):
    """Test class for KinematicsData functionality that's not covered by the _Data tests."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample 3D data (21 joints, 3 coordinates, 100 samples)
        # Each finger has 4 joints (1 base + 3 segments), and there's 1 wrist joint
        # So 1 wrist + 5 fingers * 4 joints = 21 joints total
        self.kinematics_data_3d = np.random.randn(21, 3, 100)

        # Create sample 4D data (5 chunks, 21 joints, 3 coordinates, 100 samples)
        self.kinematics_data_4d = np.random.randn(5, 21, 3, 100)

        # Create sample KinematicsData objects with 3D and 4D data
        self.kinematics_3d = KinematicsData(self.kinematics_data_3d, 100.0)
        self.kinematics_4d = KinematicsData(self.kinematics_data_4d, 100.0)

    def test_initialization(self):
        """Test initialization with different data formats."""
        # Test initialization with 3D data
        kinematics_3d = KinematicsData(self.kinematics_data_3d, 100.0)
        self.assertEqual(kinematics_3d.sampling_frequency, 100.0)
        np.testing.assert_array_equal(kinematics_3d.input_data, self.kinematics_data_3d)

        # Test initialization with 4D data
        kinematics_4d = KinematicsData(self.kinematics_data_4d, 100.0)
        self.assertEqual(kinematics_4d.sampling_frequency, 100.0)
        np.testing.assert_array_equal(kinematics_4d.input_data, self.kinematics_data_4d)

        # Test initialization with invalid data dimensions
        with self.assertRaises(ValueError):
            KinematicsData(
                np.random.randn(16, 100), 100.0
            )  # 2D array, should be at least 3D

        with self.assertRaises(ValueError):
            KinematicsData(
                np.random.randn(5, 16, 3, 100, 2), 100.0
            )  # 5D array, should be at most 4D

    def test_check_if_chunked(self):
        """Test the _check_if_chunked method specific to KinematicsData."""
        # 3D data should not be chunked
        self.assertFalse(self.kinematics_3d.is_chunked["Input"])

        # 4D data should be chunked
        self.assertTrue(self.kinematics_4d.is_chunked["Input"])

    @patch("myotorch.datatypes.KinematicsData.plot", return_value=None)
    def test_plot(self, mock_plot):
        """Test that plot method is called with the expected arguments."""
        # Test with standard parameters
        self.kinematics_3d.plot("Input", nr_of_fingers=5)
        mock_plot.assert_called_once_with("Input", nr_of_fingers=5)
        mock_plot.reset_mock()

        # Test with wrist_included=False
        self.kinematics_3d.plot("Input", nr_of_fingers=5, wrist_included=False)
        mock_plot.assert_called_once_with(
            "Input", nr_of_fingers=5, wrist_included=False
        )
        mock_plot.reset_mock()

        # Test with fewer fingers
        self.kinematics_3d.plot("Input", nr_of_fingers=3)
        mock_plot.assert_called_once_with("Input", nr_of_fingers=3)
        mock_plot.reset_mock()

        # Check that KeyError is raised for non-existent representation
        # We need to unpatch the plot method to test this
        mock_plot.side_effect = KeyError("NonExistentRepresentation")
        with self.assertRaises(KeyError):
            self.kinematics_3d.plot("NonExistentRepresentation", nr_of_fingers=5)


if __name__ == "__main__":
    unittest.main()
