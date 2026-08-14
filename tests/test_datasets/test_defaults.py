import unittest
import numpy as np
from pathlib import Path
import tempfile
import pickle
import shutil
import warnings
from unittest.mock import patch, MagicMock

from myotorch.datasets.defaults import EMBCDataset, CastelliniDataset
from myotorch.datasets.filters.emg_augmentations import (
    GaussianNoise,
    MagnitudeWarping,
    WaveletDecomposition,
)
from myotorch.datasets.filters.generic import (
    ApplyFunctionFilter,
    IndexDataFilter,
    IdentityFilter,
)
from myotorch.datasets.filters.temporal import RMSFilter, SOSFrequencyFilter


class TestDefaultDatasets(unittest.TestCase):
    """Test class for default dataset configurations."""

    def setUp(self):
        """Set up test fixtures."""
        # Suppress ResourceWarnings about unclosed files
        warnings.simplefilter("ignore", ResourceWarning)

        # Suppress DeprecationWarnings from zarr
        warnings.simplefilter("ignore", DeprecationWarning)

        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()

        # Create sample EMG data dictionary
        self.emg_data = {
            "task1": np.random.randn(320, 1000),  # 320 channels, 1000 samples
            "task2": np.random.randn(320, 1000),
            "task3": np.random.randn(320, 1000),
        }

        # Create sample ground truth kinematics data dictionary
        # Correct shape for kinematics: (n_joints, 3, n_samples)
        self.ground_truth_data = {
            "task1": np.random.randn(
                21, 3, 1000
            ),  # 21 joints, xyz coordinates, 1000 samples
            "task2": np.random.randn(21, 3, 1000),
            "task3": np.random.randn(21, 3, 1000),
        }

        # Create paths for test data
        self.emg_data_path = Path(self.temp_dir) / "emg_data.pkl"
        self.ground_truth_data_path = Path(self.temp_dir) / "ground_truth_data.pkl"
        self.save_path = Path(self.temp_dir) / "test_dataset.zarr"

        # Save test data
        with open(self.emg_data_path, "wb") as f:
            pickle.dump(self.emg_data, f)

        with open(self.ground_truth_data_path, "wb") as f:
            pickle.dump(self.ground_truth_data, f)

    def tearDown(self):
        """Tear down test fixtures."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)

    @patch("myotorch.datasets.defaults.EMGDataset")
    def test_embc_dataset_initialization(self, mock_emg_dataset):
        """Test initialization of EMBCDataset."""
        # Create the EMBC dataset
        embc_dataset = EMBCDataset(
            emg_data_path=self.emg_data_path,
            ground_truth_data_path=self.ground_truth_data_path,
            save_path=self.save_path,
            tasks_to_use=["task1", "task2"],
            debug_level=1,
            silence_zarr_warnings=True,
        )

        # Check that attributes are correctly set
        self.assertEqual(embc_dataset.emg_data_path, self.emg_data_path)
        self.assertEqual(
            embc_dataset.ground_truth_data_path, self.ground_truth_data_path
        )
        self.assertEqual(embc_dataset.save_path, self.save_path)
        self.assertEqual(embc_dataset.tasks_to_use, ["task1", "task2"])
        self.assertEqual(embc_dataset.debug_level, 1)
        self.assertEqual(embc_dataset.silence_zarr_warnings, True)

        # Check with provided data
        embc_dataset_with_data = EMBCDataset(
            emg_data_path=self.emg_data_path,
            ground_truth_data_path=self.ground_truth_data_path,
            save_path=self.save_path,
            emg_data=self.emg_data,
            ground_truth_data=self.ground_truth_data,
        )

        self.assertEqual(embc_dataset_with_data.emg_data, self.emg_data)
        self.assertEqual(
            embc_dataset_with_data.ground_truth_data, self.ground_truth_data
        )

    @patch("myotorch.datasets.defaults.EMGDataset")
    def test_embc_dataset_create(self, mock_emg_dataset):
        """Test creating the EMBC dataset configuration."""
        # Create a mock instance for the EMGDataset class
        mock_instance = MagicMock()
        mock_emg_dataset.return_value = mock_instance

        # Create the EMBC dataset
        embc_dataset = EMBCDataset(
            emg_data_path=self.emg_data_path,
            ground_truth_data_path=self.ground_truth_data_path,
            save_path=self.save_path,
            tasks_to_use=["task1", "task2"],
            debug_level=1,
            silence_zarr_warnings=True,
        )

        # Call create_dataset
        embc_dataset.create_dataset()

        # Check that EMGDataset was initialized with the correct parameters
        mock_emg_dataset.assert_called_once()
        args, kwargs = mock_emg_dataset.call_args

        # Verify key parameters were passed correctly
        self.assertEqual(kwargs["emg_data_path"], self.emg_data_path)
        self.assertEqual(kwargs["ground_truth_data_path"], self.ground_truth_data_path)
        self.assertEqual(kwargs["save_path"], self.save_path)
        self.assertEqual(kwargs["tasks_to_use"], ["task1", "task2"])
        self.assertEqual(kwargs["debug_level"], 1)
        self.assertEqual(kwargs["silence_zarr_warnings"], True)

        # Check EMBC-specific parameters
        self.assertEqual(kwargs["ground_truth_data_type"], "kinematics")
        self.assertEqual(kwargs["sampling_frequency"], 2048.0)
        self.assertEqual(kwargs["chunk_size"], 192)
        self.assertEqual(kwargs["chunk_shift"], 64)
        self.assertEqual(kwargs["testing_split_ratio"], 0.2)
        self.assertEqual(kwargs["validation_split_ratio"], 0.2)
        self.assertEqual(kwargs["amount_of_chunks_to_augment_at_once"], 500)

        # Verify augmentation pipelines
        self.assertEqual(len(kwargs["augmentation_pipelines"]), 3)
        self.assertTrue(
            isinstance(kwargs["augmentation_pipelines"][0][0], GaussianNoise)
        )
        self.assertTrue(
            isinstance(kwargs["augmentation_pipelines"][1][0], MagnitudeWarping)
        )
        self.assertTrue(
            isinstance(kwargs["augmentation_pipelines"][2][0], WaveletDecomposition)
        )

        # Verify that create_dataset was called on the instance
        mock_instance.create_dataset.assert_called_once()

    @patch("myotorch.datasets.defaults.EMGDataset")
    def test_castellini_dataset_initialization(self, mock_emg_dataset):
        """Test initialization of CastelliniDataset."""
        # Create the Castellini dataset
        castellini_dataset = CastelliniDataset(
            emg_data_path=self.emg_data_path,
            ground_truth_data_path=self.ground_truth_data_path,
            save_path=self.save_path,
            tasks_to_use=["task1", "task2"],
            debug_level=1,
            silence_zarr_warnings=True,
        )

        # Check that attributes are correctly set
        self.assertEqual(castellini_dataset.emg_data_path, self.emg_data_path)
        self.assertEqual(
            castellini_dataset.ground_truth_data_path, self.ground_truth_data_path
        )
        self.assertEqual(castellini_dataset.save_path, self.save_path)
        self.assertEqual(castellini_dataset.tasks_to_use, ["task1", "task2"])
        self.assertEqual(castellini_dataset.debug_level, 1)
        self.assertEqual(castellini_dataset.silence_zarr_warnings, True)

        # Check with provided data
        castellini_dataset_with_data = CastelliniDataset(
            emg_data_path=self.emg_data_path,
            ground_truth_data_path=self.ground_truth_data_path,
            save_path=self.save_path,
            emg_data=self.emg_data,
            ground_truth_data=self.ground_truth_data,
        )

        self.assertEqual(castellini_dataset_with_data.emg_data, self.emg_data)
        self.assertEqual(
            castellini_dataset_with_data.ground_truth_data, self.ground_truth_data
        )

    @patch("myotorch.datasets.defaults.EMGDataset")
    def test_castellini_dataset_create(self, mock_emg_dataset):
        """Test creating the Castellini dataset configuration."""
        # Create a mock instance for the EMGDataset class
        mock_instance = MagicMock()
        mock_emg_dataset.return_value = mock_instance

        # Create the Castellini dataset
        castellini_dataset = CastelliniDataset(
            emg_data_path=self.emg_data_path,
            ground_truth_data_path=self.ground_truth_data_path,
            save_path=self.save_path,
            tasks_to_use=["task1", "task2"],
            debug_level=1,
            silence_zarr_warnings=True,
        )

        # Call create_dataset
        castellini_dataset.create_dataset()

        # Check that EMGDataset was initialized with the correct parameters
        mock_emg_dataset.assert_called_once()
        args, kwargs = mock_emg_dataset.call_args

        # Verify key parameters were passed correctly
        self.assertEqual(kwargs["emg_data_path"], self.emg_data_path)
        self.assertEqual(kwargs["ground_truth_data_path"], self.ground_truth_data_path)
        self.assertEqual(kwargs["save_path"], self.save_path)
        self.assertEqual(kwargs["tasks_to_use"], ["task1", "task2"])
        self.assertEqual(kwargs["debug_level"], 1)
        self.assertEqual(kwargs["silence_zarr_warnings"], True)

        # Check Castellini-specific parameters
        self.assertEqual(kwargs["ground_truth_data_type"], "kinematics")
        self.assertEqual(kwargs["sampling_frequency"], 2048)
        self.assertEqual(kwargs["amount_of_chunks_to_augment_at_once"], 500)

        # Verify filter pipelines
        # EMG filters before chunking
        self.assertEqual(len(kwargs["emg_filter_pipeline_before_chunking"]), 1)
        self.assertEqual(len(kwargs["emg_filter_pipeline_before_chunking"][0]), 3)
        self.assertTrue(
            isinstance(
                kwargs["emg_filter_pipeline_before_chunking"][0][0], SOSFrequencyFilter
            )
        )
        self.assertTrue(
            isinstance(
                kwargs["emg_filter_pipeline_before_chunking"][0][1], SOSFrequencyFilter
            )
        )
        self.assertTrue(
            isinstance(kwargs["emg_filter_pipeline_before_chunking"][0][2], RMSFilter)
        )

        # Ground truth filters before chunking
        self.assertEqual(len(kwargs["ground_truth_filter_pipeline_before_chunking"]), 1)
        self.assertEqual(
            len(kwargs["ground_truth_filter_pipeline_before_chunking"][0]), 2
        )
        self.assertTrue(
            isinstance(
                kwargs["ground_truth_filter_pipeline_before_chunking"][0][0],
                ApplyFunctionFilter,
            )
        )
        self.assertTrue(
            isinstance(
                kwargs["ground_truth_filter_pipeline_before_chunking"][0][1],
                IndexDataFilter,
            )
        )

        # Ground truth filters after chunking
        self.assertEqual(len(kwargs["ground_truth_filter_pipeline_after_chunking"]), 1)
        self.assertEqual(
            len(kwargs["ground_truth_filter_pipeline_after_chunking"][0]), 1
        )
        self.assertTrue(
            isinstance(
                kwargs["ground_truth_filter_pipeline_after_chunking"][0][0],
                ApplyFunctionFilter,
            )
        )

        # Verify augmentation pipelines
        self.assertEqual(len(kwargs["augmentation_pipelines"]), 3)
        self.assertTrue(
            isinstance(kwargs["augmentation_pipelines"][0][0], GaussianNoise)
        )
        self.assertTrue(
            isinstance(kwargs["augmentation_pipelines"][1][0], MagnitudeWarping)
        )
        self.assertTrue(
            isinstance(kwargs["augmentation_pipelines"][2][0], WaveletDecomposition)
        )

        # Verify that create_dataset was called on the instance
        mock_instance.create_dataset.assert_called_once()

    @patch("myotorch.datasets.supervised.EMGDataset.create_dataset")
    def test_embc_dataset_integration(self, mock_create_dataset):
        """Test EMBC dataset with a mocked create_dataset method."""
        # Create the EMBC dataset
        embc_dataset = EMBCDataset(
            emg_data_path=self.emg_data_path,
            ground_truth_data_path=self.ground_truth_data_path,
            save_path=self.save_path,
            tasks_to_use=["task1", "task2"],
            debug_level=0,
            silence_zarr_warnings=True,
        )

        # Call create_dataset
        embc_dataset.create_dataset()

        # Check that create_dataset was called
        mock_create_dataset.assert_called_once()

    @patch("myotorch.datasets.supervised.EMGDataset.create_dataset")
    def test_castellini_dataset_integration(self, mock_create_dataset):
        """Test Castellini dataset with a mocked create_dataset method."""
        # Create the Castellini dataset
        castellini_dataset = CastelliniDataset(
            emg_data_path=self.emg_data_path,
            ground_truth_data_path=self.ground_truth_data_path,
            save_path=self.save_path,
            tasks_to_use=["task1", "task2"],
            debug_level=0,
            silence_zarr_warnings=True,
        )

        # Call create_dataset
        castellini_dataset.create_dataset()

        # Check that create_dataset was called
        mock_create_dataset.assert_called_once()


if __name__ == "__main__":
    unittest.main()
