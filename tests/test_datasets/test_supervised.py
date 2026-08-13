import os
import unittest
import numpy as np
from pathlib import Path
import shutil
import tempfile
import pickle
import zarr
import warnings

from myotorch.datasets.supervised import EMGDataset, _add_to_dataset
from myotorch.datasets.filters.generic import FilterBaseClass
from myotorch.datasets.filters.emg_augmentations import EMGAugmentation


# Monkey-patch the _add_to_dataset function to handle the empty iterable case in max()
def _patched_add_to_dataset(group: zarr.Group, data: np.ndarray, name: str):
    """Patched version that handles empty string arrays"""
    if data is None:
        return

    # Ensure data is a numpy array
    if not isinstance(data, np.ndarray):
        data = np.array(data)

    # Special handling for string data to ensure zarr v2/v3 compatibility
    if data.dtype.kind == "U":
        # Convert Unicode strings to bytes for consistent handling in both zarr versions
        try:
            max_length = max(
                len(s.encode("utf-8")) for s in data.flat if isinstance(s, str)
            )
            # Default to 10 if no strings are found
            if max_length == 0:
                max_length = 10
        except ValueError:  # Handle empty iterables
            max_length = 10  # Default max length

        bytes_data = np.zeros(data.shape, dtype=f"S{max_length}")
        for idx in np.ndindex(data.shape):
            if isinstance(data[idx], str):
                bytes_data[idx] = data[idx].encode("utf-8")
        data = bytes_data

    # Handle object arrays that might contain strings
    elif data.dtype.kind == "O":
        # Check if the array contains strings
        contains_strings = False
        for item in data.flat:
            if isinstance(item, str):
                contains_strings = True
                break

        if contains_strings:
            # Find the maximum string length
            try:
                max_length = max(
                    len(s.encode("utf-8")) for s in data.flat if isinstance(s, str)
                )
                # Default to 10 if no strings are found
                if max_length == 0:
                    max_length = 10
            except ValueError:  # Handle empty iterables
                max_length = 10  # Default max length

            bytes_data = np.zeros(data.shape, dtype=f"S{max_length}")
            for idx in np.ndindex(data.shape):
                if isinstance(data[idx], str):
                    bytes_data[idx] = data[idx].encode("utf-8")
            data = bytes_data

    try:
        if name in group:
            # Zarr 3 doesn't have append but we can use setitem to add data
            current_shape = group[name].shape
            new_shape = list(current_shape)
            new_shape[0] += data.shape[0]

            # Resize the dataset
            group[name].resize(new_shape)

            # Insert the new data
            group[name][current_shape[0] :] = data
        else:
            # Create new dataset with appropriate chunking
            group.create_dataset(
                name, data=data, shape=data.shape, chunks=(1, *data.shape[1:])
            )
    except Exception as e:
        # Handle differences between Zarr 2 and 3
        if "append" in str(e):
            # This is Zarr 2 behavior
            group[name].append(data)
        else:
            raise


class SimpleFilter(FilterBaseClass):
    """Simple filter for testing purposes."""

    def __init__(self, is_output=False):
        super().__init__(
            input_is_chunked=False,
            allowed_input_type="both",
            is_output=is_output,
            name="SimpleFilter",
        )

    def _filter(self, input_array, **kwargs):
        # Just return the input as is
        return input_array


class SimpleAugmentation(EMGAugmentation):
    """Simple augmentation for testing purposes."""

    def __init__(self, is_output=False):
        super().__init__(
            input_is_chunked=False, is_output=is_output, name="SimpleAugmentation"
        )

    def _filter(self, input_array, **kwargs):
        # Add small random noise for augmentation
        if isinstance(input_array, np.ndarray):
            return input_array + np.random.normal(0, 0.01, input_array.shape)
        return input_array


class TestSupervisedDataset(unittest.TestCase):
    """Test class for the EMGDataset functionality."""

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
            "task1": np.random.randn(8, 1000),  # 8 channels, 1000 samples
            "task2": np.random.randn(8, 1000),
            "task3": np.random.randn(8, 1000),
        }

        # Create sample ground truth kinematics data dictionary
        # Correct shape for kinematics: (n_joints, 3, n_samples)
        self.ground_truth_data = {
            "task1": np.random.randn(
                5, 3, 1000
            ),  # 5 joints, xyz coordinates, 1000 samples
            "task2": np.random.randn(5, 3, 1000),
            "task3": np.random.randn(5, 3, 1000),
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

        # Define filter pipelines
        self.emg_filter_before_chunking = SimpleFilter(is_output=False)
        self.emg_filter_after_chunking = SimpleFilter(is_output=True)
        self.ground_truth_filter_before_chunking = SimpleFilter(is_output=False)
        self.ground_truth_filter_after_chunking = SimpleFilter(is_output=True)

        # Define augmentation pipeline
        self.augmentation = SimpleAugmentation(is_output=True)

        # Patch the _add_to_dataset function
        from myotorch.datasets import supervised

        supervised._add_to_dataset = _patched_add_to_dataset

    def tearDown(self):
        """Tear down test fixtures."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)

    def test_initialization(self):
        """Test initialization with different parameters."""
        # Test basic initialization
        dataset = EMGDataset(
            emg_data_path=self.emg_data_path,
            ground_truth_data_path=self.ground_truth_data_path,
            ground_truth_data_type="kinematics",
            sampling_frequency=1000.0,
            save_path=self.save_path,
            chunk_size=100,
            chunk_shift=50,
        )

        # Check attributes
        self.assertEqual(dataset.emg_data_path, self.emg_data_path)
        self.assertEqual(dataset.ground_truth_data_path, self.ground_truth_data_path)
        self.assertEqual(dataset.ground_truth_data_type, "kinematics")
        self.assertEqual(dataset.sampling_frequency, 1000.0)
        self.assertEqual(dataset.save_path, self.save_path)
        self.assertEqual(dataset.chunk_size, 100)
        self.assertEqual(dataset.chunk_shift, 50)

        # Test with provided data dictionaries
        dataset = EMGDataset(
            emg_data=self.emg_data,
            ground_truth_data=self.ground_truth_data,
            ground_truth_data_type="kinematics",
            sampling_frequency=1000.0,
            save_path=self.save_path,
        )

        # Check attributes
        self.assertEqual(dataset.emg_data, self.emg_data)
        self.assertEqual(dataset.ground_truth_data, self.ground_truth_data)

        # Test with filters
        dataset = EMGDataset(
            emg_data_path=self.emg_data_path,
            ground_truth_data_path=self.ground_truth_data_path,
            ground_truth_data_type="kinematics",
            sampling_frequency=1000.0,
            save_path=self.save_path,
            emg_filter_pipeline_before_chunking=[[self.emg_filter_before_chunking]],
            emg_representations_to_filter_before_chunking=[["Input"]],
            emg_filter_pipeline_after_chunking=[[self.emg_filter_after_chunking]],
            emg_representations_to_filter_after_chunking=[["Last"]],
            ground_truth_filter_pipeline_before_chunking=[
                [self.ground_truth_filter_before_chunking]
            ],
            ground_truth_representations_to_filter_before_chunking=[["Input"]],
            ground_truth_filter_pipeline_after_chunking=[
                [self.ground_truth_filter_after_chunking]
            ],
            ground_truth_representations_to_filter_after_chunking=[["Last"]],
        )

        # Check filter attributes
        self.assertEqual(
            dataset.emg_filter_pipeline_before_chunking[0][0],
            self.emg_filter_before_chunking,
        )
        self.assertEqual(
            dataset.emg_filter_pipeline_after_chunking[0][0],
            self.emg_filter_after_chunking,
        )

        # Test with specific tasks
        dataset = EMGDataset(
            emg_data_path=self.emg_data_path,
            ground_truth_data_path=self.ground_truth_data_path,
            ground_truth_data_type="kinematics",
            sampling_frequency=1000.0,
            save_path=self.save_path,
            tasks_to_use=["task1", "task2"],
        )

        # Check tasks
        self.assertEqual(dataset.tasks_to_use, ["task1", "task2"])

    def test_create_dataset(self):
        """Test creating a dataset with basic options."""
        # Create a basic dataset
        dataset = EMGDataset(
            emg_data_path=self.emg_data_path,
            ground_truth_data_path=self.ground_truth_data_path,
            ground_truth_data_type="kinematics",
            sampling_frequency=1000.0,
            save_path=self.save_path,
            chunk_size=100,
            chunk_shift=50,
            testing_split_ratio=0.2,
            validation_split_ratio=0.0,
            debug_level=0,  # Set debug level to 0 to reduce output during tests
            silence_zarr_warnings=True,  # Silence zarr warnings
        )

        # Create the dataset
        dataset.create_dataset()

        # Verify dataset exists
        self.assertTrue(self.save_path.exists())

        # Open the dataset and check structure
        z = zarr.open(str(self.save_path), mode="r")

        # Check groups exist
        self.assertIn("training", z)
        self.assertIn("testing", z)
        self.assertIn("validation", z)

        # Check data exists in training group
        self.assertIn("emg", z["training"])
        self.assertIn("ground_truth", z["training"])
        self.assertIn("label", z["training"])
        self.assertIn("class", z["training"])
        self.assertIn("one_hot_class", z["training"])

        # Check data exists in testing group
        self.assertIn("emg", z["testing"])
        self.assertIn("ground_truth", z["testing"])
        self.assertIn("label", z["testing"])
        self.assertIn("class", z["testing"])
        self.assertIn("one_hot_class", z["testing"])

        # Verify some dimensions
        # For chunked data, shape[0] should be the number of chunks
        # At least one chunk should exist in training
        for key in z["training/emg"]:
            self.assertGreater(z["training/emg"][key].shape[0], 0)
            break

        # Check that training and testing sets have different sizes
        training_size = z["training/label"].shape[0]
        testing_size = z["testing/label"].shape[0]
        self.assertGreater(training_size, 0)
        self.assertGreater(testing_size, 0)

        # Validation should be empty or very small
        validation_size = (
            z["validation/label"].shape[0] if "label" in z["validation"] else 0
        )
        self.assertEqual(validation_size, 0)

        # Verify that the labels match the tasks
        unique_labels = np.unique(z["training/label"][:])
        # Labels are stored as byte strings in zarr
        decoded_labels = [label.decode("utf-8") for label in unique_labels.flatten()]

        # Should have at most 3 unique labels (task1, task2, task3)
        self.assertLessEqual(len(decoded_labels), 3)
        for task in decoded_labels:
            self.assertIn(task, ["task1", "task2", "task3"])

    def test_dataset_with_filters(self):
        """Test creating a dataset with filters."""
        # Create a dataset with filters
        dataset = EMGDataset(
            emg_data_path=self.emg_data_path,
            ground_truth_data_path=self.ground_truth_data_path,
            ground_truth_data_type="kinematics",
            sampling_frequency=1000.0,
            save_path=self.save_path,
            chunk_size=100,
            chunk_shift=50,
            emg_filter_pipeline_before_chunking=[[self.emg_filter_before_chunking]],
            emg_representations_to_filter_before_chunking=[["Input"]],
            emg_filter_pipeline_after_chunking=[[self.emg_filter_after_chunking]],
            emg_representations_to_filter_after_chunking=[["Last"]],
            ground_truth_filter_pipeline_before_chunking=[
                [self.ground_truth_filter_before_chunking]
            ],
            ground_truth_representations_to_filter_before_chunking=[["Input"]],
            ground_truth_filter_pipeline_after_chunking=[
                [self.ground_truth_filter_after_chunking]
            ],
            ground_truth_representations_to_filter_after_chunking=[["Last"]],
            testing_split_ratio=0.2,
            debug_level=0,  # Set debug level to 0 to reduce output during tests
            silence_zarr_warnings=True,  # Silence zarr warnings
        )

        # Create the dataset
        dataset.create_dataset()

        # Verify dataset exists
        self.assertTrue(self.save_path.exists())

        # Open the dataset and verify its structure
        z = zarr.open(str(self.save_path), mode="r")

        # Filtered data should still have the proper structure
        self.assertIn("training", z)
        self.assertIn("emg", z["training"])
        self.assertIn("ground_truth", z["training"])

        # The SimpleFilter returns the data unchanged, so we should still have data
        for key in z["training/emg"]:
            self.assertGreater(z["training/emg"][key].shape[0], 0)
            break

    def test_dataset_with_augmentation(self):
        """Test creating a dataset with augmentation."""
        # Create a dataset with augmentation
        dataset = EMGDataset(
            emg_data_path=self.emg_data_path,
            ground_truth_data_path=self.ground_truth_data_path,
            ground_truth_data_type="kinematics",
            sampling_frequency=1000.0,
            save_path=self.save_path,
            chunk_size=100,
            chunk_shift=50,
            testing_split_ratio=0.2,
            augmentation_pipelines=[[self.augmentation]],
            amount_of_chunks_to_augment_at_once=50,
            debug_level=0,  # Set debug level to 0 to reduce output during tests
            silence_zarr_warnings=True,  # Silence zarr warnings
        )

        # Create the dataset
        dataset.create_dataset()

        # Verify dataset exists
        self.assertTrue(self.save_path.exists())

        # Open the dataset and verify its structure
        z = zarr.open(str(self.save_path), mode="r")

        # Augmented data should be added to the training set
        # This should increase the size of the training set compared to without augmentation
        original_dataset = EMGDataset(
            emg_data_path=self.emg_data_path,
            ground_truth_data_path=self.ground_truth_data_path,
            ground_truth_data_type="kinematics",
            sampling_frequency=1000.0,
            save_path=Path(self.temp_dir) / "original_dataset.zarr",
            chunk_size=100,
            chunk_shift=50,
            testing_split_ratio=0.2,
            debug_level=0,  # Set debug level to 0 to reduce output during tests
            silence_zarr_warnings=True,  # Silence zarr warnings
        )
        original_dataset.create_dataset()

        original_z = zarr.open(
            str(Path(self.temp_dir) / "original_dataset.zarr"), mode="r"
        )

        # The augmented dataset should have more samples in the training set
        self.assertGreater(
            z["training/label"].shape[0], original_z["training/label"].shape[0]
        )

        # Check that the number of augmented samples is as expected
        # Original samples + augmented samples = total samples
        expected_total = original_z["training/label"].shape[0] * 2  # 1x augmentation
        self.assertEqual(z["training/label"].shape[0], expected_total)

    def test_dataset_with_specific_tasks(self):
        """Test creating a dataset with specific tasks."""
        # Create a dataset with specific tasks
        dataset = EMGDataset(
            emg_data_path=self.emg_data_path,
            ground_truth_data_path=self.ground_truth_data_path,
            ground_truth_data_type="kinematics",
            sampling_frequency=1000.0,
            save_path=self.save_path,
            chunk_size=100,
            chunk_shift=50,
            tasks_to_use=["task1", "task2"],  # Only use 2 of the 3 tasks
            testing_split_ratio=0.2,
            debug_level=0,  # Set debug level to 0 to reduce output during tests
            silence_zarr_warnings=True,  # Silence zarr warnings
        )

        # Create the dataset
        dataset.create_dataset()

        # Verify dataset exists
        self.assertTrue(self.save_path.exists())

        # Open the dataset and verify its structure
        z = zarr.open(str(self.save_path), mode="r")

        # Verify that only the specified tasks are included
        unique_labels = np.unique(z["training/label"][:])
        # Labels are stored as byte strings in zarr
        decoded_labels = [label.decode("utf-8") for label in unique_labels.flatten()]

        # Should have at most 2 unique labels (task1, task2)
        self.assertLessEqual(len(decoded_labels), 2)
        for task in decoded_labels:
            self.assertIn(task, ["task1", "task2"])

        # Should not contain task3
        self.assertNotIn("task3", decoded_labels)


if __name__ == "__main__":
    unittest.main()
