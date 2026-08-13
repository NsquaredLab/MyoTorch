# DataTypes Tests

This folder contains tests for the data type classes in the MyoTorch framework.

## Contents

- `test_data.py`: Tests for the abstract base class `_Data` in myotorch/datatypes.py
- `test_emg_data.py`: Tests for the concrete `EMGData` class in myotorch/datatypes.py
- `test_kinematics_data.py`: Tests for the concrete `KinematicsData` class in myotorch/datatypes.py
- `test_virtual_hand_kinematics.py`: Tests for the concrete `VirtualHandKinematics` class in myotorch/datatypes.py

## Overview

The tests in this folder focus on validating the functionality of the data type classes that are fundamental to the MyoTorch framework. The data types provide a unified interface for working with different kinds of data (EMG, kinematics, etc.) and managing the processing pipelines applied to this data.

## Test Structure

### `test_data.py`

This file contains tests for the abstract base class `_Data` which provides common functionality for all data types:

1. **MockFilter and MultiInputMockFilter classes**: These implement the `FilterBaseClass` interface for testing purposes, with operations that can be easily serialized/deserialized.

2. **TestData class**: A concrete implementation of the abstract `_Data` class for testing.

3. **TestDataClass**: The unittest test class containing tests for various aspects of the `_Data` API:

   - Initialization and properties
   - Applying filters (single, sequence, pipeline)
   - Data access and manipulation
   - Memory management
   - Representation history
   - String representation
   - Save and load functionality

### `test_emg_data.py`

This file contains tests specifically for the `EMGData` class, focusing on functionality not already covered by the `_Data` tests:

1. **EMG-specific initialization**: Tests for handling 2D and 3D EMG data arrays and grid layouts

2. **Grid layout validation**: Tests for various grid layout configurations and error cases

3. **EMG-specific methods**:
   - `_check_if_chunked` for EMG data
   - `_get_grid_dimensions` for grid information
   - `plot` with various grid configurations and visualization options
   - `plot_grid_layout` for visualizing electrode arrangements

### `test_kinematics_data.py`

This file contains tests specifically for the `KinematicsData` class, which handles motion capture or joint angle data:

1. **Kinematics-specific initialization**: Tests for handling 3D (joints, coordinates, samples) and 4D (chunks, joints, coordinates, samples) data formats

2. **Data validation**: Tests for confirming proper validation of input data dimensions

3. **Kinematics-specific methods**:
   - `_check_if_chunked` for kinematics data
   - `plot` with different numbers of fingers and wrist configuration options

### `test_virtual_hand_kinematics.py`

This file contains tests specifically for the `VirtualHandKinematics` class, which handles simplified hand kinematics data:

1. **VirtualHandKinematics-specific initialization**: Tests for handling 2D (9 DOFs, samples) and 3D (chunks, 9 DOFs, samples) data formats

2. **Data dimension validation**: Tests for confirming proper validation of input dimensions

3. **VirtualHandKinematics-specific methods**:
   - `_check_if_chunked` for virtual hand data
   - `plot` testing with different visualization options (number of fingers, wrist visibility)
   - Error handling for data with incorrect degrees of freedom count

## Running the Tests

To run all tests in this folder:

```bash
python -m pytest tests/test_datatypes
```

To run a specific test file with verbose output:

```bash
python -m pytest tests/test_datatypes/test_data.py -v
python -m pytest tests/test_datatypes/test_emg_data.py -v
python -m pytest tests/test_datatypes/test_kinematics_data.py -v
python -m pytest tests/test_datatypes/test_virtual_hand_kinematics.py -v
```

To run a specific test:

```bash
python -m pytest tests/test_datatypes/test_data.py::TestDataClass::test_apply_filter -v
python -m pytest tests/test_datatypes/test_emg_data.py::TestEMGData::test_initialization -v
python -m pytest tests/test_datatypes/test_kinematics_data.py::TestKinematicsData::test_check_if_chunked -v
python -m pytest tests/test_datatypes/test_virtual_hand_kinematics.py::TestVirtualHandKinematics::test_plot -v
``` 