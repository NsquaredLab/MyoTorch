.. _filters:

Transforms
==========

GPU-accelerated transforms using PyTorch named tensors.
Works on both CPU and GPU - tensors provide dimension awareness everywhere.

Base Classes
------------
.. currentmodule:: myotorch.transforms
.. autosummary::
    :toctree: generated/transforms
    :template: class.rst

    Transform

Temporal / Signal Processing
----------------------------
Temporal transforms for EMG feature extraction and signal processing.

.. currentmodule:: myotorch.transforms
.. autosummary::
    :toctree: generated/transforms
    :template: class.rst

    SlidingWindowTransform
    RMS
    MAV
    VAR
    Rectify
    Bandpass
    Highpass
    Lowpass
    Notch
    ZeroCrossings
    SlopeSignChanges
    WaveformLength
    Diff

Normalization
-------------
Normalization transforms for data preprocessing.

.. currentmodule:: myotorch.transforms
.. autosummary::
    :toctree: generated/transforms
    :template: class.rst

    ZScore
    MinMax
    Normalize
    Standardize
    InstanceNorm
    LayerNorm
    BatchNorm
    ClampRange

Generic Operations
------------------
Generic array operations.

.. currentmodule:: myotorch.transforms
.. autosummary::
    :toctree: generated/transforms
    :template: class.rst

    Reshape
    Index
    Flatten
    Squeeze
    Unsqueeze
    Transpose
    Mean
    Sum
    Stack
    Concat
    Lambda
    Identity
    Repeat
    Pad

Augmentation
------------
Data augmentation transforms.

.. currentmodule:: myotorch.transforms
.. autosummary::
    :toctree: generated/transforms
    :template: class.rst

    GaussianNoise
    MagnitudeWarp
    TimeWarp
    Dropout
    ChannelShuffle
    TimeShift
    Scale
    Cutout

Spatial / Grid-Aware
--------------------
Spatial transforms for electrode grid processing.

.. currentmodule:: myotorch.transforms
.. autosummary::
    :toctree: generated/transforms
    :template: class.rst

    SpatialFilter
    NDD
    LSD
    TSD
    IB2
