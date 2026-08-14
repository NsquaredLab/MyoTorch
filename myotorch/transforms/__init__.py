"""Transform system for MyoTorch.

GPU-accelerated transforms using PyTorch named tensors.
Works on both CPU and GPU - tensors provide dimension awareness everywhere.

Example:
-------
>>> import torch
>>> from myotorch.transforms import Pipeline, ZScore, RMS, Bandpass
>>>
>>> # Create named tensor (works on CPU or GPU)
>>> x = torch.randn(64, 2048, names=('channel', 'time'))
>>>
>>> # Pipeline with dimension-aware transforms
>>> pipeline = Pipeline([
...     Bandpass(20, 450, fs=2048, dim='time'),
...     ZScore(dim='time'),
...     RMS(window_size=200, dim='time'),
... ])
>>> y = pipeline(x)
>>>
>>> # Or on GPU
>>> x_gpu = x.cuda()
>>> y_gpu = pipeline(x_gpu)

"""

# Re-export torchvision's Compose
from torchvision.transforms import Compose

# Augmentations
from myotorch.transforms.augment import (
    ChannelShuffle,
    Cutout,
    Dropout,
    GaussianNoise,
    MagnitudeWarp,
    Scale,
    TimeShift,
    TimeWarp,
)

# Base classes and utilities
from myotorch.transforms.base import (
    TensorTransform as Transform,
)
from myotorch.transforms.base import (
    TensorTransformError as TransformError,
)
from myotorch.transforms.base import (
    align_tensors,
    emg_tensor,
    get_dim_index,
    named_tensor,
)

# Generic array operations
from myotorch.transforms.generic import (
    Concat,
    Flatten,
    Identity,
    Index,
    Lambda,
    Mean,
    Pad,
    Repeat,
    Reshape,
    Squeeze,
    Stack,
    Sum,
    Transpose,
    Unsqueeze,
)

# Normalization
from myotorch.transforms.normalize import (
    BatchNorm,
    ClampRange,
    InstanceNorm,
    LayerNorm,
    MinMax,
    Normalize,
    Standardize,
    ZScore,
)

# Spatial / grid-aware
from myotorch.transforms.spatial import (
    IB2,
    LSD,
    NDD,
    SPATIAL_KERNELS,
    TSD,
    SpatialFilter,
)

# Temporal / signal processing
from myotorch.transforms.temporal import (
    MAV,
    RMS,
    VAR,
    Bandpass,
    Diff,
    Highpass,
    Lowpass,
    Notch,
    Rectify,
    SlidingWindowTransform,
    SlopeSignChanges,
    WaveformLength,
    ZeroCrossings,
)

__all__ = [
    # Compose
    "Compose",
    # Base
    "Transform",
    "TransformError",
    "named_tensor",
    "emg_tensor",
    "get_dim_index",
    "align_tensors",
    # Temporal / signal processing
    "SlidingWindowTransform",
    "RMS",
    "MAV",
    "VAR",
    "Rectify",
    "Bandpass",
    "Highpass",
    "Lowpass",
    "Notch",
    "ZeroCrossings",
    "SlopeSignChanges",
    "WaveformLength",
    "Diff",
    # Normalization
    "ZScore",
    "MinMax",
    "Normalize",
    "InstanceNorm",
    "LayerNorm",
    "BatchNorm",
    "ClampRange",
    "Standardize",
    # Generic array ops
    "Reshape",
    "Index",
    "Flatten",
    "Squeeze",
    "Unsqueeze",
    "Transpose",
    "Mean",
    "Sum",
    "Stack",
    "Concat",
    "Lambda",
    "Identity",
    "Repeat",
    "Pad",
    # Augmentation
    "GaussianNoise",
    "MagnitudeWarp",
    "TimeWarp",
    "Dropout",
    "ChannelShuffle",
    "TimeShift",
    "Scale",
    "Cutout",
    # Spatial / grid-aware
    "SpatialFilter",
    "NDD",
    "LSD",
    "TSD",
    "IB2",
    "SPATIAL_KERNELS",
]
