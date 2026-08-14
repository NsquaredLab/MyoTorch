"""Neural network models for MyoTorch.

This module provides model architectures for EMG signal processing and
kinematics prediction.

Example:
-------
>>> from myotorch.models import RaulNetV17
>>> model = RaulNetV17(
...     learning_rate=1e-4,
...     nr_of_input_channels=2,
...     input_length__samples=192,
...     nr_of_outputs=60,
...     cnn_encoder_channels=(32, 16, 16),
...     mlp_encoder_channels=(64, 64),
...     event_search_kernel_length=31,
...     event_search_kernel_stride=8,
... )

"""

# RaulNet model family
# Components
from myotorch.models.components import (
    SAU,
    SMU,
    EuclideanDistance,
    PSerf,
    WeightedSum,
)
from myotorch.models.raul_net import RaulNetV16, RaulNetV17, RaulNetV18, RaulNetV19

__all__ = [
    # Models
    "RaulNetV16",
    "RaulNetV17",
    "RaulNetV18",
    "RaulNetV19",
    # Components
    "EuclideanDistance",
    "PSerf",
    "SAU",
    "SMU",
    "WeightedSum",
]
