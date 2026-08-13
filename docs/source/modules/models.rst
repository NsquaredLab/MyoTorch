Models
======

RaulNet Models
--------------

RaulNet is a family of CNN-based models for decoding hand kinematics from
high-density EMG signals.

V17 (Latest)
^^^^^^^^^^^^

The V17 model is the latest version with lazy configuration and TorchScript support.

.. currentmodule:: myotorch.models.raul_net.v17
.. autosummary::
    :toctree: generated/models
    :template: class.rst

    RaulNetV17

V16
^^^

The V16 model was used in the MyoGestic paper (Simpetru et al., 2024).

.. currentmodule:: myotorch.models.raul_net.v16
.. autosummary::
    :toctree: generated/models
    :template: class.rst

    RaulNetV16


Components
----------

Activation Functions
^^^^^^^^^^^^^^^^^^^^

Custom learnable activation functions for neural networks.

.. currentmodule:: myotorch.models.components.activation_functions
.. autosummary::
    :toctree: generated/models
    :template: class.rst

    PSerf
    SAU
    SMU

Loss Functions
^^^^^^^^^^^^^^

Custom loss functions for kinematics prediction.

.. currentmodule:: myotorch.models.components.losses
.. autosummary::
    :toctree: generated/models
    :template: class.rst

    EuclideanDistance

Utilities
^^^^^^^^^

Utility modules for model building.

.. currentmodule:: myotorch.models.components.utils
.. autosummary::
    :toctree: generated/models
    :template: class.rst

    WeightedSum
