Datasets
========

The datasets module provides a layered architecture for data handling:

- **Base Layer**: WindowedDataset handles zarr I/O, windowing, caching
- **Paradigm Layer**: SupervisedDataset for supervised learning
- **Integration Layer**: DataModule for Lightning integration
- **Storage Layer**: DatasetCreator and Modality for creating datasets

Storage
-------
.. currentmodule:: myotorch.datasets
.. autosummary::
    :toctree: generated/datasets
    :template: class.rst

    DatasetCreator
    Modality

Base Dataset
------------
.. currentmodule:: myotorch.datasets.base
.. autosummary::
    :toctree: generated/datasets
    :template: class.rst

    WindowedDataset

Paradigms
---------
.. currentmodule:: myotorch.datasets.paradigms
.. autosummary::
    :toctree: generated/datasets
    :template: class.rst

    SupervisedDataset

Integration
-----------
.. currentmodule:: myotorch.datasets.datamodule
.. autosummary::
    :toctree: generated/datasets
    :template: class.rst

    DataModule

Utilities
---------
.. currentmodule:: myotorch.datasets.utils
.. autosummary::
    :toctree: generated/datasets
    :template: class.rst

    DataSplitter
    DatasetFormatter

Presets
-------
Pre-configured transforms for published papers.

.. currentmodule:: myotorch.datasets.presets
.. autosummary::
    :toctree: generated/datasets
    :template: class.rst

    EMBCConfig

.. autosummary::
    :toctree: generated/datasets

    embc_train_transform
    embc_eval_transform
    embc_target_transform
    embc_kinematics_transform
