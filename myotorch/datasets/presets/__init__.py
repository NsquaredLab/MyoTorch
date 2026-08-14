"""Pre-configured transform pipelines for published papers and benchmarks.

Each submodule provides configuration classes and transform functions
matching specific research papers.

Available Presets
-----------------
embc : EMBC 2022 paper configuration
    Simpetru, R.C., et al., 2022. Accurate Continuous Prediction of
    14 Degrees of Freedom of the Hand from Myoelectrical Signals.
"""

from myotorch.datasets.presets.embc import (
    EMBCConfig,
    embc_eval_transform,
    embc_kinematics_transform,
    embc_target_transform,
    embc_train_transform,
)

__all__ = [
    "EMBCConfig",
    "embc_eval_transform",
    "embc_kinematics_transform",
    "embc_target_transform",
    "embc_train_transform",
]
