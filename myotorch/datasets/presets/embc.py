"""EMBC 2022 paper configuration and transforms.

Pre-configured transform pipelines matching the EMBC 2022 paper:

    Simpetru, R.C., et al., 2022. Accurate Continuous Prediction of
    14 Degrees of Freedom of the Hand from Myoelectrical Signals
    through Convolutive Deep Learning. EMBC 2022, pp. 702-706.

Example:
-------
>>> from myotorch.datasets import DatasetCreator, Modality
>>> from myotorch.datasets.presets.embc import embc_kinematics_transform
>>>
>>> creator = DatasetCreator(
...     modalities={
...         "emg": Modality(path="emg.pkl", dims=("channel", "time")),
...         "kinematics": Modality(
...             path="kinematics.pkl",
...             dims=("dof", "time"),
...             transform=embc_kinematics_transform(),
...         ),
...     },
...     ...
... )

"""

from __future__ import annotations

from dataclasses import dataclass

from myotorch.transforms import (
    Compose,
    Flatten,
    GaussianNoise,
    Identity,
    Index,
    Lowpass,
    MagnitudeWarp,
    Mean,
    Stack,
)


@dataclass
class EMBCConfig:
    """Configuration matching EMBC 2022 paper.

    References
    ----------
    [1] Simpetru, R.C., et al., 2022. Accurate Continuous Prediction of
        14 Degrees of Freedom of the Hand from Myoelectrical Signals
        through Convolutive Deep Learning. EMBC 2022, pp. 702-706.

    """

    sampling_frequency: float = 2048.0
    window_size: int = 192
    lowpass_cutoff: float = 20.0
    lowpass_order: int = 4
    n_electrode_grids: int = 5
    test_ratio: float = 0.2
    val_ratio: float = 0.2


def embc_kinematics_transform() -> Compose:
    """Pre-storage transform for kinematics (EMBC paper).

    Converts (21, 3, time) -> (60, time) by:
    1. Flattening joints*xyz -> (63, time)
    2. Removing wrist (first 3 values) -> (60, time)

    Returns
    -------
    Compose
        Transform to set as Modality.transform for kinematics.

    """
    return Compose(
        [
            Flatten(start_dim=0, end_dim=1),  # (21, 3, time) -> (63, time)
            Index(slice(3, None), dim="channel"),  # Remove wrist -> (60, time)
        ]
    )


def embc_train_transform(
    config: EMBCConfig | None = None,
    augmentation: str = "noise",
) -> Compose:
    """Training-time transform for EMG (EMBC paper).

    Creates dual representation by stacking raw and filtered signals.

    Parameters
    ----------
    config : EMBCConfig | None
        Configuration (uses defaults if None).
    augmentation : str
        Augmentation: "noise", "warp", or "none".

    Returns
    -------
    Compose
        Transform pipeline producing (representation, channel, time).

    """
    cfg = config or EMBCConfig()

    transforms = [
        # Stack into (representation, channel, time)
        Stack(
            {
                "raw": Identity(),
                "filtered": Lowpass(
                    cfg.lowpass_cutoff, fs=cfg.sampling_frequency, dim="time"
                ),
            },
            dim="representation",
        ),
    ]

    if augmentation == "noise":
        transforms.append(GaussianNoise(std=0.1))
    elif augmentation == "warp":
        transforms.append(MagnitudeWarp(sigma=0.35, n_knots=6, dim="time"))

    return Compose(transforms)


def embc_eval_transform(config: EMBCConfig | None = None) -> Compose:
    """Evaluation-time transform for EMG (EMBC paper, no augmentation).

    Parameters
    ----------
    config : EMBCConfig | None
        Configuration (uses defaults if None).

    Returns
    -------
    Compose
        Transform pipeline producing (representation, channel, time).

    """
    cfg = config or EMBCConfig()

    return Compose(
        [
            Stack(
                {
                    "raw": Identity(),
                    "filtered": Lowpass(
                        cfg.lowpass_cutoff, fs=cfg.sampling_frequency, dim="time"
                    ),
                },
                dim="representation",
            ),
        ]
    )


def embc_target_transform() -> Mean:
    """Target transform: average kinematics over window.

    Returns
    -------
    Mean
        Transform that averages over time: (60, time) -> (60,).

    """
    return Mean(dim="time")
