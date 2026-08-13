"""RaulNetV17 model for EMG-to-kinematics decoding."""

from __future__ import annotations

from typing import Any

import lightning as L
import numpy as np
import torch
from torch import nn, optim


def _ceil_div(n: int, d: int) -> int:
    """Compute ceiling division: ceil(n/d)."""
    return (n + d - 1) // d


class RaulNetV17(L.LightningModule):
    """Model for decoding kinematics from EMG data.

    Attributes
    ----------
    learning_rate : float
        The learning rate.
    nr_of_input_channels : int
        The number of input channels.
    nr_of_outputs : int
        The number of outputs.
    cnn_encoder_channels : tuple[int, int, int]
        Tuple containing 3 integers defining the cnn encoder channels.
    mlp_encoder_channels : tuple[int, int]
        Tuple containing 2 integers defining the mlp encoder channels.
    event_search_kernel_length : int
        Integer that sets the length of the kernels searching for action potentials.
    event_search_kernel_stride : int
        Integer that sets the stride of the kernels searching for action potentials.
    training_means : Optional[np.ndarray]
        The means of the training data. The shape is (1, nr_of_input_channels, 1, 1, 1).
    training_stds : Optional[np.ndarray]
        The standard deviations of the training data. The shape is (1, nr_of_input_channels, 1, 1, 1).

    """

    def __init__(
        self,
        learning_rate: float,
        nr_of_input_channels: int,
        input_length__samples: int,
        nr_of_outputs: int,
        cnn_encoder_channels: tuple[int, int, int],
        mlp_encoder_channels: tuple[int, int],
        event_search_kernel_length: int,
        event_search_kernel_stride: int,
        nr_of_electrode_grids: int = 3,
        nr_of_electrodes_per_grid: int = 36,
        inference_only: bool = False,
        training_means: np.ndarray | None = None,
        training_stds: np.ndarray | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.nr_of_input_channels = nr_of_input_channels
        self.nr_of_outputs = nr_of_outputs
        self.input_length__samples = input_length__samples

        self.cnn_encoder_channels = cnn_encoder_channels
        self.mlp_encoder_channels = mlp_encoder_channels
        self.event_search_kernel_length = event_search_kernel_length
        self.event_search_kernel_stride = event_search_kernel_stride

        self.nr_of_electrode_grids = nr_of_electrode_grids
        self.nr_of_electrodes_per_grid = nr_of_electrodes_per_grid

        self.inference_only = inference_only

        self.training_means = training_means
        self.training_stds = training_stds

        # make self.training_means and self.training_stds tensors as parameters
        if self.training_means is not None:
            self.training_means = torch.from_numpy(self.training_means).float()
        if self.training_stds is not None:
            self.training_stds = torch.from_numpy(self.training_stds).float()

        self.criterion = nn.L1Loss()

        self.cnn_encoder = None
        self.mlp = None

        self.model = None

    def configure_model(self) -> None:
        if self.cnn_encoder is not None:
            return

        cnn_encoder = nn.Sequential(
            nn.Conv3d(
                self.nr_of_input_channels,
                self.cnn_encoder_channels[0],
                kernel_size=(1, 1, int(self.event_search_kernel_length)),
                stride=(1, 1, self.event_search_kernel_stride),
                padding=(0, 0, int(self.event_search_kernel_length / 2)),
                groups=self.nr_of_input_channels,
            ),
            nn.GELU(approximate="tanh"),
            nn.InstanceNorm3d(self.cnn_encoder_channels[0]),
            nn.Dropout3d(p=0.20),
            nn.Conv3d(
                self.cnn_encoder_channels[0],
                self.cnn_encoder_channels[1],
                kernel_size=(
                    self.nr_of_electrode_grids,
                    _ceil_div(self.nr_of_electrodes_per_grid, 2),
                    18,
                ),
                dilation=(1, 2, 1),
                padding=(
                    _ceil_div(self.nr_of_electrode_grids, 2),
                    _ceil_div(self.nr_of_electrodes_per_grid, 4),
                    0,
                ),
                padding_mode="circular",
            ),
            nn.GELU(approximate="tanh"),
            nn.InstanceNorm3d(self.cnn_encoder_channels[1]),
            nn.Conv3d(
                self.cnn_encoder_channels[1],
                self.cnn_encoder_channels[2],
                kernel_size=(
                    self.nr_of_electrode_grids,
                    _ceil_div(self.nr_of_electrodes_per_grid, 7),
                    1,
                ),
            ),
            nn.GELU(approximate="tanh"),
            nn.InstanceNorm3d(self.cnn_encoder_channels[2]),
            nn.Flatten(),
        )

        self.cnn_encoder = cnn_encoder

        mlp = nn.Sequential(
            nn.Linear(
                cnn_encoder(
                    torch.rand(
                        (
                            1,
                            self.nr_of_input_channels,
                            self.nr_of_electrode_grids,
                            self.nr_of_electrodes_per_grid,
                            self.input_length__samples,
                        ),
                    ),
                )
                .detach()
                .shape[1],
                self.mlp_encoder_channels[0],
            ),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.mlp_encoder_channels[0], self.mlp_encoder_channels[1]),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.mlp_encoder_channels[1], self.nr_of_outputs),
        )

        self.mlp = mlp

        model = nn.Sequential(self.cnn_encoder, self.mlp)

        self.model = torch.jit.script(model)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self._reshape_and_normalize(inputs)
        return self.model(x)

    def _reshape_and_normalize(self, inputs: torch.Tensor) -> torch.Tensor:
        x = torch.stack(inputs.split(self.nr_of_electrodes_per_grid, dim=2), dim=2)

        if self.training_means is not None and self.training_stds is not None:
            if self.training_means.device != x.device:
                self.training_means = self.training_means.to(x.device)
                self.training_stds = self.training_stds.to(x.device)

            return (x - self.training_means) / (self.training_stds + 1e-15)

        return (x - x.mean(dim=(3, 4), keepdim=True)) / (
            x.std(dim=(3, 4), keepdim=True, unbiased=True) + 1e-15
        )

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            amsgrad=True,
            weight_decay=0.32,
            fused=True,
        )

        onecycle_scheduler = {
            "scheduler": optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.learning_rate * (10**1.5),
                total_steps=self.trainer.estimated_stepping_batches,
                anneal_strategy="cos",
                three_phase=False,
                div_factor=10**1.5,
                final_div_factor=1e3,
            ),
            "name": "OneCycleLR",
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [onecycle_scheduler]

    def training_step(
        self,
        train_batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> dict[str, Any] | None:
        inputs, ground_truths = train_batch
        ground_truths = ground_truths.flatten(start_dim=1)

        prediction = self(inputs)

        scores_dict = {"loss": self.criterion(prediction, ground_truths)}

        if scores_dict["loss"].isnan().item():
            return None

        self.log_dict(
            scores_dict,
            prog_bar=True,
            logger=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log_dict(
            {f"train/{k}": v for k, v in scores_dict.items()},
            prog_bar=False,
            logger=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )

        return scores_dict

    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> dict[str, Any]:
        inputs, ground_truths = batch
        ground_truths = ground_truths.flatten(start_dim=1)

        prediction = self(inputs)
        scores_dict = {"val_loss": self.criterion(prediction, ground_truths)}

        self.log_dict(
            scores_dict,
            prog_bar=True,
            logger=False,
            on_epoch=True,
            sync_dist=True,
        )

        self.log_dict(
            {f"val/{k}": v for k, v in scores_dict.items()},
            prog_bar=False,
            logger=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )

        return scores_dict

    def test_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> dict[str, Any]:
        inputs, ground_truths = batch
        ground_truths = ground_truths.flatten(start_dim=1)

        prediction = self(inputs)
        scores_dict = {"loss": self.criterion(prediction, ground_truths)}

        self.log_dict(
            scores_dict,
            prog_bar=True,
            logger=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log_dict(
            {f"test/{k}": v for k, v in scores_dict.items()},
            prog_bar=False,
            logger=True,
            on_epoch=False,
            on_step=True,
            sync_dist=True,
        )

        return scores_dict
