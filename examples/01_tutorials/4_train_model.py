"""
Training Pipeline
=================

This example shows the complete training pipeline from raw data to model training.
Each step is explained and demonstrated.
"""

# %%
from pathlib import Path

import torch

# Get the path to the data file
# Find data directory relative to myotorch package (works in all contexts)
import myotorch
_pkg_dir = Path(myotorch.__file__).parent.parent
DATA_DIR = _pkg_dir / "examples" / "data"
if not DATA_DIR.exists():
    DATA_DIR = Path.cwd() / "examples" / "data"

# Determine device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Step 1: Create Dataset with Preprocessing
# ------------------------------------------
# Use DatasetCreator with Modality transforms for pre-storage processing.
# Here we use the EMBC paper configuration as an example.

from myotorch.datasets import DatasetCreator, Modality, embc_kinematics_transform

print("=" * 60)
print("STEP 1: Dataset Creation")
print("=" * 60)

creator = DatasetCreator(
    modalities={
        # EMG: raw continuous data (320 channels from 5 electrode grids)
        "emg": Modality(
            path=DATA_DIR / "emg.pkl",
            dims=("channel", "time"),
        ),
        # Kinematics: apply transform to flatten and remove wrist
        # (21, 3, time) -> (60, time)
        "kinematics": Modality(
            path=DATA_DIR / "kinematics.pkl",
            dims=("dof", "time"),
            transform=embc_kinematics_transform(),
        ),
    },
    sampling_frequency=2048.0,
    tasks_to_use=["1", "2"],
    save_path=DATA_DIR / "tutorial_dataset.zip",
    test_ratio=0.2,
    val_ratio=0.2,
    debug_level=1,
)
creator.create()

# %%
# Step 2: Define Training Transforms
# -----------------------------------
# Transforms are applied on-the-fly during training.
# - embc_train_transform: Creates dual representation (raw + lowpass) + augmentation
# - embc_eval_transform: Same processing without augmentation

from myotorch.datasets import embc_eval_transform, embc_target_transform, embc_train_transform

print()
print("=" * 60)
print("STEP 2: Training Transforms")
print("=" * 60)

# Training: dual representation + noise augmentation
train_tf = embc_train_transform(augmentation="noise")
print(f"Train transform: {train_tf}")

# Validation: dual representation only (no augmentation)
val_tf = embc_eval_transform()
print(f"Val transform: {val_tf}")

# Target: average kinematics over window -> single prediction per DOF
target_tf = embc_target_transform()
print(f"Target transform: {target_tf}")

# %%
# Step 3: Create DataModule
# --------------------------
# DataModule handles:
# - Loading from zarr directly to tensors (GPU if available)
# - On-the-fly windowing (no pre-chunking needed)
# - Input/target selection (decided at training time, not storage time)
# - Transform application
# - Batching and DataLoader creation

from myotorch.datasets import DataModule

print()
print("=" * 60)
print("STEP 3: DataModule Setup")
print("=" * 60)

dm = DataModule(
    data_path=DATA_DIR / "tutorial_dataset.zip",
    # Select which modalities are inputs vs targets
    inputs=["emg"],
    targets=["kinematics"],
    # Windowing parameters
    window_size=192,  # ~94ms at 2048Hz
    window_stride=64,  # For val/test (deterministic sliding window)
    n_windows_per_epoch=500,  # For training (random positions) - small for demo
    # Transforms (applied on-the-fly)
    train_transform=train_tf,
    val_transform=val_tf,
    target_transform=target_tf,  # Average kinematics over window
    # DataLoader settings
    batch_size=32,
    num_workers=0,  # Set to 4+ for parallel loading
    # Device: load directly to GPU if available
    device=DEVICE,
)

# Setup creates the datasets
dm.setup("fit")

print(f"Training samples per epoch: {len(dm.train_dataloader()) * dm.batch_size}")
print(f"Validation batches: {len(dm.val_dataloader())}")

# %%
# Step 4: Inspect Batch Structure
# --------------------------------
# With single input/target, DataModule returns tensors directly
# (for compatibility with existing models).
# With multiple inputs/targets, it returns dicts.

print()
print("=" * 60)
print("STEP 4: Batch Structure")
print("=" * 60)

batch = next(iter(dm.train_dataloader()))
emg_batch, kin_batch = batch

print(f"EMG input shape: {emg_batch.shape}")
print(f"EMG input device: {emg_batch.device}")
print(f"Kinematics target shape: {kin_batch.shape}")

# EMG shape explanation (with Stack transform):
# - Batch size: 32
# - Representations: 2 (raw, filtered)
# - Channels: varies based on data
# - Time: 192 samples
print()
print("EMG shape = (batch, representation, channel, time) via Stack transform")

# %%
# Step 5: Create Model
# ---------------------
# RaulNetV16 expects:
# - Input: (batch, 2, channels, time) - 2 representations
# - Output: (batch, 60) - 60 DOF predictions

print()
print("=" * 60)
print("STEP 5: Model Setup")
print("=" * 60)

from myotorch.models import RaulNetV16

# Get actual channel count from data
n_channels = emg_batch.shape[2]
n_grids = n_channels // 64 if n_channels >= 64 else 1

model = RaulNetV16(
    learning_rate=1e-4,
    nr_of_input_channels=2,  # raw + filtered
    input_length__samples=192,
    nr_of_outputs=60,  # 60 DOF
    nr_of_electrode_grids=n_grids,
    nr_of_electrodes_per_grid=64,
    cnn_encoder_channels=(4, 1, 1),
    mlp_encoder_channels=(8, 8),
    event_search_kernel_length=31,
    event_search_kernel_stride=8,
)

# Move model to same device as data
model = model.to(DEVICE)

print(f"Model: {model.__class__.__name__}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# %%
# Step 6: Forward Pass Test
# --------------------------
# Named tensors are used during transforms for dimension-awareness,
# but stripped in the collate function before passing to the model.

print()
print("=" * 60)
print("STEP 6: Forward Pass Test")
print("=" * 60)

print(f"DataModule output: {emg_batch.shape}")
print("(Names stripped in collate - ready for model)")

# Quick forward pass test
model.eval()
with torch.no_grad():
    output = model(emg_batch)
print(f"Model output: {output.shape}")

# %%
# Step 7: Training Loop
# ----------------------
# Use PyTorch Lightning Trainer for training.
# Note: For real training, increase max_epochs and n_windows_per_epoch.

import lightning as L

print()
print("=" * 60)
print("STEP 7: Training (1 epoch, 500 windows)")
print("=" * 60)

torch.set_float32_matmul_precision("medium")  # For performance on some CPUs

trainer = L.Trainer(
    accelerator="auto",
    devices=1,
    precision="32",  # Use 32-bit for CPU compatibility
    max_epochs=1,
    log_every_n_steps=5,
    logger=False,
    enable_checkpointing=False,
    enable_progress_bar=True,
)

# Train for 1 epoch
trainer.fit(model, datamodule=dm)

# %%
# Summary
# -------
# The complete pipeline:
#
# 1. **DatasetCreator** - Store continuous data with pre-processing transforms
# 2. **Modality.transform** - Pre-storage transforms (e.g., flatten kinematics)
# 3. **DataModule** - Load directly to GPU, window, select inputs/targets
# 4. **train_transform** - On-the-fly transforms (filtering, augmentation)
# 5. **Model** - Your neural network
#
# Key benefits:
# - Modular: swap transforms without changing dataset
# - Efficient: zarr -> GPU loading (kvikio if available)
# - Flexible: input/target selection at training time
# - Named tensors: dimension-aware transforms

print()
print("=" * 60)
print("PIPELINE COMPLETE")
print("=" * 60)
