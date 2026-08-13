"""
Temporal Signal Processing
==========================

This example demonstrates temporal transforms for EMG signal processing.
All transforms work with PyTorch named tensors for dimension-aware operations
that run on both CPU and GPU.
"""

# %%
# Loading Data
# ------------
# Load EMG data and wrap as a named tensor.

import pickle as pkl
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import myotorch

# Get the path to the data file
# Find data directory relative to myotorch package (works in all contexts)
import myotorch
_pkg_dir = Path(myotorch.__file__).parent.parent
DATA_DIR = _pkg_dir / "examples" / "data"
if not DATA_DIR.exists():
    DATA_DIR = Path.cwd() / "examples" / "data"

with open(DATA_DIR / "emg.pkl", "rb") as f:
    emg_data = pkl.load(f)

# Create named tensor with myotorch
SAMPLING_FREQ = 2044
emg = myotorch.emg_tensor(emg_data["1"], fs=SAMPLING_FREQ)

print(f"EMG data loaded: {emg.names} {emg.shape}")
print(f"Device: {emg.device}")

# Use fivethirtyeight style for all plots
plt.style.use("fivethirtyeight")

# %%
# Visualizing Raw Signal
# ----------------------
# Plot one channel of the raw EMG signal.

channel = 0
time_sec = 5
samples = int(time_sec * SAMPLING_FREQ)

plt.figure(figsize=(12, 4))
plt.plot(emg[channel, :samples].rename(None).numpy())
plt.title("Raw EMG Signal (Channel 0)")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.show()

# %%
# 1. Frequency Filtering
# ----------------------
# Bandpass filtering to extract the useful EMG frequency band (20-450 Hz).

from myotorch.transforms import Bandpass, Compose

# Create bandpass filter - explicitly operates on "time" dimension
bandpass = Bandpass(low=20, high=450, fs=SAMPLING_FREQ, dim="time")
print(f"Transform: {bandpass}")

# Apply the filter
filtered = bandpass(emg)
print(f"Output names: {filtered.names}")

# Visualize
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(emg[channel, :samples].rename(None).numpy())
plt.title("Raw EMG Signal")
plt.ylabel("Amplitude")

plt.subplot(2, 1, 2)
plt.plot(filtered[channel, :samples].rename(None).numpy())
plt.title("Bandpass Filtered EMG (20-450 Hz)")
plt.xlabel("Samples")
plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()

# %%
# 2. Rectification
# ----------------
# Full-wave rectification converts negative values to positive.

from myotorch.transforms import Rectify

# Rectification is element-wise
rectify = Rectify()
rectified = rectify(filtered)

plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(filtered[channel, :samples].rename(None).numpy())
plt.title("Bandpass Filtered EMG")
plt.ylabel("Amplitude")

plt.subplot(2, 1, 2)
plt.plot(rectified[channel, :samples].rename(None).numpy())
plt.title("Rectified EMG")
plt.xlabel("Samples")
plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()

# %%
# 3. RMS Feature Extraction
# -------------------------
# Root Mean Square (RMS) represents signal power over time windows.

from myotorch.transforms import RMS

# RMS with sliding window - operates on "time" dimension
rms = RMS(window_size=200, dim="time")  # ~100ms window
rms_feature = rms(rectified)

print(f"RMS output shape: {rms_feature.shape}")  # Reduced time dimension

plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(rectified[channel, :samples].rename(None).numpy(), alpha=0.7)
plt.title("Rectified EMG")
plt.ylabel("Amplitude")

plt.subplot(2, 1, 2)
# Adjust for reduced samples
rms_samples = min(samples // 200, rms_feature.shape[-1])
plt.plot(rms_feature[channel, :rms_samples].rename(None).numpy(), linewidth=2)
plt.title("RMS Envelope (200 sample window)")
plt.xlabel("Windows")
plt.ylabel("RMS Amplitude")

plt.tight_layout()
plt.show()

# %%
# 4. Composes: Chaining Transforms
# ---------------------------------
# Combine multiple transforms into a preprocessing pipeline.
# Uses torchvision.transforms.Compose under the hood.

pipeline = Compose([
    Bandpass(low=20, high=450, fs=SAMPLING_FREQ, dim="time"),
    Rectify(),
    RMS(window_size=200, dim="time"),
])

print(f"Compose: {pipeline}")

# Apply entire pipeline
processed = pipeline(emg)
print(f"Input: {emg.names} {emg.shape}")
print(f"Output: {processed.names} {processed.shape}")

# %%
# 5. Mean Absolute Value (MAV)
# ----------------------------
# Another common EMG feature - moving average of rectified signal.

from myotorch.transforms import MAV

mav = MAV(window_size=200, dim="time")
mav_feature = mav(rectified)

# Compare RMS and MAV
plt.figure(figsize=(12, 6))

# Normalize for comparison
rms_data = rms_feature[channel].rename(None)
mav_data = mav_feature[channel].rename(None)
rms_norm = (rms_data - rms_data.min()) / (rms_data.max() - rms_data.min())
mav_norm = (mav_data - mav_data.min()) / (mav_data.max() - mav_data.min())

n_windows = min(50, len(rms_norm))
plt.plot(rms_norm[:n_windows].numpy(), label="RMS", linewidth=2)
plt.plot(mav_norm[:n_windows].numpy(), label="MAV", linewidth=2, alpha=0.8)
plt.title("RMS vs MAV Features (Normalized)")
plt.xlabel("Windows")
plt.ylabel("Normalized Amplitude")
plt.legend()
plt.show()

# %%
# 6. Lowpass vs Highpass
# ----------------------
# Compare different filter types.

from myotorch.transforms import Highpass, Lowpass

lowpass = Lowpass(cutoff=50, fs=SAMPLING_FREQ, dim="time")
highpass = Highpass(cutoff=50, fs=SAMPLING_FREQ, dim="time")

low_filtered = lowpass(emg)
high_filtered = highpass(emg)

plt.figure(figsize=(12, 10))

plt.subplot(3, 1, 1)
plt.plot(emg[channel, :samples].rename(None).numpy())
plt.title("Raw EMG")
plt.ylabel("Amplitude")

plt.subplot(3, 1, 2)
plt.plot(low_filtered[channel, :samples].rename(None).numpy())
plt.title("Lowpass Filtered (< 50 Hz)")
plt.ylabel("Amplitude")

plt.subplot(3, 1, 3)
plt.plot(high_filtered[channel, :samples].rename(None).numpy())
plt.title("Highpass Filtered (> 50 Hz)")
plt.xlabel("Samples")
plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()

# %%
# 7. Normalization
# ----------------
# Z-score normalization for consistent scaling.

from myotorch.transforms import ZScore

# Z-score normalize each channel over time
zscore = ZScore(dim="time")
normalized = zscore(processed)

processed_data = processed.rename(None)
print(f"Before normalization:")
print(f"\tMean: {float(processed_data.mean()):.4f}")
print(f"\tStd:  {float(processed_data.std()):.4f}")

normalized_data = normalized.rename(None)
print(f"\nAfter normalization:")
print(f"\tMean: {float(normalized_data.mean()):.6f}")
print(f"\tStd:  {float(normalized_data.std()):.6f}")

# %%
# 8. Data Augmentation
# --------------------
# Add noise and warping for training augmentation.

from myotorch.transforms import GaussianNoise, MagnitudeWarp

# Add Gaussian noise
noise_aug = GaussianNoise(std=0.1)
noisy = noise_aug(emg)

# Magnitude warping
warp_aug = MagnitudeWarp(sigma=0.2, n_knots=4, dim="time")
warped = warp_aug(emg)

plt.figure(figsize=(12, 10))

plt.subplot(3, 1, 1)
plt.plot(emg[channel, :samples].rename(None).numpy())
plt.title("Original EMG")
plt.ylabel("Amplitude")

plt.subplot(3, 1, 2)
plt.plot(noisy[channel, :samples].rename(None).numpy())
plt.title("With Gaussian Noise")
plt.ylabel("Amplitude")

plt.subplot(3, 1, 3)
plt.plot(warped[channel, :samples].rename(None).numpy())
plt.title("With Magnitude Warping")
plt.xlabel("Samples")
plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()

# %%
# 9. Multi-feature Extraction
# ---------------------------
# Extract multiple features using Stack.

from myotorch.transforms import Stack

# Create multiple feature extractors
feature_pipeline = Compose([
    Bandpass(low=20, high=450, fs=SAMPLING_FREQ, dim="time"),
    Rectify(),
])

# Apply bandpass + rectify first
preprocessed = feature_pipeline(emg)

# Then extract multiple features
features = Stack({
    "rms": RMS(window_size=200, dim="time"),
    "mav": MAV(window_size=200, dim="time"),
    "var": myotorch.transforms.VAR(window_size=200, dim="time"),
}, dim="feature")

multi_features = features(preprocessed)
print(f"Multi-feature output: {multi_features.names} {multi_features.shape}")

# %%
# 10. GPU Acceleration
# --------------------
# Move data to GPU for faster processing.

if torch.cuda.is_available():
    # Move to GPU
    emg_gpu = emg.cuda()
    print(f"EMG on GPU: {emg_gpu.device}")

    # Apply pipeline on GPU
    processed_gpu = pipeline(emg_gpu)
    print(f"Processed on GPU: {processed_gpu.device}")
else:
    print("CUDA not available - using CPU")

# %%
# 11. Creating Custom Transforms
# ------------------------------
# Extend the Transform base class for custom processing.

from myotorch.transforms import Transform

class MedianFilter(Transform):
    """Custom median filter transform using PyTorch.

    Parameters
    ----------
    kernel_size : int
        Size of the median filter kernel. Must be odd.
    dim : str
        Dimension to filter along.
    """

    def __init__(self, kernel_size: int = 5, dim: str = "time", **kwargs):
        super().__init__(dim=dim, **kwargs)
        if kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd")
        self.kernel_size = kernel_size

    def _apply(self, x: torch.Tensor) -> torch.Tensor:
        from myotorch.transforms.base import get_dim_index
        dim_idx = get_dim_index(x, self.dim)
        names = x.names

        x = x.rename(None)

        # Unfold to get sliding windows
        x_unfolded = x.unfold(dim_idx, self.kernel_size, 1)

        # Compute median over the last dimension (the window)
        result = x_unfolded.median(dim=-1).values

        # Pad to maintain size
        pad_total = self.kernel_size - 1
        pad_before = pad_total // 2
        pad_after = pad_total - pad_before

        # Move target dim to last position for padding
        result = result.movedim(dim_idx, -1)
        original_shape = result.shape

        # F.pad with mode='replicate' requires 3D+ input
        # Reshape to 3D: (batch, 1, time)
        result = result.reshape(-1, 1, result.shape[-1])
        result = torch.nn.functional.pad(result, (pad_before, pad_after), mode='replicate')

        # Restore original shape and move dim back
        result = result.reshape(*original_shape[:-1], -1)
        result = result.movedim(-1, dim_idx)

        if names[0] is not None:
            result = result.rename(*names)

        return result

# Use in a pipeline
custom_pipeline = Compose([
    Bandpass(low=20, high=450, fs=SAMPLING_FREQ, dim="time"),
    MedianFilter(kernel_size=11, dim="time"),
    ZScore(dim="time"),
])

custom_output = custom_pipeline(emg)
print(f"Custom pipeline output: {custom_output.names} {custom_output.shape}")

# Visualize
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(emg[channel, :samples].rename(None).numpy())
plt.title("Raw EMG")
plt.ylabel("Amplitude")

plt.subplot(2, 1, 2)
plt.plot(custom_output[channel, :samples].rename(None).numpy())
plt.title("Custom Compose (Bandpass + Median + ZScore)")
plt.xlabel("Samples")
plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()

# %%
# Summary
# -------
# Key temporal transforms:
#
# - **Bandpass/Lowpass/Highpass/Notch** - FFT-based frequency filtering
# - **Rectify** - Full-wave rectification
# - **RMS/MAV/VAR** - Sliding window feature extraction
# - **ZScore/MinMax** - Normalization
# - **GaussianNoise/MagnitudeWarp/TimeWarp** - Augmentation
#
# All transforms:
# - Work with PyTorch named tensors
# - Are dimension-aware via the `dim` parameter
# - Run on both CPU and GPU
# - Can be combined with torchvision.transforms.Compose (or Compose)
