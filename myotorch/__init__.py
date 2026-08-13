"""MyoTorch - The AI toolkit for myocontrol research.

MyoTorch is a cutting-edge research companion for unlocking the secrets hidden within
biomechanical data. It's specifically designed for exploring the complex interplay
between electromyography (EMG) signals, kinematics (movement), and kinetics (forces).

Leveraging PyTorch and PyTorch Lightning, MyoTorch provides:
- Data loaders and preprocessing filters tailored for biomechanical signals
- Peer-reviewed AI models and components for analysis and prediction tasks
- Essential utilities to streamline the research workflow

MyoTorch aims to accelerate research in predicting movement from muscle activity,
analyzing forces during motion, and developing novel AI approaches for biomechanical challenges.

Note: MyoTorch is built for research and is continuously evolving.
"""

from __future__ import annotations

import importlib.metadata
import os

import toml

# Initialize zarr with zarrs codec pipeline (must be done before any zarr imports)
from myotorch.io import zarr_io as _zarr_io  # noqa: F401

# Try multiple methods to get the version
try:
    # Method 1: Try to read from pyproject.toml first
    package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pyproject_path = os.path.join(package_root, "pyproject.toml")

    if os.path.exists(pyproject_path):
        pyproject_data = toml.load(pyproject_path)
        __version__ = pyproject_data.get("project", {}).get("version", "unknown")

    # Method 2: If that fails or version is still unknown, try importlib.metadata
    if __version__ == "unknown":
        __version__ = importlib.metadata.version("MyoTorch")

except Exception:
    # If all methods fail, we at least have a default
    __version__ = "unknown"

# Re-export commonly used utilities for convenient access
from myotorch.transforms.base import emg_tensor, named_tensor  # noqa: E402
