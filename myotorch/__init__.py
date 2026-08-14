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

# Submodules and convenience re-exports resolve lazily (PEP 562): importing
# myotorch stays cheap (~30 ms instead of ~2.2 s pulling torch + lightning).
_LAZY_SUBMODULES = ("datasets", "datatypes", "io", "models", "tracking", "transforms")
_LAZY_ATTRS = {"emg_tensor": "myotorch.transforms.base",
               "named_tensor": "myotorch.transforms.base"}

__all__ = [*_LAZY_SUBMODULES, *_LAZY_ATTRS, "__version__"]

_zarr_initialized = False


def _ensure_zarr_init():
    """zarr codec-pipeline config must run before myotorch touches zarr."""
    global _zarr_initialized
    if not _zarr_initialized:
        import myotorch.io.zarr_io  # noqa: F401
        _zarr_initialized = True


def __getattr__(name):
    import importlib

    if name in _LAZY_SUBMODULES:
        _ensure_zarr_init()
        return importlib.import_module(f"myotorch.{name}")
    if name in _LAZY_ATTRS:
        _ensure_zarr_init()
        return getattr(importlib.import_module(_LAZY_ATTRS[name]), name)
    raise AttributeError(f"module 'myotorch' has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
