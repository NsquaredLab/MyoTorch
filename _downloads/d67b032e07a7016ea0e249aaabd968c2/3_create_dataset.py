"""
Creating a dataset
===========================

This example shows how to create a multi-modal dataset for training.

# sphinx_gallery_defer_exec
"""

# %%
# Creating a Dataset with Multiple Modalities
# --------------------------------------------
# MyoTorch stores continuous data with named dimensions (xarray + zarr).
# Any number of modalities can be stored - you decide what's input vs
# target at training time, not storage time.
#
from pathlib import Path

from myotorch.datasets import DatasetCreator, Modality

# Get the path to the data file
# Find data directory relative to myotorch package (works in all contexts)
import myotorch
_pkg_dir = Path(myotorch.__file__).parent.parent
DATA_DIR = _pkg_dir / "examples" / "data"
if not DATA_DIR.exists():
    DATA_DIR = Path.cwd() / "examples" / "data"

# Create dataset with multiple modalities
creator = DatasetCreator(
    modalities={
        "emg": Modality(
            path=DATA_DIR / "emg.pkl",
            dims=("channel", "time"),
        ),
        "kinematics": Modality(
            path=DATA_DIR / "kinematics.pkl",
            dims=("joint", "xyz", "time"),
        ),
    },
    sampling_frequency=2044.0,
    tasks_to_use=["1", "2"],
    save_path=DATA_DIR / "dataset.zip",
    test_ratio=0.2,
    val_ratio=0.2,
    debug_level=1,
)

creator.create()
