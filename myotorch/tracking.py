"""Experiment tracking with MLflow.

This module provides helpers for tracking experiments, logging artifacts,
and maintaining reproducibility across training runs.

Example:
-------
>>> from myotorch.tracking import create_logger, log_predictions
>>>
>>> logger = create_logger(
...     model_name="RaulNetV17",
...     dataset_path="data/sub1_dataset.zip",
... )
>>> trainer = L.Trainer(logger=logger, ...)
>>> trainer.fit(model, datamodule=dm)
>>>
>>> # After inference, save predictions for visualization
>>> log_predictions(logger, predictions, ground_truths, split="validation")

"""

from __future__ import annotations

import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from lightning.pytorch.loggers import MLFlowLogger


def _get_default_tracking_uri() -> str:
    """Get the default MLflow tracking URI (project root mlflow.db)."""
    # Find project root by looking for pyproject.toml
    current = Path.cwd()
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return f"sqlite:///{parent / 'mlflow.db'}"
    # Fallback to current directory
    return "sqlite:///mlflow.db"


DEFAULT_TRACKING_URI = _get_default_tracking_uri()


def get_git_info() -> dict[str, str]:
    """Get current git commit hash and branch."""
    info = {}
    try:
        info["git_commit"] = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()[:8]
        )
        info["git_branch"] = (
            subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
        # Check if working directory is clean
        result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
        info["git_dirty"] = "true" if result.stdout.strip() else "false"
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return info


def create_logger(
    model_name: str,
    dataset_path: Path | str,
    experiment_name: str = "myotorch",
    run_name: str | None = None,
    tracking_uri: str | None = None,
    log_model: Literal[True, False, "all"] = True,
    **extra_tags: Any,
) -> "MLFlowLogger":
    """Create an MLflow logger with standard MyoTorch configuration.

    Parameters
    ----------
    model_name : str
        Name of the model being trained (e.g., "RaulNetV17").
    dataset_path : Path | str
        Path to the dataset file.
    experiment_name : str
        MLflow experiment name. Default: "myotorch".
    run_name : str | None
        Custom run name. If None, generates "{model_name}_{timestamp}".
    tracking_uri : str
        MLflow tracking URI. Default: SQLite database.
    log_model : bool | "all"
        Whether to log model checkpoints. True logs at end (recommended),
        "all" logs every checkpoint during training (wasteful).
    **extra_tags : Any
        Additional tags to log with the experiment.

    Returns
    -------
    MLFlowLogger
        Configured logger for PyTorch Lightning Trainer.

    Example
    -------
    >>> logger = create_logger("RaulNetV17", "data/dataset.zip")
    >>> trainer = L.Trainer(logger=logger)

    """
    from lightning.pytorch.loggers import MLFlowLogger

    if run_name is None:
        run_name = f"{model_name}_{datetime.now():%Y%m%d_%H%M%S}"

    # Build tags
    tags = {
        "model": model_name,
        "dataset": str(Path(dataset_path).name),
        "dataset_path": str(Path(dataset_path).resolve()),
        **get_git_info(),
        **extra_tags,
    }

    return MLFlowLogger(
        experiment_name=experiment_name,
        run_name=run_name,
        tracking_uri=tracking_uri or DEFAULT_TRACKING_URI,
        log_model=log_model,
        tags=tags,
    )


def log_predictions(
    logger: "MLFlowLogger",
    predictions: np.ndarray,
    ground_truths: np.ndarray,
    split: str = "validation",
    extra_metadata: dict[str, Any] | None = None,
) -> str:
    """Log predictions and ground truths as MLflow artifacts.

    Saves a .npz file containing predictions, ground truths, and metadata
    that can be loaded later for visualization.

    Parameters
    ----------
    logger : MLFlowLogger
        The MLflow logger from training.
    predictions : np.ndarray
        Model predictions array.
    ground_truths : np.ndarray
        Ground truth labels array.
    split : str
        Data split name (e.g., "validation", "testing").
    extra_metadata : dict | None
        Additional metadata to store with predictions.

    Returns
    -------
    str
        Path to the logged artifact.

    Example
    -------
    >>> log_predictions(logger, pred_joints, gt_joints, split="validation")

    """
    import mlflow

    metadata = {
        "split": split,
        "predictions_shape": predictions.shape,
        "ground_truths_shape": ground_truths.shape,
        "timestamp": datetime.now().isoformat(),
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    # Save to temporary file, then log as artifact
    with tempfile.TemporaryDirectory() as tmpdir:
        artifact_path = Path(tmpdir) / f"predictions_{split}.npz"
        np.savez(
            artifact_path,
            predictions=predictions,
            ground_truths=ground_truths,
            metadata=metadata,
        )

        # Log to MLflow
        mlflow.log_artifact(str(artifact_path), artifact_path="predictions")

    return f"predictions/predictions_{split}.npz"


def load_predictions(
    run_id: str,
    split: str = "validation",
    tracking_uri: str | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load predictions from an MLflow run.

    Parameters
    ----------
    run_id : str
        MLflow run ID.
    split : str
        Data split to load.
    tracking_uri : str
        MLflow tracking URI.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, dict]
        Predictions, ground truths, and metadata.

    Example
    -------
    >>> preds, gts, meta = load_predictions("abc123", split="validation")

    """
    import mlflow

    mlflow.set_tracking_uri(tracking_uri or DEFAULT_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    # Download artifact
    artifact_path = client.download_artifacts(run_id, f"predictions/predictions_{split}.npz")
    data = np.load(artifact_path, allow_pickle=True)

    return data["predictions"], data["ground_truths"], data["metadata"].item()


def list_runs(
    experiment_name: str = "myotorch",
    tracking_uri: str | None = None,
) -> list[dict[str, Any]]:
    """List all runs in an experiment.

    Parameters
    ----------
    experiment_name : str
        Name of the experiment.
    tracking_uri : str
        MLflow tracking URI.

    Returns
    -------
    list[dict]
        List of run info dictionaries with run_id, run_name, status, and tags.

    """
    import mlflow

    mlflow.set_tracking_uri(tracking_uri or DEFAULT_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return []

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
    )

    return [
        {
            "run_id": run.info.run_id,
            "run_name": run.info.run_name,
            "status": run.info.status,
            "start_time": run.info.start_time,
            "tags": run.data.tags,
            "metrics": run.data.metrics,
        }
        for run in runs
    ]


def get_latest_run(
    experiment_name: str = "myotorch",
    tracking_uri: str | None = None,
) -> dict[str, Any] | None:
    """Get the most recent run from an experiment.

    Returns
    -------
    dict | None
        Run info dictionary, or None if no runs exist.

    """
    runs = list_runs(experiment_name, tracking_uri)
    return runs[0] if runs else None


def log_source_files(
    *file_paths: Path | str,
    logger: "MLFlowLogger | None" = None,
    run_id: str | None = None,
    artifact_subdir: str = "source",
) -> None:
    """Log source files as artifacts for reproducibility.

    Parameters
    ----------
    *file_paths : Path | str
        Paths to files to log (e.g., training script, dataset creator).
    logger : MLFlowLogger | None
        Lightning MLFlowLogger instance. If provided, uses its run.
    run_id : str | None
        MLflow run ID. Alternative to logger parameter.
    artifact_subdir : str
        Subdirectory in artifacts to store files. Default: "source".

    Example
    -------
    >>> logger = create_logger(...)
    >>> log_source_files("train.py", "create_dataset.py", logger=logger)

    """
    import mlflow

    # Get run_id from logger if provided
    if logger is not None:
        run_id = logger.run_id

    if run_id is None:
        raise ValueError("Must provide either logger or run_id")

    mlflow.set_tracking_uri(DEFAULT_TRACKING_URI)

    with mlflow.start_run(run_id=run_id):
        for file_path in file_paths:
            path = Path(file_path)
            if path.exists():
                mlflow.log_artifact(str(path), artifact_path=artifact_subdir)
            else:
                import warnings
                warnings.warn(f"Source file not found: {path}", stacklevel=2)


def log_config(
    config: dict[str, Any],
    filename: str = "config.json",
) -> None:
    """Log a configuration dictionary as a JSON artifact.

    Parameters
    ----------
    config : dict
        Configuration dictionary to save.
    filename : str
        Filename for the artifact. Default: "config.json".

    Example
    -------
    >>> log_config({
    ...     "window_size": 192,
    ...     "transforms": ["bandpass", "notch"],
    ...     "dataset_params": {...},
    ... })

    """
    import json
    import mlflow
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / filename
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2, default=str)
        mlflow.log_artifact(str(config_path))
