"""Rich console formatting utilities for dataset creation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree


@dataclass
class DatasetConfig:
    """Configuration for dataset creation display."""

    emg_data_path: Path
    ground_truth_data_path: Path
    ground_truth_data_type: str
    sampling_frequency: float
    save_path: Path
    chunk_size: int
    chunk_shift: int
    test_ratio: float
    val_ratio: float
    augmentation_batch_size: int
    debug_level: int
    silence_warnings: bool


class DatasetFormatter:
    """Handles Rich console output for dataset creation.

    Extracts all formatting and display logic from EMGDataset to
    provide a clean separation of concerns.

    Parameters
    ----------
    console : Console | None
        Rich console instance. If None, creates a new one.
    debug_level : int
        Debug level (0=none, 1=text, 2=text+graphs).

    Examples
    --------
    >>> formatter = DatasetFormatter(debug_level=1)
    >>> formatter.print_header()
    >>> formatter.print_config(config)
    >>> formatter.print_summary(dataset)

    """

    def __init__(self, console: Console | None = None, debug_level: int = 0):
        self.console = console or Console(color_system=None, highlight=False)
        self.debug_level = debug_level

    def should_print(self, level: int = 1) -> bool:
        """Check if output should be printed at the given level."""
        return self.debug_level >= level

    def print_header(self, title: str = "STARTING DATASET CREATION") -> None:
        """Print a section header."""
        if not self.should_print():
            return
        self.console.rule(title)
        self.console.print()

    def print_config(self, config: DatasetConfig) -> None:
        """Print dataset configuration table."""
        if not self.should_print():
            return

        table = Table(
            title="Dataset Configuration",
            show_header=True,
            box=box.ROUNDED,
            padding=(0, 2),
        )
        table.add_column("Parameter", width=30)
        table.add_column("Value")

        table.add_row("EMG data path", str(config.emg_data_path))
        table.add_row("Ground truth data path", str(config.ground_truth_data_path))
        table.add_row("Ground truth data type", config.ground_truth_data_type)
        table.add_row("Sampling frequency (Hz)", str(config.sampling_frequency))
        table.add_row("Save path", str(config.save_path))
        table.add_row("Chunk size", str(config.chunk_size))
        table.add_row("Chunk shift", str(config.chunk_shift))
        table.add_row("Testing split ratio", str(config.test_ratio))
        table.add_row("Validation split ratio", str(config.val_ratio))
        table.add_row("Augmentation batch size", str(config.augmentation_batch_size))
        table.add_row("Debug level", str(config.debug_level))
        table.add_row("Silence Zarr warnings", str(config.silence_warnings))

        self.console.print(table)
        self.console.print()

    def print_tasks_info(self, tasks: list[str]) -> None:
        """Print information about tasks to process."""
        if not self.should_print():
            return
        self.console.print(f"Processing {len(tasks)} tasks: {', '.join(tasks)}")
        self.console.print()

    def print_data_structure(
        self,
        emg_data: dict[str, np.ndarray],
        ground_truth_data: dict[str, np.ndarray],
    ) -> None:
        """Print data structure tree."""
        if not self.should_print():
            return

        tree = Tree("Dataset Structure")

        emg_branch = tree.add("EMG Data")
        for i, (k, v) in enumerate(list(emg_data.items())[:5]):
            emg_branch.add(f"Task {k}: Shape {v.shape}")
        if len(emg_data) > 5:
            emg_branch.add(f"... {len(emg_data) - 5} more tasks")

        gt_branch = tree.add("Ground Truth Data")
        for i, (k, v) in enumerate(list(ground_truth_data.items())[:5]):
            gt_branch.add(f"Task {k}: Shape {v.shape}")
        if len(ground_truth_data) > 5:
            gt_branch.add(f"... {len(ground_truth_data) - 5} more tasks")

        self.console.print(tree)
        self.console.print()

    def print_data_panel(self, data: Any, title: str) -> None:
        """Print a data object in a styled panel."""
        if not self.should_print():
            return

        panel = Panel.fit(
            str(data),
            title=title,
            box=box.ROUNDED,
            padding=(0, 2),
        )
        self.console.print(panel)

    def print_section(self, title: str) -> None:
        """Print a section label."""
        if not self.should_print():
            return
        self.console.rule(title)
        self.console.print()

    def print_action(self, action: str) -> None:
        """Print an action being performed."""
        if not self.should_print():
            return
        self.console.print(action)

    def print_split_sizes(
        self,
        training_sizes: list[int],
        testing_sizes: list[int],
        validation_sizes: list[int],
    ) -> None:
        """Print dataset split sizes table."""
        if not self.should_print():
            return

        table = Table(
            title="Dataset Split Sizes",
            show_header=True,
            box=box.ROUNDED,
            padding=(0, 2),
            width=40,
        )
        table.add_column("Split")
        table.add_column("Sizes")

        table.add_row("Training", str(training_sizes))
        table.add_row("Testing", str(testing_sizes))
        table.add_row("Validation", str(validation_sizes))

        self.console.print(table)
        self.console.print()

    def print_augmentation_config(
        self,
        num_pipelines: int,
        pipeline_names: list[str],
        batch_size: int,
        training_size: int,
    ) -> None:
        """Print augmentation configuration."""
        if not self.should_print():
            return

        table = Table(
            title="Augmentation Configuration",
            show_header=True,
            box=box.ROUNDED,
            padding=(0, 2),
        )
        table.add_column("Parameter", width=30)
        table.add_column("Value")

        table.add_row("Total augmentation pipelines", str(num_pipelines))
        table.add_row("Pipelines", "\n".join(pipeline_names))
        table.add_row("Chunks to augment at once", str(batch_size))
        table.add_row("Total training samples", str(training_size))

        self.console.print(table)
        self.console.print()

    def print_summary(self, dataset: zarr.Group) -> None:
        """Print final dataset summary."""
        if not self.should_print():
            return

        # Calculate sizes
        sizes = self._calculate_sizes(dataset)

        self.console.rule("DATASET CREATION COMPLETED")
        self.console.print()

        # Summary table
        table = Table(
            title="Dataset Summary",
            show_header=True,
            box=box.ROUNDED,
            padding=(0, 2),
            width=60,
        )
        table.add_column("Metric", width=30)
        table.add_column("Value")

        table.add_row(
            "Training samples",
            str(
                dataset["training/label"].shape[0]
                if "label" in dataset["training"]
                else 0
            ),
        )
        table.add_row(
            "Testing samples",
            str(
                dataset["testing/label"].shape[0]
                if "label" in dataset["testing"]
                else 0
            ),
        )
        table.add_row(
            "Validation samples",
            str(
                dataset["validation/label"].shape[0]
                if "label" in dataset["validation"]
                else 0
            ),
        )
        table.add_row("Total dataset size", f"{sizes['total']:.2f} MB")

        for split, size_mb in sizes["splits"].items():
            table.add_row(f"{split.capitalize()} split size", f"{size_mb:.2f} MB")

        self.console.print(table)
        self.console.print()

        # Structure tree
        self._print_structure_tree(dataset)

        self.console.rule("Dataset Creation Successfully Completed!")

    def _calculate_sizes(self, dataset: zarr.Group) -> dict:
        """Calculate dataset sizes in MB."""
        total_bytes = 0
        split_sizes = {}

        for split in ["training", "testing", "validation"]:
            split_bytes = 0
            for group in ["emg", "ground_truth"]:
                if group in dataset[split]:
                    for k in dataset[split][group]:
                        arr = dataset[split][group][k]
                        item_size = np.dtype(arr.dtype).itemsize
                        arr_size = np.prod(arr.shape) * item_size
                        split_bytes += arr_size
                        total_bytes += arr_size
            split_sizes[split] = split_bytes / (1024 * 1024)

        return {
            "total": total_bytes / (1024 * 1024),
            "splits": split_sizes,
        }

    def _print_structure_tree(self, dataset: zarr.Group) -> None:
        """Print dataset structure as a tree."""
        tree = Tree("Dataset Structure")

        for split in ["training", "testing", "validation"]:
            if "emg" in dataset[split]:
                emg_sizes = {
                    k: dataset[f"{split}/emg"][k].shape for k in dataset[f"{split}/emg"]
                }
                if emg_sizes:
                    split_branch = tree.add(split.capitalize())
                    emg_branch = split_branch.add("EMG Representations")
                    for k, shape in emg_sizes.items():
                        emg_branch.add(f"{k}: {shape}")

        self.console.print(tree)
        self.console.print()
