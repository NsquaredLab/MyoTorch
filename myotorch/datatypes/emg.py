"""EMG data type and grid layout utilities."""

from __future__ import annotations

import numpy as np
import xarray as xr
from matplotlib import pyplot as plt

from myotorch.datatypes.base import _Data


def emg_xarray(
    data,
    grid_layouts: list[np.ndarray] | None = None,
    fs: float = 2048.0,
    dims: tuple[str, ...] = ("channel", "time"),
    **attrs,
) -> xr.DataArray:
    """Create an EMG DataArray with grid layouts and metadata.

    This is the recommended way to create EMG data for use with transforms.
    Grid layouts are stored in attrs for spatial transforms to use.

    Parameters
    ----------
    data : np.ndarray
        EMG data array.
    grid_layouts : list[np.ndarray] | None
        List of 2D arrays mapping grid positions to channel indices.
        Each array element contains the electrode index (0-based), or -1 for gaps.
    fs : float
        Sampling frequency in Hz.
    dims : tuple[str, ...]
        Dimension names. Default: ("channel", "time").
    **attrs
        Additional attributes to store.

    Returns
    -------
    xr.DataArray
        EMG DataArray with grid_layouts and sampling_frequency in attrs.

    Examples
    --------
    >>> import numpy as np
    >>> from myotorch.datatypes import emg_xarray, create_grid_layout
    >>>
    >>> # Create grid layouts
    >>> grid1 = create_grid_layout(8, 8)  # 64 electrodes
    >>> grid2 = create_grid_layout(4, 4)  # 16 electrodes
    >>> grid2[grid2 >= 0] += 64  # Offset indices
    >>>
    >>> # Create EMG array with grid info
    >>> emg = emg_xarray(
    ...     data,
    ...     grid_layouts=[grid1, grid2],
    ...     fs=2048.0,
    ... )
    >>>
    >>> # Use with transforms
    >>> from myotorch.transforms import NDD, Pipeline, Bandpass
    >>> pipeline = Pipeline([
    ...     Bandpass(20, 450, fs=2048, dim="time"),
    ...     NDD(grids="all"),
    ... ])
    >>> filtered = pipeline(emg)

    """
    all_attrs = {"sampling_frequency": fs, **attrs}
    if grid_layouts is not None:
        all_attrs["grid_layouts"] = grid_layouts

    return xr.DataArray(data, dims=dims, attrs=all_attrs)


def create_grid_layout(
    rows: int,
    cols: int,
    n_electrodes: int | None = None,
    fill_pattern: str = "row",
    missing_indices: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Creates a grid layout based on specified parameters.

    Parameters
    ----------
    rows : int
        Number of rows in the grid.
    cols : int
        Number of columns in the grid.
    n_electrodes : int, optional
        Number of electrodes in the grid. If None, will be set to rows*cols minus
        the number of missing indices. Default is None.
    fill_pattern : str, optional
        Pattern to fill the grid. Options are 'row' (row-wise) or 'column' (column-wise).
        Default is 'row'.
    missing_indices : list[tuple[int, int]], optional
        List of (row, col) indices that should be left empty (-1). Default is None.

    Returns
    -------
    np.ndarray
        2D array representing the grid layout.

    Raises
    ------
    ValueError
        If the parameters are invalid.

    Examples
    --------
    >>> import numpy as np
    >>> from myotorch.datatypes import create_grid_layout
    >>>
    >>> # Create a 4*4 grid with row-wise numbering (0-15)
    >>> grid1 = create_grid_layout(4, 4, fill_pattern='row')
    >>> print(grid1)
    [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]
     [12 13 14 15]]
    >>>
    >>> # Create a 4*4 grid with column-wise numbering (0-15)
    >>> grid2 = create_grid_layout(4, 4, fill_pattern='column')
    >>> print(grid2)
    [[ 0  4  8 12]
     [ 1  5  9 13]
     [ 2  6 10 14]
     [ 3  7 11 15]]
    >>>
    >>> # Create a 3*3 grid with only 8 electrodes (missing bottom-right)
    >>> grid3 = create_grid_layout(3, 3, 8, 'row',
    ...                           missing_indices=[(2, 2)])
    >>> print(grid3)
    [[ 0  1  2]
     [ 3  4  5]
     [ 6  7 -1]]

    """
    # Initialize grid with -1 (gaps)
    grid = np.full((rows, cols), -1, dtype=int)

    # Process missing indices
    if missing_indices is None:
        missing_indices = []

    missing_positions = set(
        (r, c) for r, c in missing_indices if 0 <= r < rows and 0 <= c < cols
    )
    max_electrodes = rows * cols - len(missing_positions)

    # Validate n_electrodes
    if n_electrodes is None:
        n_electrodes = max_electrodes
    elif n_electrodes > max_electrodes:
        raise ValueError(
            f"Number of electrodes ({n_electrodes}) exceeds available positions "
            f"({max_electrodes} = {rows}*{cols} - {len(missing_positions)} missing)",
        )

    # Fill the grid based on the pattern
    electrode_idx = 0
    if fill_pattern.lower() == "row":
        for r in range(rows):
            for c in range(cols):
                if (r, c) not in missing_positions and electrode_idx < n_electrodes:
                    grid[r, c] = electrode_idx
                    electrode_idx += 1
    elif fill_pattern.lower() == "column":
        for c in range(cols):
            for r in range(rows):
                if (r, c) not in missing_positions and electrode_idx < n_electrodes:
                    grid[r, c] = electrode_idx
                    electrode_idx += 1
    else:
        raise ValueError(
            f"Invalid fill pattern: {fill_pattern}. Use 'row' or 'column'.",
        )

    return grid


class EMGData(_Data):
    """Class for storing EMG data.

    Parameters
    ----------
    input_data : np.ndarray
        The raw EMG data. The shape of the array should be (n_channels, n_samples) or (n_chunks, n_channels, n_samples).#

        .. important:: The class will only accept 2D or 3D arrays.
        There is no way to check if you actually have it in (n_chunks, n_samples) or (n_chunks, n_channels, n_samples) format.
        Please make sure to provide the correct shape of the data.

    sampling_frequency : float
        The sampling frequency of the EMG data.
    grid_layouts : list[np.ndarray] | None, optional
        List of 2D arrays specifying the exact electrode arrangement for each grid.
        Each array element contains the electrode index (0-based).

        .. note:: All electrodes numbers must be unique and non-negative. The numbers must be contiguous (0 to n) spread over however many grids.

        Default is None.

    Attributes
    ----------
    input_data : np.ndarray
        The raw EMG data. The shape of the array should be (n_channels, n_samples) or (n_chunks, n_channels, n_samples).
    sampling_frequency : float
        The sampling frequency of the EMG data.
    grid_layouts : list[np.ndarray] | None
        List of 2D arrays specifying the exact electrode arrangement for each grid.
        Each array element contains the electrode index (0-based).
    processed_data : dict[str, np.ndarray]
        A dictionary where the keys are the names of filters applied
        to the EMG data and the values are the processed EMG data.

    Raises
    ------
    ValueError
        If the shape of the raw EMG data is not (n_channels, n_samples) or (n_chunks, n_channels, n_samples).
        If the grid layouts are not provided or are not valid.

    Examples
    --------
    >>> import numpy as np
    >>> from myotorch.datatypes import EMGData, create_grid_layout
    >>>
    >>> # Create sample EMG data (16 channels, 1000 samples)
    >>> emg_data = np.random.randn(16, 1000)
    >>> sampling_freq = 2000  # 2000 Hz
    >>>
    >>> # Create a basic EMGData object
    >>> emg = EMGData(emg_data, sampling_freq)
    >>>
    >>> # Create an EMGData object with grid layouts
    >>> # Define a 4*4 electrode grid with row-wise numbering
    >>> grid = create_grid_layout(4, 4, fill_pattern='row')
    >>> emg_with_grid = EMGData(emg_data, sampling_freq, grid_layouts=[grid])

    """

    def __init__(
        self,
        input_data: np.ndarray,
        sampling_frequency: float,
        grid_layouts: list[np.ndarray] | None = None,
    ):
        if input_data.ndim not in (2, 3):
            raise ValueError(
                "The shape of the raw EMG data should be (n_channels, n_samples) or (n_chunks, n_channels, n_samples).",
            )
        super().__init__(
            input_data,
            sampling_frequency,
            nr_of_dimensions_when_unchunked=3,
        )

        self.grid_layouts = None  # Initialize to None first

        # Process and validate grid layouts if provided
        if grid_layouts is not None:
            # Transform to list if it is a numpy array
            if isinstance(grid_layouts, np.ndarray):
                grid_layouts = list(grid_layouts)

            for i, layout in enumerate(grid_layouts):
                if not isinstance(layout, np.ndarray) or layout.ndim != 2:
                    raise ValueError(f"Grid layout {i + 1} must be a 2D numpy array")

                # Check that not all elements are -1
                if np.all(layout == -1):
                    raise ValueError(
                        f"Grid layout {i + 1} contains all -1 values, indicating no electrodes!",
                    )

                # Check for duplicate electrode indices
                valid_indices = layout[layout >= 0]
                if len(np.unique(valid_indices)) != len(valid_indices):
                    raise ValueError(
                        f"Grid layout {i + 1} contains duplicate electrode indices",
                    )

            # Store the validated grid layouts
            self.grid_layouts = grid_layouts

    def _get_grid_dimensions(self) -> list[tuple[int, int, int]]:
        """Get dimensions and electrode counts for each grid.

        Returns
        -------
        list[tuple[int, int, int]]
            List of (rows, cols, electrodes) tuples for each grid, or empty list if no grid layouts are available.

        """
        if self.grid_layouts is None:
            return []

        return [
            (layout.shape[0], layout.shape[1], np.sum(layout >= 0))
            for layout in self.grid_layouts
        ]

    def plot(
        self,
        representation: str,
        nr_of_grids: int | None = None,
        nr_of_electrodes_per_grid: int | None = None,
        scaling_factor: float | list[float] = 20.0,
        use_grid_layouts: bool = True,
    ):
        """Plots the data for a specific representation.

        Parameters
        ----------
        representation : str
            The representation to plot.
        nr_of_grids : int | None, optional
            The number of electrode grids to plot. If None and grid_layouts is provided,
            will use the number of grids in grid_layouts. Default is None.
        nr_of_electrodes_per_grid : int | None, optional
            The number of electrodes per grid to plot. If None, will be determined from data shape
            or grid_layouts if available. Default is None.
        scaling_factor : float | list[float], optional
            The scaling factor for the data. The default is 20.0.
            If a list is provided, the scaling factor for each grid is used.
        use_grid_layouts : bool, optional
            Whether to use the grid_layouts for plotting. Default is True.
            If False, will use the nr_of_grids and nr_of_electrodes_per_grid parameters.

        """
        data = self[representation]

        # Use grid_layouts if available and requested
        if self.grid_layouts is not None and use_grid_layouts:
            grid_dimensions = self._get_grid_dimensions()

            if nr_of_grids is not None and nr_of_grids != len(self.grid_layouts):
                print(
                    f"Warning: nr_of_grids ({nr_of_grids}) does not match grid_layouts length "
                    f"({len(self.grid_layouts)}). Using grid_layouts.",
                )

            nr_of_grids = len(self.grid_layouts)
            electrodes_per_grid = [dims[2] for dims in grid_dimensions]
        else:
            # Auto-determine nr_of_grids if not provided
            if nr_of_grids is None:
                nr_of_grids = 1

            # Auto-determine nr_of_electrodes_per_grid if not provided
            if nr_of_electrodes_per_grid is None:
                if self.is_chunked[representation]:
                    total_electrodes = data.shape[1]
                else:
                    total_electrodes = data.shape[0]

                # Try to determine a sensible default
                nr_of_electrodes_per_grid = total_electrodes // nr_of_grids

            electrodes_per_grid = [nr_of_electrodes_per_grid] * nr_of_grids

        # Prepare scaling factors
        if isinstance(scaling_factor, float):
            scaling_factor = [scaling_factor] * nr_of_grids

        assert len(scaling_factor) == nr_of_grids, (
            "The number of scaling factors should be equal to the number of grids."
        )

        fig = plt.figure(figsize=(5 * nr_of_grids, 6))

        # Calculate electrode index offset for each grid
        electrode_offsets = [0]
        for i in range(len(electrodes_per_grid) - 1):
            electrode_offsets.append(electrode_offsets[-1] + electrodes_per_grid[i])

        # Make a subplot for each grid
        for grid_idx in range(nr_of_grids):
            ax = fig.add_subplot(1, nr_of_grids, grid_idx + 1)

            grid_title = f"Grid {grid_idx + 1}"
            if self.grid_layouts is not None and use_grid_layouts:
                rows, cols, _ = grid_dimensions[grid_idx]
                grid_title += f" ({rows}*{cols})"
            ax.set_title(grid_title)

            offset = electrode_offsets[grid_idx]
            n_electrodes = electrodes_per_grid[grid_idx]

            for electrode_idx in range(n_electrodes):
                data_idx = offset + electrode_idx
                if self.is_chunked[representation]:
                    # Handle chunked data - plot first chunk for visualization
                    ax.plot(
                        data[0, data_idx]
                        + electrode_idx * data[0].mean() * scaling_factor[grid_idx],
                    )
                else:
                    ax.plot(
                        data[data_idx]
                        + electrode_idx * data.mean() * scaling_factor[grid_idx],
                    )

            ax.set_xlabel("Time (samples)")
            ax.set_ylabel("Electrode #")

            # Set the y-axis ticks to the electrode numbers beginning from 1
            mean_val = (
                data[0].mean() if self.is_chunked[representation] else data.mean()
            )
            ax.set_yticks(
                np.arange(0, n_electrodes) * mean_val * scaling_factor[grid_idx],
                np.arange(1, n_electrodes + 1),
            )

            # Only for grid 1 keep the y-axis label
            if grid_idx != 0:
                ax.set_ylabel("")

        plt.tight_layout()
        plt.show()

    def plot_grid_layout(
        self,
        grid_idx: int = 0,
        show_indices: bool = True,
        cmap: plt.cm.ScalarMappable | None = None,
        figsize: tuple[float, float] | None = None,
        title: str | None = None,
        colorbar: bool = True,
        grid_color: str = "black",
        grid_alpha: float = 0.7,
        text_color: str = "white",
        text_fontsize: int = 10,
        text_fontweight: str = "bold",
        highlight_electrodes: list[int] | None = None,
        highlight_color: str = "red",
        save_path: str | None = None,
        dpi: int = 150,
        return_fig: bool = False,
        ax: plt.Axes | None = None,
        autoshow: bool = True,
    ):
        """Plots the 2D layout of a specific electrode grid with enhanced visualization.

        Parameters
        ----------
        grid_idx : int, optional
            The index of the grid to plot. Default is 0.
        show_indices : bool, optional
            Whether to show the electrode indices in the plot. Default is True.
        cmap : plt.cm.ScalarMappable | None, optional
            Custom colormap to use for visualization. If None, a default viridis colormap is used.
        figsize : tuple[float, float] | None, optional
            Custom figure size as (width, height) in inches. If None, size is calculated based on grid dimensions.
            Ignored if an existing axes object is provided.
        title : str | None, optional
            Custom title for the plot. If None, a default title showing grid dimensions is used.
        colorbar : bool, optional
            Whether to show a colorbar. Default is True.
        grid_color : str, optional
            Color of the grid lines. Default is "black".
        grid_alpha : float, optional
            Transparency of grid lines (0-1). Default is 0.7.
        text_color : str, optional
            Color of the electrode indices text. Default is "white".
        text_fontsize : int, optional
            Font size for electrode indices. Default is 10.
        text_fontweight : str, optional
            Font weight for electrode indices. Default is "bold".
        highlight_electrodes : list[int] | None, optional
            List of electrode indices to highlight. Default is None.
        highlight_color : str, optional
            Color to use for highlighting electrodes. Default is "red".
        save_path : str | None, optional
            Path to save the figure. If None, figure is not saved. Default is None.
        dpi : int, optional
            DPI for saved figure. Default is 150.
        return_fig : bool, optional
            Whether to return the figure and axes. Default is False.
        ax : plt.Axes | None, optional
            Existing axes object to plot on. If None, a new figure and axes will be created.
        autoshow : bool, optional
            Whether to automatically show the figure. Default is True.
            Set to False when plotting multiple grids on the same figure.

        Returns
        -------
        tuple[plt.Figure, plt.Axes] | None
            Figure and axes objects if return_fig is True.

        Raises
        ------
        ValueError
            If grid_layouts is not available or the grid_idx is out of range.

        """
        if self.grid_layouts is None:
            raise ValueError("Cannot plot grid layout: grid_layouts not provided.")

        if grid_idx < 0 or grid_idx >= len(self.grid_layouts):
            raise ValueError(
                f"Grid index {grid_idx} out of range (0 to {len(self.grid_layouts) - 1}).",
            )

        # Get the grid layout
        grid = self.grid_layouts[grid_idx]
        rows, cols = grid.shape

        # Get number of electrodes
        n_electrodes = np.sum(grid >= 0)

        # Set default title if not provided
        if title is None:
            title = f"Grid {grid_idx + 1} layout ({rows}*{cols}) with {n_electrodes} electrodes"

        # Create a masked array for plotting
        masked_grid = np.ma.masked_less(grid, 0)

        # Create figure and axes if not provided
        if ax is None:
            # Calculate optimal figure size if not provided
            if figsize is None:
                # Scale based on grid dimensions with minimum size
                width = max(6, cols * 0.75 + 2)
                height = max(5, rows * 0.75 + 1)
                if colorbar:
                    width += 1  # Add space for colorbar
                figsize = (width, height)

            fig, ax = plt.subplots(figsize=figsize)
        else:
            # Get the figure object from the provided axes
            fig = ax.figure

        # Setup colormap
        if cmap is None:
            cmap = plt.cm.viridis
            cmap.set_bad("white", 1.0)

        # Create custom norm to ensure integer values are centered in color bands
        norm = plt.Normalize(vmin=-0.5, vmax=np.max(grid) + 0.5)

        # Plot the grid with improved visuals
        im = ax.imshow(masked_grid, cmap=cmap, norm=norm, interpolation="nearest")

        # Add colorbar if requested
        if colorbar:
            cbar = plt.colorbar(im, ax=ax, pad=0.01)
            cbar.set_label("Electrode Index")
            # Add tick labels only at integer positions
            cbar.set_ticks(np.arange(0, np.max(grid) + 1))

        # Improve grid lines
        # Major ticks at electrode centers
        ax.set_xticks(np.arange(0, cols, 1))
        ax.set_yticks(np.arange(0, rows, 1))
        # Minor ticks at grid boundaries
        ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)

        # Hide major tick labels for cleaner look
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        # Apply grid styling
        ax.grid(
            which="minor",
            color=grid_color,
            linestyle="-",
            linewidth=1,
            alpha=grid_alpha,
        )
        ax.tick_params(which="minor", bottom=False, left=False)

        # Add axis labels
        ax.set_xlabel("Columns", fontsize=text_fontsize + 1)
        ax.set_ylabel("Rows", fontsize=text_fontsize + 1)

        # Add electrode numbers with improved styling
        if show_indices:
            for i in range(rows):
                for j in range(cols):
                    if grid[i, j] >= 0:
                        # Create a dictionary for text properties
                        text_props = {
                            "ha": "center",
                            "va": "center",
                            "color": text_color,
                            "fontsize": text_fontsize,
                            "fontweight": text_fontweight,
                        }

                        # Add highlight if this electrode is in highlight list
                        if highlight_electrodes and grid[i, j] in highlight_electrodes:
                            # Draw a circle around highlighted electrodes
                            circle = plt.Circle(
                                (j, i),
                                0.4,
                                fill=False,
                                edgecolor=highlight_color,
                                linewidth=2,
                                alpha=0.8,
                            )
                            ax.add_patch(circle)
                            # Change text properties for highlighted electrodes
                            text_props["fontweight"] = "extra bold"

                        # Add the electrode index text
                        ax.text(j, i, str(grid[i, j]), **text_props)

        # Add a title with improved styling
        ax.set_title(title, fontsize=text_fontsize + 4, pad=10)

        # Set aspect ratio to be equal
        ax.set_aspect("equal")

        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

        # Show the figure if autoshow is True
        if autoshow:
            plt.tight_layout()
            plt.show()

        # Return figure and axes if requested
        if return_fig:
            return fig, ax
        return None
