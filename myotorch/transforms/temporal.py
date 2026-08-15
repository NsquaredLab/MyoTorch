"""GPU-accelerated temporal transforms using PyTorch.

All transforms work with named tensors and run on any device (CPU, CUDA, MPS).

Filter implementations:
- Bandpass, Highpass, Lowpass: scipy-designed Butterworth SOS run through
  ``torchaudio.functional.lfilter`` (unclamped, with edge warmup) by default.
  The legacy clamped torchaudio biquad cascade is available via
  ``design="biquad_cascade"`` / the ``.legacy()`` classmethods.
- Notch: Uses FFT-based filtering for sharp, precise narrow-band removal
  (ideal for powerline interference at 50/60 Hz).

Example:
-------
>>> import torch
>>> from myotorch.transforms import RMS, Bandpass, ZScore, Compose
>>>
>>> # Create EMG tensor on GPU
>>> emg = torch.randn(64, 20000, device='cuda', names=('channel', 'time'))
>>>
>>> # GPU-accelerated pipeline
>>> pipeline = Compose([
...     Bandpass(20, 450, fs=2048, dim='time'),
...     RMS(window_size=200, dim='time'),
...     ZScore(dim='time'),
... ])
>>> processed = pipeline(emg)  # All on GPU

"""

from __future__ import annotations

import math
import warnings

import torch
import torchaudio.functional as AF

from myotorch.transforms.base import TensorTransform, get_dim_index


class SlidingWindowTransform(TensorTransform):
    """Base class for sliding window transforms (GPU-accelerated).

    Handles the common pattern of unfold + reduce over sliding windows.
    Subclasses only need to implement `_compute_window` to define the
    window-wise computation.

    Parameters
    ----------
    window_size : int
        Window size in samples.
    stride : int | None
        Stride between windows. If None, uses window_size (non-overlapping).
    dim : str
        Dimension to compute over.

    """

    def __init__(
        self,
        window_size: int,
        stride: int | None = None,
        dim: str = "time",
        **kwargs,
    ):
        super().__init__(dim=dim, **kwargs)
        self.window_size = window_size
        self.stride = stride or window_size

    def _compute_window(self, x_unfolded: torch.Tensor) -> torch.Tensor:
        """Compute the window-wise statistic.

        Parameters
        ----------
        x_unfolded : torch.Tensor
            Unfolded tensor with windows in the last dimension.
            Shape: (..., n_windows, window_size)

        Returns
        -------
        torch.Tensor
            Reduced tensor. Shape: (..., n_windows)

        """
        raise NotImplementedError("Subclasses must implement _compute_window")

    def _apply(self, x: torch.Tensor) -> torch.Tensor:
        dim_idx = get_dim_index(x, self.dim)
        names = x.names

        x_unfolded = x.rename(None).unfold(dim_idx, self.window_size, self.stride)
        result = self._compute_window(x_unfolded)

        if names[0] is not None:
            result = result.rename(*names)

        return result


class RMS(SlidingWindowTransform):
    """Root Mean Square over sliding windows (GPU-accelerated).

    Uses unfold for efficient sliding window computation on GPU.

    Parameters
    ----------
    window_size : int
        Window size in samples.
    stride : int | None
        Stride between windows. If None, uses window_size (non-overlapping).
    dim : str
        Dimension to compute RMS over.

    Examples
    --------
    >>> x = torch.randn(64, 2048, device='cuda', names=('channel', 'time'))
    >>> rms = RMS(window_size=200, dim='time')
    >>> y = rms(x)  # Shape: (64, 10)

    """

    def _compute_window(self, x_unfolded: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.mean(x_unfolded**2, dim=-1))


class MAV(SlidingWindowTransform):
    """Mean Absolute Value over sliding windows (GPU-accelerated).

    Parameters
    ----------
    window_size : int
        Window size in samples.
    stride : int | None
        Stride between windows.
    dim : str
        Dimension to compute MAV over.

    """

    def _compute_window(self, x_unfolded: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(x_unfolded), dim=-1)


class VAR(SlidingWindowTransform):
    """Variance over sliding windows (GPU-accelerated).

    Parameters
    ----------
    window_size : int
        Window size in samples.
    stride : int | None
        Stride between windows.
    dim : str
        Dimension to compute variance over.

    """

    def _compute_window(self, x_unfolded: torch.Tensor) -> torch.Tensor:
        return torch.var(x_unfolded, dim=-1)


class Rectify(TensorTransform):
    """Full-wave rectification (absolute value).

    Parameters
    ----------
    dim : str
        Dimension name (not used, but kept for API consistency).

    """

    def _apply(self, x: torch.Tensor) -> torch.Tensor:
        return torch.abs(x)


class _ButterworthFilter(TensorTransform):
    """Shared machinery for Bandpass/Highpass/Lowpass (GPU-accelerated).

    Two designs:

    - ``design="butter"`` (default): proper Butterworth SOS computed once via
      ``scipy.signal.butter``, run section-by-section through
      ``torchaudio.functional.lfilter(clamp=False)``. Torch-differentiable,
      runs on any device. ``Q`` is ignored (the Butterworth design fixes it).
    - ``design="biquad_cascade"``: the legacy loop over torchaudio
      ``*_biquad`` calls. torchaudio hardcodes output clamping to [-1, 1] on
      this path, so it saturates any signal outside that range (raw EMG spans
      thousands of units — 94.7% of samples saturated in measurement), and its
      zero filter state per call makes ~half of a 320-sample window startup
      transient. Kept only so existing checkpoints keep their exact
      preprocessing; construct it via the ``.legacy()`` classmethods.

    ``warmup="edge"`` prepends ``warmup_len`` copies of the first sample
    before filtering and crops them after — equivalent to steady-state
    (``lfilter_zi``-style) initialisation for step inputs, which makes
    windowed output match a persistent-state streaming filter within a couple
    percent. ``warmup=None`` restores the legacy zero-state transient.
    ``warmup_len`` defaults to ``ceil(2 * fs / lowest_cutoff)``.
    """

    def __init__(
        self,
        *,
        fs: float,
        order: int,
        Q: float,
        dim: str,
        clamp: bool,
        design: str,
        warmup: str | None,
        warmup_len: int | None,
        **kwargs,
    ):
        super().__init__(dim=dim, **kwargs)
        self.fs = fs
        self.order = order
        self.Q = Q
        self.clamp = clamp
        self.design = design
        self.warmup = warmup

        if design not in ("butter", "biquad_cascade"):
            raise ValueError(f"design must be 'butter' or 'biquad_cascade', got {design!r}")
        if warmup not in (None, "edge"):
            raise ValueError(f"warmup must be None or 'edge', got {warmup!r}")

        if warmup_len is not None:
            self.warmup_len = int(warmup_len)
        else:
            self.warmup_len = math.ceil(2 * fs / self._lowest_cutoff())

        self._sos: torch.Tensor | None = None
        if design == "butter":
            self._sos = torch.as_tensor(self._design_sos(), dtype=torch.float64)
        self._warned_clamp = False

    def _lowest_cutoff(self) -> float:
        """Lowest cutoff in Hz — sets the slowest time constant for warmup."""
        raise NotImplementedError

    def _design_sos(self):
        """Return the scipy Butterworth SOS array, shape (n_sections, 6)."""
        raise NotImplementedError

    def _legacy_biquads(self, x: torch.Tensor) -> torch.Tensor:
        """The pre-fix torchaudio biquad cascade (clamped, zero state)."""
        raise NotImplementedError

    def _apply(self, x: torch.Tensor) -> torch.Tensor:
        dim_idx = get_dim_index(x, self.dim)
        names = x.names

        x = x.rename(None)

        # Move time dimension to last if not already
        if dim_idx != x.ndim - 1:
            x = x.movedim(dim_idx, -1)

        # torchaudio's biquad path hardcodes clamping regardless of self.clamp
        clamps = self.clamp or self.design == "biquad_cascade"
        if clamps and not self._warned_clamp:
            peak = x.abs().max().item()
            if peak > 1:
                warnings.warn(
                    f"{self.name}: input exceeds [-1, 1] (max abs {peak:.3g}) but the "
                    "filter clamps its output to [-1, 1], so the output will saturate. "
                    "Rescale the input, or use the default design='butter' with "
                    "clamp=False (only .legacy() configs need the clamped path).",
                    stacklevel=2,
                )
                self._warned_clamp = True

        n_pad = self.warmup_len if self.warmup == "edge" else 0
        if n_pad:
            x = torch.cat(
                [x[..., :1].expand(*x.shape[:-1], n_pad), x], dim=-1
            )

        if self.design == "butter":
            sos = self._sos.to(device=x.device, dtype=x.dtype)
            for section in sos:
                x = AF.lfilter(
                    x, a_coeffs=section[3:], b_coeffs=section[:3], clamp=self.clamp
                )
        else:
            x = self._legacy_biquads(x)

        if n_pad:
            x = x[..., n_pad:]

        # Move time dimension back
        if dim_idx != x.ndim - 1:
            x = x.movedim(-1, dim_idx)

        if names[0] is not None:
            x = x.rename(*names)

        return x


class Bandpass(_ButterworthFilter):
    """Bandpass filter (GPU-accelerated).

    Default is an unclamped Butterworth SOS design with edge warmup; see
    ``_ButterworthFilter`` for the design/clamp/warmup semantics and why the
    legacy clamped biquad cascade is unsafe for raw EMG.

    Parameters
    ----------
    low : float
        Low cutoff frequency in Hz.
    high : float
        High cutoff frequency in Hz.
    fs : float
        Sampling frequency in Hz.
    order : int
        Filter order. For ``design="butter"`` this is scipy's bandpass order
        (the resulting filter has 2*order poles); for the legacy cascade it is
        the number of biquad passes per edge. Default 4.
    Q : float
        Quality factor for the legacy biquad cascade only; ignored by
        ``design="butter"``. Default 0.707.
    dim : str
        Dimension to filter over.
    clamp : bool
        Clamp filter output to [-1, 1]. Default False.
    design : str
        "butter" (default) or "biquad_cascade" (legacy).
    warmup : str | None
        "edge" (default) or None (legacy zero-state transient).
    warmup_len : int | None
        Warmup samples; default ``ceil(2 * fs / low)``.

    Examples
    --------
    >>> x = torch.randn(64, 2048, device='cuda', names=('channel', 'time'))
    >>> bp = Bandpass(20, 450, fs=2048, dim='time')
    >>> y = bp(x)

    """

    def __init__(
        self,
        low: float,
        high: float,
        fs: float,
        order: int = 4,
        Q: float = 0.707,
        dim: str = "time",
        clamp: bool = False,
        design: str = "butter",
        warmup: str | None = "edge",
        warmup_len: int | None = None,
        **kwargs,
    ):
        self.low = low
        self.high = high
        super().__init__(
            fs=fs, order=order, Q=Q, dim=dim, clamp=clamp, design=design,
            warmup=warmup, warmup_len=warmup_len, **kwargs,
        )

    @classmethod
    def legacy(
        cls,
        low: float,
        high: float,
        fs: float,
        order: int = 4,
        Q: float = 0.707,
        dim: str = "time",
    ) -> Bandpass:
        """Pre-fix configuration: clamped torchaudio biquad cascade, no warmup.

        Bit-for-bit identical to the original Bandpass. Existing checkpoints
        were trained with this preprocessing and depend on it.
        """
        return cls(
            low, high, fs, order=order, Q=Q, dim=dim,
            clamp=True, design="biquad_cascade", warmup=None,
        )

    def _lowest_cutoff(self) -> float:
        return self.low

    def _design_sos(self):
        from scipy.signal import butter

        return butter(
            self.order, [self.low, self.high], btype="bandpass",
            fs=self.fs, output="sos",
        )

    def _legacy_biquads(self, x: torch.Tensor) -> torch.Tensor:
        # Apply highpass at low cutoff (removes frequencies below low)
        for _ in range(self.order):
            x = AF.highpass_biquad(x, self.fs, self.low, self.Q)

        # Apply lowpass at high cutoff (removes frequencies above high)
        for _ in range(self.order):
            x = AF.lowpass_biquad(x, self.fs, self.high, self.Q)

        return x


class Highpass(_ButterworthFilter):
    """Highpass filter (GPU-accelerated).

    Default is an unclamped Butterworth SOS design with edge warmup; see
    ``_ButterworthFilter`` for the design/clamp/warmup semantics and why the
    legacy clamped biquad cascade is unsafe for raw EMG.

    Parameters
    ----------
    cutoff : float
        Cutoff frequency in Hz.
    fs : float
        Sampling frequency in Hz.
    order : int
        Filter order. For ``design="butter"`` this is the Butterworth order;
        for the legacy cascade it is the number of biquad passes. Default 4.
    Q : float
        Quality factor for the legacy biquad cascade only; ignored by
        ``design="butter"``. Default 0.707.
    dim : str
        Dimension to filter over.
    clamp : bool
        Clamp filter output to [-1, 1]. Default False.
    design : str
        "butter" (default) or "biquad_cascade" (legacy).
    warmup : str | None
        "edge" (default) or None (legacy zero-state transient).
    warmup_len : int | None
        Warmup samples; default ``ceil(2 * fs / cutoff)``.

    """

    def __init__(
        self,
        cutoff: float,
        fs: float,
        order: int = 4,
        Q: float = 0.707,
        dim: str = "time",
        clamp: bool = False,
        design: str = "butter",
        warmup: str | None = "edge",
        warmup_len: int | None = None,
        **kwargs,
    ):
        self.cutoff = cutoff
        super().__init__(
            fs=fs, order=order, Q=Q, dim=dim, clamp=clamp, design=design,
            warmup=warmup, warmup_len=warmup_len, **kwargs,
        )

    @classmethod
    def legacy(
        cls,
        cutoff: float,
        fs: float,
        order: int = 4,
        Q: float = 0.707,
        dim: str = "time",
    ) -> Highpass:
        """Pre-fix configuration: clamped torchaudio biquad cascade, no warmup.

        Bit-for-bit identical to the original Highpass. Existing checkpoints
        were trained with this preprocessing and depend on it.
        """
        return cls(
            cutoff, fs, order=order, Q=Q, dim=dim,
            clamp=True, design="biquad_cascade", warmup=None,
        )

    def _lowest_cutoff(self) -> float:
        return self.cutoff

    def _design_sos(self):
        from scipy.signal import butter

        return butter(self.order, self.cutoff, btype="highpass", fs=self.fs, output="sos")

    def _legacy_biquads(self, x: torch.Tensor) -> torch.Tensor:
        # Apply biquad filter multiple times for higher order
        for _ in range(self.order):
            x = AF.highpass_biquad(x, self.fs, self.cutoff, self.Q)

        return x


class Lowpass(_ButterworthFilter):
    """Lowpass filter (GPU-accelerated).

    Default is an unclamped Butterworth SOS design with edge warmup; see
    ``_ButterworthFilter`` for the design/clamp/warmup semantics. The old
    default (torchaudio biquad cascade) clamps every pass to [-1, 1] —
    saturating 94.7% of raw EMG samples in measurement — and starts from zero
    state each call, leaving ~48% of a 320-sample window as startup transient.

    Parameters
    ----------
    cutoff : float
        Cutoff frequency in Hz.
    fs : float
        Sampling frequency in Hz.
    order : int
        Filter order. For ``design="butter"`` this is the Butterworth order;
        for the legacy cascade it is the number of biquad passes. Default 4.
    Q : float
        Quality factor for the legacy biquad cascade only; ignored by
        ``design="butter"``. Default 0.707.
    dim : str
        Dimension to filter over.
    clamp : bool
        Clamp filter output to [-1, 1]. Default False.
    design : str
        "butter" (default) or "biquad_cascade" (legacy).
    warmup : str | None
        "edge" (default) or None (legacy zero-state transient).
    warmup_len : int | None
        Warmup samples; default ``ceil(2 * fs / cutoff)``.

    """

    def __init__(
        self,
        cutoff: float,
        fs: float,
        order: int = 4,
        Q: float = 0.707,
        dim: str = "time",
        clamp: bool = False,
        design: str = "butter",
        warmup: str | None = "edge",
        warmup_len: int | None = None,
        **kwargs,
    ):
        self.cutoff = cutoff
        super().__init__(
            fs=fs, order=order, Q=Q, dim=dim, clamp=clamp, design=design,
            warmup=warmup, warmup_len=warmup_len, **kwargs,
        )

    @classmethod
    def legacy(
        cls,
        cutoff: float,
        fs: float,
        order: int = 4,
        Q: float = 0.707,
        dim: str = "time",
    ) -> Lowpass:
        """Pre-fix configuration: clamped torchaudio biquad cascade, no warmup.

        Bit-for-bit identical to the original Lowpass. Existing checkpoints
        were trained with this preprocessing and depend on it — load them with
        this config, not the new default.
        """
        return cls(
            cutoff, fs, order=order, Q=Q, dim=dim,
            clamp=True, design="biquad_cascade", warmup=None,
        )

    def _lowest_cutoff(self) -> float:
        return self.cutoff

    def _design_sos(self):
        from scipy.signal import butter

        return butter(self.order, self.cutoff, btype="lowpass", fs=self.fs, output="sos")

    def _legacy_biquads(self, x: torch.Tensor) -> torch.Tensor:
        # Apply biquad filter multiple times for higher order
        for _ in range(self.order):
            x = AF.lowpass_biquad(x, self.fs, self.cutoff, self.Q)

        return x


class Notch(TensorTransform):
    """Notch filter using FFT (GPU-accelerated).

    Removes a specific frequency (e.g., powerline interference at 50/60 Hz).
    Uses FFT-based approach which provides sharp, precise frequency removal
    ideal for narrow-band interference like powerline noise.

    Parameters
    ----------
    freq : float
        Center frequency to remove in Hz.
    width : float
        Width of the notch in Hz (default: 2 Hz).
    fs : float
        Sampling frequency in Hz.
    dim : str
        Dimension to filter over.

    """

    def __init__(
        self,
        freq: float,
        width: float = 2.0,
        fs: float = 2048.0,
        dim: str = "time",
        **kwargs,
    ):
        super().__init__(dim=dim, **kwargs)
        self.freq = freq
        self.width = width
        self.fs = fs

    def _apply(self, x: torch.Tensor) -> torch.Tensor:
        dim_idx = get_dim_index(x, self.dim)
        names = x.names
        n_samples = x.shape[dim_idx]

        x = x.rename(None)

        X = torch.fft.rfft(x, dim=dim_idx)
        freqs = torch.fft.rfftfreq(n_samples, 1 / self.fs, device=x.device)

        # Create notch (inverse of bandpass around freq)
        half_width = self.width / 2
        mask = ~((freqs >= self.freq - half_width) & (freqs <= self.freq + half_width))
        mask = mask.float()

        shape = [1] * x.ndim
        shape[dim_idx] = -1
        mask = mask.view(*shape)

        X_filtered = X * mask
        x_filtered = torch.fft.irfft(X_filtered, n=n_samples, dim=dim_idx)

        if names[0] is not None:
            x_filtered = x_filtered.rename(*names)

        return x_filtered


class ZeroCrossings(SlidingWindowTransform):
    """Count zero crossings in sliding windows (GPU-accelerated).

    Parameters
    ----------
    window_size : int
        Window size in samples.
    stride : int | None
        Stride between windows.
    dim : str
        Dimension to analyze.

    """

    def _compute_window(self, x_unfolded: torch.Tensor) -> torch.Tensor:
        signs = torch.sign(x_unfolded)
        return torch.sum(torch.abs(torch.diff(signs, dim=-1)) > 0, dim=-1).float()


class SlopeSignChanges(SlidingWindowTransform):
    """Count slope sign changes in sliding windows (GPU-accelerated).

    Parameters
    ----------
    window_size : int
        Window size in samples.
    stride : int | None
        Stride between windows.
    dim : str
        Dimension to analyze.

    """

    def _compute_window(self, x_unfolded: torch.Tensor) -> torch.Tensor:
        slopes = torch.diff(x_unfolded, dim=-1)
        signs = torch.sign(slopes)
        return torch.sum(torch.abs(torch.diff(signs, dim=-1)) > 0, dim=-1).float()


class WaveformLength(SlidingWindowTransform):
    """Waveform length over sliding windows (GPU-accelerated).

    Sum of absolute differences between consecutive samples.

    Parameters
    ----------
    window_size : int
        Window size in samples.
    stride : int | None
        Stride between windows.
    dim : str
        Dimension to analyze.

    """

    def _compute_window(self, x_unfolded: torch.Tensor) -> torch.Tensor:
        return torch.sum(torch.abs(torch.diff(x_unfolded, dim=-1)), dim=-1)


class Diff(TensorTransform):
    """Compute differences along a dimension.

    Parameters
    ----------
    n : int
        Number of times to differentiate.
    dim : str
        Dimension to differentiate over.

    """

    def __init__(self, n: int = 1, dim: str = "time", **kwargs):
        super().__init__(dim=dim, **kwargs)
        self.n = n

    def _apply(self, x: torch.Tensor) -> torch.Tensor:
        dim_idx = get_dim_index(x, self.dim)
        names = x.names

        x = x.rename(None)

        for _ in range(self.n):
            x = torch.diff(x, dim=dim_idx)

        if names[0] is not None:
            x = x.rename(*names)

        return x
