"""Acceptance tests for the Lowpass clamp + transient fix.

The old default (torchaudio biquad cascade) clamped every pass to [-1, 1] and
started from zero filter state per call. The new default is an unclamped
scipy-designed Butterworth SOS with an edge-replication warmup; ``.legacy()``
must reproduce the old path bit-for-bit because checkpoints depend on it.
"""

import warnings

import numpy as np
import pytest
import torch
import torchaudio.functional as AF
from scipy.signal import butter, sosfilt

from myotorch.transforms import Lowpass

FS = 2048.0
CUTOFF = 20.0


@pytest.fixture
def x():
    torch.manual_seed(42)
    return torch.randn(8, 320) * 500  # EMG-scale amplitudes, far outside [-1, 1]


def test_scipy_equivalence(x):
    """New default design matches scipy sosfilt within 1e-4 relative L2."""
    # warmup=None so both sides start from identical (zero) filter state
    y = Lowpass(CUTOFF, fs=FS, warmup=None)(x).numpy()

    sos = butter(4, CUTOFF, btype="lowpass", fs=FS, output="sos")
    ref = sosfilt(sos, x.numpy().astype(np.float64), axis=-1)

    rel_l2 = np.linalg.norm(y - ref) / np.linalg.norm(ref)
    assert rel_l2 < 1e-4, f"relative L2 vs scipy: {rel_l2:.3g}"


def test_no_saturation(x):
    """Output range follows the input scale; nothing is pinned to +/-1 rails."""
    lp = Lowpass(CUTOFF, fs=FS)
    y = lp(x)

    assert y.abs().max() > 10, "output collapsed towards [-1, 1]"
    assert not (y.abs() == 1.0).any(), "samples sitting exactly on the clamp rails"
    # A clamp is a nonlinearity: doubling the input must double the output.
    assert torch.allclose(lp(2 * x), 2 * y, rtol=1e-5, atol=1e-6)


def test_legacy_bit_exact(x):
    """Lowpass.legacy() equals the original 4x lowpass_biquad loop exactly."""
    ref = x
    for _ in range(4):
        ref = AF.lowpass_biquad(ref, FS, CUTOFF, 0.707)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # legacy clamp warning is tested elsewhere
        y = Lowpass.legacy(CUTOFF, fs=FS)(x)

    assert torch.equal(y, ref)


def test_warmup_effectiveness():
    """Edge warmup makes windowed output track a persistent (continuous) filter."""
    window = 320
    n_windows = 10
    t = torch.arange(n_windows * window, dtype=torch.float64) / FS
    # Smooth in-band content at EMG scale
    signal = 500 * (
        torch.sin(2 * torch.pi * 1.5 * t) + 0.5 * torch.sin(2 * torch.pi * 4.0 * t)
    )

    lp_edge = Lowpass(CUTOFF, fs=FS)
    lp_zero = Lowpass(CUTOFF, fs=FS, warmup=None)

    # Persistent reference: the whole recording filtered once
    ref = lp_edge(signal)

    start = window // 4  # window-start region for the improvement ratio
    tail_devs, start_err_edge, start_err_zero = [], [], []
    for i in range(1, n_windows):  # skip window 0: it shares the reference's own start
        sl = slice(i * window, (i + 1) * window)
        y_edge = lp_edge(signal[sl])
        y_zero = lp_zero(signal[sl])
        r = ref[sl]

        half = window // 2
        tail_devs.append(
            ((y_edge[half:] - r[half:]).abs().mean() / r[half:].abs().mean()).item()
        )
        start_err_edge.append((y_edge[:start] - r[:start]).abs().mean().item())
        start_err_zero.append((y_zero[:start] - r[:start]).abs().mean().item())

    mean_tail_dev = np.mean(tail_devs)
    assert mean_tail_dev < 0.05, f"tail deviation {mean_tail_dev:.3%} >= 5%"

    improvement = np.mean(start_err_zero) / np.mean(start_err_edge)
    assert improvement > 3, f"warmup start-region improvement only {improvement:.2f}x"


def test_gradient_flow():
    """Backward through the new default path yields finite gradients."""
    torch.manual_seed(0)
    x = (torch.randn(4, 320) * 500).requires_grad_()

    y = Lowpass(CUTOFF, fs=FS)(x)
    y.sum().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_clamp_warning_fires_exactly_once(x):
    """clamp=True with out-of-range input warns on the first call only."""
    lp = Lowpass(CUTOFF, fs=FS, clamp=True)

    with pytest.warns(UserWarning, match="clamps") as record:
        lp(x)
    assert len([w for w in record if "clamps" in str(w.message)]) == 1

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        lp(x)  # second call must stay silent
