"""Acceptance tests for the Lowpass clamp + transient fix.

The old default (torchaudio biquad cascade) clamped every pass to [-1, 1] and
started from zero filter state per call. The new default is an unclamped
zero-state scipy-designed Butterworth SOS with real recording history; ``.legacy()``
must reproduce the old path bit-for-bit because checkpoints depend on it.
"""

import warnings

import numpy as np
import pytest
import torch
import torchaudio.functional as AF
from scipy.signal import butter, sosfilt

from myotorch.transforms import Bandpass, Highpass, Lowpass

FS = 2048.0
CUTOFF = 20.0


@pytest.fixture
def x():
    torch.manual_seed(42)
    return torch.randn(8, 320) * 500  # EMG-scale amplitudes, far outside [-1, 1]


@pytest.mark.parametrize("offset", [-50, 50])
@pytest.mark.parametrize("kind", ["lowpass", "highpass", "bandpass"])
def test_scipy_equivalence(kind, offset):
    """Default coefficients and zero-state output match scipy on raw EMG."""
    transform = {
        "lowpass": Lowpass(20, fs=FS),
        "highpass": Highpass(20, fs=FS),
        "bandpass": Bandpass(20, 450, fs=FS),
    }[kind]
    cutoff = [20, 450] if kind == "bandpass" else 20
    sos = butter(4, cutoff, btype=kind, fs=FS, output="sos")
    np.testing.assert_array_equal(transform._sos.numpy(), sos)
    xn = (np.random.default_rng(42).normal(0, 90, (8, 320)) + offset).astype(np.float32)
    ref = sosfilt(sos, xn.astype(np.float64), axis=-1)
    y = transform(torch.from_numpy(xn)).numpy()
    assert np.sqrt(np.mean((y - ref) ** 2)) / ref.std() < 1e-4


@pytest.mark.parametrize("amplitude", [-500, 500])
def test_step_keeps_amplitude(amplitude):
    x = torch.full((2, 2048), float(amplitude))
    y = Lowpass(CUTOFF, fs=FS)(x)
    torch.testing.assert_close(y[:, -32:], x[:, -32:], rtol=1e-4, atol=0)


@pytest.mark.parametrize("transform", [Lowpass(20, FS), Highpass(20, FS), Bandpass(20, 450, FS)])
def test_context_samples(transform):
    assert transform.context_samples == 410
    assert not hasattr(transform, "warmup")
    assert not hasattr(transform, "warmup_len")


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


def test_real_history_matches_continuous_filter():
    """Every cropped window tracks continuous filtering; isolated windows do not."""
    rng = np.random.default_rng(42)
    xn = (rng.normal(0, 90, (4, 20_000)) + 50).astype(np.float32)
    lp = Lowpass(CUTOFF, fs=FS)
    context, window = lp.context_samples, 320
    ref = sosfilt(lp._sos.numpy(), xn.astype(np.float64), axis=-1)
    starts = range(context, xn.shape[-1] - window + 1, window)
    history = torch.from_numpy(np.stack([xn[:, p-context:p+window] for p in starts]))
    isolated = history[..., -window:]
    expected = np.stack([ref[:, p:p+window] for p in starts])
    filtered = lp(history).numpy()[..., -window:]
    # Mean absolute deviation in units of the continuous signal's standard deviation.
    deviations = np.mean(np.abs(filtered - expected), axis=(1, 2)) / ref.std()
    assert np.all(deviations < 1e-4), deviations.max()
    without_history = lp(isolated).numpy()
    assert np.mean(np.abs(without_history - expected)) / ref.std() > 0.05


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
