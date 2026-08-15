"""Regression tests for the Lowpass/Highpass/Bandpass clamp + transient fix.

Shipped defect: the torchaudio biquad cascade clamps every pass to [-1, 1]
(saturating raw EMG) and starts from zero filter state per call. The new
default is unclamped Butterworth SOS with edge warmup; ``.legacy()`` must
stay bit-for-bit identical to the old code because checkpoints depend on it.
"""

import warnings

import numpy as np
import pytest
import torch
import torchaudio.functional as AF
from scipy.signal import butter, sosfilt

from myotorch.transforms import Bandpass, Highpass, Lowpass

FS = 2048.0


@pytest.fixture
def x():
    torch.manual_seed(0)
    return torch.randn(8, 320) * 500  # EMG-scale amplitudes, far outside [-1, 1]


def test_legacy_bit_for_bit(x):
    """.legacy() reproduces the pre-fix biquad cascades exactly."""
    y = x
    for _ in range(4):
        y = AF.lowpass_biquad(y, FS, 20.0, 0.707)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert torch.equal(Lowpass.legacy(20, fs=FS)(x), y)

        y = x
        for _ in range(4):
            y = AF.highpass_biquad(y, FS, 20.0, 0.707)
        assert torch.equal(Highpass.legacy(20, fs=FS)(x), y)

        y = x
        for _ in range(4):
            y = AF.highpass_biquad(y, FS, 20.0, 0.707)
        for _ in range(4):
            y = AF.lowpass_biquad(y, FS, 450.0, 0.707)
        assert torch.equal(Bandpass.legacy(20, 450, fs=FS)(x), y)


@pytest.mark.parametrize(
    "transform,sos",
    [
        (Lowpass(20, fs=FS), butter(4, 20, btype="lowpass", fs=FS, output="sos")),
        (Highpass(20, fs=FS), butter(4, 20, btype="highpass", fs=FS, output="sos")),
        (
            Bandpass(20, 450, fs=FS),
            butter(4, [20, 450], btype="bandpass", fs=FS, output="sos"),
        ),
    ],
)
def test_butter_matches_scipy_and_does_not_clamp(transform, sos, x):
    """Default design matches scipy sosfilt (same edge padding) within 1e-4 rel."""
    y = transform(x).numpy()
    assert np.abs(y).max() > 10  # not clamped to [-1, 1]

    pad = transform.warmup_len
    xn = x.numpy().astype(np.float64)
    xp = np.concatenate([np.repeat(xn[:, :1], pad, axis=1), xn], axis=1)
    ref = sosfilt(sos, xp, axis=-1)[:, pad:]
    assert np.abs(y - ref).max() / np.abs(ref).max() < 1e-4


def test_clamp_warns_once_on_out_of_range_input(x):
    lp = Lowpass.legacy(20, fs=FS)
    with pytest.warns(UserWarning, match="clamps"):
        lp(x)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        lp(x)  # second call must not warn again

    # unclamped default never warns
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        Lowpass(20, fs=FS)(x)
