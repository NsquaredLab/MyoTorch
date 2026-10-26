"""Real history, split boundaries, and compatibility of windowed datasets."""

import json
import warnings
from zipfile import ZipFile

import numpy as np
import pytest
from scipy.signal import butter, sosfilt

from myotorch.datasets import DataModule, DatasetCreator, Modality, SupervisedDataset, WindowedDataset
from myotorch.transforms import Compose, Index, Lowpass, Stack

W, CONTEXT, FS = 320, 410, 2048
SPLITS = ("training", "testing", "validation")


@pytest.fixture(scope="module")
def dataset_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("context") / "data.zip"
    rng = np.random.default_rng(42)
    inputs = {"task1": rng.normal(0, 90, (3, 20_003)) + 50,
              "task2": rng.normal(0, 90, (3, 21_431)) - 50}
    targets = {task: np.arange(x.shape[-1])[None, :] for task, x in inputs.items()}
    DatasetCreator(
        {"emg": Modality(data=inputs),
         "kinematics": Modality(data=targets, dims=("joint", "time"))},
        save_path=path,
        time_chunk_size=2048,
    ).create()
    return path


def open_dataset(path, **kwargs):
    return SupervisedDataset(path, window_size=W, window_stride=W, **kwargs)


def boundaries(ds, rec_idx):
    task = ds._recording_tasks[rec_idx]
    return [0, *ds._split_group.attrs["splices"][task], ds._recording_lengths[rec_idx]]


def test_creator_records_splices_and_ratios(dataset_path):
    ds = open_dataset(dataset_path)
    assert ds._store.attrs["test_ratio"] == 0.2
    assert ds._store.attrs["val_ratio"] == 0.2
    for split in SPLITS:
        for task, n in [("task1", 20_003), ("task2", 21_431)]:
            n_test = 2 * int(n * 0.2 / 2)
            expected = {"training": [n // 2 - n_test // 2],
                        "testing": [n_test // 2 - int(n_test * 0.2 / 2)],
                        "validation": []}
            assert ds._store[split].attrs["splices"][task] == expected[split]


@pytest.mark.parametrize("split", SPLITS)
@pytest.mark.parametrize("cache", [True, False])
def test_context_windows_never_cross_splices(dataset_path, split, cache):
    ds = open_dataset(dataset_path, split=split, context=CONTEXT, cache_in_ram=cache)
    expected_positions = []
    for rec_idx in range(len(ds._recording_tasks)):
        edges = boundaries(ds, rec_idx)
        for start, end in zip(edges, edges[1:]):
            expected_positions.extend((rec_idx, p) for p in range(start + CONTEXT, end - W + 1, W))
    assert [ds._get_deterministic_position(i) for i in range(len(ds))] == expected_positions
    for idx, (rec_idx, pos) in enumerate(expected_positions):
        inputs, targets = ds[idx]
        task = ds._recording_tasks[rec_idx]
        assert inputs["emg"].shape[-1] == W + CONTEXT
        assert targets["kinematics"].shape[-1] == W
        np.testing.assert_array_equal(inputs["emg"], ds._split_group["emg"][task][:, pos-CONTEXT:pos+W])
        np.testing.assert_array_equal(targets["kinematics"], ds._split_group["kinematics"][task][:, pos:pos+W])


@pytest.mark.parametrize("split", SPLITS)
def test_random_positions_have_full_history_in_one_piece(dataset_path, split):
    ds = SupervisedDataset(dataset_path, split=split, window_size=W,
                           n_windows=1000, seed=7, context=CONTEXT)
    for _ in range(len(ds)):
        rec_idx, pos = ds._sample_random_position()
        edges = boundaries(ds, rec_idx)
        assert any(pos - CONTEXT >= start and pos + W <= end
                   for start, end in zip(edges, edges[1:]))


@pytest.mark.parametrize("split", SPLITS)
def test_zero_context_keeps_old_positions_and_values(dataset_path, split):
    ds = open_dataset(dataset_path, split=split, context=0)
    expected = [(i, p) for i, length in enumerate(ds._recording_lengths)
                for p in range(0, length-W+1, W)]
    assert [ds._get_deterministic_position(i) for i in range(len(ds))] == expected
    for idx, (i, pos) in enumerate(expected):
        inputs, targets = ds[idx]
        for mod, value in {**inputs, **targets}.items():
            np.testing.assert_array_equal(value, ds._split_group[mod][ds._recording_tasks[i]][..., pos:pos+W])
    random_ds = SupervisedDataset(dataset_path, split=split, window_size=W,
                                  n_windows=1000, seed=9, context=0)
    rng = np.random.default_rng(9)
    counts = np.array(random_ds._recording_lengths) - W + 1
    for _ in range(1000):
        old_pos = rng.integers(0, counts.sum())
        rec_idx = 0
        while old_pos >= counts[rec_idx]:
            old_pos -= counts[rec_idx]
            rec_idx += 1
        assert random_ds._sample_random_position() == (rec_idx, old_pos)


def test_explicit_splice_control(dataset_path):
    safe = open_dataset(dataset_path, respect_splices=True)
    old = open_dataset(dataset_path)
    assert len(safe) < len(old)
    for idx in range(len(safe)):
        i, p = safe._get_deterministic_position(idx)
        edges = boundaries(safe, i)
        assert any(p >= a and p+W <= b for a, b in zip(edges, edges[1:]))
    unrestricted = open_dataset(dataset_path, context=CONTEXT, respect_splices=False)
    assert len(unrestricted) == sum((n-CONTEXT-W)//W+1 for n in unrestricted._recording_lengths)


def test_context_modalities_and_targets(dataset_path):
    ds = WindowedDataset(dataset_path, window_size=W, window_stride=W, context=CONTEXT)
    assert all(arr.shape[-1] == W+CONTEXT for arr in ds[0].values())
    ds = WindowedDataset(dataset_path, window_size=W, window_stride=W,
                         context=CONTEXT, context_modalities=["emg"])
    assert ds[0]["kinematics"].shape[-1] == W
    ds = open_dataset(dataset_path, context=CONTEXT, context_modalities=[])
    assert ds[0][0]["emg"].shape[-1] == W
    with pytest.raises(ValueError, match="target"):
        open_dataset(dataset_path, context=CONTEXT, context_modalities=["kinematics"])
    with pytest.raises(ValueError, match="modalit"):
        open_dataset(dataset_path, context=CONTEXT, context_modalities=["missing"])


def strip_metadata(source, target, ratios=True):
    with ZipFile(source) as src, ZipFile(target, "w") as dst:
        for name in dict.fromkeys(src.namelist()):
            content = src.read(name)
            if name in ["zarr.json", *(f"{s}/zarr.json" for s in SPLITS)]:
                metadata = json.loads(content)
                metadata["attributes"].pop("splices", None)
                if ratios:
                    metadata["attributes"].pop("test_ratio", None)
                    metadata["attributes"].pop("val_ratio", None)
                content = json.dumps(metadata).encode()
            dst.writestr(name, content)


@pytest.mark.parametrize("split", SPLITS)
def test_old_dataset_infers_exact_splices_with_warning(dataset_path, tmp_path, split):
    path = tmp_path / "old.zip"
    strip_metadata(dataset_path, path)
    with pytest.warns(UserWarning, match=r"assuming test_ratio=0.2.*val_ratio=0.2"):
        old = open_dataset(path, split=split, context=CONTEXT)
    new = open_dataset(dataset_path, split=split, context=CONTEXT)
    assert len(old) == len(new)
    assert old._valid_ranges == new._valid_ranges
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        open_dataset(path, split=split, context=0)


def test_custom_ratios_and_odd_lengths_are_inferred(tmp_path):
    path = tmp_path / "custom.zip"
    DatasetCreator({"emg": Modality(data={"odd": np.arange(100_003)[None, :]})},
                   save_path=path, test_ratio=0.3, val_ratio=0.1,
                   time_chunk_size=8192).create()
    old_path = tmp_path / "custom_old.zip"
    strip_metadata(path, old_path, ratios=False)
    for split in SPLITS:
        kwargs = dict(split=split, window_size=W, window_stride=W, context=CONTEXT)
        new = WindowedDataset(path, **kwargs)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            old = WindowedDataset(old_path, **kwargs)
        assert old._valid_ranges == new._valid_ranges


@pytest.mark.parametrize("split", SPLITS)
def test_filter_stack_matches_continuous_split(dataset_path, split):
    pipeline = Stack({"raw": Index(slice(-W, None), dim="time"),
                      "filtered": Compose([Lowpass(20, FS), Index(slice(-W, None), dim="time")])})
    ds = open_dataset(dataset_path, split=split, context=CONTEXT, transform=pipeline, device="cpu")
    sos = butter(4, 20, fs=FS, output="sos")
    refs = {task: sosfilt(sos, ds._split_group["emg"][task][:].astype(np.float64), axis=-1)
            for task in ds._recording_tasks}
    for idx in range(len(ds)):
        i, pos = ds._get_deterministic_position(idx)
        task = ds._recording_tasks[i]
        inputs, targets = ds[idx]
        stacked = inputs["emg"].rename(None).numpy()
        assert inputs["emg"].names == ("representation", "channel", "time")
        np.testing.assert_array_equal(stacked[0], ds._split_group["emg"][task][:, pos:pos+W])
        error = np.abs(stacked[1] - refs[task][:, pos:pos+W]).mean() / refs[task].std()
        assert error < 1e-4, error
        assert targets["kinematics"].shape[-1] == W


def test_datamodule_passes_context_to_all_splits(dataset_path):
    dm = DataModule(dataset_path, window_size=W, window_stride=W, n_windows_per_epoch=10,
                    context=CONTEXT, num_workers=0, device="cpu")
    dm.setup()
    for ds in [dm.train_dataset, dm.val_dataset, dm.test_dataset]:
        assert ds.context == CONTEXT
        inputs, targets = ds[0]
        assert inputs["emg"].shape[-1] == W+CONTEXT
        assert targets["kinematics"].shape[-1] == W
    x, y = next(iter(dm.train_dataloader()))
    assert x.shape[-1] == W+CONTEXT
    assert y.shape[-1] == W


@pytest.mark.parametrize("context", [-1, 1.5])
def test_invalid_context_rejected(dataset_path, context):
    with pytest.raises(ValueError, match="context"):
        open_dataset(dataset_path, context=context)


def test_no_piece_long_enough(dataset_path):
    with pytest.raises(ValueError, match="context"):
        open_dataset(dataset_path, split="validation", context=10_000)


def test_out_of_bounds_error_names_variable(dataset_path):
    ds = open_dataset(dataset_path)
    with pytest.raises(ValueError, match="emg/task1"):
        ds._load_window("emg/task1", ds._recording_lengths[0], "emg")


@pytest.mark.parametrize(
    "n,test_ratio,val_ratio",
    [(101, 0, 0.2), (101, 0.2, 0), (101, 0.3, 0.1),
     (101, 0.2, 0.2), (9, 0.2, 0.2), (101, 1, 0)],
)
def test_splice_helper_matches_actual_discontinuities(tmp_path, n, test_ratio, val_ratio):
    from myotorch.datasets.utils.splitter import splice_positions

    source = np.arange(n)[None, :]
    creator = DatasetCreator(
        {"emg": Modality(data=source)}, save_path=tmp_path / "unused.zip",
        test_ratio=test_ratio, val_ratio=val_ratio,
    )
    arrays = creator._split_continuous(source)
    positions = splice_positions(n, test_ratio, val_ratio)
    for split, arr in zip(SPLITS, arrays):
        actual = [] if arr is None else (np.flatnonzero(np.diff(arr[0]) != 1) + 1).tolist()
        assert positions[split] == actual
