import numpy as np
import pandas as pd
import pytest
import torch

import utils
from model import build_model

META = {
    "name": "tiny", "model": "UCNN", "sampling_hz": 32.0, "cut_start": 10, "cut_end": 10,
    "sensors": ["a", "b", "c"], "sensor_descriptions": {}, "mean": [0, 0, 0], "std": [1, 1, 1],
    "classes": ["NO MAN", "OTHER MAN", "X"], "man_min_duration_steps": {"X": 64},
    "hps": {"ACTIVATION": "Relu", "BN": False, "NLAYERS": 2, "BASECHANNELS": 4, "KERNEL_SIZE": 3,
            "STATIC": False, "DILATION": 1, "POOL_KS": [2, 2], "PADDING_MODE": "reflect",
            "WINDOW_SIZE": 64, "OUTPUT_STEPS": 16, "CONTEXT_SIZE": 0,
            "BOTTLENECK_SELFATTENTION": True, "NUM_HEADS": 2, "BOTTLENECK_UNITS": 1},
}


@pytest.fixture
def bundle():
    torch.manual_seed(0)
    m = build_model(META).eval()
    return utils.ModelBundle("tiny", META, m, torch.device("cpu"))


def test_window_starts_cover_everything():
    for n in [64, 65, 100, 128, 129, 1000]:
        starts = utils.window_starts(n, 64)
        covered = np.zeros(n, bool)
        for s in starts:
            covered[s:s + 64] = True
        assert covered.all() and starts[-1] + 64 == n


@pytest.mark.parametrize("n", [30, 64, 100, 1000])
def test_predict_covers_every_step(bundle, n):
    p = utils.predict_proba(bundle, np.random.randn(n, 3).astype(np.float32), batch_size=3)
    assert p.shape == (n, 3)
    np.testing.assert_allclose(p.sum(1), 1, atol=1e-5)


def test_prepare_flight_checks_and_orders_columns(bundle):
    df = pd.DataFrame(np.random.randn(500, 4), columns=["c", "extra", "a", "b"])
    f = utils.prepare_flight(df, "F1", bundle)
    assert list(f.raw.columns) == ["a", "b", "c"] and len(f.x) == 480
    with pytest.raises(utils.InputError):
        utils.prepare_flight(df.drop(columns=["b"]), "F1", bundle)


def test_csv_roundtrip_uses_time_column(bundle):
    df = pd.DataFrame(np.random.randn(500, 3), columns=["a", "b", "c"])
    df.insert(0, "time", 100 + np.arange(500) / 32)
    parsed = utils.read_flight_file(df.to_csv(index=False).encode(), "FAN1_x.csv")
    f = utils.prepare_flight(parsed, "FAN1", bundle)
    assert f.time[0] == 0 and abs(f.time[1] - 1 / 32) < 1e-9


def test_end_to_end(bundle):
    flight = utils.prepare_flight(pd.DataFrame(np.random.randn(400, 3), columns=["a", "b", "c"]), "F", bundle)
    res = utils.classify_flight(bundle, flight, utils.pp_params(bundle, True))
    assert len(res.labels) == len(flight.x)
    seg = utils.segments_df(res, META["classes"])
    assert set(seg.columns) >= {"manoeuvre", "start_s", "duration_s"}
    utils.plot_flight(res, flight.raw, flight.time, ["a"], {}, META["classes"])


GT_META = {**META, "gt": {"column": "MANOEUVERS", "id_to_name": {"0": "NO MAN", "7": "X alias", "9": "Unused"},
                          "alias": {"X alias": "X"}}}


def test_gt_filename_and_encoding(bundle):
    assert utils.is_gt_filename("FAN1205_Y.parquet") and not utils.is_gt_filename("FAN1205_X.parquet")
    b = utils.ModelBundle("tiny", GT_META, bundle.model, bundle.device)
    ids = utils.encode_gt(pd.Series([0.0, 7.0, 9.0, 42.0, np.nan]), b)
    assert ids.tolist() == [0, 2, 1, 1, -1]          # unknown / unselected -> OTHER MAN, NaN -> ignored
    assert utils.encode_gt(pd.Series(["NO MAN", "X alias", "zzz"]), b).tolist() == [0, 2, 1]
    with pytest.raises(utils.InputError):            # numeric IDs need the mapping
        utils.encode_gt(pd.Series([1.0]), bundle)


def test_gt_alignment_follows_cut(bundle):
    t = 100 + np.arange(500) / 32
    df = pd.DataFrame(np.random.randn(500, 3), columns=["a", "b", "c"], index=pd.Index(t, name="time"))
    f = utils.prepare_flight(df, "F", bundle)
    gt = pd.Series(np.arange(500), index=t)
    assert utils.align_gt(gt, f, 32).tolist() == list(range(10, 490))            # positional
    assert utils.align_gt(gt.iloc[5:], f, 32).tolist() == list(range(10, 490))   # by time


def test_scores_match_definition():
    y, p = np.array([0, 0, 1, 1, 2, -1]), np.array([0, 1, 1, 1, 0, 2])
    cm = utils.confusion_matrix(y, p, 3)
    assert cm.sum() == 5
    s, per_class = utils.scores_from_confusion(cm, ["n", "o", "x"])
    assert s["accuracy"] == pytest.approx(3 / 5)
    f1 = [0.5, 0.8, 0.0]
    assert s["f1_weighted"] == pytest.approx((2 * f1[0] + 2 * f1[1] + f1[2]) / 5)
    assert s["f1_macro"] == pytest.approx(np.mean(f1))


def test_plot_with_ground_truth(bundle):
    flight = utils.prepare_flight(pd.DataFrame(np.random.randn(400, 3), columns=["a", "b", "c"]), "F", bundle)
    res = utils.classify_flight(bundle, flight, None)
    gt = np.zeros(len(flight.x), int); gt[50:150] = 2
    fig = utils.plot_flight(res, flight.raw, flight.time, ["a"], {}, META["classes"], gt=gt)
    assert any(tr.name == "pred ≠ GT" for tr in fig.data) or (res.labels[gt >= 0] == gt).all()
    utils.plot_flight(None, flight.raw, flight.time, ["a"], {}, META["classes"], gt=gt)  # GT before running


def test_read_from_folder_only_needed_columns(tmp_path, bundle):
    t = 100 + np.arange(300) / 32
    df = pd.DataFrame(np.random.randn(300, 5), columns=["a", "b", "c", "x1", "x2"], index=pd.Index(t, name="time"))
    df.to_parquet(tmp_path / "F7_X.parquet")
    df[["a"]].rename(columns={"a": "MANOEUVERS"}).to_parquet(tmp_path / "F7_Y.parquet")
    files = utils.list_flight_files(tmp_path)
    assert list(files) == ["F7"]                                   # _Y is not a flight
    read = utils.read_flight_path(files["F7"], ["a", "b", "c", "MANOEUVERS"])
    assert list(read.columns) == ["a", "b", "c"] and read.index[0] == 100   # time kept, extras skipped
    assert utils.list_flight_files(tmp_path / "missing") == {}
    df.to_parquet(tmp_path / "F7.parquet")                         # ambiguous name: _X still wins
    assert utils.list_flight_files(tmp_path)["F7"].name == "F7_X.parquet"
