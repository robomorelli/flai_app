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
