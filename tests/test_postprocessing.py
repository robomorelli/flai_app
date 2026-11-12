import numpy as np

from postprocessing import (PostProcessingParams, filter_and_fill_out, merge_sequences_with_threshold,
                            nonzero_segments, post_process, unify_predictions)


def test_all_no_man_does_not_crash():
    p = np.zeros(1000, dtype=int)
    assert (post_process(p, 128, PostProcessingParams()) == 0).all()
    assert (merge_sequences_with_threshold(p, 100) == 0).all()


def test_empty():
    assert post_process(np.array([], dtype=int), 128, PostProcessingParams()).size == 0


def test_segments():
    p = np.array([0, 1, 1, 2, 0, 0, 3])
    assert nonzero_segments(p) == [(1, 2, 1), (3, 3, 2), (6, 6, 3)]


def test_unify_uses_mode():
    p = np.array([0, 2, 2, 2, 5, 0])
    assert unify_predictions(p).tolist() == [0, 2, 2, 2, 2, 0]


def test_merge_same_label():
    p = np.array([1] * 5 + [0] * 3 + [1] * 5)
    assert (merge_sequences_with_threshold(p, 3) == 1).all()
    assert merge_sequences_with_threshold(p, 2)[6] == 0


def test_merge_keeps_last_segment_with_different_label():
    # the training-repo version dropped the final segment in this case
    p = np.array([1] * 10 + [0] * 2 + [2] * 3)
    out = merge_sequences_with_threshold(p, 5, only_same_label=False)
    assert (out == 1).all()
    p2 = np.array([1] * 10 + [0] * 20 + [2] * 3)
    out2 = merge_sequences_with_threshold(p2, 5, only_same_label=False)
    assert out2[-1] == 2


def test_fill_never_wraps_around():
    p = np.zeros(50, dtype=int)
    p[0:5] = 3  # segment at the very beginning, will be expanded
    out = filter_and_fill_out(p, {3: 100}, min_perc=0.01, fixed_th=0, fill_perc=0.4)
    assert out[-1] == 0          # negative indices would have written the tail
    assert out[0] == 3


def test_fill_at_end_no_index_error():
    p = np.zeros(50, dtype=int)
    p[-5:] = 3
    out = filter_and_fill_out(p, {3: 100}, min_perc=0.01, fixed_th=0, fill_perc=0.4)
    assert out[-1] == 3 and out.size == 50


def test_short_segment_dropped():
    p = np.zeros(1000, dtype=int)
    p[100:150] = 4
    out = filter_and_fill_out(p, {}, min_perc=0.05, fixed_th=228, fill_perc=0.4)
    assert (out == 0).all()
