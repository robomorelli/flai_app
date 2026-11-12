"""
Post-processing of per-step manoeuvre predictions (label 0 == NO MAN).

Same three stages as the training repo (unify -> merge -> filter/fill), with the
edge cases fixed:
  * empty input / all NO MAN no longer crash (IndexError on segment_bounds[0]);
  * the last segment is no longer dropped when it has a different label;
  * fill/expand never writes outside the array (negative indices wrapped around);
  * applied per flight, never across the boundary of two concatenated flights.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

Segment = Tuple[int, int, int]  # (start, end_inclusive, label)


@dataclass
class PostProcessingParams:
    merge_gap_steps: int = 100          # max NO MAN gap merged between two segments
    min_perc: float = 0.05              # drop segment if shorter than min_perc * class min duration
    fill_perc: float = 0.4              # expand segment if shorter than fill_perc * class min duration
    extra_min_steps: int = 100          # classes without stats: drop if shorter than output_steps + this
    min_durations: Dict[int, int] = field(default_factory=dict)  # class id -> min duration (steps)


def nonzero_segments(pred: np.ndarray) -> List[Segment]:
    """Contiguous runs of the same non-zero label."""
    pred = np.asarray(pred)
    if pred.size == 0:
        return []
    change = np.flatnonzero(np.diff(pred)) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change - 1, [pred.size - 1]))
    return [(int(s), int(e), int(pred[s])) for s, e in zip(starts, ends) if pred[s] != 0]


def _nonzero_blocks(pred: np.ndarray) -> List[Tuple[int, int]]:
    """Contiguous runs of non-zero values (labels may differ inside)."""
    nz = np.concatenate(([0], (pred != 0).astype(np.int8), [0]))
    d = np.diff(nz)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))  # [start, end)


def _mode(values: np.ndarray, tie_break: int) -> int:
    labels, counts = np.unique(values, return_counts=True)
    best = labels[counts == counts.max()]
    return int(tie_break) if tie_break in best else int(best[0])


def unify_predictions(pred: np.ndarray) -> np.ndarray:
    """Inside each non-zero block, replace mixed labels with the block's mode."""
    pred = np.array(pred, copy=True)
    for s, e in _nonzero_blocks(pred):
        block = pred[s:e]
        if np.unique(block).size > 1:
            pred[s:e] = _mode(block, tie_break=int(block[0]))
    return pred


def merge_sequences_with_threshold(pred: np.ndarray, threshold: int, only_same_label: bool = True) -> np.ndarray:
    """
    Merge consecutive segments separated by a NO MAN gap <= threshold.
    only_same_label=True: merge only equal labels.
    only_same_label=False: merge also different labels, the merged segment takes the
    duration-weighted mode (ties -> label of the left segment).
    """
    pred = np.asarray(pred)
    segs = nonzero_segments(pred)
    if not segs:
        return pred.copy()

    merged: List[Segment] = []
    cur_s, cur_e, cur_l = segs[0]
    cur_counts: Dict[int, int] = {cur_l: cur_e - cur_s + 1}
    for s, e, l in segs[1:]:
        gap = s - cur_e - 1
        can_merge = gap <= threshold and (l == cur_l or not only_same_label)
        if can_merge:
            cur_e = e
            cur_counts[l] = cur_counts.get(l, 0) + (e - s + 1)
            top = max(cur_counts.values())
            if cur_counts.get(cur_l, 0) != top:
                cur_l = max(cur_counts, key=cur_counts.get)
        else:
            merged.append((cur_s, cur_e, cur_l))
            cur_s, cur_e, cur_l = s, e, l
            cur_counts = {l: e - s + 1}
    merged.append((cur_s, cur_e, cur_l))

    out = np.zeros_like(pred)
    for s, e, l in merged:
        out[s:e + 1] = l
    return out


def filter_and_fill_out(pred: np.ndarray, min_durations: Dict[int, int], min_perc: float,
                        fixed_th: int, fill_perc: float) -> np.ndarray:
    """
    Drop segments that are too short for their class; symmetrically expand segments
    that are shorter than fill_perc * class min duration.
    """
    pred = np.array(pred, copy=True)
    n = pred.size
    for s, e, label in nonzero_segments(pred):
        length = e - s + 1
        min_dur = min_durations.get(label)
        if min_dur is None:
            if length < fixed_th:
                pred[s:e + 1] = 0
            continue
        if length < int(min_dur * min_perc):
            pred[s:e + 1] = 0
            continue
        ratio = length / min_dur
        if ratio < fill_perc:
            half = int((fill_perc - ratio) * min_dur) // 2
            pred[max(0, s - half):min(n, e + 1 + half)] = label
    return pred


def post_process(pred: np.ndarray, output_steps: int, params: PostProcessingParams) -> np.ndarray:
    pred = unify_predictions(pred)
    pred = merge_sequences_with_threshold(pred, params.merge_gap_steps, only_same_label=True)
    pred = merge_sequences_with_threshold(pred, params.merge_gap_steps, only_same_label=False)
    pred = filter_and_fill_out(pred, params.min_durations, min_perc=params.min_perc,
                               fixed_th=output_steps + params.extra_min_steps, fill_perc=params.fill_perc)
    return pred
