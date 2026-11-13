"""
Inference pipeline: model bundle loading, file parsing, preprocessing,
windowed prediction, segment extraction and plotting.

Nothing here depends on Streamlit, so it can be tested and reused from scripts.
"""
from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch

# Streamlit's file watcher trips on torch.classes (noisy 'Examining the path of torch.classes' logs).
torch.classes.__path__ = []
from plotly.subplots import make_subplots

import config
from model import build_model
from postprocessing import PostProcessingParams, nonzero_segments, post_process


# --------------------------------------------------------------------------- model

@dataclass
class ModelBundle:
    name: str
    meta: dict
    model: torch.nn.Module
    device: torch.device

    @property
    def sensors(self) -> List[str]:
        return self.meta["sensors"]

    @property
    def classes(self) -> List[str]:
        return self.meta["classes"]

    @property
    def window(self) -> int:
        return int(self.meta["hps"]["WINDOW_SIZE"])

    @property
    def output_steps(self) -> int:
        return int(self.meta["hps"]["OUTPUT_STEPS"])

    @property
    def hz(self) -> float:
        return float(self.meta["sampling_hz"])


def available_models(model_dir: Path = config.CLASSIFICATION_MODEL_DIR) -> List[str]:
    """Bundles that have both <name>.json and <name>.pt."""
    if not model_dir.exists():
        return []
    return sorted(p.stem for p in model_dir.glob("*.json") if p.with_suffix(".pt").exists())


def read_model_meta(name: str, model_dir: Path = config.CLASSIFICATION_MODEL_DIR) -> dict:
    with open(model_dir / f"{name}.json") as f:
        return json.load(f)


def get_device(pref: str = config.DEVICE) -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(name: str, model_dir: Path = config.CLASSIFICATION_MODEL_DIR, device: Optional[str] = None) -> ModelBundle:
    meta = read_model_meta(name, model_dir)
    dev = get_device(device or config.DEVICE)
    state = torch.load(model_dir / f"{name}.pt", map_location="cpu", weights_only=True)
    model = build_model(meta)
    model.load_state_dict(state, strict=True)
    model.to(dev).eval()
    return ModelBundle(name=name, meta=meta, model=model, device=dev)


# --------------------------------------------------------------------------- input

class InputError(ValueError):
    """Problem with an uploaded file, meant to be shown to the user."""


def flight_name_from_filename(filename: str) -> str:
    return Path(filename).name.split(".")[0].split("_")[0]


def read_flight_file(data: bytes, filename: str) -> pd.DataFrame:
    """Parse an uploaded CSV/Parquet file. The time axis (if any) becomes the index."""
    ext = Path(filename).suffix.lower()
    buf = io.BytesIO(data)
    if ext == ".parquet":
        df = pd.read_parquet(buf)
    elif ext == ".csv":
        df = pd.read_csv(buf)
        df = df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed:")])
    else:
        raise InputError(f"{filename}: unsupported format '{ext}' (use .csv or .parquet)")
    return _set_time_index(df)


def _set_time_index(df: pd.DataFrame) -> pd.DataFrame:
    time_col = next((c for c in config.TIME_COLUMNS if c in df.columns), None)
    return df.set_index(time_col) if time_col is not None else df


def read_flight_path(path, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Read a flight file from disk. For parquet only ``columns`` (+ any time column) are read,
    which matters for wide files (hundreds of sensors); missing columns are simply skipped."""
    path = Path(path)
    if path.suffix.lower() == ".parquet" and columns is not None:
        import pyarrow.parquet as pq

        have = set(pq.read_schema(path).names)
        cols = [c for c in list(columns) + list(config.TIME_COLUMNS) if c in have]
        return _set_time_index(pd.read_parquet(path, columns=cols))
    return read_flight_file(path.read_bytes(), path.name)


def list_flight_files(folder) -> Dict[str, Path]:
    """Flight name -> signal file (.parquet/.csv, never a `_Y` label file) in ``folder``.
    If a flight has several candidates, `<flight>_X.*` wins, then parquet over csv."""
    if not folder or not Path(folder).expanduser().is_dir():
        return {}
    rank = lambda p: (not p.name.split(".")[0].upper().endswith("_X"), p.suffix.lower() != ".parquet", p.name)
    out = {}
    for p in sorted(Path(folder).expanduser().iterdir(), key=rank):
        if p.is_file() and p.suffix.lower() in (".parquet", ".csv") and not is_gt_filename(p.name):
            out.setdefault(flight_name_from_filename(p.name), p)
    return dict(sorted(out.items()))


@dataclass
class PreparedFlight:
    name: str
    raw: pd.DataFrame          # selected sensors, physical units, after cut
    x: np.ndarray              # normalized [T, C] float32
    time: np.ndarray           # seconds (from file if available, else step / hz)
    warnings: List[str] = field(default_factory=list)
    src_start: int = 0         # position of raw[0] in the source file (after the transient cut)
    n_source: int = 0          # rows in the source file


def prepare_flight(df: pd.DataFrame, name: str, bundle: ModelBundle, cut: bool = True) -> PreparedFlight:
    """Select + order sensors, cut transients, fill gaps, normalize with training stats."""
    warnings: List[str] = []
    missing = [s for s in bundle.sensors if s not in df.columns]
    if missing:
        raise InputError(f"{name}: {len(missing)} sensors required by the model are missing: {', '.join(missing)}")

    raw = df[bundle.sensors].apply(pd.to_numeric, errors="coerce")
    n_source, src_start = len(raw), 0

    cs, ce = int(bundle.meta.get("cut_start", 0)), int(bundle.meta.get("cut_end", 0))
    if cut and (cs or ce):
        if len(raw) > cs + ce + bundle.output_steps:
            raw = raw.iloc[cs:len(raw) - ce]
            src_start = cs
        else:
            warnings.append(f"flight too short to cut {cs}+{ce} transient steps: cut skipped")

    n_nan = int(raw.isna().sum().sum())
    if n_nan:
        warnings.append(f"{n_nan} missing/non-numeric values filled with the nearest valid value")
        raw = raw.ffill().bfill().fillna(0.0)

    if len(raw) < bundle.window:
        warnings.append(f"flight shorter than one model window ({len(raw)} < {bundle.window} steps): "
                        "padded, predictions less reliable")

    mean = np.asarray(bundle.meta["mean"], dtype=np.float32)
    std = np.asarray(bundle.meta["std"], dtype=np.float32)
    x = ((raw.to_numpy(dtype=np.float32) - mean) / std).astype(np.float32)

    idx = raw.index
    if pd.api.types.is_numeric_dtype(idx) and not isinstance(idx, pd.RangeIndex):
        time = idx.to_numpy(dtype=float)
        time = time - time[0]
    else:
        time = np.arange(len(raw)) / bundle.hz
    return PreparedFlight(name=name, raw=raw, x=x, time=time, warnings=warnings,
                          src_start=src_start, n_source=n_source)


# --------------------------------------------------------------------------- ground truth

def is_gt_filename(filename: str) -> bool:
    """`FAN1205_Y.parquet` -> True (training-repo convention: <flight>_X = signals, <flight>_Y = labels)."""
    return Path(filename).name.split(".")[0].upper().endswith("_Y")


def gt_column(df: pd.DataFrame, bundle: ModelBundle) -> Optional[str]:
    cands = [bundle.meta.get("gt", {}).get("column")] + list(config.GT_COLUMNS)
    return next((c for c in cands if c and c in df.columns), None)


def default_gt_dir(bundle: ModelBundle) -> str:
    """FLAI_GT_DIR if set, else the first existing training-data folder for the bundle's aircraft."""
    if config.GT_DIR:
        return config.GT_DIR
    for c in config.GT_DIR_CANDIDATES:
        p = Path(c.format(aircraft=bundle.meta.get("aircraft", "")))
        if p.is_dir():
            return str(p)
    return ""


def find_gt_file(flight_name: str, folder) -> Optional[Path]:
    """<folder>/<flight>_Y.parquet (or .csv), if it exists."""
    if not folder:
        return None
    folder = Path(folder).expanduser()
    for ext in (".parquet", ".csv"):
        p = folder / f"{flight_name}_Y{ext}"
        if p.is_file():
            return p
    return None


def encode_gt(values: pd.Series, bundle: ModelBundle) -> np.ndarray:
    """Labels -> class ids with the training-time mapping; -1 where the label is missing.

    Numeric IDs go through the bundle's ``gt.id_to_name`` (manoeuvers.csv), names through
    ``gt.alias`` (chosen_man_*.json). Anything not mapped becomes OTHER MAN, as in training.
    Class order from the export: [NO MAN, OTHER MAN, <manoeuvres>].
    """
    classes = bundle.classes
    gt = bundle.meta.get("gt", {})
    no_man, other = classes[0], classes[1]
    ids = {c: i for i, c in enumerate(classes)}
    alias = gt.get("alias") or {c: c for c in classes[2:]}
    id_to_name = {int(float(k)): v for k, v in gt.get("id_to_name", {}).items()}
    numeric = pd.api.types.is_numeric_dtype(values)
    if numeric and not id_to_name:
        raise InputError("ground truth contains numeric manoeuvre IDs but the model bundle has no ID -> name "
                         "mapping (`gt.id_to_name`): re-export it with scripts/export_checkpoint.py")

    valid = values.notna().to_numpy()
    lut = {}
    for u in pd.unique(values[valid]):
        name = id_to_name.get(int(u), other) if numeric else str(u).strip()
        lut[u] = ids[name if name == no_man else alias.get(name, other)]
    out = np.full(len(values), -1, dtype=int)
    out[valid] = values[valid].map(lut).to_numpy(dtype=int)
    return out


def align_gt(gt: pd.Series, flight: PreparedFlight, hz: float) -> pd.Series:
    """Ground-truth labels on the flight's steps (same transient cut). Same length as the source
    file -> positional; otherwise matched on the time index (nearest within half a step)."""
    n = len(flight.raw)
    if len(gt) == flight.n_source:
        return gt.iloc[flight.src_start:flight.src_start + n].reset_index(drop=True)
    idx = flight.raw.index
    if pd.api.types.is_numeric_dtype(gt.index) and pd.api.types.is_numeric_dtype(idx) \
            and not isinstance(idx, pd.RangeIndex):
        g = gt[~gt.index.duplicated()].sort_index()
        return g.reindex(idx, method="nearest", tolerance=0.5 / hz).reset_index(drop=True)
    raise InputError(f"{flight.name}: ground truth has {len(gt)} rows, flight has {flight.n_source}, "
                     "and there is no time column to align them")


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> np.ndarray:
    """[k, k] counts, rows = ground truth, cols = prediction. Steps with y_true < 0 are ignored."""
    m = y_true >= 0
    return np.bincount(k * y_true[m] + y_pred[m], minlength=k * k).reshape(k, k)


def scores_from_confusion(cm: np.ndarray, classes: List[str]) -> Tuple[dict, pd.DataFrame]:
    """Step-level accuracy / F1 (weighted, macro) like sklearn, plus a per-class table."""
    tp = np.diag(cm).astype(float)
    support, predicted = cm.sum(1).astype(float), cm.sum(0).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(predicted > 0, tp / predicted, 0.0)
        rec = np.where(support > 0, tp / support, 0.0)
        f1 = np.where(support + predicted > 0, 2 * tp / (support + predicted), 0.0)
    n = support.sum()
    present = (support + predicted) > 0
    summary = {
        "steps": int(n),
        "accuracy": float(tp.sum() / n) if n else float("nan"),
        "f1_weighted": float((f1 * support).sum() / n) if n else float("nan"),
        "f1_macro": float(f1[present].mean()) if present.any() else float("nan"),
    }
    per_class = pd.DataFrame({"class": classes, "precision": prec.round(3), "recall": rec.round(3),
                              "f1": f1.round(3), "support_steps": support.astype(int),
                              "predicted_steps": predicted.astype(int)})
    return summary, per_class[present].reset_index(drop=True)


# --------------------------------------------------------------------------- inference

def window_starts(n: int, window: int) -> List[int]:
    """Non-overlapping windows covering [0, n); the last one is aligned to the end."""
    if n <= window:
        return [0]
    starts = list(range(0, n - window + 1, window))
    if starts[-1] + window < n:
        starts.append(n - window)
    return starts


@torch.no_grad()
def predict_proba(bundle: ModelBundle, x: np.ndarray, batch_size: int = config.BATCH_SIZE,
                  progress=None) -> np.ndarray:
    """Per-step class probabilities [T, K] for one flight, covering every step."""
    n, w, k = len(x), bundle.window, len(bundle.classes)
    if n < w:  # edge padding to a full window
        x = np.pad(x, ((0, w - n), (0, 0)), mode="edge")
    starts = window_starts(len(x), w)

    probs = np.zeros((len(x), k), dtype=np.float32)
    covered = 0
    for b in range(0, len(starts), batch_size):
        chunk = starts[b:b + batch_size]
        xb = torch.from_numpy(np.stack([x[s:s + w] for s in chunk])).to(bundle.device)
        p = torch.softmax(bundle.model(xb).float(), dim=-1).cpu().numpy()  # [B, w/out, K]
        p = np.repeat(p, bundle.output_steps, axis=1)                      # [B, w', K]
        for s, ps in zip(chunk, p):
            ps = np.pad(ps, ((0, w - len(ps)), (0, 0)), mode="edge") if len(ps) < w else ps[:w]
            # end-aligned last window overlaps the previous one: only fill uncovered steps
            off = max(covered - s, 0)
            probs[s + off:s + w] = ps[off:]
            covered = s + w
        if progress is not None:
            progress(min(1.0, (b + len(chunk)) / len(starts)))
    return probs[:n]


@dataclass
class FlightResult:
    name: str
    time: np.ndarray
    raw: pd.DataFrame
    proba: np.ndarray
    labels_raw: np.ndarray
    labels: np.ndarray
    warnings: List[str]

    def steps_df(self, classes: List[str], gt: Optional[np.ndarray] = None) -> pd.DataFrame:
        conf = self.proba[np.arange(len(self.labels)), self.labels]
        df = pd.DataFrame({
            "flight": self.name,
            "step": np.arange(len(self.labels)),
            "time_s": np.round(self.time, 4),
            "class_id": self.labels,
            "class_name": np.asarray(classes)[self.labels],
            "confidence": np.round(conf, 4),
            "class_id_raw": self.labels_raw,
        })
        if gt is not None:
            df["gt_class_id"] = gt
            df["gt_class_name"] = np.where(gt >= 0, np.asarray(classes)[np.maximum(gt, 0)], "")
        return df


def segments_df(res: FlightResult, classes: List[str]) -> pd.DataFrame:
    rows = []
    for s, e, l in nonzero_segments(res.labels):
        rows.append({
            "flight": res.name,
            "manoeuvre": classes[l],
            "class_id": l,
            "start_s": round(float(res.time[s]), 2),
            "end_s": round(float(res.time[e]), 2),
            "duration_s": round(float(res.time[e] - res.time[s]), 2),
            "start_step": s,
            "end_step": e,
            "mean_confidence": round(float(res.proba[s:e + 1, l].mean()), 3),
        })
    cols = ["flight", "manoeuvre", "class_id", "start_s", "end_s", "duration_s",
            "start_step", "end_step", "mean_confidence"]
    return pd.DataFrame(rows, columns=cols)


def pp_params(bundle: ModelBundle, use_class_durations: bool, **overrides) -> PostProcessingParams:
    durations = {}
    if use_class_durations:
        ids = {c: i for i, c in enumerate(bundle.classes)}
        durations = {ids[k]: int(v) for k, v in bundle.meta.get("man_min_duration_steps", {}).items() if k in ids}
    return PostProcessingParams(min_durations=durations, **overrides)


def classify_flight(bundle: ModelBundle, flight: PreparedFlight, params: Optional[PostProcessingParams],
                    progress=None) -> FlightResult:
    proba = predict_proba(bundle, flight.x, progress=progress)
    labels_raw = proba.argmax(axis=1)
    labels = post_process(labels_raw, bundle.output_steps, params) if params is not None else labels_raw
    return FlightResult(flight.name, flight.time, flight.raw, proba, labels_raw, labels, list(flight.warnings))


# --------------------------------------------------------------------------- plotting

def _decimate(n: int, max_points: int) -> np.ndarray:
    """Indices keeping min and max of each bucket, so peaks survive decimation."""
    if n <= max_points:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, max_points).astype(int))


def _minmax_decimate(y: np.ndarray, max_points: int) -> np.ndarray:
    n = len(y)
    if n <= max_points:
        return np.arange(n)
    buckets = max_points // 2
    edges = np.linspace(0, n, buckets + 1).astype(int)
    idx = []
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        seg = y[a:b]
        i1, i2 = a + int(np.nanargmin(seg)), a + int(np.nanargmax(seg))
        idx.extend(sorted((i1, i2)))
    return np.asarray(idx)


def class_colors(classes: List[str]) -> Dict[str, str]:
    import plotly.express as px

    palette = px.colors.qualitative.Dark24 + px.colors.qualitative.Light24
    return {c: palette[i % len(palette)] for i, c in enumerate(classes)}


def _zscore(raw: pd.DataFrame, s: str, normalize: bool):
    """(physical values, plotted values) for one sensor; z-score uses whole-flight stats so the
    scale does not jump while moving the window."""
    y = raw[s].to_numpy(dtype=float)
    if not normalize:
        return y, y
    sd = np.nanstd(y)
    return y, (y - np.nanmean(y)) / (sd if sd > 0 else 1.0)


def _timeline(fig: go.Figure, labels: np.ndarray, time: np.ndarray, a: int, b: int, classes: List[str],
              colors: Dict[str, str], seen: set, row: int, what: str, proba: Optional[np.ndarray] = None,
              bands: bool = False):
    """One coloured box per manoeuvre segment overlapping [a, b)."""
    for s, e, l in nonzero_segments(labels):
        if e < a or s >= b:
            continue
        c = classes[l]
        x0, x1 = float(time[s]), float(time[e])
        if bands:
            fig.add_vrect(x0=x0, x1=x1, fillcolor=colors[c], opacity=0.15, line_width=0, row=1, col=1)
        conf = f"<br>conf {proba[s:e + 1, l].mean():.2f}" if proba is not None else ""
        fig.add_trace(go.Scatter(
            x=[x0, x1, x1, x0, x0], y=[0, 0, 1, 1, 0], fill="toself", mode="lines",
            line=dict(width=0), fillcolor=colors[c], name=c, legendgroup=c, showlegend=c not in seen,
            legendgrouptitle_text="Manoeuvres" if not seen else None,
            hovertemplate=f"<b>{c}</b> ({what})<br>{x0:.1f}s → {x1:.1f}s ({x1 - x0:.1f}s){conf}<extra></extra>"),
            row=row, col=1)
        seen.add(c)


def plot_flight(res: Optional[FlightResult], raw: pd.DataFrame, time: np.ndarray, sensors: List[str],
                descriptions: Dict[str, str], classes: Optional[List[str]] = None,
                t_range: Optional[tuple] = None, max_points: Optional[int] = None,
                show_confidence: bool = True, normalize: bool = True,
                gt: Optional[np.ndarray] = None) -> go.Figure:
    """Detail view: all signals overlaid in one panel + prediction timeline (+ ground truth) below.

    Only samples inside ``t_range`` (seconds, inclusive) are drawn, at full resolution unless
    ``max_points`` is given (then min/max decimation). Hover shows physical units.
    ``gt``: per-step ground-truth class ids (-1 = unknown); drawn under the prediction, with
    mismatching steps marked in red.
    """
    has_res = res is not None and classes is not None
    has_gt = gt is not None and classes is not None
    a, b = (0, len(time)) if t_range is None else (int(np.searchsorted(time, t_range[0], "left")),
                                                    int(np.searchsorted(time, t_range[1], "right")))
    b = max(b, a + 1)
    t = time[a:b]
    rows = 1 + has_res + has_gt
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        row_heights=[1.0] + [0.2] * (rows - 1))

    for s in sensors:
        y, y_plot = _zscore(raw, s, normalize)
        y, y_plot = y[a:b], y_plot[a:b]
        idx = _minmax_decimate(y, max_points) if max_points else np.arange(len(y))
        label = descriptions.get(s, s)
        short = label if len(label) <= 40 else label[:39] + "…"
        fig.add_trace(go.Scattergl(x=t[idx], y=y_plot[idx], customdata=y[idx], mode="lines", name=short,
                                   legendgroup="signals", legendgrouptitle_text="Signals", line=dict(width=1),
                                   hovertemplate=f"{label}: %{{customdata:.3f}}<extra></extra>"),
                      row=1, col=1)
    fig.update_yaxes(title_text="z-score" if normalize else "value", title_font_size=10, row=1, col=1)

    colors = class_colors(classes) if classes is not None else {}
    seen: set = set()
    row = 2
    if has_res:
        _timeline(fig, res.labels, time, a, b, classes, colors, seen, row, "prediction", res.proba, bands=True)
        if show_confidence:
            conf = res.proba[np.arange(a, b), res.labels[a:b]]
            fig.add_trace(go.Scattergl(x=t, y=conf, mode="lines", name="confidence",
                                       line=dict(color="rgba(80,80,80,0.6)", width=1)), row=row, col=1)
        fig.update_yaxes(range=[0, 1.05], title_text="prediction", title_font_size=10, row=row, col=1)
        row += 1
    if has_gt:
        _timeline(fig, gt, time, a, b, classes, colors, seen, row, "ground truth")
        if has_res:
            g, p = gt[a:b], res.labels[a:b]
            wrong = np.where((g >= 0) & (g != p), 1.0, np.nan)
            if np.isfinite(wrong).any():
                fig.add_trace(go.Scattergl(x=t, y=wrong * 1.03, mode="lines", name="pred ≠ GT",
                                           line=dict(color="crimson", width=4), hoverinfo="skip"),
                              row=row, col=1)
        fig.update_yaxes(range=[0, 1.08], title_text="ground truth", title_font_size=10, row=row, col=1)
    for r in range(2, rows + 1):
        fig.update_yaxes(showticklabels=False, row=r, col=1)

    fig.update_xaxes(range=[float(t[0]), float(t[-1])], row=1, col=1)
    fig.update_xaxes(title_text="time [s]", row=rows, col=1)
    fig.update_layout(height=520 + 130 * (rows - 1), margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(groupclick="toggleitem"), hovermode="x unified")
    return fig


def plot_overview(res: Optional[FlightResult], raw: pd.DataFrame, time: np.ndarray, sensors: List[str],
                  classes: Optional[List[str]], t_range: tuple,
                  max_points: int = config.MAX_PLOT_POINTS) -> go.Figure:
    """Whole-flight navigator: compact (decimated) signals, manoeuvre bands, current window highlighted."""
    fig = go.Figure()
    for s in sensors:
        _, y = _zscore(raw, s, True)
        idx = _minmax_decimate(y, max_points)
        fig.add_trace(go.Scattergl(x=time[idx], y=y[idx], mode="lines", line=dict(width=0.7),
                                   opacity=0.6, hoverinfo="skip", showlegend=False))
    if res is not None and classes is not None:
        colors = class_colors(classes)
        for s, e, l in nonzero_segments(res.labels):
            fig.add_vrect(x0=float(time[s]), x1=float(time[e]), fillcolor=colors[classes[l]],
                          opacity=0.25, line_width=0)
    fig.add_vrect(x0=t_range[0], x1=t_range[1], fillcolor="rgba(0,0,0,0)", line=dict(color="crimson", width=2))
    fig.update_xaxes(range=[float(time[0]), float(time[-1])], title_text="time [s]", fixedrange=True)
    fig.update_yaxes(visible=False, fixedrange=True)
    fig.update_layout(height=170, margin=dict(l=10, r=10, t=10, b=10), hovermode=False)
    return fig


def plot_confusion(cm_norm: np.ndarray, labels: List[str]) -> go.Figure:
    """Row-normalized confusion matrix (rows = ground truth)."""
    short = [l if len(l) <= 28 else l[:27] + "…" for l in labels]
    fig = go.Figure(go.Heatmap(z=cm_norm, x=short, y=short, colorscale="Blues", zmin=0, zmax=1,
                               text=np.round(cm_norm, 2), texttemplate="%{text}", textfont_size=9,
                               hovertemplate="GT %{y}<br>pred %{x}<br>%{z:.2f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", title_text="ground truth")
    fig.update_xaxes(title_text="prediction", tickangle=40)
    fig.update_layout(height=max(400, 32 * len(labels) + 150), margin=dict(l=10, r=10, t=10, b=10))
    return fig


def plot_occurrences(res: FlightResult, time: np.ndarray, classes: List[str], cls, gt: Optional[np.ndarray] = None,
                     numbers: Optional[Dict[int, int]] = None, title: Optional[str] = None) -> go.Figure:
    """Whole-flight strip for one or more manoeuvres (``cls``: id or set of ids): their predicted
    occurrences in colour (other manoeuvres in grey) and, if available, the same manoeuvres in the
    ground truth. ``numbers``: start step -> label number written above each predicted occurrence."""
    cls = {cls} if isinstance(cls, (int, np.integer)) else set(cls)
    colors = class_colors(classes)
    numbers = numbers or {}
    fig = go.Figure()
    rows = [("prediction", res.labels, 1.0)] + ([("ground truth", gt, 0.0)] if gt is not None else [])
    for what, labels, y0 in rows:
        for s, e, l in nonzero_segments(labels):
            x0, x1 = float(time[s]), float(time[e])
            mine, pred = l in cls, what == "prediction"
            num = numbers.get(s) if mine and pred else None
            fig.add_trace(go.Scatter(
                x=[x0, x1, x1, x0, x0], y=[y0, y0, y0 + 0.8, y0 + 0.8, y0], fill="toself", mode="lines",
                line=dict(width=0), fillcolor=colors[classes[l]] if mine else "rgba(160,160,160,0.35)",
                name=classes[l], legendgroup=classes[l], showlegend=False,
                hovertemplate=(f"<b>{'#%d ' % num if num else ''}{classes[l]}</b> ({what})"
                               f"<br>{x0:.1f}s → {x1:.1f}s ({x1 - x0:.1f}s)<extra></extra>")))
            if num:
                fig.add_annotation(x=(x0 + x1) / 2, y=y0 + 0.8, text=f"#{num}", showarrow=False,
                                   yanchor="bottom", font=dict(size=10))
    if len(cls) > 1:  # legend: one entry per selected manoeuvre
        for c in sorted(cls):
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=classes[c],
                                     marker=dict(color=colors[classes[c]], size=10, symbol="square")))
    fig.update_yaxes(tickvals=[y0 + 0.4 for _, _, y0 in rows], ticktext=[w for w, _, _ in rows],
                     range=[-0.1 if gt is not None else 0.9, 2.1], fixedrange=True)
    fig.update_xaxes(range=[float(time[0]), float(time[-1])], title_text="time [s]")
    fig.update_layout(height=(150 if gt is not None else 120) + (25 if title else 0),
                      margin=dict(l=10, r=10, t=30 if title else 10, b=10),
                      title=dict(text=title, font_size=12) if title else None,
                      legend=dict(orientation="h", y=1.02, yanchor="bottom", x=1, xanchor="right"))
    return fig
