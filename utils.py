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
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
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

    time_col = next((c for c in config.TIME_COLUMNS if c in df.columns), None)
    if time_col is not None:
        df = df.set_index(time_col)
    return df


@dataclass
class PreparedFlight:
    name: str
    raw: pd.DataFrame          # selected sensors, physical units, after cut
    x: np.ndarray              # normalized [T, C] float32
    time: np.ndarray           # seconds (from file if available, else step / hz)
    warnings: List[str] = field(default_factory=list)


def prepare_flight(df: pd.DataFrame, name: str, bundle: ModelBundle, cut: bool = True) -> PreparedFlight:
    """Select + order sensors, cut transients, fill gaps, normalize with training stats."""
    warnings: List[str] = []
    missing = [s for s in bundle.sensors if s not in df.columns]
    if missing:
        raise InputError(f"{name}: {len(missing)} sensors required by the model are missing: {', '.join(missing)}")

    raw = df[bundle.sensors].apply(pd.to_numeric, errors="coerce")

    cs, ce = int(bundle.meta.get("cut_start", 0)), int(bundle.meta.get("cut_end", 0))
    if cut and (cs or ce):
        if len(raw) > cs + ce + bundle.output_steps:
            raw = raw.iloc[cs:len(raw) - ce]
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
    return PreparedFlight(name=name, raw=raw, x=x, time=time, warnings=warnings)


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

    def steps_df(self, classes: List[str]) -> pd.DataFrame:
        conf = self.proba[np.arange(len(self.labels)), self.labels]
        return pd.DataFrame({
            "flight": self.name,
            "step": np.arange(len(self.labels)),
            "time_s": np.round(self.time, 4),
            "class_id": self.labels,
            "class_name": np.asarray(classes)[self.labels],
            "confidence": np.round(conf, 4),
            "class_id_raw": self.labels_raw,
        })


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


def plot_flight(res: Optional[FlightResult], raw: pd.DataFrame, time: np.ndarray, sensors: List[str],
                descriptions: Dict[str, str], classes: Optional[List[str]] = None,
                max_points: int = config.MAX_PLOT_POINTS, show_confidence: bool = True) -> go.Figure:
    """Signals in physical units (one row each, shared time axis) + manoeuvre timeline."""
    has_res = res is not None and classes is not None
    n_sig = max(len(sensors), 1)
    rows = n_sig + (1 if has_res else 0)
    heights = [1.0] * n_sig + ([0.6] if has_res else [])
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=heights)

    for r, s in enumerate(sensors, start=1):
        y = raw[s].to_numpy()
        idx = _minmax_decimate(y, max_points)
        label = descriptions.get(s, s)
        fig.add_trace(go.Scattergl(x=time[idx], y=y[idx], mode="lines", name=label, showlegend=False,
                                   line=dict(width=1), hovertemplate=f"{label}<br>t=%{{x:.1f}}s<br>%{{y:.3f}}<extra></extra>"),
                      row=r, col=1)
        fig.update_yaxes(title_text=label if len(label) <= 28 else label[:27] + "…", title_font_size=10, row=r, col=1)

    if has_res:
        colors = class_colors(classes)
        segs = nonzero_segments(res.labels)
        seen = set()
        for s, e, l in segs:
            c = classes[l]
            x0, x1 = float(time[s]), float(time[e])
            for r in range(1, n_sig + 1):
                fig.add_vrect(x0=x0, x1=x1, fillcolor=colors[c], opacity=0.15, line_width=0, row=r, col=1)
            fig.add_trace(go.Scatter(
                x=[x0, x1, x1, x0, x0], y=[0, 0, 1, 1, 0], fill="toself", mode="lines",
                line=dict(width=0), fillcolor=colors[c], name=c, legendgroup=c, showlegend=c not in seen,
                hovertemplate=f"<b>{c}</b><br>{x0:.1f}s → {x1:.1f}s ({x1 - x0:.1f}s)"
                              f"<br>conf {res.proba[s:e + 1, l].mean():.2f}<extra></extra>"),
                row=rows, col=1)
            seen.add(c)
        if show_confidence:
            conf = res.proba[np.arange(len(res.labels)), res.labels]
            idx = _decimate(len(conf), max_points)
            fig.add_trace(go.Scattergl(x=time[idx], y=conf[idx], mode="lines", name="confidence",
                                       line=dict(color="rgba(80,80,80,0.6)", width=1)), row=rows, col=1)
        fig.update_yaxes(range=[0, 1.05], title_text="manoeuvre / conf.", title_font_size=10, row=rows, col=1)

    fig.update_xaxes(title_text="time [s]", row=rows, col=1)
    fig.update_layout(height=max(320, 150 * rows), margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", y=1.02, yanchor="bottom"), hovermode="x unified")
    return fig
