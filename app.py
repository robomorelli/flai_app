"""Streamlit UI for flight manoeuvre recognition.  Run:  streamlit run app.py"""
import hashlib
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import config
import utils

st.set_page_config(page_title="Manoeuvre recognition", page_icon="✈️", layout="wide",
                   initial_sidebar_state="expanded")
st.title("Manoeuvre recognition")


# --------------------------------------------------------------------------- cached steps

@st.cache_resource(show_spinner="Loading model…")
def get_model(name: str) -> utils.ModelBundle:
    return utils.load_model(name)


@st.cache_data(show_spinner=False, max_entries=32)
def parse_file(data: bytes, filename: str) -> pd.DataFrame:
    return utils.read_flight_file(data, filename)


@st.cache_data(show_spinner=False, max_entries=64)
def read_flight_path(path: str, mtime: float, columns: tuple) -> pd.DataFrame:
    return utils.read_flight_path(path, list(columns))


@st.cache_data(show_spinner=False, max_entries=64)
def parse_path(path: str, mtime: float) -> pd.DataFrame:  # mtime: invalidates the cache when the file changes
    return utils.read_flight_file(Path(path).read_bytes(), path)


# --------------------------------------------------------------------------- sidebar: model

st.sidebar.header("Model")
st.sidebar.selectbox("Task", config.TASKS)

models = utils.available_models()
if not models:
    st.error(f"No model bundle found in `{config.CLASSIFICATION_MODEL_DIR}`. "
             "Export one with `scripts/export_checkpoint.py` (see README).")
    st.stop()

default_idx = models.index(config.DEFAULT_MODEL) if config.DEFAULT_MODEL in models else 0
model_name = st.sidebar.selectbox("Model", models, index=default_idx)

try:
    bundle = get_model(model_name)
except Exception as ex:  # show the real reason, then stop: nothing below can work
    st.error(f"Unable to load model '{model_name}': {ex}")
    st.stop()

meta = bundle.meta
with st.sidebar.expander("Model card", expanded=False):
    st.markdown(
        f"- **type**: {meta['model']} — aircraft {meta.get('aircraft', '?')}\n"
        f"- **trained**: {meta.get('trained_at', '?')} (epoch {meta.get('epoch', '?')})\n"
        f"- **{meta.get('validation_metric', 'val score')}**: {meta.get('validation_score', '?')}\n"
        f"- **window**: {bundle.window} steps = {bundle.window / bundle.hz:.0f}s @ {bundle.hz:g} Hz\n"
        f"- **resolution**: 1 prediction / {bundle.output_steps} steps "
        f"({bundle.output_steps / bundle.hz:g}s)\n"
        f"- **device**: {bundle.device}\n"
        f"- **sensors**: {len(bundle.sensors)} — **classes**: {len(bundle.classes)}"
    )
    st.caption(f"source: {meta.get('source_checkpoint', '')}")
    splits = meta.get("splits") or {}
    if splits:
        st.markdown("**Data split** (" + ", ".join(f"{k} {len(v)}" for k, v in splits.items()) + " flights)"
                    + (f" — `{meta['splits_source']}`" if meta.get("splits_source") else ""))
        st.markdown(f"Test flights ({len(splits.get('test', []))}), never seen in training:")
        st.code(" ".join(sorted(splits.get("test", []))), language=None, wrap_lines=True)
        with st.popover("Train / validation flights"):
            for k in ("train", "val"):
                st.markdown(f"**{k}** ({len(splits.get(k, []))})")
                st.code(" ".join(sorted(splits.get(k, []))), language=None, wrap_lines=True)
    else:
        st.caption("Train/val/test flights not recorded in this bundle (re-export to add them).")

st.sidebar.header("Post-processing")
do_pp = st.sidebar.toggle("Apply post-processing", value=True)
use_durations = st.sidebar.toggle(
    "Use per-class min durations", value=False, disabled=not do_pp,
    help="Uses the training-set min duration of each manoeuvre to drop/extend segments. "
         "Off reproduces the setting the reported test F1 was measured with.")
with st.sidebar.expander("Advanced", expanded=False):
    merge_gap = st.number_input("Merge gap [steps]", 0, 10_000, config.PP_MERGE_GAP_STEPS, disabled=not do_pp)
    extra_min = st.number_input("Min segment beyond resolution [steps]", 0, 10_000, config.PP_EXTRA_MIN_STEPS,
                                disabled=not do_pp)
    cut = st.checkbox(f"Cut transients ({meta.get('cut_start', 0)} + {meta.get('cut_end', 0)} steps)", value=True,
                      help="Same start/end cut used at training time.")

st.sidebar.header("Ground truth")
use_gt = st.sidebar.toggle(
    "Compare with ground truth", value=False,
    help="Looks for the labels of each flight and compares them with the prediction. Search order: a "
         "label column inside the flight file, an uploaded `<flight>_Y.parquet`, then the folder below.")
gt_dir = st.sidebar.text_input("Ground-truth folder (`<flight>_Y.parquet`)", value=utils.default_gt_dir(bundle),
                               disabled=not use_gt, placeholder="e.g. …/flai/data/flights/M346/PAR/parquet")

# --------------------------------------------------------------------------- sidebar: data

SPLIT_OF = {f: k for k, fl in (meta.get("splits") or {}).items() for f in fl}

st.sidebar.header("Flights")
source = st.sidebar.radio("Source", ["Upload files", "Data folder"], horizontal=True,
                          help="**Data folder** lists the flights of a folder on this machine (e.g. the training "
                               "data) and can show only the test flights of the model.")
uploads, gt_uploads, folder_files = [], {}, {}
if source == "Upload files":
    uploads = st.sidebar.file_uploader("Flight files (.csv or .parquet, one per flight; `<flight>_Y` files = labels)",
                                       type=["csv", "parquet"], accept_multiple_files=True) or []
    gt_uploads = {utils.flight_name_from_filename(u.name): u for u in uploads if utils.is_gt_filename(u.name)}
    uploads = [u for u in uploads if not utils.is_gt_filename(u.name)]
else:
    data_dir = st.sidebar.text_input("Data folder (`<flight>_X.parquet`)", value=utils.default_gt_dir(bundle))
    avail = utils.list_flight_files(data_dir)
    split_opts = [k for k in ("test", "val", "train") if SPLIT_OF] + ["all"]
    split = st.sidebar.selectbox(
        "Show", split_opts, format_func=lambda k: {"test": "test flights only (never seen in training)",
                                                   "val": "validation flights", "train": "training flights",
                                                   "all": "all flights"}[k],
        disabled=not avail)
    options = sorted(n for n in avail if split == "all" or SPLIT_OF.get(n) == split)
    if "folder_flights" in st.session_state:  # keep only what the current filter still offers
        st.session_state["folder_flights"] = [n for n in st.session_state["folder_flights"] if n in options]

    def _select_all(opts=options):
        st.session_state["folder_flights"] = list(opts)

    picked_flights = st.sidebar.multiselect(
        f"Flights ({len(options)} available)", options, key="folder_flights",
        format_func=lambda n: n if split != "all" or n not in SPLIT_OF else f"{n} ({SPLIT_OF[n]})")
    st.sidebar.button(f"Select all {len(options)}", on_click=_select_all, disabled=not options,
                      use_container_width=True)
    if data_dir and not avail:
        st.sidebar.warning(f"No flight files in `{data_dir}`.")
    folder_files = {n: avail[n] for n in picked_flights}

if not uploads and not folder_files:
    st.info("Upload one or more flight files from the sidebar, or pick them from a data folder (**Source**). "
            f"Each file must contain the {len(bundle.sensors)} sensor columns used by the model (an optional "
            "`time` column is used as the time axis).")
    with st.expander("Required sensors"):
        st.dataframe(pd.DataFrame({"sensor": bundle.sensors,
                                   "description": [meta["sensor_descriptions"].get(s, "") for s in bundle.sensors]}),
                     hide_index=True, use_container_width=True)
    st.stop()


def _add_flight(name: str, label: str, load, key: str):
    try:
        df = load()
        flights[name] = (utils.prepare_flight(df, name, bundle, cut=cut), key)
        if (col := utils.gt_column(df, bundle)) is not None:
            inline_gt[name] = df[col]
    except utils.InputError as ex:
        errors.append(str(ex))
    except Exception as ex:
        errors.append(f"{label}: cannot read file ({ex})")


flights, errors, inline_gt = {}, [], {}
for up in uploads:
    data = up.getvalue()
    h = hashlib.md5(data).hexdigest()
    name = utils.flight_name_from_filename(up.name)
    if name in flights:
        name = f"{name}_{h[:6]}"
    _add_flight(name, up.name, lambda d=data, n=up.name: parse_file(d, n), h)
if folder_files:
    cols = tuple(bundle.sensors) + tuple(c for c in [meta.get("gt", {}).get("column"), *config.GT_COLUMNS] if c)
    with st.spinner(f"Reading {len(folder_files)} flight(s)…"):
        for name, path in folder_files.items():
            mt = path.stat().st_mtime
            _add_flight(name, path.name, lambda p=str(path), m=mt: read_flight_path(p, m, cols), f"{path}:{mt}")

for e in errors:
    st.error(e)
if not flights:
    st.stop()

# --------------------------------------------------------------------------- run

params = utils.pp_params(bundle, use_durations, merge_gap_steps=int(merge_gap),
                         extra_min_steps=int(extra_min)) if do_pp else None
run_key = (model_name, cut, do_pp, use_durations, int(merge_gap), int(extra_min),
           tuple(sorted((n, h) for n, (_, h) in flights.items())))

results = st.session_state.get("results") if st.session_state.get("run_key") == run_key else None

c1, c2 = st.columns([1, 4])
run = c1.button("Run classification", type="primary", use_container_width=True)
c2.caption(f"{len(flights)} flight(s) loaded: " + ", ".join(
    f"{n} ({len(f.x) / bundle.hz / 60:.1f} min" + (f", {SPLIT_OF[n]} set" if n in SPLIT_OF else "") + ")"
    for n, (f, _) in flights.items()))

if run:
    results = {}
    bar = st.progress(0.0, text="Classifying…")
    for i, (name, (flight, _)) in enumerate(flights.items()):
        def cb(p, i=i, name=name):
            bar.progress((i + p) / len(flights), text=f"Classifying {name}…")
        try:
            results[name] = utils.classify_flight(bundle, flight, params, progress=cb)
        except Exception as ex:
            st.error(f"{name}: classification failed ({ex})")
    bar.empty()
    st.session_state["results"], st.session_state["run_key"] = results, run_key
elif st.session_state.get("results") and st.session_state.get("run_key") != run_key:
    st.warning("Inputs or settings changed: press **Run classification** to update the results.")

# --------------------------------------------------------------------------- ground truth

def load_gt(name: str, flight: utils.PreparedFlight):
    """(class ids per step or None, source description or error message)."""
    try:
        if name in inline_gt:
            series, src = inline_gt[name], "label column in the flight file"
        else:
            if name in gt_uploads:
                up = gt_uploads[name]
                gdf, src = parse_file(up.getvalue(), up.name), f"uploaded {up.name}"
            elif (path := utils.find_gt_file(name, gt_dir)) is not None:
                gdf, src = parse_path(str(path), path.stat().st_mtime), str(path)
            else:
                where = f"uploads or `{gt_dir}`" if gt_dir else "uploads (no ground-truth folder set)"
                return None, f"no `{name}_Y` file found in {where}"
            col = utils.gt_column(gdf, bundle)
            if col is None:
                return None, f"{src}: no label column ({', '.join(config.GT_COLUMNS)})"
            series = gdf[col]
        aligned = utils.align_gt(series, flight, bundle.hz)
        return utils.encode_gt(aligned, bundle), src
    except Exception as ex:
        return None, f"cannot use ground truth ({ex})"


gts, gt_msgs = {}, {}
if not use_gt:  # cheap check (no file read) so the user knows labels are there
    n_avail = sum(n in gt_uploads or n in inline_gt or utils.find_gt_file(n, gt_dir) is not None for n in flights)
    if n_avail:
        st.sidebar.info(f"Ground truth available for {n_avail}/{len(flights)} loaded flight(s): "
                        "enable **Compare with ground truth** to use it.")
if use_gt:
    for name, (flight, _) in flights.items():
        g, msg = load_gt(name, flight)
        gt_msgs[name] = msg
        if g is not None:
            gts[name] = g

# --------------------------------------------------------------------------- results

classes = bundle.classes
descr = meta.get("sensor_descriptions", {})
FIT = "fit manoeuvre"
WIDTHS = {"30 s": 30, "1 min": 60, "2 min": 120, "5 min": 300, "10 min": 600, "30 min": 1800, "whole flight": None}


def focus_on(fl: str, s: float, e: float, label: str):
    """Point the flight viewer at one manoeuvre: select its flight, fit the window on it (with margin)
    and highlight it. Used as a widget callback, so it may change other widgets' state."""
    T = float(flights[fl][0].time[-1])
    pad = max(10.0, 0.5 * (e - s))
    w = min(e - s + 2 * pad, T)
    st.session_state["view_flight"] = fl
    st.session_state["focus"] = dict(flight=fl, start=s, end=e, label=label, width=w)
    st.session_state["win"] = FIT
    st.session_state["pos"] = round(min(max(s - pad, 0.0), max(T - w, 0.0)), 1)


def focus_selected(table_key: str, segs: pd.DataFrame):
    rows = st.session_state[table_key]["selection"]["rows"]
    if rows and rows[0] < len(segs):
        r = segs.iloc[rows[0]]
        label = (f"#{r['#']} " if "#" in segs else "") + r["manoeuvre"]
        focus_on(r["flight"], float(r["start_s"]), float(r["end_s"]), label)


def clear_focus():
    f = st.session_state.pop("focus", None)
    if f and st.session_state.get("win") == FIT:  # keep roughly the same zoom with a preset width
        st.session_state["win"] = next((l for l, v in WIDTHS.items() if v and v >= f["width"]), "whole flight")


def show_manoeuvres(picks: pd.DataFrame, segs: pd.DataFrame):
    """Where the selected (flight, manoeuvre) pairs were detected: one whole-flight strip per flight
    (selected manoeuvres coloured, numbered) and one table of all occurrences. Clicking an occurrence
    opens it in the flight viewer (the only detailed signal chart of the page)."""
    occ = segs.merge(picks[["flight", "manoeuvre"]], on=["flight", "manoeuvre"])
    occ = occ.sort_values(["flight", "start_s"]).reset_index(drop=True)
    occ.insert(0, "#", np.arange(1, len(occ) + 1))
    by_man = occ.groupby("manoeuvre")["duration_s"].agg(["size", "sum"])
    st.markdown("#### " + " · ".join(f"{m}: {int(r['size'])}× ({r['sum']:.0f}s)" for m, r in by_man.iterrows()))
    for fl, o in occ.groupby("flight", sort=False):
        cls = {classes.index(m) for m in o["manoeuvre"].unique()}
        st.plotly_chart(utils.plot_occurrences(results[fl], flights[fl][0].time, classes, cls, gts.get(fl),
                                               numbers=dict(zip(o["start_step"], o["#"])), title=fl),
                        use_container_width=True, key=f"occfig_{fl}", config={"displayModeBar": False})
    cols = (["flight"] if occ["flight"].nunique() > 1 else []) + \
           (["manoeuvre"] if occ["manoeuvre"].nunique() > 1 else [])
    st.dataframe(occ[["#"] + cols + ["start_s", "end_s", "duration_s", "mean_confidence"]],
                 hide_index=True, use_container_width=True, selection_mode="single-row",
                 key="occ_table", on_select=partial(focus_selected, "occ_table", occ))
    st.caption("Click an occurrence to open it in the **Flight viewer** below: window fitted on it, "
               "manoeuvre highlighted with a dotted box.")


if results:
    all_segments = pd.concat([utils.segments_df(r, classes) for r in results.values()], ignore_index=True)
    st.subheader("Detected manoeuvres")
    m1, m2, m3 = st.columns(3)
    m1.metric("Flights", len(results))
    m2.metric("Manoeuvres", len(all_segments))
    m3.metric("Distinct types", all_segments["manoeuvre"].nunique() if len(all_segments) else 0)
    if len(all_segments):
        summary = all_segments.groupby(["flight", "manoeuvre"]).agg(
            count=("manoeuvre", "size"), total_s=("duration_s", "sum")).reset_index()
        picked = st.dataframe(summary, hide_index=True, use_container_width=True, on_select="rerun",
                              selection_mode="multi-row", key="summary_table")
        rows = [r for r in picked.selection.rows if r < len(summary)]
        if rows:
            if len({summary.loc[r, "manoeuvre"] for r in rows}) > 6:
                st.caption("Many manoeuvres selected: the strips get hard to read, a few at a time works best.")
            with st.container(border=True):
                show_manoeuvres(summary.iloc[rows], all_segments)
        else:
            st.caption("Select one or more rows (checkbox on the left) to see where those manoeuvres were "
                       "detected.")
    d1, d2 = st.columns(2)
    d1.download_button("Download segments (CSV)", all_segments.to_csv(index=False).encode(),
                       "manoeuvre_segments.csv", "text/csv", use_container_width=True)
    steps = pd.concat([r.steps_df(classes, gts.get(n)) for n, r in results.items()], ignore_index=True)
    d2.download_button("Download per-step predictions (CSV)", steps.to_csv(index=False).encode(),
                       "predictions_per_step.csv", "text/csv", use_container_width=True)

st.subheader("Flight viewer")
names = list(flights)
if st.session_state.get("view_flight") not in names:
    st.session_state["view_flight"] = names[0]
name = st.selectbox("Flight", names, key="view_flight",
                    format_func=lambda n: n + (f" ({SPLIT_OF[n]} set)" if n in SPLIT_OF else ""))
flight = flights[name][0]
for w in flight.warnings:
    st.warning(w)
s1, s2 = st.columns([4, 1])
chosen = s1.multiselect("Signals (default: all sensors used for training)", bundle.sensors,
                        default=bundle.sensors, key="sig", format_func=lambda s: f"{s} — {descr.get(s, '')}")
norm = s2.toggle("Normalize signals (z-score)", value=True, key="norm",
                 help="Sensors have different units: normalizing makes them comparable on one axis. "
                      "Hover always shows the physical value.")
T = float(flight.time[-1])
cs, ce = int(meta.get("cut_start", 0)), int(meta.get("cut_end", 0))
cut_note = (f" First {cs} and last {ce} steps ({cs / bundle.hz:.0f}s + {ce / bundle.hz:.0f}s) are cut "
            "as transients; t=0 is the first step after the cut."
            if cut and (cs or ce) and not any("cut skipped" in w for w in flight.warnings) else "")

# ---- highlighted manoeuvre (set by clicking a table row), only for the flight it belongs to
focus = st.session_state.get("focus")
focus = focus if focus and focus["flight"] == name else None
if focus:
    f1, f2 = st.columns([5, 1])
    f1.markdown(f"Highlighted: **{focus['label']}** — {focus['start']:.1f}s → {focus['end']:.1f}s "
                f"({focus['end'] - focus['start']:.1f}s), dotted box in the chart.")
    f2.button("Clear highlight", on_click=clear_focus, use_container_width=True)

# ---- window on the flight: full resolution inside, navigator below
widths = ({FIT: focus["width"]} if focus else {}) | WIDTHS
if st.session_state.get("win") not in widths:
    st.session_state["win"] = "2 min"
wlabel = st.radio("Window", list(widths), horizontal=True, key="win")
w = min(widths[wlabel] or T, T)
max_start = max(T - w, 0.0)
if "pos" not in st.session_state:
    st.session_state["pos"] = max_start / 2          # start centred on the flight
st.session_state["pos"] = min(max(float(st.session_state["pos"]), 0.0), max_start)
t0 = st.session_state["pos"]
t_range = (t0, t0 + w)

res = results.get(name) if results else None
n_win = int(np.searchsorted(flight.time, t_range[1], "right") - np.searchsorted(flight.time, t_range[0]))
st.caption(f"Showing {t_range[0]:.0f}s → {t_range[1]:.0f}s of {T:.0f}s ({T / 60:.1f} min): "
           f"{n_win} samples per signal, every sample drawn (no decimation).{cut_note}")
if n_win * max(len(chosen), 1) > 2_000_000:
    st.info("Many points in this window: the chart may be slow. Use a shorter window for smoother interaction.")
fig = utils.plot_flight(res, flight.raw, flight.time, chosen, descr, classes,
                        t_range=t_range, normalize=norm, gt=gts.get(name))
if focus and focus["end"] >= t_range[0] and focus["start"] <= t_range[1]:
    fig.add_vrect(x0=focus["start"], x1=focus["end"], line=dict(color="black", width=1.5, dash="dot"),
                  row=1, col=1)
st.plotly_chart(fig, use_container_width=True, key="fig")

if max_start > 0:
    def shift(d=0.0, m=max_start):
        st.session_state["pos"] = min(max(st.session_state["pos"] + d, 0.0), m)
    b1, b2, b3 = st.columns([1, 10, 1])
    b1.button("◀", key="prev", on_click=shift, kwargs=dict(d=-w), use_container_width=True, help="Previous window")
    b2.slider("Window start [s]", 0.0, max_start, step=max(round(1 / bundle.hz, 3), 0.1),
              key="pos", label_visibility="collapsed")
    b3.button("▶", key="next", on_click=shift, kwargs=dict(d=w), use_container_width=True, help="Next window")
st.plotly_chart(utils.plot_overview(res, flight.raw, flight.time, chosen, classes if res else None, t_range),
                use_container_width=True, key="ov", config={"displayModeBar": False})
st.caption("Overview of the whole flight (red box = window above). Simplified for display only.")
if res is not None:
    segs = utils.segments_df(res, classes)
    st.dataframe(segs.drop(columns=["flight", "class_id", "start_step", "end_step"]), hide_index=True,
                 use_container_width=True, selection_mode="single-row", key=f"segs_{name}",
                 on_select=partial(focus_selected, f"segs_{name}", segs))
    st.caption(f"All manoeuvres of {name}: click one to fit the window on it and highlight it.")

if use_gt:
    st.subheader("Comparison with ground truth")
    missing = {n: m for n, m in gt_msgs.items() if n not in gts}
    for n, m in missing.items():
        st.warning(f"{n}: {m}")
    compared = [n for n in gts if results and n in results]
    if not gts:
        st.info("No ground truth found for the loaded flights.")
    elif not compared:
        st.info(f"Ground truth found for {len(gts)} flight(s): press **Run classification** to compare.")
    else:
        k = len(classes)
        rows, cm_all, cm_raw_all = [], np.zeros((k, k), int), np.zeros((k, k), int)
        for n in compared:
            g, r = gts[n], results[n]
            cm, cm_raw = utils.confusion_matrix(g, r.labels, k), utils.confusion_matrix(g, r.labels_raw, k)
            cm_all, cm_raw_all = cm_all + cm, cm_raw_all + cm_raw
            sm, sm_raw = utils.scores_from_confusion(cm, classes)[0], utils.scores_from_confusion(cm_raw, classes)[0]
            rows.append({"flight": n, "split": SPLIT_OF.get(n, "unseen"), "accuracy": sm["accuracy"], "F1 weighted": sm["f1_weighted"],
                         "F1 macro": sm["f1_macro"], "F1 weighted (no post-proc.)": sm_raw["f1_weighted"],
                         "labelled steps": sm["steps"], "unlabelled steps": int((g < 0).sum()),
                         "source": gt_msgs[n]})
        summary, per_class = utils.scores_from_confusion(cm_all, classes)
        summary_raw = utils.scores_from_confusion(cm_raw_all, classes)[0]
        g1, g2, g3, g4 = st.columns(4)
        g1.metric("Flights compared", len(compared))
        g2.metric("Accuracy", f"{summary['accuracy']:.3f}")
        g3.metric("F1 weighted", f"{summary['f1_weighted']:.3f}",
                  delta=f"{summary['f1_weighted'] - summary_raw['f1_weighted']:+.3f} vs no post-proc.",
                  delta_color="normal" if do_pp else "off")
        g4.metric("F1 macro", f"{summary['f1_macro']:.3f}")
        st.caption("Step-level scores on all compared flights pooled, same transient cut as the prediction. "
                   + ("Prediction = post-processed labels." if do_pp else "Post-processing is off."))
        st.dataframe(pd.DataFrame(rows).round(3), hide_index=True, use_container_width=True)
        seen = [n for n in compared if SPLIT_OF.get(n) in ("train", "val")]
        if seen:
            st.warning(f"{', '.join(seen)}: used in training/validation, scores are optimistic. Use test "
                       "flights (see the model card) for an honest estimate.")
        with st.expander("Per-class scores and confusion matrix"):
            st.dataframe(per_class, hide_index=True, use_container_width=True)
            present = [i for i in range(k) if cm_all[i].sum() or cm_all[:, i].sum()]
            cmn = cm_all[np.ix_(present, present)]
            cmn = cmn / np.maximum(cmn.sum(1, keepdims=True), 1)
            st.plotly_chart(utils.plot_confusion(cmn, [classes[i] for i in present]), use_container_width=True)
