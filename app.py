"""Streamlit UI for flight manoeuvre recognition.  Run:  streamlit run app.py"""
import hashlib

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

# --------------------------------------------------------------------------- sidebar: data

st.sidebar.header("Flights")
uploads = st.sidebar.file_uploader("Flight files (.csv or .parquet, one per flight)", type=["csv", "parquet"],
                                   accept_multiple_files=True)
if not uploads:
    st.info("Upload one or more flight files from the sidebar. Each file must contain the "
            f"{len(bundle.sensors)} sensor columns used by the model (an optional `time` column is used "
            "as the time axis).")
    with st.expander("Required sensors"):
        st.dataframe(pd.DataFrame({"sensor": bundle.sensors,
                                   "description": [meta["sensor_descriptions"].get(s, "") for s in bundle.sensors]}),
                     hide_index=True, use_container_width=True)
    st.stop()

flights, errors = {}, []
for up in uploads:
    data = up.getvalue()
    name = utils.flight_name_from_filename(up.name)
    if name in flights:
        name = f"{name}_{hashlib.md5(data).hexdigest()[:6]}"
    try:
        df = parse_file(data, up.name)
        flights[name] = (utils.prepare_flight(df, name, bundle, cut=cut), hashlib.md5(data).hexdigest())
    except utils.InputError as ex:
        errors.append(str(ex))
    except Exception as ex:
        errors.append(f"{up.name}: cannot read file ({ex})")

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
    f"{n} ({len(f.x) / bundle.hz / 60:.1f} min)" for n, (f, _) in flights.items()))

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

# --------------------------------------------------------------------------- results

classes = bundle.classes
descr = meta.get("sensor_descriptions", {})

if results:
    all_segments = pd.concat([utils.segments_df(r, classes) for r in results.values()], ignore_index=True)
    st.subheader("Detected manoeuvres")
    m1, m2, m3 = st.columns(3)
    m1.metric("Flights", len(results))
    m2.metric("Manoeuvres", len(all_segments))
    m3.metric("Distinct types", all_segments["manoeuvre"].nunique() if len(all_segments) else 0)
    if len(all_segments):
        st.dataframe(all_segments.groupby(["flight", "manoeuvre"]).agg(
            count=("manoeuvre", "size"), total_s=("duration_s", "sum")).reset_index(),
            hide_index=True, use_container_width=True)
    d1, d2 = st.columns(2)
    d1.download_button("Download segments (CSV)", all_segments.to_csv(index=False).encode(),
                       "manoeuvre_segments.csv", "text/csv", use_container_width=True)
    steps = pd.concat([r.steps_df(classes) for r in results.values()], ignore_index=True)
    d2.download_button("Download per-step predictions (CSV)", steps.to_csv(index=False).encode(),
                       "predictions_per_step.csv", "text/csv", use_container_width=True)

st.subheader("Flights")
tabs = st.tabs(list(flights))
for tab, (name, (flight, _)) in zip(tabs, flights.items()):
    with tab:
        for w in flight.warnings:
            st.warning(w)
        default = bundle.sensors[:config.DEFAULT_PLOTTED_SENSORS]
        chosen = st.multiselect("Signals", bundle.sensors, default=default, key=f"sig_{name}",
                                format_func=lambda s: f"{s} — {descr.get(s, '')}")
        res = results.get(name) if results else None
        fig = utils.plot_flight(res, flight.raw, flight.time, chosen, descr, classes if res else None)
        st.plotly_chart(fig, use_container_width=True, key=f"fig_{name}")
        if res is not None:
            st.dataframe(utils.segments_df(res, classes), hide_index=True, use_container_width=True)
