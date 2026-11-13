# FLAI app — manoeuvre recognition

Streamlit app that classifies flight manoeuvres from sensor time series using a
model trained in the `flai` training repo. It runs **on its own**: it does not
import the training code, pytorch-lightning or wandb.

```
app.py              Streamlit UI
config.py           paths and defaults (absolute paths, env overrides)
utils.py            pipeline: loading, preprocessing, windowed inference, segments, plots
model.py            standalone UCNN (same parameter names as the Lightning model)
postprocessing.py   segment smoothing (unify / merge / filter-fill)
weights/classification/<name>.pt + <name>.json   exported model bundles
scripts/export_checkpoint.py                     .ckpt (training repo) -> bundle
tests/
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Model bundles

The app loads `weights/classification/<name>.pt` (plain `state_dict`) and
`<name>.json` (sensors and their order, training mean/std, classes, window,
output resolution, hyper-parameters, sampling rate, transient cut, per-class
min durations, provenance). Weights are not committed (see `.gitignore`).

Export a new checkpoint **once**, with an environment where the training repo is importable
(the original `.ckpt` pickles classes from `src.*`):

```bash
python scripts/export_checkpoint.py --flai-repo ../flai \
    --ckpt ../flai/app/weights/classification/ucnn.ckpt --name ucnn
```

The script refuses to export if the sensor order or number of classes is
inconsistent, and checks that the standalone model gives the same output as the
training model.

Current bundle `ucnn`: sweep `wandering-sound-6` (2025-06-07, epoch 10,
validation_f1_par 0.77). On the 38 test flights the app pipeline gives
weighted F1 0.876 (post-processing on, per-class durations off), matching the
training-repo evaluation.

## Input files

One file per flight, `.parquet` or `.csv`. It must contain all sensor columns
listed in the model card (extra columns are ignored, order does not matter). A
`time` column / index is used as time axis, otherwise `step / sampling_hz`.
The flight name is the file name before the first `_` (`FAN1205_X.parquet` -> `FAN1205`).

Instead of uploading, **Source → Data folder** lists the flights of a local folder (default: the
training data next to this app, or `FLAI_GT_DIR`) and reads only the model's sensor columns.
The **Show** filter uses the train/val/test split stored in the bundle: by default only the test
flights (never seen in training) are offered; **Select all** picks every flight shown.

## Ground truth (optional)

Off by default. Enable **Compare with ground truth** in the sidebar; for each flight the app
looks for labels in this order: a `MANOEUVERS` column in the flight file itself, an uploaded
`<flight>_Y.parquet` (upload it together with `<flight>_X.parquet`), then
`<folder>/<flight>_Y.parquet` in the folder set in the sidebar (default: `FLAI_GT_DIR`, e.g.
`../flai/data/flights/M346/PAR/parquet`).
Labels are decoded as in training (ID -> name via `manoeuvers.csv`, name -> class via the
`chosen_man_*.json` aliases, everything else -> OTHER MAN), using the `gt` section of the
bundle JSON, and get the same transient cut as the signals.
The app shows step-level accuracy / F1 (pooled and per flight, also without post-processing),
per-class scores, the confusion matrix, a ground-truth timeline under the prediction with
mismatches in red, and adds `gt_class_*` columns to the per-step CSV.

## Outputs

* table of detected manoeuvres (start, end, duration, mean confidence) per flight;
* select one or more rows of the summary table to see where those manoeuvres were detected (one strip per flight,
  with numbered occurrences, ground truth if enabled, zoom on each occurrence); click a row of a
  flight's segment table to move that flight's window onto it;
* interactive plot: signals in physical units + manoeuvre timeline + confidence;
* CSV downloads: segments and per-step predictions.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

Environment variables: `FLAI_MODEL_DIR`, `FLAI_GT_DIR`, `FLAI_DEFAULT_MODEL`, `FLAI_DEVICE` (`auto|cpu|cuda|mps`), `FLAI_BATCH_SIZE`.
