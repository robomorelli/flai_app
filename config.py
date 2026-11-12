"""App configuration. All paths are absolute and independent of the working directory."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Exported model bundles (<name>.pt + <name>.json), see scripts/export_checkpoint.py
CLASSIFICATION_MODEL_DIR = Path(os.environ.get("FLAI_MODEL_DIR", ROOT / "weights" / "classification"))
DEFAULT_MODEL = os.environ.get("FLAI_DEFAULT_MODEL", "ucnn")

TASKS = ["Manoeuvre classification"]

# Inference
BATCH_SIZE = int(os.environ.get("FLAI_BATCH_SIZE", 8))
DEVICE = os.environ.get("FLAI_DEVICE", "auto")  # auto | cpu | cuda | mps

# Columns that may carry the time axis in uploaded files
TIME_COLUMNS = ("time", "Time", "TIME", "t")

# Plotting
MAX_PLOT_POINTS = 6000          # per trace, after min/max decimation
DEFAULT_PLOTTED_SENSORS = 4

# Post-processing defaults (steps at the model sampling rate)
PP_MERGE_GAP_STEPS = 100
PP_MIN_PERC = 0.05
PP_FILL_PERC = 0.4
PP_EXTRA_MIN_STEPS = 100
