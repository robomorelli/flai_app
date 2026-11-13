"""
Export a Lightning checkpoint from the flai training repo into a self-contained
bundle usable by this app (no dependency on the training code at runtime).

Must be run ONCE with an environment that can import the flai training repo,
because the original .ckpt pickles enums/classes from ``src.*``:

    python scripts/export_checkpoint.py \
        --flai-repo /path/to/flai \
        --ckpt /path/to/flai/app/weights/classification/ucnn.ckpt \
        --name ucnn

Output (in weights/classification/):
    <name>.pt    -> plain state_dict (loadable with torch.load(weights_only=True))
    <name>.json  -> metadata: sensors (ordered), normalization stats, classes,
                    window/output steps, hyper-parameters, sampling rate, ground-truth
                    label decoding (ID -> name -> class), ...
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

APP_ROOT = Path(__file__).resolve().parents[1]


def _read_json(path):
    with open(path) as f:
        return json.load(f)


def _to_plain(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.integer, np.floating)):
        return v.item()
    if hasattr(v, "value") and hasattr(v, "name"):  # Enum
        return v.name
    return v


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--flai-repo", required=True, type=Path)
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--name", default="ucnn")
    p.add_argument("--sampling-hz", type=float, default=32.0)
    p.add_argument("--cut-start", type=int, default=None, help="default: training CUT_FLIGHT_TIME_START")
    p.add_argument("--cut-end", type=int, default=None, help="default: training CUT_FLIGHT_TIME_END")
    p.add_argument("--out-dir", type=Path, default=APP_ROOT / "weights" / "classification")
    args = p.parse_args()

    flai = args.flai_repo.resolve()
    sys.path.insert(0, str(flai))
    os.chdir(flai)  # relative data paths stored in the checkpoint are relative to the repo root
    import src.constants as cst  # noqa: E402  (needed to unpickle the checkpoint)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    ext, hps_raw = ckpt["external"], ckpt["hps"]
    hps = {k: _to_plain(v["val"]) for k, v in hps_raw.items()}

    mu, std = ext["MU_PAR"], ext["STD_PAR"]
    sensors = list(mu.index)
    assert list(std.index) == sensors

    # sanity check: the sensor order used at training time == JSON_PAR_PATH order
    par_json = _read_json(ext["JSON_PAR_PATH"])
    if list(par_json.keys()) != sensors:
        raise RuntimeError("Sensor order in MU_PAR differs from JSON_PAR_PATH: refusing to export")

    man_json = _read_json(ext["JSON_MAN_PATH"])
    classes = [cst.NO_MAN_LABEL, cst.OTHER_MAN_LABEL] + list(man_json.keys())

    n_out = ckpt["state_dict"]["segment_classifier.conv.weight"].shape[0]
    if n_out != len(classes):
        raise RuntimeError(f"Classifier has {n_out} outputs but {len(classes)} classes were derived")

    stats_path = Path(ext["JSON_MAN_PATH"]).parent / "man_stats.json"
    man_stats = _read_json(stats_path) if stats_path.exists() else {}
    # [min, max, mean, count] durations in steps, keyed by class name
    min_durations = {k: v[0] for k, v in man_stats.items() if k in classes}

    # ground-truth decoding as in training (dataloader_utils.load_man): ID -> name via
    # manoeuvers.csv, then alias -> class via JSON_MAN_PATH, anything else -> OTHER MAN
    import csv  # noqa: E402

    map_csv = Path(ext.get("TARGET_MAPPING_CSV_PATH") or Path(ext["JSON_MAN_PATH"]).parent / cst.TARGET_MAPPING)
    id_to_name = {}
    if map_csv.exists():
        with open(map_csv) as f:
            id_to_name = {str(int(r["ID"])): r["NAME"] for r in csv.DictReader(f)}
    else:
        print(f"WARNING: {map_csv} not found: numeric ground-truth IDs cannot be decoded by the app")
    gt = {"column": ext.get("TARGET_COL", cst.MAN_COL_IDS), "id_to_name": id_to_name,
          "alias": {a: cls for cls, aliases in man_json.items() for a in aliases}}

    # flights used for train / val / test (shown in the app's model card)
    splits = {}
    if ext.get("JSON_FLIGHT_PATH") and Path(ext["JSON_FLIGHT_PATH"]).exists():
        fj = _read_json(ext["JSON_FLIGHT_PATH"])
        splits = {k: list(fj.get(k, [])) for k in ("train", "val", "test")}
    else:
        print("WARNING: JSON_FLIGHT_PATH not found: the bundle will not list train/val/test flights")

    best_score = None
    for k, v in ckpt.get("callbacks", {}).items():
        if "ModelCheckpoint" in k:
            best_score = float(v.get("best_model_score")) if v.get("best_model_score") is not None else None
            source = v.get("best_model_path")

    meta = {
        "name": args.name,
        "model": ext["MODEL"].name,
        "aircraft": _to_plain(ext["AIRCRAFT"]),
        "project": ext.get("PROJECT_NAME"),
        "trained_at": ext.get("NOW"),
        "source_checkpoint": source if best_score is not None else str(args.ckpt),
        "epoch": ckpt.get("epoch"),
        "validation_metric": ext.get("METRIC_CHECKPOINT_MODEL"),
        "validation_score": best_score,
        "sampling_hz": args.sampling_hz,
        "cut_start": args.cut_start if args.cut_start is not None else cst.CUT_FLIGHT_TIME_START,
        "cut_end": args.cut_end if args.cut_end is not None else cst.CUT_FLIGHT_TIME_END,
        "sensors": sensors,
        "sensor_descriptions": {s: par_json[s] for s in sensors},
        "mean": [float(mu[s]) for s in sensors],
        "std": [float(std[s]) for s in sensors],
        "classes": classes,
        "man_min_duration_steps": min_durations,
        "hps": hps,
        "gt": gt,
        "splits": splits,
        "splits_source": Path(ext["JSON_FLIGHT_PATH"]).name if ext.get("JSON_FLIGHT_PATH") else None,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt["state_dict"], args.out_dir / f"{args.name}.pt")
    with open(args.out_dir / f"{args.name}.json", "w") as f:
        json.dump(meta, f, indent=2)

    # verify: standalone model == original Lightning model on a random window
    sys.path.insert(0, str(APP_ROOT))
    from model import build_model  # noqa: E402

    standalone = build_model(meta)
    standalone.load_state_dict(ckpt["state_dict"], strict=True)
    standalone.eval()

    from src.config import Configuration  # noqa: E402

    cf = Configuration()
    cf.init_from_yaml(ext, hps_raw)
    original = cf.MODEL.value.load_from_checkpoint(str(args.ckpt), conf=cf, hps=cf.HPS, map_location="cpu")
    original.eval()

    x = torch.randn(2, hps["WINDOW_SIZE"], len(sensors))
    with torch.no_grad():
        a = standalone(x)
        b = original(x)[0]
    max_diff = (a - b).abs().max().item()
    print(f"output shape {tuple(a.shape)}  max |diff| vs training model = {max_diff:.2e}")
    if max_diff > 1e-4:
        raise RuntimeError("Standalone model does not match the training model")
    print(f"exported -> {args.out_dir / args.name}.pt / .json")


if __name__ == "__main__":
    main()
