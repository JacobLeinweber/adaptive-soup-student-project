"""Hyperparameter search (pre-experiment) on the six cross-dataset transfers, following AdaTime's sweep; seed 52.

Every run continues training the supplied seed-52 source-only checkpoint of its transfer with one method
(Deep CORAL, DANN, DSAN) and one combination of learning rate, source weight and alignment weight from
configs/hp_search_cross_dataset.json (written by make_hp_search_config.py). Everything else is Stage 2:
30 epochs, batch size 32, weight decay 1e-4, the model after the last epoch, the same data pipeline.

The runs go combination by combination (combination k for all 6 transfers x 3 methods, then k + 1), so a
search stopped at any time leaves complete combinations behind.

Start (finished runs are skipped, so the same command also continues an interrupted search):
    python run_hp_search.py --device cuda
Tests:
    python run_hp_search.py --device mps --check-only
    python run_hp_search.py --device mps --combinations 0-1 --epochs 1 --output-dir /tmp/hp_test

Before the first run the script checks the Python packages, the device, the git state of the code and the
configuration, every input file (size and SHA-256), the data and checkpoints themselves and the free disk
space, and stops with a list of all problems. A failing run is logged and skipped (and retried at the next
start); the search stops by itself only if the device breaks, the code changes, the disk fills up or
18 runs fail in a row.

Per run, in <output-dir>/runs/c<combination>/<method>/<transfer>/:
    model_last.pt, discriminator_last.pt (DANN)   checkpoints, not in git (.gitignore)
    train_log.csv                                  losses per epoch
    outputs.npz                                    predictions of five splits, source-val CE per window, DEV weights
    report_only.json                               target-train and target-test scores, for the report only
    result.json                                    written last: setting, provenance and criteria (SRC, DEV, FST, oracle)
In <output-dir>, rewritten after every run: criteria.csv (one row per finished run, no target train or test),
report_only_target.csv (target train and test), progress.json; appended: search_log.txt, errors.log.

Target labels: training gets target windows only (uda.data.load_unlabeled). Target labels are read through
uda.evaluation.true_labels and used after training only: FST (5 fixed windows per class of target val), the
oracle on target val, and the report-only scores of target train and target test.
"""
from __future__ import annotations

import sys

if sys.version_info < (3, 11):
    sys.exit(f"START CHECK FAILED: Python 3.11 or newer is needed (this is Python {sys.version.split()[0]}).")

import argparse
import csv
import io
import json
import math
import os
import platform
import shutil
import subprocess
import time
import traceback
import warnings
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    import numpy as np
    import sklearn
    import torch
except ImportError as missing:
    sys.exit(f"START CHECK FAILED: a Python package is missing ({missing}). "
             "Install the project requirements: python -m pip install -r requirements.txt")

from src.training.experiments import Transfer
from src.training.model import SelfHARCNN
from src.training.train_transfer import ROOT
from src.training.utils import json_default
from uda.criteria import features_and_logits, per_window_ce
from uda.data import check_output_dir, checkpoint_dir, get_transfer, load_label_map, load_labeled, load_source_only_model, load_unlabeled, sha256
from uda.evaluation import true_labels
from uda.hp_search import (ALIGNMENT_LOSS, OUTPUT_SPLITS, Combination, combinations, config_digest, config_problems,
                           dev_with_raw, load_config, split_scores, train_run)
from uda.runs import clean_commit

DEFAULT_CONFIG = ROOT / "configs" / "hp_search_cross_dataset.json"
DEFAULT_OUTPUT = ROOT / "results" / "hp_search" / "cross_dataset_seed52"
RUN_BYTES = 1.2e6         # disk space of one run (checkpoint 0.86 MB, discriminator, outputs, json), with a margin
DISK_MARGIN = 0.3e9       # kept free on top at the start
MIN_FREE_BYTES = 0.2e9    # the search stops if less than this is free
MAX_CONSECUTIVE_FAILURES = 18
LFS_POINTER = b"version https://git-lfs.github.com/spec/v1"
EXIT_START_CHECK, EXIT_DEVICE, EXIT_FAILURES, EXIT_CODE_CHANGED, EXIT_DISK, EXIT_INTERRUPTED = 2, 3, 4, 5, 6, 130
# Epoch-log entry of the alignment loss of each method.
ALIGNMENT_LOG_KEY = {"deep_coral": "coral", "dann": "domain_loss", "dsan": "lmmd"}
TARGET_LABELS_USED = ("fst_*: 5 labelled target-val windows per class (few-shot); oracle_target_val_*: all of target "
                      "val (reference only); target train and target test only in report_only.json")


# ---------------------------------------------------------------- console, log and files

def configure_console() -> None:
    """Never fail on a character the console cannot show (for example on a Windows code page)."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace", line_buffering=True)
        except (AttributeError, ValueError):
            pass


class Log:
    """Time-stamped messages to the console and to search_log.txt; epoch lines go to the file only."""

    def __init__(self) -> None:
        self.handle = None

    def open(self, path: Path) -> None:
        self.handle = path.open("a", encoding="utf-8")

    def _file(self, line: str) -> None:
        if self.handle is not None:
            try:
                self.handle.write(line + "\n")
                self.handle.flush()
            except OSError:
                pass

    def __call__(self, message: str) -> None:
        line = f"{datetime.now():%Y-%m-%d %H:%M:%S}  {message}"
        try:
            print(line, flush=True)
        except OSError:
            pass
        self._file(line)

    def detail(self, message: str) -> None:
        self._file(f"{datetime.now():%Y-%m-%d %H:%M:%S}    {message}")


def write_atomic(path: Path, data: bytes) -> None:
    """Write next to `path` and rename, so `path` is either complete or untouched.

    The rename is retried for a few seconds: on Windows a virus scanner or an open viewer can lock a file briefly.
    """
    tmp = path.with_name(f"{path.stem}.tmp{path.suffix}")
    with tmp.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    for attempt in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.5)


def json_bytes(payload) -> bytes:
    return (json.dumps(payload, indent=1, default=json_default) + "\n").encode("utf-8")


def torch_bytes(state: dict) -> bytes:
    buffer = io.BytesIO()
    torch.save({key: value.detach().cpu() for key, value in state.items()}, buffer)
    return buffer.getvalue()


def npz_bytes(**arrays) -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


def csv_bytes(rows: list[dict]) -> bytes:
    columns = list(dict.fromkeys(key for row in rows for key in row))
    text = io.StringIO()
    writer = csv.DictWriter(text, fieldnames=columns, restval="", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return text.getvalue().encode("utf-8")


# ---------------------------------------------------------------- plan

@dataclass(frozen=True)
class Run:
    combination: Combination
    method: str
    transfer: Transfer

    @property
    def key(self) -> tuple:
        return self.combination.id, self.method, self.transfer.name

    @property
    def label(self) -> str:
        return f"c{self.combination.id:03d} {self.method:<10} {self.transfer.name}"

    def folder(self, out: Path) -> Path:
        return out / "runs" / f"c{self.combination.id:03d}" / self.method / self.transfer.name


def parse_ids(values: list[str]) -> list[int]:
    """Combination ids from '0 1 5' or ranges like '0-9'."""
    ids = []
    for value in values:
        first, _, last = value.partition("-")
        ids.extend(range(int(first), int(last or first) + 1))
    return sorted(set(ids))


def build_plan(config: dict, ids: list[int] | None, methods: list[str], transfers: list[Transfer]) -> list[Run]:
    """Combination by combination; inside a combination transfer by transfer, method by method."""
    by_method = {method: {c.id: c for c in combinations(config, method)} for method in methods}
    available = set.intersection(*(set(table) for table in by_method.values()))
    chosen = sorted(available) if ids is None else ids
    unknown = [i for i in chosen if i not in available]
    if unknown:
        raise SystemExit(f"Unknown combination ids: {unknown} (available: 0-{max(available)})")
    return [Run(by_method[method][i], method, transfer) for i in chosen for transfer in transfers for method in methods]


# ---------------------------------------------------------------- start checks

def check_versions() -> list[str]:
    problems = []
    if int(np.__version__.split(".")[0]) < 2:
        problems.append(f"numpy 2 or newer is needed to read the prepared data (found {np.__version__}): "
                        "python -m pip install -r requirements.txt")
    if int(torch.__version__.split(".")[0]) < 2:
        problems.append(f"PyTorch 2 or newer is needed (found {torch.__version__}).")
    return problems


def device_info(device: torch.device) -> dict:
    info = {"device": device.type, "torch": torch.__version__, "cuda": torch.version.cuda,
            "python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__,
            "platform": platform.platform()}
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(torch.cuda.current_device())
        info.update({"device_name": properties.name, "device_memory_gb": round(properties.total_memory / 1e9, 1),
                     "cudnn": torch.backends.cudnn.version(), "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                     "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                     "cudnn_deterministic": torch.backends.cudnn.deterministic,
                     "cudnn_benchmark": torch.backends.cudnn.benchmark})
    elif device.type == "mps":
        info["device_name"] = f"Apple MPS ({platform.machine()})"
    else:
        info["device_name"] = f"CPU ({platform.processor() or platform.machine()})"
    return info


def check_device(name: str) -> tuple[list[str], dict]:
    if name == "cuda" and not torch.cuda.is_available():
        return [f"--device cuda, but CUDA is not available (PyTorch {torch.__version__}, built for CUDA "
                f"{torch.version.cuda}). Use a CUDA build of PyTorch or check the GPU driver."], {}
    if name == "mps" and not torch.backends.mps.is_available():
        return ["--device mps, but MPS is not available on this machine."], {}
    device = torch.device(name)
    try:
        model = SelfHARCNN((150, 18), 5).to(device).train()
        model(torch.zeros(4, 150, 18, device=device)).sum().backward()
        if device.type == "cuda":
            torch.cuda.synchronize()
    except Exception as error:
        return [f"--device {name}: a test forward and backward pass failed: {error!r}"], {}
    return [], device_info(device)


def device_healthy(device: torch.device) -> bool:
    try:
        value = (torch.ones(8, device=device) * 2).sum().item()
        if device.type == "cuda":
            torch.cuda.synchronize()
        return value == 16
    except Exception:
        return False


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.STDOUT)


def check_code(config_path: Path) -> tuple[list[str], str | None]:
    """Code (uda/ and every .py file) and configuration must equal the last commit; returns that commit."""
    try:
        commit = clean_commit()
    except FileNotFoundError:
        return ["git was not found. Run the search in a git checkout of the repository."], None
    except subprocess.CalledProcessError as error:
        return [f"git failed ({error}). Run the search in a git checkout of the repository."], None
    except RuntimeError as error:
        return [f"{str(error).strip()}\n    -> commit, stash or move these files: the search runs on committed code only."], None
    try:
        relative = config_path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return [f"The configuration must be a file of this repository: {config_path}"], commit
    try:
        git("ls-files", "--error-unmatch", "--", relative)
        status = git("status", "--porcelain", "--", relative)
    except subprocess.CalledProcessError:
        return [f"The configuration is not committed: {relative}"], commit
    if status.strip():
        return [f"The configuration differs from the last commit: {relative}"], commit
    return [], commit


def check_input_files(config: dict) -> list[str]:
    problems = []
    for entry in config["inputs"]:
        path = ROOT / entry["path"]
        if not path.is_file():
            problems.append(f"missing: {entry['path']}")
            continue
        with path.open("rb") as handle:
            head = handle.read(len(LFS_POINTER))
        if head == LFS_POINTER:
            problems.append(f"Git LFS pointer instead of the real file: {entry['path']}")
        elif entry["stored"] == "git-lfs":
            size = path.stat().st_size
            if size != entry["bytes"]:
                problems.append(f"unexpected size {size} bytes (expected {entry['bytes']}): {entry['path']}")
            elif sha256(path) != entry["sha256"]:
                problems.append(f"content differs from the expected version (SHA-256): {entry['path']}")
    if problems:
        problems.append(f"-> unzip {config['inputs_zip']} into the repository root ({ROOT}), replacing existing files")
    return problems


class Inputs:
    """Everything the runs read, loaded once: labelled source data, target windows without labels, and,
    kept apart and used for scoring only, the true labels of every transfer."""

    def __init__(self) -> None:
        self.source: dict[str, tuple[dict, dict]] = {}
        self.source_label_maps: dict[str, dict] = {}
        self.target_windows: dict[str, dict] = {}
        self.scoring_labels: dict[str, dict] = {}
        self.label_maps: dict[str, dict] = {}
        self.checkpoint_sha256: dict[str, str] = {}


def load_inputs(config: dict, transfers: list[Transfer]) -> tuple[Inputs, list[str]]:
    data, problems = Inputs(), []
    family = config["family"]
    for transfer in transfers:
        try:
            label_map = load_label_map(transfer)
            data.label_maps[transfer.name] = label_map
            if transfer.source not in data.source:
                x, y, meta = load_labeled(family, transfer.source)
                data.source[transfer.source] = (x, y)
                data.source_label_maps[transfer.source] = meta["label_map"]
            if data.source_label_maps[transfer.source] != label_map:
                problems.append(f"{transfer.name}: the label map of the source data differs from the checkpoint")
            if transfer.target not in data.target_windows:
                data.target_windows[transfer.target] = load_unlabeled(family, transfer.target)  # windows only
            x, y = data.source[transfer.source]
            windows = data.target_windows[transfer.target]
            for name, arrays in ((transfer.source, x), (transfer.target, windows)):
                for split in ("train", "val", "test"):
                    if arrays[split].ndim != 3 or arrays[split].shape[1:] != (150, 18) or not len(arrays[split]):
                        problems.append(f"{name} {split}: unexpected window array of shape {arrays[split].shape}")
            if any(len(x[split]) != len(y[split]) for split in ("train", "val", "test")):
                problems.append(f"{transfer.source}: windows and labels differ in number")
            labels = true_labels(transfer)  # for scoring after training; also checks both label maps
            if len(labels["target_val"]) != len(windows["val"]) or len(labels["target_test"]) != len(windows["test"]):
                problems.append(f"{transfer.target}: target windows and labels differ in number")
            data.scoring_labels[transfer.name] = labels
            load_source_only_model(transfer)  # strict loading of the checkpoint
            data.checkpoint_sha256[transfer.name] = sha256(checkpoint_dir(transfer) / "model_last.pt")
        except Exception as error:
            problems.append(f"{transfer.name}: loading data or checkpoint failed: {error!r}")
    for transfer in transfers:
        selection = config["few_shot"]["selection"].get(transfer.target)
        labels = data.scoring_labels.get(transfer.name, {}).get("target_val")
        if selection is None:
            problems.append(f"configuration: no few-shot windows for {transfer.target}")
        elif labels is not None:
            indices = np.asarray(selection["indices"], dtype=np.int64)
            if (len(labels) != selection["n_target_val"] or indices.max() >= len(labels)
                    or not np.array_equal(labels[indices], selection["labels"])):
                problems.append(f"{transfer.target}: the few-shot windows of the configuration do not match the target-val labels")
    return data, problems


def smoke_test(config: dict, data: Inputs, transfer: Transfer, methods: list[str], device: torch.device) -> list[str]:
    """Two training steps of every method on 64 windows on the device, then outputs, scores and DEV.

    Catches problems of the device with the loss functions before the night starts. The real runs seed
    themselves again (seed_everything in every training loop), so this does not change any result.
    """
    problems = []
    source_x, source_y = data.source[transfer.source]
    target_x = data.target_windows[transfer.target]
    num_classes = len(data.label_maps[transfer.name])
    for method in methods:
        try:
            model = load_source_only_model(transfer)
            train_run(method, combinations(config, method)[0], model, source_x["train"][:64], source_y["train"][:64],
                      target_x["train"][:64], epochs=1, batch_size=config["fixed"]["batch_size"],
                      weight_decay=config["fixed"]["weight_decay"], seed=config["seed"], device=device, log=lambda line: None)
            features, logits = features_and_logits(model, source_x["val"][:64], device)
            target_features, _ = features_and_logits(model, target_x["train"][:64], device)
            if not np.isfinite(logits).all():
                problems.append(f"smoke test {method} on {device.type}: non-finite outputs")
            split_scores("smoke", logits, source_y["val"][:64], num_classes)
            dev_with_raw(features, target_features, features, per_window_ce(logits, source_y["val"][:64]), config["dev"]["seed"])
        except Exception as error:
            problems.append(f"smoke test {method} on {device.type} failed: {error!r}")
    return problems


def read_finished(run: Run, out: Path) -> dict | None:
    """result.json of a finished run (it is written last), or None."""
    folder = run.folder(out)
    try:
        result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    needed = ["model_last.pt", "train_log.csv", "outputs.npz", "report_only.json"]
    if run.method == "dann":
        needed.append("discriminator_last.pt")
    return result if all((folder / name).is_file() for name in needed) else None


def same_setting(result: dict, run: Run, epochs: int) -> bool:
    c = run.combination
    return (result.get("combination") == c.id and result.get("method") == run.method
            and result.get("transfer") == run.transfer.name and result.get("epochs") == epochs
            and result.get("learning_rate") == c.learning_rate and result.get("source_ce_weight") == c.source_ce_weight
            and result.get("alignment_weight") == c.alignment_weight)


def check_output(out: Path, config: dict, epochs: int, commit: str | None) -> list[str]:
    record = out / "search_config.json"
    if not record.is_file():
        return []
    try:
        saved = json.loads(record.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return [f"{record} cannot be read ({error}). Use a new --output-dir."]
    problems = []
    if saved.get("config_sha256") != config_digest(config):
        problems.append(f"{out} belongs to another search configuration. Use a new --output-dir.")
    if saved.get("epochs") != epochs:
        problems.append(f"{out} holds runs with {saved.get('epochs')} epochs, this call uses {epochs}. Use a new --output-dir.")
    if commit is not None and saved.get("repository_commit") != commit:
        problems.append(f"{out} was started with commit {saved.get('repository_commit')}, this checkout is {commit}. "
                        "Check out that commit or use a new --output-dir.")
    return problems


def free_bytes(path: Path) -> int:
    while not path.exists():
        path = path.parent
    return shutil.disk_usage(path).free


# ---------------------------------------------------------------- one run

def execute(run: Run, out: Path, data: Inputs, config: dict, epochs: int, device: torch.device, env: dict,
            commit: str, log: Log) -> tuple[dict, dict]:
    """Train, save and score one run; returns its criteria row and its report-only row."""
    started, tic = datetime.now(timezone.utc), time.perf_counter()
    transfer, method, combination = run.transfer, run.method, run.combination
    folder = run.folder(out)
    folder.mkdir(parents=True, exist_ok=True)
    for name in ("result.json", "error.json"):
        (folder / name).unlink(missing_ok=True)
    fixed, seed = config["fixed"], config["seed"]
    num_classes = len(data.label_maps[transfer.name])
    checkpoint = checkpoint_dir(transfer) / "model_last.pt"
    if sha256(checkpoint) != data.checkpoint_sha256[transfer.name]:
        raise RuntimeError(f"The source-only checkpoint changed during the search: {checkpoint}")
    model = load_source_only_model(transfer, checkpoint)
    source_x, source_y = data.source[transfer.source]
    target_x = data.target_windows[transfer.target]  # windows only

    history, discriminator = train_run(method, combination, model, source_x["train"], source_y["train"], target_x["train"],
                                       epochs=epochs, batch_size=fixed["batch_size"], weight_decay=fixed["weight_decay"],
                                       seed=seed, device=device, log=lambda line: log.detail(f"{run.label}  {line}"))
    train_seconds = time.perf_counter() - tic
    write_atomic(folder / "model_last.pt", torch_bytes(model.state_dict()))  # HAR predictor only
    if discriminator is not None:
        write_atomic(folder / "discriminator_last.pt", torch_bytes(discriminator.state_dict()))
    write_atomic(folder / "train_log.csv", csv_bytes(history))

    # Label-free outputs of the final model, eval mode.
    tic_eval = time.perf_counter()
    features, logits = {}, {}
    for split in ("source_train", *OUTPUT_SPLITS):
        domain, part = split.split("_")
        features[split], logits[split] = features_and_logits(model, (source_x if domain == "source" else target_x)[part], device)
    source_val_ce = per_window_ce(logits["source_val"], source_y["val"])
    alignment_key = ALIGNMENT_LOG_KEY[method]
    last = history[-1]
    row = {
        "search": config["name"], "combination": combination.id, "origin": combination.origin, "method": method,
        "transfer": transfer.name, "family": transfer.family, "source": transfer.source, "target": transfer.target,
        "seed": seed, "learning_rate": combination.learning_rate, "source_ce_weight": combination.source_ce_weight,
        "alignment_weight": combination.alignment_weight, "alignment_loss": ALIGNMENT_LOSS[method],
        "epochs": epochs, "batch_size": fixed["batch_size"], "weight_decay": fixed["weight_decay"],
        "steps_per_epoch": history[0]["steps"], "n_source_train": len(source_x["train"]),
        "n_target_train": len(target_x["train"]), "n_source_val": len(source_x["val"]),
        "first_source_ce": history[0]["source_ce"], "final_source_ce": last["source_ce"],
        "final_alignment_loss": last[alignment_key], "final_total_loss": last["total"],
        "nonfinite_training_loss": any(not math.isfinite(r[k]) for r in history for k in ("source_ce", "total", alignment_key)),
        "nonfinite_outputs": not all(np.isfinite(values).all() for values in logits.values()),
    }
    if method == "dann":
        row.update({"final_disc_acc_source": last["disc_acc_source"], "final_disc_acc_target": last["disc_acc_target"]})
    if method == "dsan":
        row.update({"final_pseudo_label_confidence": last["pseudo_label_confidence"],
                    "lmmd_nan_steps": sum(r["lmmd_nan_steps"] for r in history)})
    row.update(split_scores("src", logits["source_val"], source_y["val"], num_classes))
    row.update(split_scores("source_test", logits["source_test"], source_y["test"], num_classes))
    try:
        dev, weights = dev_with_raw(features["source_train"], features["target_train"], features["source_val"],
                                    source_val_ce, config["dev"]["seed"])
        row.update({**dev, "dev_seed": config["dev"]["seed"], "dev_error": ""})
    except Exception as error:  # DEV trains an auxiliary classifier; its failure must not cost the run
        weights = np.full(len(source_val_ce), np.nan)
        row.update({"dev_risk": math.nan, "dev_seed": config["dev"]["seed"], "dev_error": repr(error)})

    # Scoring with target labels, after training.
    labels = data.scoring_labels[transfer.name]
    shots = np.asarray(config["few_shot"]["selection"][transfer.target]["indices"], dtype=np.int64)
    row.update(split_scores("fst", logits["target_val"][shots], labels["target_val"][shots], num_classes))
    row.update(split_scores("oracle_target_val", logits["target_val"], labels["target_val"], num_classes))
    row["target_labels_used"] = TARGET_LABELS_USED
    report = {key: row[key] for key in ("search", "combination", "origin", "method", "transfer", "learning_rate",
                                         "source_ce_weight", "alignment_weight", "epochs")}
    report.update(split_scores("target_train", logits["target_train"], labels["target_train"], num_classes))
    report.update(split_scores("target_test", logits["target_test"], labels["target_test"], num_classes))
    report["note"] = "report only: never use these values to select a combination, a method or a run"

    eval_seconds = time.perf_counter() - tic_eval
    row.update({**env, "repository_commit": commit,
                "init_checkpoint": checkpoint.relative_to(ROOT).as_posix(),
                "init_checkpoint_sha256": data.checkpoint_sha256[transfer.name],
                "started_utc": started.isoformat(), "finished_utc": datetime.now(timezone.utc).isoformat(),
                "train_seconds": round(train_seconds, 2), "eval_seconds": round(eval_seconds, 2),
                "total_seconds": round(time.perf_counter() - tic, 2)})
    predictions = {f"y_pred_{split}": logits[split].argmax(1).astype(np.int64) for split in OUTPUT_SPLITS}
    write_atomic(folder / "outputs.npz", npz_bytes(**predictions, source_val_ce=source_val_ce,
                                                   dev_weights=np.asarray(weights, dtype=np.float64)))
    write_atomic(folder / "report_only.json", json_bytes(report))
    write_atomic(folder / "result.json", json_bytes(row))  # last: marks the run as finished
    return row, report


# ---------------------------------------------------------------- progress

class Progress:
    """Finished runs, complete combinations and an estimate of the remaining time."""

    def __init__(self, plan: list[Run]) -> None:
        self.plan = plan
        self.finished: set[tuple] = set()
        self.seconds: dict[tuple, list[float]] = defaultdict(list)
        self.per_combination = defaultdict(int)
        for run in plan:
            self.per_combination[run.combination.id] += 1

    def done(self, run: Run, seconds: float | None) -> None:
        self.finished.add(run.key)
        if seconds is not None:
            self.seconds[run.method, run.transfer.name].append(seconds)

    def remaining(self) -> list[Run]:
        return [run for run in self.plan if run.key not in self.finished]

    def complete_combinations(self) -> list[int]:
        counts = defaultdict(int)
        for run in self.plan:
            if run.key in self.finished:
                counts[run.combination.id] += 1
        return sorted(i for i, n in counts.items() if n == self.per_combination[i])

    def eta_seconds(self) -> float | None:
        means = {key: sum(values) / len(values) for key, values in self.seconds.items() if values}
        if not means:
            return None
        fallback = sum(means.values()) / len(means)
        return sum(means.get((run.method, run.transfer.name), fallback) for run in self.remaining())


def describe_ids(ids: list[int]) -> str:
    if not ids:
        return "none"
    parts, start = [], ids[0]
    for previous, current in zip(ids, ids[1:] + [None]):
        if current != previous + 1:
            parts.append(f"{start}" if start == previous else f"{start}-{previous}")
            start = current
    return ", ".join(parts)


def hours(seconds: float | None) -> str:
    if seconds is None:
        return "after the first runs"
    finish = datetime.now() + timedelta(seconds=seconds)
    return f"{seconds / 3600:.1f} h (about {finish:%a %H:%M})"


# ---------------------------------------------------------------- main

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", required=True, choices=["cuda", "mps", "cpu"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, help=f"default: {DEFAULT_OUTPUT.relative_to(ROOT)}")
    parser.add_argument("--combinations", nargs="+", metavar="ID", help="ids or ranges, e.g. 0-9 (default: all)")
    parser.add_argument("--methods", nargs="+", choices=["deep_coral", "dann", "dsan"])
    parser.add_argument("--transfers", nargs="+", metavar="TRANSFER")
    parser.add_argument("--epochs", type=int, help="tests only: other number of epochs (needs its own --output-dir)")
    parser.add_argument("--check-only", action="store_true", help="run the start checks and stop")
    return parser.parse_args()


def fail(log: Log, problems: list[str]) -> int:
    log("START CHECK FAILED. Nothing was trained. Problems:")
    for problem in problems:
        log(f"  - {problem}")
    log("Fix these and run the same command again.")
    return EXIT_START_CHECK


def main() -> int:
    configure_console()
    warnings.filterwarnings("ignore", module="sklearn")  # degenerate predictions of diverged runs
    args = parse_args()
    log = Log()
    log(f"Hyperparameter search: {args.config}")
    problems = check_versions()
    try:
        config = load_config(args.config)
    except (OSError, ValueError) as error:
        return fail(log, problems + [f"configuration {args.config} cannot be read: {error}"])
    problems += config_problems(config)
    if problems:
        return fail(log, problems)
    methods = args.methods or config["methods"]
    if any(name not in config["transfers"] for name in args.transfers or []):
        return fail(log, [f"--transfers: choose from {config['transfers']}"])
    transfers = [get_transfer(name) for name in (args.transfers or config["transfers"])]
    epochs = args.epochs or config["fixed"]["epochs"]
    if epochs != config["fixed"]["epochs"] and args.output_dir is None:
        return fail(log, ["--epochs other than the configured value needs its own --output-dir"])
    plan = build_plan(config, parse_ids(args.combinations) if args.combinations else None, methods, transfers)
    out = check_output_dir(args.output_dir or DEFAULT_OUTPUT)

    device_problems, env = check_device(args.device)
    code_problems, commit = check_code(args.config)
    file_problems = check_input_files(config)
    problems = device_problems + code_problems + file_problems
    data = None
    if not file_problems:
        log("Loading data and checkpoints ...")
        data, load_problems = load_inputs(config, transfers)
        problems += load_problems
        if not problems:
            log("Smoke test: two training steps of every method on the device ...")
            problems += smoke_test(config, data, transfers[0], methods, torch.device(args.device))
    problems += check_output(out, config, epochs, commit)
    progress = Progress(plan)
    for run in plan:
        result = read_finished(run, out)
        if result is not None:
            if not same_setting(result, run, epochs):
                problems.append(f"{run.folder(out)} holds a run with another setting. Use a new --output-dir.")
            else:
                progress.done(run, result.get("total_seconds") if result.get("device") == args.device else None)
    remaining = len(progress.remaining())
    free = free_bytes(out)
    needed = remaining * RUN_BYTES + DISK_MARGIN
    if free < needed:
        problems.append(f"not enough disk space: {free / 1e9:.2f} GB free, about {needed / 1e9:.2f} GB needed "
                        f"for {remaining} runs ({RUN_BYTES / 1e6:.1f} MB per run)")
    if problems:
        return fail(log, problems)

    log(f"START CHECK PASSED. Device {env['device']} ({env['device_name']}), PyTorch {env['torch']}, commit {commit[:10]}")
    log(f"Planned {len(plan)} runs ({len(plan) // max(1, len(methods) * len(transfers))} combinations x "
        f"{len(transfers)} transfers x {len(methods)} methods, {epochs} epochs); finished {len(progress.finished)}, "
        f"remaining {remaining}. Disk: {free / 1e9:.1f} GB free, about {remaining * RUN_BYTES / 1e9:.2f} GB needed.")
    log(f"Output: {out}")
    if args.check_only:
        return 0

    out.mkdir(parents=True, exist_ok=True)
    log.open(out / "search_log.txt")
    log(f"Session start: {len(progress.finished)} of {len(plan)} runs finished, device {env['device_name']}, commit {commit}")
    if not (out / "search_config.json").is_file():
        write_atomic(out / "search_config.json", json_bytes({
            "config": args.config.resolve().relative_to(ROOT).as_posix(), "config_sha256": config_digest(config),
            "epochs": epochs, "repository_commit": commit, "created_utc": datetime.now(timezone.utc).isoformat()}))

    rows, reports = {}, {}
    for path in sorted((out / "runs").glob("c*/*/*/result.json")):
        try:
            rows[path.parent] = json.loads(path.read_text(encoding="utf-8"))
            reports[path.parent] = json.loads((path.parent / "report_only.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            rows.pop(path.parent, None)
    order = {(m, t.name): (i, j) for j, m in enumerate(config["methods"]) for i, t in enumerate(map(get_transfer, config["transfers"]))}

    def sort_key(row: dict) -> tuple:
        return row["combination"], *order.get((row["method"], row["transfer"]), (99, 99))

    def write_tables() -> None:
        try:
            write_atomic(out / "criteria.csv", csv_bytes(sorted(rows.values(), key=sort_key)))
            write_atomic(out / "report_only_target.csv", csv_bytes(sorted(reports.values(), key=sort_key)))
        except OSError as error:  # e.g. a table open in a spreadsheet program; the run files are complete
            log(f"warning: tables not written ({error}); they are rewritten after the next run")

    def write_progress(current: str, failed: list[str]) -> None:
        complete = progress.complete_combinations()
        try:
            write_atomic(out / "progress.json", json_bytes({
                "updated": datetime.now().isoformat(timespec="seconds"), "search": config["name"],
                "device": env["device_name"], "repository_commit": commit, "planned_runs": len(plan),
                "finished_runs": len(progress.finished), "remaining_runs": len(progress.remaining()),
                "complete_combinations": complete, "n_complete_combinations": len(complete),
                "failed_this_session": failed, "current": current, "eta_hours": (
                    None if progress.eta_seconds() is None else round(progress.eta_seconds() / 3600, 2))}))
        except OSError as error:
            log(f"warning: progress.json not written ({error})")

    failed: list[str] = []
    consecutive = 0
    device = torch.device(args.device)
    position = len(progress.finished)
    announced = set(progress.complete_combinations())
    checked_combination = None
    try:
        for run in plan:
            if run.key in progress.finished:
                continue
            if free_bytes(out) < MIN_FREE_BYTES:
                log(f"STOPPED: less than {MIN_FREE_BYTES / 1e9:.1f} GB disk space left. Free space, then run the same command again.")
                return EXIT_DISK
            if run.combination.id != checked_combination:  # the code must stay the committed one (once per combination)
                checked_combination = run.combination.id
                try:
                    now = clean_commit()
                except RuntimeError as error:
                    log(f"STOPPED: the code changed during the search:\n{error}")
                    return EXIT_CODE_CHANGED
                except Exception as error:  # a passing git hiccup must not stop the night
                    log(f"warning: git check failed ({error!r}); continuing with commit {commit[:10]}")
                    now = commit
                if now != commit:
                    log(f"STOPPED: HEAD moved from {commit} to {now} during the search.")
                    return EXIT_CODE_CHANGED
            position += 1
            write_progress(run.label, failed)
            c = run.combination
            tic = time.perf_counter()
            try:
                row, report = execute(run, out, data, config, epochs, device, env, commit, log)
            except KeyboardInterrupt:
                raise
            except Exception as error:
                consecutive += 1
                failed.append(run.label)
                log(f"FAILED {run.label}: {type(error).__name__}: {error} (details in errors.log; retried at the next start)")
                trace = traceback.format_exc()
                try:
                    with (out / "errors.log").open("a", encoding="utf-8") as handle:
                        handle.write(f"\n=== {datetime.now():%Y-%m-%d %H:%M:%S} {run.label}\n{trace}")
                    write_atomic(run.folder(out) / "error.json", json_bytes({"run": run.label, "error": repr(error), "traceback": trace}))
                except OSError:
                    pass
                if device.type == "cuda":
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                if not device_healthy(device):
                    log(f"STOPPED: the device {args.device} does not work any more. Run the same command again "
                        "(restart the computer first if it keeps failing).")
                    return EXIT_DEVICE
                if consecutive >= MAX_CONSECUTIVE_FAILURES:
                    log(f"STOPPED: {consecutive} runs failed in a row; see errors.log.")
                    return EXIT_FAILURES
                continue
            consecutive = 0
            seconds = time.perf_counter() - tic
            progress.done(run, seconds)
            rows[run.folder(out)] = row
            reports[run.folder(out)] = report
            write_tables()
            write_progress("", failed)
            log(f"[{position:4d}/{len(plan)}] {run.label:<42} lr {c.learning_rate:.0e} src {c.source_ce_weight:6.3f} "
                f"align {c.alignment_weight:6.3f} | {seconds:5.0f} s | SRC ce {row['src_ce']:.3f} macro-F1 "
                f"{row['src_macro_f1']:.3f} | remaining {len(progress.remaining())}, ETA {hours(progress.eta_seconds())}")
            complete = progress.complete_combinations()
            if c.id in complete and c.id not in announced:
                announced.add(c.id)
                log(f"Combination {c.id} complete. Complete combinations: {len(complete)} ({describe_ids(complete)})")
        write_progress("", failed)
        complete = progress.complete_combinations()
        log(f"Search finished: {len(progress.finished)} of {len(plan)} runs, {len(failed)} failed in this session, "
            f"complete combinations: {len(complete)} ({describe_ids(complete)}).")
        if failed:
            log("Run the same command again to retry the failed runs.")
        return 0
    except KeyboardInterrupt:
        write_progress("", failed)
        complete = progress.complete_combinations()
        log(f"Stopped by Ctrl+C. Finished runs are kept ({len(progress.finished)} of {len(plan)}; complete "
            f"combinations: {len(complete)} ({describe_ids(complete)})). The interrupted run is redone at the next start.")
        return EXIT_INTERRUPTED


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:  # during the start checks; nothing was written
        print("Stopped by Ctrl+C before the first run.", flush=True)
        sys.exit(EXIT_INTERRUPTED)
