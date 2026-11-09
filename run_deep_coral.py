"""Deep CORAL and the lambda = 0 controls, starting from the supplied source-only checkpoints.

Examples:
    python run_deep_coral.py --transfer PAMAP2_to_RealWorld
    python run_deep_coral.py --all
    python run_deep_coral.py --all --control               # lambda = 0, source only
    python run_deep_coral.py --all --control-target-pass   # lambda = 0, target windows still pass through the model

Target data enter only as windows (features.pkl); this script never opens a target
evaluation_labels.pkl. Scoring happens afterwards in make_results_tables.py.
"""
from __future__ import annotations

import argparse
import csv
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from src.training.experiments import TRANSFERS, Transfer
from src.training.train_transfer import ROOT
from src.training.utils import SEED, write_json
from uda.data import (check_output_dir, checkpoint_dir, get_transfer, load_label_map, load_labeled,
                      load_source_only_model, load_unlabeled, sha256)
from uda.evaluation import SPLITS, predict
from uda.training import train_continued

ADATIME = {
    "repository": "https://github.com/emadeldeen24/AdaTime",
    "commit": "2d9be6bd0542e150518afaf8aa0c6ea0823b4a3d",
    "loss": "models/loss.py:109-127 (class CORAL)",
    "forward_passes": "algorithms/algorithms.py:188-205 (Deep_Coral.training_epoch)",
    "weights": "configs/hparams.py:71-76 (class HAR = UCIHAR, 'Deep_Coral')",
}
CORAL_WEIGHT = 4.23035475456397        # lambda: AdaTime HAR Deep_Coral "coral_wt"
SOURCE_CE_WEIGHT = 0.1013209750429822  # AdaTime HAR Deep_Coral "src_cls_loss_wt"
# Baseline configuration (defaults of run_transfer, identical in all supplied config.json files).
BASELINE = {"epochs": 30, "batch_size": 32, "lr": 3e-4, "weight_decay": 1e-4, "seed": SEED}
# Run variants. With target_pass the target windows go through the model in train mode, so they
# update the BatchNorm statistics; only coral_weight > 0 adds a CORAL gradient on top.
MODES = {
    "deep_coral": {"folder": "deep_coral", "method": "deep_coral",
                   "coral_weight": CORAL_WEIGHT, "target_pass": True},
    "control": {"folder": "source_only_continued", "method": "source_only_continued (lambda = 0)",
                "coral_weight": 0.0, "target_pass": False},
    "control_target_pass": {"folder": "lambda0_target_pass",
                            "method": "lambda0_target_pass (lambda = 0, target windows pass through the model)",
                            "coral_weight": 0.0, "target_pass": True},
}
NONDETERMINISM = ("Trained on Apple MPS. PyTorch does not guarantee bitwise reproducible results on MPS; "
                  "a repeated run with the same seed can differ slightly.")


def run_one(transfer: Transfer, mode: dict, output_root: Path, device: torch.device, overwrite: bool) -> None:
    out = output_root / transfer.name
    if (out / "config.json").is_file() and not overwrite:
        print(f"skip {transfer.name}: finished run in {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")
        return
    print(f"=== {mode['method']}: {transfer.name}", flush=True)
    started = datetime.now(timezone.utc)
    checkpoint = checkpoint_dir(transfer) / "model_last.pt"
    checkpoint_hash = sha256(checkpoint)
    label_map = load_label_map(transfer)
    model = load_source_only_model(transfer)

    source_x, source_y, source_meta = load_labeled(transfer.family, transfer.source)
    if source_meta["label_map"] != label_map:
        raise ValueError(f"Source label map does not match the checkpoint: {transfer.name}")
    target_x = load_unlabeled(transfer.family, transfer.target)  # windows only, no labels

    tic = time.perf_counter()
    history = train_continued(
        model, source_x["train"], source_y["train"], target_x["train"] if mode["target_pass"] else None,
        coral_weight=mode["coral_weight"], source_ce_weight=SOURCE_CE_WEIGHT, device=device,
        **BASELINE, log=lambda line: print(line, flush=True),
    )
    train_seconds = time.perf_counter() - tic

    (out / "predictions").mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, out / "model_last.pt")
    with (out / "train_log.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    for split in SPLITS:
        domain, part = split.split("_")
        x = source_x[part] if domain == "source" else target_x[part]
        np.savez_compressed(out / "predictions" / f"{split}.npz", y_pred=predict(model, x, device))

    if sha256(checkpoint) != checkpoint_hash:
        raise RuntimeError(f"The supplied checkpoint changed: {checkpoint}")
    write_json(out / "config.json", {
        "method": mode["method"],
        "transfer": transfer.name, "family": transfer.family,
        "source": transfer.source, "target": transfer.target, "label_map": label_map,
        "init_checkpoint": str(checkpoint.relative_to(ROOT)), "init_checkpoint_sha256": checkpoint_hash,
        "coral_weight": mode["coral_weight"], "source_ce_weight": SOURCE_CE_WEIGHT,
        "optimizer": "Adam (torch.optim.Adam, L2 weight decay)", **BASELINE,
        "steps_per_epoch": history[0]["steps"],
        "n_source_train": len(source_x["train"]), "n_target_train": len(target_x["train"]),
        "target_input": "target train windows only (features.pkl), no labels" if mode["target_pass"] else "none",
        "forward_passes": ("separate source and target passes in train mode, source first"
                           if mode["target_pass"] else "source only"),
        "model_selection": "last epoch", "adatime": ADATIME,
        "device": str(device), "torch": torch.__version__, "numpy": np.__version__,
        "python": platform.python_version(), "platform": platform.platform(),
        "nondeterminism": NONDETERMINISM,
        "repository_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "started_utc": started.isoformat(), "finished_utc": datetime.now(timezone.utc).isoformat(),
        "train_seconds": round(train_seconds, 1),
    })
    print(f"saved {out} ({train_seconds:.0f}s training)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--transfer", choices=[t.name for t in TRANSFERS])
    which.add_argument("--all", action="store_true", help="all 12 transfers, one after another")
    variant = parser.add_mutually_exclusive_group()
    variant.add_argument("--control", action="store_true", help="lambda = 0 control, source only")
    variant.add_argument("--control-target-pass", action="store_true",
                         help="lambda = 0 control in which target windows still pass through the model")
    parser.add_argument("--output-dir", type=Path, help="default: the folder of the variant under results/")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--overwrite", action="store_true", help="retrain even if a finished run exists")
    args = parser.parse_args()
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is not available")
    mode = MODES["control" if args.control else "control_target_pass" if args.control_target_pass else "deep_coral"]
    output_root = check_output_dir(args.output_dir or ROOT / "results" / mode["folder"])
    transfers = TRANSFERS if args.all else [get_transfer(args.transfer)]
    for transfer in transfers:
        run_one(transfer, mode, output_root, torch.device(args.device), args.overwrite)


if __name__ == "__main__":
    main()
