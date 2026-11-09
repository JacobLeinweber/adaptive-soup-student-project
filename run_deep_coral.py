"""Deep CORAL and the lambda = 0 controls on the five seeds, each from the source-only model of its seed.

Examples:
    python run_deep_coral.py --all --all-variants                     # all seeds and variants
    python run_deep_coral.py --transfer PAMAP2_to_RealWorld --seeds 100
    python run_deep_coral.py --all --seeds 52 --control               # lambda = 0, source only
    python run_deep_coral.py --all --seeds 52 --control-target-pass   # lambda = 0, target windows still pass through the model

Seed 52 starts from the supplied checkpoint in models/, the other seeds from results/source_only/.
Results: results/<variant>/seed_<seed>/<transfer>/. Finished runs are skipped, so an interrupted
call can simply be repeated.

Target data enter only as windows (features.pkl); this script never opens a target
evaluation_labels.pkl. Scoring happens afterwards in make_seed_tables.py.
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from src.training.experiments import TRANSFERS, Transfer
from src.training.train_transfer import ROOT
from src.training.utils import write_json
from uda.data import check_output_dir, get_transfer, load_label_map, load_labeled, load_source_only_model, load_unlabeled, sha256
from uda.evaluation import SPLITS, predict
from uda.runs import BASELINE, SEEDS, clean_commit, environment, is_complete, source_only_checkpoint, write_train_log
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


def run_one(transfer: Transfer, mode: dict, seed: int, results_root: Path, device: torch.device, overwrite: bool) -> None:
    out = results_root / mode["folder"] / f"seed_{seed}" / transfer.name
    if is_complete(out, SPLITS) and not overwrite:
        print(f"skip {mode['folder']} seed {seed} {transfer.name}: finished run in {out}", flush=True)
        return
    commit = clean_commit()
    print(f"=== {mode['method']}, seed {seed}: {transfer.name}", flush=True)
    (out / "config.json").unlink(missing_ok=True)
    started = datetime.now(timezone.utc)
    checkpoint = source_only_checkpoint(transfer, seed)
    checkpoint_hash = sha256(checkpoint)
    label_map = load_label_map(transfer)
    model = load_source_only_model(transfer, checkpoint)

    source_x, source_y, source_meta = load_labeled(transfer.family, transfer.source)
    if source_meta["label_map"] != label_map:
        raise ValueError(f"Source label map does not match the checkpoint: {transfer.name}")
    target_x = load_unlabeled(transfer.family, transfer.target)  # windows only, no labels

    tic = time.perf_counter()
    history = train_continued(
        model, source_x["train"], source_y["train"], target_x["train"] if mode["target_pass"] else None,
        coral_weight=mode["coral_weight"], source_ce_weight=SOURCE_CE_WEIGHT, seed=seed, device=device,
        **BASELINE, log=lambda line: print(line, flush=True),
    )
    train_seconds = time.perf_counter() - tic

    (out / "predictions").mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, out / "model_last.pt")
    write_train_log(out / "train_log.csv", history)
    for split in SPLITS:
        domain, part = split.split("_")
        x = source_x[part] if domain == "source" else target_x[part]
        np.savez_compressed(out / "predictions" / f"{split}.npz", y_pred=predict(model, x, device))

    if sha256(checkpoint) != checkpoint_hash:
        raise RuntimeError(f"The source-only checkpoint changed: {checkpoint}")
    write_json(out / "config.json", {
        "method": mode["method"],
        "transfer": transfer.name, "family": transfer.family,
        "source": transfer.source, "target": transfer.target, "label_map": label_map, "seed": seed,
        "init_checkpoint": str(checkpoint.relative_to(ROOT)), "init_checkpoint_sha256": checkpoint_hash,
        "coral_weight": mode["coral_weight"], "source_ce_weight": SOURCE_CE_WEIGHT,
        "optimizer": "Adam (torch.optim.Adam, L2 weight decay)", **BASELINE,
        "steps_per_epoch": history[0]["steps"],
        "n_source_train": len(source_x["train"]), "n_target_train": len(target_x["train"]),
        "target_input": "target train windows only (features.pkl), no labels" if mode["target_pass"] else "none",
        "forward_passes": ("separate source and target passes in train mode, source first"
                           if mode["target_pass"] else "source only"),
        "model_selection": "last epoch", "adatime": ADATIME,
        **environment(device), "repository_commit": commit,
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
    variant.add_argument("--all-variants", action="store_true", help="Deep CORAL and both controls")
    parser.add_argument("--seeds", type=int, nargs="+", choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--output-dir", type=Path, help="default: results/")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--overwrite", action="store_true", help="retrain even if a finished run exists")
    args = parser.parse_args()
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is not available")
    if args.all_variants:
        modes = list(MODES.values())
    else:
        modes = [MODES["control" if args.control else "control_target_pass" if args.control_target_pass else "deep_coral"]]
    results_root = check_output_dir(args.output_dir or ROOT / "results")
    transfers = TRANSFERS if args.all else [get_transfer(args.transfer)]
    for seed in args.seeds:
        for mode in modes:
            for transfer in transfers:
                run_one(transfer, mode, seed, results_root, torch.device(args.device), args.overwrite)


if __name__ == "__main__":
    main()
