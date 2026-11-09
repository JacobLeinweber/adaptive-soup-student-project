"""DANN and its control on the five seeds, each from the source-only model of its seed.

Examples:
    python run_dann.py --all --all-variants                  # all seeds, DANN and control
    python run_dann.py --transfer PAMAP2_to_RealWorld --seeds 52
    python run_dann.py --all --seeds 52 --control            # domain weight 0, target windows still pass through the model

Seed 52 starts from the supplied checkpoint in models/, the other seeds from results/source_only/.
Results: results/dann/seed_<seed>/<transfer>/ and results/dann_lambda0_target_pass/seed_<seed>/<transfer>/.
Finished runs are skipped, so an interrupted call can simply be repeated.

Target data enter only as windows (features.pkl); this script never opens a target
evaluation_labels.pkl. Scoring happens afterwards in a separate script.
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
from uda.dann import DISCRIMINATOR_BETAS, DISCRIMINATOR_HIDDEN, train_dann
from uda.data import check_output_dir, get_transfer, load_label_map, load_labeled, load_source_only_model, load_unlabeled, sha256
from uda.evaluation import SPLITS, predict
from uda.runs import BASELINE, SEEDS, clean_commit, environment, is_complete, source_only_checkpoint, write_train_log

ADATIME = {
    "repository": "https://github.com/emadeldeen24/AdaTime",
    "commit": "2d9be6bd0542e150518afaf8aa0c6ea0823b4a3d",
    "discriminator": "models/models.py:229-248 (class Discriminator), disc_hid_dim configs/data_model_configs.py:41",
    "gradient_reversal": "models/models.py:252-261 (class ReverseLayerF)",
    "reversal_schedule": "algorithms/algorithms.py:321,327-328 (DANN.training_epoch), formula taken literally",
    "training_step": "algorithms/algorithms.py:284-371 (class DANN)",
    "weights": "configs/hparams.py:42-47 (class HAR = UCIHAR, 'DANN')",
}
DOMAIN_WEIGHT = 2.943729820531079   # AdaTime HAR DANN "domain_loss_wt"
SOURCE_CE_WEIGHT = 5.1390077646202  # AdaTime HAR DANN "src_cls_loss_wt"
MODES = {
    "dann": {"folder": "dann", "method": "dann", "domain_weight": DOMAIN_WEIGHT},
    "control": {"folder": "dann_lambda0_target_pass",
                "method": "dann_lambda0_target_pass (domain weight 0, no discriminator, target windows pass through the model)",
                "domain_weight": 0.0},
}


def is_finished(out: Path, mode: dict) -> bool:
    """A run writes config.json last; a DANN run must also hold its discriminator."""
    return is_complete(out, SPLITS) and (mode["domain_weight"] == 0 or (out / "discriminator_last.pt").is_file())


def run_one(transfer: Transfer, mode: dict, seed: int, results_root: Path, device: torch.device, overwrite: bool) -> None:
    out = results_root / mode["folder"] / f"seed_{seed}" / transfer.name
    if is_finished(out, mode) and not overwrite:
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
    history, discriminator = train_dann(
        model, source_x["train"], source_y["train"], target_x["train"],
        domain_weight=mode["domain_weight"], source_ce_weight=SOURCE_CE_WEIGHT, seed=seed, device=device,
        **BASELINE, log=lambda line: print(line, flush=True),
    )
    train_seconds = time.perf_counter() - tic

    (out / "predictions").mkdir(parents=True, exist_ok=True)
    # model_last.pt holds only the HAR predictor (features + classifier); the discriminator stays
    # in its own file, because UDA-specific heads are not part of a later soup.
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, out / "model_last.pt")
    if discriminator is not None:
        torch.save({k: v.detach().cpu() for k, v in discriminator.state_dict().items()}, out / "discriminator_last.pt")
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
        "domain_weight": mode["domain_weight"], "source_ce_weight": SOURCE_CE_WEIGHT,
        "optimizer": "Adam (torch.optim.Adam, L2 weight decay, default betas)", **BASELINE,
        "discriminator": None if discriminator is None else {
            "hidden_dim": DISCRIMINATOR_HIDDEN,
            "optimizer": "Adam, same lr and weight decay as the network", "betas": list(DISCRIMINATOR_BETAS),
            "reversal_schedule": "AdaTime formula, practically alpha = 1 from the first step",
        },
        "steps_per_epoch": history[0]["steps"],
        "n_source_train": len(source_x["train"]), "n_target_train": len(target_x["train"]),
        "target_input": "target train windows only (features.pkl), no labels",
        "forward_passes": "separate source and target passes in train mode, source first",
        "model_selection": "last epoch", "adatime": ADATIME,
        **environment(device), "repository_commit": commit,
        "started_utc": started.isoformat(), "finished_utc": datetime.now(timezone.utc).isoformat(),
        "train_seconds": round(train_seconds, 1),
    })
    print(f"saved {out} ({train_seconds:.0f}s training)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--transfer", nargs="+", choices=[t.name for t in TRANSFERS])
    which.add_argument("--all", action="store_true", help="all 12 transfers, one after another")
    variant = parser.add_mutually_exclusive_group()
    variant.add_argument("--control", action="store_true", help="only the control dann_lambda0_target_pass")
    variant.add_argument("--all-variants", action="store_true", help="DANN and the control")
    parser.add_argument("--seeds", type=int, nargs="+", choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--output-dir", type=Path, help="default: results/")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--overwrite", action="store_true", help="retrain even if a finished run exists")
    args = parser.parse_args()
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is not available")
    modes = list(MODES.values()) if args.all_variants else [MODES["control" if args.control else "dann"]]
    results_root = check_output_dir(args.output_dir or ROOT / "results")
    transfers = TRANSFERS if args.all else [get_transfer(name) for name in args.transfer]
    for seed in args.seeds:
        for mode in modes:
            for transfer in transfers:
                run_one(transfer, mode, seed, results_root, torch.device(args.device), args.overwrite)


if __name__ == "__main__":
    main()
