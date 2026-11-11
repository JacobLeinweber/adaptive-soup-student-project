"""Sensitivity to the alignment weight in the paper convention: Deep CORAL, DANN and DSAN with source weight 1.

Loss per step for all three methods: 1 * CE(source) + lambda * alignment loss, lambda = factor * base lambda.
The base lambda of a method is AdaTime's alignment weight divided by AdaTime's source-loss weight for
HAR (UCIHAR), so that factor 1 has the same ratio of the two terms as the runs with the AdaTime weights.
Everything else is unchanged: start model, optimiser, budget, forward passes, DANN reversal factor and
discriminator, constant DSAN weight.

The shared control (lambda = 0, source weight 1, target forward pass, no discriminator) is the same
for all three methods: with these weights the CORAL, DANN and DSAN loops compute the same updates
(checked bitwise on MHEALTH_to_PAMAP2).
It is run with the DANN loop.

Examples:
    python run_paper_convention.py --all                                  # 3 methods x 5 factors and the control
    python run_paper_convention.py --all --methods dann --factors 0.5 1
    python run_paper_convention.py --transfer PAMAP2_to_RealWorld --methods control

Only seed 52 by default; it starts from the supplied checkpoint in models/.
Results: results/paper_convention/<method>/x<factor>/seed_<seed>/<transfer>/ and
results/paper_convention/control/seed_<seed>/<transfer>/. Finished runs are skipped, so an
interrupted call can simply be repeated.

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

import run_dann
import run_deep_coral
import run_dsan
from src.training.experiments import TRANSFERS, Transfer
from src.training.train_transfer import ROOT
from src.training.utils import SEED, write_json
from uda.dann import DISCRIMINATOR_BETAS, DISCRIMINATOR_HIDDEN, train_dann
from uda.data import check_output_dir, get_transfer, load_label_map, load_labeled, load_source_only_model, load_unlabeled, sha256
from uda.dsan import KERNEL_MUL, KERNEL_NUM, train_dsan
from uda.evaluation import SPLITS, predict
from uda.runs import BASELINE, SEEDS, clean_commit, environment, is_complete, source_only_checkpoint, write_train_log
from uda.training import train_continued

CONVENTION = "paper: loss = 1 * CE(source) + lambda * alignment loss, lambda = factor * base lambda"
SOURCE_WEIGHT = 1.0
FACTORS = (0.25, 0.5, 1.0, 2.0, 4.0)
# Source validation is saved as well: the label-free source risk (SRC) is computed on it.
RUN_SPLITS = ("source_val", *SPLITS)
# AdaTime weights for HAR (configs/hparams.py, commit 2d9be6bd), taken from the run scripts of the AdaTime runs.
METHODS = {
    "deep_coral": {"alignment_loss": "CORAL", "adatime": run_deep_coral.ADATIME,
                   "adatime_alignment_weight": run_deep_coral.CORAL_WEIGHT,
                   "adatime_source_weight": run_deep_coral.SOURCE_CE_WEIGHT},
    "dann": {"alignment_loss": "domain loss (discriminator CE on source + target, gradient reversal)",
             "adatime": run_dann.ADATIME,
             "adatime_alignment_weight": run_dann.DOMAIN_WEIGHT, "adatime_source_weight": run_dann.SOURCE_CE_WEIGHT},
    "dsan": {"alignment_loss": "LMMD", "adatime": run_dsan.ADATIME,
             "adatime_alignment_weight": run_dsan.LMMD_WEIGHT, "adatime_source_weight": run_dsan.SOURCE_CE_WEIGHT},
}
CONTROL = "control"
RESULTS = ROOT / "results" / "paper_convention"


def base_lambda(method: str) -> float:
    """AdaTime alignment weight / AdaTime source-loss weight."""
    return METHODS[method]["adatime_alignment_weight"] / METHODS[method]["adatime_source_weight"]


def factor_name(factor: float) -> str:
    return f"x{factor:g}"


def run_folder(results_root: Path, method: str, factor: float | None, seed: int, transfer: Transfer) -> Path:
    if method == CONTROL:
        return results_root / CONTROL / f"seed_{seed}" / transfer.name
    return results_root / method / factor_name(factor) / f"seed_{seed}" / transfer.name


def is_finished(out: Path, method: str) -> bool:
    """A run writes config.json last; a DANN run must also hold its discriminator."""
    return is_complete(out, RUN_SPLITS) and (method != "dann" or (out / "discriminator_last.pt").is_file())


def train(method: str, weight: float, model, source_x, source_y, target_x, seed: int, device: torch.device):
    """Continue training in place with source weight 1; return the epoch log and the discriminator (DANN only)."""
    common = dict(source_ce_weight=SOURCE_WEIGHT, seed=seed, device=device, log=lambda line: print(line, flush=True), **BASELINE)
    if method == "deep_coral":
        return train_continued(model, source_x, source_y, target_x, coral_weight=weight, **common), None
    if method == "dsan":
        return train_dsan(model, source_x, source_y, target_x, lmmd_weight=weight, **common), None
    # DANN, and the shared control (weight 0: no discriminator, no domain loss, target windows still pass through).
    return train_dann(model, source_x, source_y, target_x, domain_weight=weight, **common)


def method_details(method: str, weight: float) -> dict:
    """The settings of the method that are not loss weights, as in the config.json of the AdaTime runs."""
    if method == "dann" or method == CONTROL:
        return {"discriminator": None if weight == 0 else {
            "hidden_dim": DISCRIMINATOR_HIDDEN,
            "optimizer": "Adam, same lr and weight decay as the network", "betas": list(DISCRIMINATOR_BETAS),
            "reversal_schedule": "AdaTime formula, practically alpha = 1 from the first step"}}
    if method == "dsan":
        return {"lmmd": {
            "kernel_num": KERNEL_NUM, "kernel_mul": KERNEL_MUL,
            "bandwidth": "mean squared distance over all pairs of the stacked batch, no gradient",
            "pseudo_labels": "softmax outputs of the model on the target batch of the step, no gradient",
            "weight_schedule": "none (constant weight, as in AdaTime)"}}
    return {}


def run_one(transfer: Transfer, method: str, factor: float | None, seed: int, results_root: Path,
            device: torch.device, overwrite: bool) -> None:
    out = run_folder(results_root, method, factor, seed, transfer)
    label = CONTROL if method == CONTROL else f"{method} {factor_name(factor)}"
    if is_finished(out, method) and not overwrite:
        print(f"skip {label} seed {seed} {transfer.name}: finished run in {out}", flush=True)
        return
    commit = clean_commit()
    base = None if method == CONTROL else base_lambda(method)
    weight = 0.0 if method == CONTROL else factor * base
    print(f"=== {label} (lambda {weight:.6g}), seed {seed}: {transfer.name}", flush=True)
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
    history, discriminator = train(method, weight, model, source_x["train"], source_y["train"], target_x["train"], seed, device)
    train_seconds = time.perf_counter() - tic

    (out / "predictions").mkdir(parents=True, exist_ok=True)
    # model_last.pt holds only the HAR predictor (features + classifier); the DANN discriminator has its own file.
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, out / "model_last.pt")
    if discriminator is not None:
        torch.save({k: v.detach().cpu() for k, v in discriminator.state_dict().items()}, out / "discriminator_last.pt")
    write_train_log(out / "train_log.csv", history)
    for split in RUN_SPLITS:
        domain, part = split.split("_")
        x = source_x[part] if domain == "source" else target_x[part]
        np.savez_compressed(out / "predictions" / f"{split}.npz", y_pred=predict(model, x, device))

    if sha256(checkpoint) != checkpoint_hash:
        raise RuntimeError(f"The source-only checkpoint changed: {checkpoint}")
    if method == CONTROL:
        weights = {"factor": 0.0, "base_lambda": None, "alignment_weight": 0.0,
                   "alignment_loss": "none (shared control of Deep CORAL, DANN and DSAN)",
                   "control_loop": "train_dann with domain weight 0; the CORAL and DSAN loops give bitwise the same model (checked on MHEALTH_to_PAMAP2)"}
    else:
        weights = {"factor": factor, "base_lambda": base, "alignment_weight": weight,
                   "alignment_loss": METHODS[method]["alignment_loss"],
                   "base_lambda_from": {"adatime_alignment_weight": METHODS[method]["adatime_alignment_weight"],
                                        "adatime_source_weight": METHODS[method]["adatime_source_weight"],
                                        "file": METHODS[method]["adatime"]["weights"]}}
    write_json(out / "config.json", {
        "method": method, "convention": CONVENTION,
        "transfer": transfer.name, "family": transfer.family,
        "source": transfer.source, "target": transfer.target, "label_map": label_map, "seed": seed,
        "init_checkpoint": str(checkpoint.relative_to(ROOT)), "init_checkpoint_sha256": checkpoint_hash,
        **weights, "source_ce_weight": SOURCE_WEIGHT,
        "optimizer": "Adam (torch.optim.Adam, L2 weight decay, default betas)", **BASELINE,
        **method_details(method, weight),
        "steps_per_epoch": history[0]["steps"],
        "n_source_train": len(source_x["train"]), "n_target_train": len(target_x["train"]),
        "target_input": "target train windows only (features.pkl), no labels",
        "forward_passes": "separate source and target passes in train mode, source first",
        "saved_predictions": list(RUN_SPLITS),
        "model_selection": "last epoch", "adatime": METHODS[method]["adatime"] if method != CONTROL else None,
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
    parser.add_argument("--methods", nargs="+", choices=[*METHODS, CONTROL], default=[*METHODS, CONTROL])
    parser.add_argument("--factors", type=float, nargs="+", choices=FACTORS, default=list(FACTORS),
                        help="multiples of the base lambda (not used for the control)")
    parser.add_argument("--seeds", type=int, nargs="+", choices=SEEDS, default=[SEED])
    parser.add_argument("--output-dir", type=Path, help="default: results/paper_convention/")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--overwrite", action="store_true", help="retrain even if a finished run exists")
    args = parser.parse_args()
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is not available")
    results_root = check_output_dir(args.output_dir or RESULTS)
    transfers = TRANSFERS if args.all else [get_transfer(name) for name in args.transfer]
    for seed in args.seeds:
        for method in args.methods:
            for factor in [None] if method == CONTROL else args.factors:
                for transfer in transfers:
                    run_one(transfer, method, factor, seed, results_root, torch.device(args.device), args.overwrite)


if __name__ == "__main__":
    main()
