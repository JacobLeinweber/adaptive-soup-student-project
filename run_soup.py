"""Uniform soups of the final checkpoints, their output ensembles and the source-val predictions of the ingredients.

For every transfer and seed, and for both soups (the three methods and the three controls, see uda/soup.py):

1. Ingredients. The three runs must be finished and must have started from the same source-only checkpoint
   (init_checkpoint_sha256 in their config.json, equal to the checkpoint on disk), otherwise the script stops.
   Their logits are computed from the saved checkpoints on the five splits (source val, source test, target
   train, val, test). The source-val predictions are saved for the label-free criterion SRC: the runs of
   stage 2 saved none. This is prediction only, without training and without target data.
2. Soup. The ingredients are averaged with create_method_soup.average_state_dicts (soup script of the
   project partner): every weight and the BatchNorm statistics with equal weight 1/3.
   bn_avg keeps the averaged BatchNorm statistics; bn_target re-estimates them afterwards in one pass over
   the target-train windows (no labels), the weights stay unchanged.
3. Ensemble. The mean of the softmax outputs of the three ingredients; no new weights.

Target data enter only as windows (features.pkl); this script never opens a target evaluation_labels.pkl.
Scoring happens afterwards in make_soup_tables.py.

Results:
    results/soup/<soup_methods|soup_controls>/<bn_avg|bn_target>/seed_<seed>/<transfer>/
        model_last.pt, config.json, predictions/<split>.npz
    results/ensemble/<methods|controls>/seed_<seed>/<transfer>/
        config.json, predictions/<split>.npz
    results/source_val_predictions/<ingredient>/seed_<seed>/<transfer>/
        config.json, source_val.npz
Every predictions file holds y_pred; source_val.npz also holds the logits (soups, ingredients) or the mean
probabilities (ensembles), from which SRC is computed. Finished outputs are skipped, so an interrupted call
can simply be repeated.

Examples:
    python run_soup.py --all
    python run_soup.py --transfer PAMAP2_to_RealWorld --seeds 52 --soups soup_methods
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from create_method_soup import average_state_dicts
from src.training.experiments import TRANSFERS, Transfer
from src.training.model import SelfHARCNN
from src.training.train_transfer import ROOT
from src.training.utils import write_json
from uda.data import check_output_dir, get_transfer, load_label_map, load_labeled, load_unlabeled, sha256
from uda.evaluation import SPLITS
from uda.runs import SEEDS, clean_commit, environment, is_complete, source_only_checkpoint
from uda.soup import BN_BATCH_SIZE, BN_VARIANTS, ENSEMBLES, SOUPS, logits, recompute_bn, softmax

RESULTS = ROOT / "results"
PREDICTION_SPLITS = ("source_val", *SPLITS)
AVERAGING_AUTHOR = "create_method_soup.py by the project partner (Denis Petrov), averaging step average_state_dicts"


def soup_dir(root: Path, soup: str, bn: str, seed: int, transfer: Transfer) -> Path:
    return root / "soup" / soup / bn / f"seed_{seed}" / transfer.name


def ensemble_dir(root: Path, soup: str, seed: int, transfer: Transfer) -> Path:
    return root / "ensemble" / ENSEMBLES[soup] / f"seed_{seed}" / transfer.name


def source_val_dir(root: Path, ingredient: str, seed: int, transfer: Transfer) -> Path:
    return root / "source_val_predictions" / ingredient / f"seed_{seed}" / transfer.name


def has_config(folder: Path) -> bool:
    """Every output writes its config.json last."""
    try:
        json.loads((folder / "config.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    return True


def finished(root: Path, soup: str, seed: int, transfer: Transfer) -> bool:
    folders = [soup_dir(root, soup, bn, seed, transfer) for bn in BN_VARIANTS]
    folders += [ensemble_dir(root, soup, seed, transfer)]
    folders += [source_val_dir(root, ingredient, seed, transfer) for ingredient in SOUPS[soup]]
    return all(has_config(folder) for folder in folders) and all(
        (folder / "model_last.pt").is_file() for folder in folders[:len(BN_VARIANTS)])


def ingredient_info(ingredient: str, seed: int, transfer: Transfer, init_hash: str) -> tuple[Path, dict]:
    """Folder and provenance of a finished ingredient run; stops if it started from another checkpoint."""
    folder = RESULTS / ingredient / f"seed_{seed}" / transfer.name
    if not is_complete(folder, SPLITS):
        raise FileNotFoundError(f"Ingredient run not finished: {folder}")
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    if config["seed"] != seed or config["transfer"] != transfer.name:
        raise ValueError(f"{folder}: config.json names seed {config['seed']} / {config['transfer']}")
    if config["init_checkpoint_sha256"] != init_hash:
        raise ValueError(f"{folder} started from another checkpoint ({config['init_checkpoint_sha256']}, expected {init_hash})")
    checkpoint = folder / "model_last.pt"
    return folder, {"variant": ingredient, "run_dir": str(folder.relative_to(ROOT)),
                    "checkpoint": str(checkpoint.relative_to(ROOT)), "checkpoint_sha256": sha256(checkpoint),
                    "init_checkpoint_sha256": config["init_checkpoint_sha256"],
                    "repository_commit": config.get("repository_commit"), "device": config["device"],
                    "train_seconds": config.get("train_seconds")}


def build_model(state: dict, num_classes: int) -> SelfHARCNN:
    model = SelfHARCNN((150, 18), num_classes)
    model.load_state_dict(state, strict=True)
    return model.eval()


def split_windows(source_x: dict, target_x: dict) -> dict[str, np.ndarray]:
    return {split: (source_x if split.startswith("source") else target_x)[split.split("_")[1]] for split in PREDICTION_SPLITS}


def save_predictions(folder: Path, outputs: dict[str, np.ndarray], kind: str) -> None:
    """y_pred for every split; source_val.npz also keeps the logits or probabilities for SRC."""
    (folder / "predictions").mkdir(parents=True, exist_ok=True)
    for split, values in outputs.items():
        extra = {kind: values.astype(np.float32)} if split == "source_val" else {}
        np.savez_compressed(folder / "predictions" / f"{split}.npz", y_pred=values.argmax(1), **extra)


def common_config(transfer: Transfer, seed: int, label_map: dict, checkpoint: Path, init_hash: str) -> dict:
    return {"transfer": transfer.name, "family": transfer.family, "source": transfer.source, "target": transfer.target,
            "label_map": label_map, "seed": seed,
            "init_checkpoint": str(checkpoint.relative_to(ROOT)), "init_checkpoint_sha256": init_hash}


def run_one(transfer: Transfer, seed: int, soup: str, root: Path, device: torch.device, commit: str, overwrite: bool) -> None:
    if finished(root, soup, seed, transfer) and not overwrite:
        print(f"skip {soup} seed {seed} {transfer.name}: finished", flush=True)
        return
    print(f"=== {soup}, seed {seed}: {transfer.name}", flush=True)
    started = datetime.now(timezone.utc)
    label_map = load_label_map(transfer)
    num_classes = len(label_map)
    checkpoint = source_only_checkpoint(transfer, seed)
    init_hash = sha256(checkpoint)
    source_x, _, source_meta = load_labeled(transfer.family, transfer.source)  # source labels are not used here
    if source_meta["label_map"] != label_map:
        raise ValueError(f"Source label map does not match the checkpoint: {transfer.name}")
    target_x = load_unlabeled(transfer.family, transfer.target)  # windows only, no labels
    windows = split_windows(source_x, target_x)
    base = {**common_config(transfer, seed, label_map, checkpoint, init_hash), "target_labels_used": False}
    env = {**environment(device), "repository_commit": commit}

    # 1. Ingredients: provenance, start checkpoint, logits from the saved checkpoints.
    infos, states, outputs, agreement = [], [], [], []
    tic = time.perf_counter()
    for ingredient in SOUPS[soup]:
        folder, info = ingredient_info(ingredient, seed, transfer, init_hash)
        state = torch.load(folder / "model_last.pt", map_location="cpu", weights_only=True)
        model = build_model(state, num_classes).to(device)
        output = {split: logits(model, windows[split], device) for split in PREDICTION_SPLITS}
        # The runs predicted on their own device; the argmax recomputed here should match their saved predictions.
        info["prediction_agreement_with_saved"] = {}
        for split in SPLITS:
            with np.load(folder / "predictions" / f"{split}.npz") as stored:
                info["prediction_agreement_with_saved"][split] = float((stored["y_pred"] == output[split].argmax(1)).mean())
        agreement.append(min(info["prediction_agreement_with_saved"].values()))
        infos.append(info)
        states.append(state)
        outputs.append(output)
    ingredient_seconds = time.perf_counter() - tic

    for info, output in zip(infos, outputs):
        out = source_val_dir(root, info["variant"], seed, transfer)
        (out / "config.json").unlink(missing_ok=True)
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / "source_val.npz", y_pred=output["source_val"].argmax(1),
                            logits=output["source_val"].astype(np.float32))
        write_json(out / "config.json", {
            "kind": "source-val predictions of a stage-2 run, computed afterwards from its checkpoint (decision D5)",
            "variant": info["variant"], **base, "run_dir": info["run_dir"], "checkpoint": info["checkpoint"],
            "checkpoint_sha256": info["checkpoint_sha256"], "run_repository_commit": info["repository_commit"],
            "procedure": "prediction only: eval mode, no gradients, batches of 512; no training, no target data",
            "used_for": "label-free criterion SRC (source risk on source validation)",
            "saved": "source_val.npz: y_pred and logits (float32)", **env,
            "finished_utc": datetime.now(timezone.utc).isoformat()})

    # 2. Soup: uniform average of the ingredients, then the two BatchNorm variants.
    tic = time.perf_counter()
    soup_state = average_state_dicts(states)
    average_seconds = time.perf_counter() - tic
    counters = {key: sorted({int(state[key]) for state in states}) for key in soup_state if key.endswith("num_batches_tracked")}
    for bn in BN_VARIANTS:
        out = soup_dir(root, soup, bn, seed, transfer)
        (out / "config.json").unlink(missing_ok=True)
        out.mkdir(parents=True, exist_ok=True)
        model = build_model({k: v.clone() for k, v in soup_state.items()}, num_classes).to(device)
        bn_seconds, bn_details = 0.0, {"batchnorm": "running statistics averaged like the weights (1/3 each)"}
        if bn == "bn_target":
            tic = time.perf_counter()
            n_batches = recompute_bn(model, target_x["train"], seed, device)
            bn_seconds = time.perf_counter() - tic
            bn_details = {"batchnorm": "running statistics re-estimated on target train after averaging; weights unchanged",
                          "bn_recompute": {"data": "target train windows (features.pkl), no labels",
                                           "procedure": "torch.optim.swa_utils.update_bn: reset running_mean/running_var/"
                                                        "num_batches_tracked, momentum None (cumulative average over the "
                                                        "batches), one pass in train mode without gradients (dropout active)",
                                           "batch_size": BN_BATCH_SIZE, "n_batches": n_batches,
                                           "n_windows": len(target_x["train"]),
                                           "order": f"random permutation from torch.Generator().manual_seed({seed})",
                                           "dropout_seed": seed}}
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, out / "model_last.pt")
        tic = time.perf_counter()
        save_predictions(out, {split: logits(model, windows[split], device) for split in PREDICTION_SPLITS}, "logits")
        predict_seconds = time.perf_counter() - tic
        write_json(out / "config.json", {
            "method": "uniform_soup", "soup": soup, "bn_variant": bn, **base,
            "ingredients": infos, "ingredient_weights": [1 / len(infos)] * len(infos),
            "averaging": {"code": AVERAGING_AUTHOR,
                          "rule": "arithmetic mean of every floating-point state_dict entry in float32 (weights, biases, "
                                  "BatchNorm affine parameters and running statistics); integer entries "
                                  "(num_batches_tracked) by integer division, dtype kept",
                          "num_batches_tracked_of_ingredients": counters},
            **bn_details, "averaged_part": "HAR predictor only (features + classifier); no discriminator",
            "target_input": "none" if bn == "bn_avg" else "target train windows only (features.pkl), no labels",
            "saved_predictions": list(PREDICTION_SPLITS),
            "seconds": {"load_ingredients_and_logits": round(ingredient_seconds, 3), "average": round(average_seconds, 4),
                        "bn_recompute": round(bn_seconds, 3), "predict": round(predict_seconds, 3)},
            **env, "started_utc": started.isoformat(), "finished_utc": datetime.now(timezone.utc).isoformat()})

    # 3. Output ensemble: mean softmax of the three ingredients.
    out = ensemble_dir(root, soup, seed, transfer)
    (out / "config.json").unlink(missing_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    tic = time.perf_counter()
    probabilities = {split: np.mean([softmax(output[split]) for output in outputs], axis=0) for split in PREDICTION_SPLITS}
    save_predictions(out, probabilities, "prob")
    write_json(out / "config.json", {
        "method": "output_ensemble", "ensemble": ENSEMBLES[soup], "belongs_to_soup": soup, **base,
        "ingredients": infos, "rule": "mean of the softmax outputs of the ingredients (eval mode, each with its own "
                                      "BatchNorm statistics), prediction = argmax",
        "target_input": "none", "saved_predictions": list(PREDICTION_SPLITS),
        "min_prediction_agreement_with_saved": min(agreement),
        "seconds": {"load_ingredients_and_logits": round(ingredient_seconds, 3), "combine": round(time.perf_counter() - tic, 3)},
        **env, "started_utc": started.isoformat(), "finished_utc": datetime.now(timezone.utc).isoformat()})
    if sha256(checkpoint) != init_hash:
        raise RuntimeError(f"The source-only checkpoint changed: {checkpoint}")
    print(f"saved {soup} seed {seed} {transfer.name} (ingredients {ingredient_seconds:.1f}s, "
          f"min agreement with saved predictions {min(agreement):.4f})", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--transfer", nargs="+", choices=[t.name for t in TRANSFERS])
    which.add_argument("--all", action="store_true", help="all 12 transfers")
    parser.add_argument("--seeds", type=int, nargs="+", choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--soups", nargs="+", choices=list(SOUPS), default=list(SOUPS))
    parser.add_argument("--output-dir", type=Path, help="default: results/")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--overwrite", action="store_true", help="recompute even if finished outputs exist")
    args = parser.parse_args()
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is not available")
    root = check_output_dir(args.output_dir or RESULTS)
    commit = clean_commit()
    device = torch.device(args.device)
    transfers = TRANSFERS if args.all else [get_transfer(name) for name in args.transfer]
    for seed in args.seeds:
        for transfer in transfers:
            for soup in args.soups:
                run_one(transfer, seed, soup, root, device, commit, args.overwrite)
    if clean_commit() != commit:
        raise RuntimeError("The code commit changed while the soups were computed (E34)")


if __name__ == "__main__":
    main()
