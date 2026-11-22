"""Write the configuration of the hyperparameter search on the cross-dataset transfers (run once, then commit).

The file fixes everything that the search must not draw anew on another machine:
    - the search space per method, taken from AdaTime's sweep configuration (sources below),
    - the combinations: 0 = Stage-2 setting, 1 = AdaTime hparams.py for HAR, 2..99 = random draws,
    - the input files with size and SHA-256 (checked by run_hp_search.py before it starts),
    - the few-shot target windows (5 per class from target val), drawn once with a fixed seed.

AdaTime: https://github.com/emadeldeen24/AdaTime, commit 2d9be6bd0542e150518afaf8aa0c6ea0823b4a3d (branch adatime_v2).

Example: python make_hp_search_config.py      # writes configs/hp_search_cross_dataset.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import run_dann
import run_deep_coral
import run_dsan
from src.training.experiments import TRANSFERS
from src.training.train_transfer import ROOT
from src.training.utils import SEED
from uda.data import checkpoint_dir, load_label_map, load_labeled, sha256
from uda.hp_search import ALIGNMENT_LOSS, METHODS, few_shot_indices
from uda.runs import BASELINE

FAMILY = "cross_dataset"
CONFIG = ROOT / "configs" / "hp_search_cross_dataset.json"
INPUTS_ZIP = "hp_search_inputs_cross_dataset_seed52.zip"
N_COMBINATIONS = 100   # ids 0..99: two reference combinations and 98 random draws
SAMPLING_SEED = SEED   # random draws: numpy default_rng([SAMPLING_SEED, index of the method in METHODS])
FEW_SHOT_PER_CLASS = 5
FEW_SHOT_SEED = SEED
DEV_SEED = SEED

ADATIME = {
    "repository": "https://github.com/emadeldeen24/AdaTime",
    "commit": "2d9be6bd0542e150518afaf8aa0c6ea0823b4a3d",
    "sweep_space": "configs/sweep_params.py:11-92 (sweep_alg_hparams); DANN 12-16, Deep_Coral 37-41, DSAN 65-69",
    "sweep_uses_only_alg_space": "trainers/sweep.py:51-56: the wandb sweep gets sweep_alg_hparams[da_method] only",
    "unused_train_space": "configs/sweep_params.py:1-10 (sweep_train_hparams: num_epochs, batch_size, learning_rate, "
                          "disc_lr, weight_decay, step_size, gamma, optimizer) is defined but not used by the sweep",
    "strategy_and_metric": "main_sweep.py:32 (hp_search_strategy 'random'), main_sweep.py:34 (metric_to_minimize 'src_risk')",
    "one_setting_for_all_scenarios": "trainers/sweep.py:71-96: every sweep run trains all scenarios of the dataset "
                                     "with the same hyperparameters",
    "hparams_har": "configs/hparams.py:12-18 (train_params: num_epochs 40, batch_size 32, weight_decay 1e-4, "
                   "step_size 50, lr_decay 0.5), 42-47 (DANN), 56-62 (DSAN), 71-76 (Deep_Coral)",
    "losses_read": "algorithms/algorithms.py:200-201 (Deep_Coral: coral_wt, src_cls_loss_wt), 359-360 (DANN: "
                   "src_cls_loss_wt, domain_loss_wt), 624-625 (DSAN: domain_loss_wt, src_cls_loss_wt)",
    "discriminator_optimizer": "algorithms/algorithms.py:306-310: same learning_rate and weight_decay as the network, betas (0.5, 0.99)",
    "few_shot": "trainers/abstract_trainer.py:147-148 and dataloader/dataloader.py:106-128: 5 windows per class "
                "from the target test split, drawn with torch.randperm",
}
LEARNING_RATES = [1e-2, 5e-3, 1e-3, 5e-4]
# Search space per method in the names of our code, with AdaTime's name and the line of sweep_params.py.
SPACE = {
    "deep_coral": {
        "learning_rate": {"values": LEARNING_RATES, "adatime_name": "learning_rate", "source": "configs/sweep_params.py:38"},
        "source_ce_weight": {"distribution": "uniform", "min": 0.1, "max": 10.0, "adatime_name": "src_cls_loss_wt",
                             "source": "configs/sweep_params.py:39"},
        "alignment_weight": {"distribution": "uniform", "min": 0.1, "max": 10.0, "adatime_name": "coral_wt",
                             "source": "configs/sweep_params.py:40"},
    },
    "dann": {
        "learning_rate": {"values": LEARNING_RATES, "adatime_name": "learning_rate", "source": "configs/sweep_params.py:13"},
        "source_ce_weight": {"distribution": "uniform", "min": 0.1, "max": 10.0, "adatime_name": "src_cls_loss_wt",
                             "source": "configs/sweep_params.py:14"},
        "alignment_weight": {"distribution": "uniform", "min": 0.01, "max": 10.0, "adatime_name": "domain_loss_wt",
                             "source": "configs/sweep_params.py:15"},
    },
    "dsan": {
        "learning_rate": {"values": LEARNING_RATES, "adatime_name": "learning_rate", "source": "configs/sweep_params.py:66"},
        "source_ce_weight": {"distribution": "uniform", "min": 0.1, "max": 10.0, "adatime_name": "src_cls_loss_wt",
                             "source": "configs/sweep_params.py:67"},
        "alignment_weight": {"distribution": "uniform", "min": 0.01, "max": 10.0, "adatime_name": "domain_loss_wt",
                             "source": "configs/sweep_params.py:68 (given there as mmd_wt)",
                             "note": "AdaTime's sweep varies mmd_wt for DSAN, but its DSAN loss reads domain_loss_wt "
                                     "(algorithms/algorithms.py:624; fixed at 1.59 in configs/hparams.py:60), so in "
                                     "AdaTime the LMMD weight is never searched. The range given for mmd_wt is applied "
                                     "to the weight the loss uses (our lmmd_weight = AdaTime domain_loss_wt)."},
    },
}
# AdaTime weights for HAR (configs/hparams.py), taken from the Stage-2 run scripts; learning rates of hparams.py.
REFERENCE = {
    "deep_coral": {"source_ce_weight": run_deep_coral.SOURCE_CE_WEIGHT, "alignment_weight": run_deep_coral.CORAL_WEIGHT,
                   "adatime_learning_rate": 5e-4, "hparams": "configs/hparams.py:71-76"},
    "dann": {"source_ce_weight": run_dann.SOURCE_CE_WEIGHT, "alignment_weight": run_dann.DOMAIN_WEIGHT,
             "adatime_learning_rate": 1e-3, "hparams": "configs/hparams.py:42-47"},
    "dsan": {"source_ce_weight": run_dsan.SOURCE_CE_WEIGHT, "alignment_weight": run_dsan.LMMD_WEIGHT,
             "adatime_learning_rate": 1e-3, "hparams": "configs/hparams.py:56-62 (domain_loss_wt 1.59; mmd_wt 2.087 is not read)"},
}
DEVIATIONS = [
    "Epochs fixed at 30 (Stage 2) instead of AdaTime's 40 (configs/hparams.py:13): the runs continue training the "
    "source-only model instead of training from scratch. AdaTime's sweep does not vary the epochs or the batch size "
    "either: sweep_train_hparams (configs/sweep_params.py:1-10) is not passed to the sweep (trainers/sweep.py:55).",
    "Batch size fixed at 32 (Stage 2 = AdaTime HAR, configs/hparams.py:14); weight decay fixed at 1e-4 "
    "(AdaTime HAR alg_hparams, not part of AdaTime's alg sweep).",
    "No learning-rate scheduler. AdaTime builds StepLR(step_size 50, gamma 0.5), which never steps within 40 epochs.",
    "DSAN: the range AdaTime gives for mmd_wt is applied to domain_loss_wt, the weight the DSAN loss reads.",
    "Combination 0 is the Stage-2 setting: AdaTime's loss weights with learning rate 3e-4 (the learning rate of the "
    "supplied source-only training) instead of AdaTime's 5e-4 (Deep CORAL) or 1e-3 (DANN, DSAN). Stage 2 ran with "
    "3e-4. Combination 1 is AdaTime hparams.py including its learning rate.",
    "Random draws use numpy instead of wandb; as in wandb's random search, every parameter is drawn independently "
    "(values: uniform over the list; uniform: U(min, max)).",
    "SRC is computed on source val (AdaTime: source test, trainers/abstract_trainer.py:246), so that source test "
    "stays untouched by any selection.",
    "FST uses 5 windows per class from target val (AdaTime: from target test, trainers/abstract_trainer.py:147), "
    "drawn once with a fixed seed and the same for every run; target test is never used for a selection.",
    "One seed (52) and one setting for all six transfers, as AdaTime uses one setting for all scenarios of a dataset.",
]


def random_combinations(method: str) -> list[dict]:
    rng = np.random.default_rng([SAMPLING_SEED, METHODS.index(method)])
    space = SPACE[method]
    rows = []
    for combination_id in range(2, N_COMBINATIONS):
        lr = space["learning_rate"]["values"][int(rng.integers(len(space["learning_rate"]["values"])))]
        source_weight = float(rng.uniform(space["source_ce_weight"]["min"], space["source_ce_weight"]["max"]))
        alignment_weight = float(rng.uniform(space["alignment_weight"]["min"], space["alignment_weight"]["max"]))
        rows.append({"id": combination_id, "origin": "random", "learning_rate": lr,
                     "source_ce_weight": source_weight, "alignment_weight": alignment_weight})
    return rows


def all_combinations(method: str) -> list[dict]:
    reference = REFERENCE[method]
    weights = {"source_ce_weight": reference["source_ce_weight"], "alignment_weight": reference["alignment_weight"]}
    return [{"id": 0, "origin": "stage2", "learning_rate": BASELINE["lr"], **weights},
            {"id": 1, "origin": "adatime_hparams", "learning_rate": reference["adatime_learning_rate"], **weights},
            *random_combinations(method)]


def inputs(transfers) -> list[dict]:
    """Every file the search reads, with size and SHA-256. 'git-lfs' files are not in the fork and come with the input ZIP."""
    datasets = sorted({name for t in transfers for name in (t.source, t.target)})
    paths = []
    for dataset in datasets:
        folder = Path("data") / "prepared" / FAMILY / dataset
        paths += [folder / "features.pkl", folder / "evaluation_labels.pkl"]
    paths += [Path("data") / "prepared" / FAMILY / "metadata" / f"{d}.json" for d in datasets]
    for transfer in transfers:
        folder = checkpoint_dir(transfer).relative_to(ROOT)
        paths += [folder / "model_last.pt", folder / "config.json"]
    entries = []
    for path in paths:
        stored = "git-lfs" if path.suffix in (".pkl", ".pt") else "git"
        entry = {"path": path.as_posix(), "stored": stored}
        if stored == "git-lfs":  # text files tracked in git may get other line endings on checkout, so only these are hashed
            entry.update({"bytes": (ROOT / path).stat().st_size, "sha256": sha256(ROOT / path)})
        entries.append(entry)
    return entries


def few_shot(transfers) -> dict:
    """Few-shot target windows per target dataset: indices into target val and their labels."""
    selection = {}
    for dataset in sorted({t.target for t in transfers}):
        maps = {json.dumps(load_label_map(t), sort_keys=True) for t in transfers if dataset in (t.source, t.target)}
        _, labels, meta = load_labeled(FAMILY, dataset)
        if len(maps) != 1 or json.dumps(meta["label_map"], sort_keys=True) not in maps:
            raise SystemExit(f"Label maps of {dataset} and its checkpoints differ")
        num_classes = len(meta["label_map"])
        indices, taken = few_shot_indices(labels["val"], num_classes, FEW_SHOT_PER_CLASS, FEW_SHOT_SEED)
        counts = np.bincount(np.asarray(labels["val"], dtype=np.int64), minlength=num_classes)
        selection[dataset] = {"n_target_val": int(len(labels["val"])), "class_counts_target_val": counts.tolist(),
                              "taken_per_class": taken, "indices": indices.tolist(),
                              "labels": np.asarray(labels["val"])[indices].astype(int).tolist()}
    return selection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=CONFIG)
    args = parser.parse_args()
    transfers = [t for t in TRANSFERS if t.family == FAMILY]
    config = {
        "name": "hp_search_cross_dataset_seed52",
        "description": "Pre-experiment: random hyperparameter search following AdaTime's sweep, cross-dataset "
                       "transfers, seed 52, every run starting from the supplied seed-52 source-only checkpoint.",
        "family": FAMILY,
        "transfers": [t.name for t in transfers],
        "methods": list(METHODS),
        "seed": SEED,
        "start_model": "supplied source-only checkpoint of the transfer: models/cross_dataset/<transfer>/model_last.pt",
        "fixed": {"epochs": BASELINE["epochs"], "batch_size": BASELINE["batch_size"],
                  "weight_decay": BASELINE["weight_decay"], "optimizer": "Adam (L2 weight decay, default betas)",
                  "model_selection": "model after the last epoch",
                  "pipeline": "as Stage 2: supervisor data IO and splits, separate source and target passes in train "
                              "mode, epoch = source batches, target cycled"},
        "order": "combination by combination: combination k for all transfers x methods, then k + 1",
        "sampling": {"strategy": "random", "n_combinations": N_COMBINATIONS, "reference_ids": [0, 1],
                     "random_ids": [2, N_COMBINATIONS - 1], "seed": SAMPLING_SEED,
                     "generator": "numpy default_rng([seed, index of the method in methods]); per combination in this "
                                  "order: learning rate, source weight, alignment weight",
                     "same_combinations_for_all_transfers": True},
        "space": SPACE,
        "alignment_loss": ALIGNMENT_LOSS,
        "reference_combinations": {
            "0": "stage2: AdaTime loss weights for HAR with learning rate 3e-4, exactly the Stage-2 setting "
                 "(device check against the Stage-2 runs of seed 52)",
            "1": "adatime_hparams: AdaTime configs/hparams.py for HAR including its learning rate",
            "weights": REFERENCE},
        "adatime": ADATIME,
        "deviations_from_adatime": DEVIATIONS,
        "few_shot": {"split": "target_val", "per_class": FEW_SHOT_PER_CLASS, "seed": FEW_SHOT_SEED,
                     "note": "uses target labels (few-shot); the same windows for every run with this target",
                     "selection": few_shot(transfers)},
        "dev": {"seed": DEV_SEED, "implementation": "uda.criteria.dev_weights and dev_value, as uda.criteria.dev_risk",
                "domain_data": "96-dim eval-mode features: source train (label 1) vs target train (label 0)",
                "validation": "source val; errors = cross-entropy per window",
                "raw_quantities": "outputs.npz of every run: source_val_ce and dev_weights per source-val window"},
        "inputs_zip": INPUTS_ZIP,
        "inputs": inputs(transfers),
        "combinations": {method: all_combinations(method) for method in METHODS},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, indent=1) + "\n", encoding="utf-8")
    print(f"Saved {args.output.relative_to(ROOT)}: {len(METHODS)} methods x {N_COMBINATIONS} combinations, "
          f"{len(config['inputs'])} input files")


if __name__ == "__main__":
    main()
