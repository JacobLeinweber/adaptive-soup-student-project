"""Iterative Model Soup during training across Deep CORAL, DANN, and DSAN methods.

For a given seed and transfer, initializes three models from the same source-only checkpoint
(one each for Deep CORAL, DANN, and DSAN). In each epoch:
1. Trains each model for 1 epoch using its respective loss/algorithm.
2. Creates an arithmetic mean soup of the three model state_dicts (using average_state_dicts).
3. Updates all three models with the new soup weights.
4. Continues to the next epoch until the final epoch, where the final soup is saved as model_last.pt.

Examples:
    python run_iterative_soup.py --transfer PAMAP2_to_RealWorld --seeds 52
    python run_iterative_soup.py --all --seeds 52
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from create_method_soup import average_state_dicts
from src.training.experiments import TRANSFERS, Transfer
from src.training.train_transfer import ROOT
from src.training.utils import write_json
from uda.dann import DISCRIMINATOR_BETAS, DISCRIMINATOR_HIDDEN, DomainDiscriminator, GradientReversal, reversal_factor
from uda.data import check_output_dir, get_transfer, load_label_map, load_labeled, load_source_only_model, load_unlabeled, sha256
from uda.evaluation import SPLITS, predict
from uda.runs import BASELINE, SEEDS, clean_commit, environment, is_complete, source_only_checkpoint, write_train_log
from uda.training import CyclicTargetSampler, extract_features, coral_loss
from uda.dsan import lmmd_loss

ADATIME = {
    "repository": "https://github.com/emadeldeen24/AdaTime",
    "commit": "2d9be6bd0542e150518afaf8aa0c6ea0823b4a3d",
    "description": "Iterative multi-method soup (Deep CORAL + DANN + DSAN) updated per epoch",
}

# Hyperparameters derived from individual method scripts
CORAL_WEIGHT = 4.23035475456397
CORAL_CE_WEIGHT = 0.1013209750429822

DANN_DOMAIN_WEIGHT = 2.943729820531079
DANN_CE_WEIGHT = 5.1390077646202

DSAN_LMMD_WEIGHT = 1.59
DSAN_CE_WEIGHT = 1.8744909939900247


def train_epoch_coral(model, optimizer, x_source, y_source, x_target, coral_weight, source_ce_weight, batch_size, source_generator, target_sampler, device):
    model.train()
    xs = torch.as_tensor(x_source, dtype=torch.float32, device=device)
    ys = torch.as_tensor(y_source, dtype=torch.long, device=device)
    xt = torch.as_tensor(x_target, dtype=torch.float32, device=device)
    
    ce_sum, total_sum, steps = 0.0, 0.0, 0
    for order in torch.randperm(len(xs), generator=source_generator).split(batch_size):
        index = order.to(device)
        source_features = extract_features(model, xs[index])
        ce = torch.nn.functional.cross_entropy(model.classifier(source_features), ys[index])
        loss = source_ce_weight * ce
        
        target_index = target_sampler.take(len(order)).to(device)
        target_features = extract_features(model, xt[target_index])
        coral = coral_loss(source_features, target_features)
        loss = loss + coral_weight * coral
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        ce_sum += ce.item()
        total_sum += loss.item()
        steps += 1
    return total_sum / steps


def train_epoch_dann(model, discriminator, opt_model, opt_disc, x_source, y_source, x_target, domain_weight, source_ce_weight, batch_size, source_generator, target_sampler, epoch, total_epochs, device):
    model.train()
    discriminator.train()
    xs = torch.as_tensor(x_source, dtype=torch.float32, device=device)
    ys = torch.as_tensor(y_source, dtype=torch.long, device=device)
    xt = torch.as_tensor(x_target, dtype=torch.float32, device=device)
    num_batches = max(len(xs), len(xt)) // batch_size
    
    total_sum, steps = 0.0, 0
    for step, order in enumerate(torch.randperm(len(xs), generator=source_generator).split(batch_size)):
        index = order.to(device)
        source_features = extract_features(model, xs[index])
        ce = torch.nn.functional.cross_entropy(model.classifier(source_features), ys[index])
        
        target_index = target_sampler.take(len(order)).to(device)
        target_features = extract_features(model, xt[target_index])
        
        loss = source_ce_weight * ce
        alpha = reversal_factor(step, epoch + 1, num_batches, total_epochs)
        source_domain = discriminator(GradientReversal.apply(source_features, alpha))
        target_domain = discriminator(GradientReversal.apply(target_features, alpha))
        domain_loss = (torch.nn.functional.cross_entropy(source_domain, torch.ones(len(order), dtype=torch.long, device=device))
                       + torch.nn.functional.cross_entropy(target_domain, torch.zeros(len(order), dtype=torch.long, device=device)))
        loss = loss + domain_weight * domain_loss
        
        opt_model.zero_grad()
        opt_disc.zero_grad()
        loss.backward()
        opt_model.step()
        opt_disc.step()
        
        total_sum += loss.item()
        steps += 1
    return total_sum / steps


def train_epoch_dsan(model, optimizer, x_source, y_source, x_target, lmmd_weight, source_ce_weight, batch_size, source_generator, target_sampler, device):
    model.train()
    xs = torch.as_tensor(x_source, dtype=torch.float32, device=device)
    ys = torch.as_tensor(y_source, dtype=torch.long, device=device)
    xt = torch.as_tensor(x_target, dtype=torch.float32, device=device)
    
    total_sum, steps = 0.0, 0
    for order in torch.randperm(len(xs), generator=source_generator).split(batch_size):
        index = order.to(device)
        source_features = extract_features(model, xs[index])
        ce = torch.nn.functional.cross_entropy(model.classifier(source_features), ys[index])
        
        target_index = target_sampler.take(len(order)).to(device)
        target_features = extract_features(model, xt[target_index])
        target_probabilities = torch.nn.functional.softmax(model.classifier(target_features), dim=1)
        
        loss = source_ce_weight * ce
        lmmd, _, _ = lmmd_loss(source_features, target_features, ys[index], target_probabilities)
        loss = loss + lmmd_weight * lmmd
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_sum += loss.item()
        steps += 1
    return total_sum / steps


def run_iterative_soup_one(transfer: Transfer, seed: int, results_root: Path, device: torch.device, overwrite: bool) -> None:
    out = results_root / "iterative_soup" / f"seed_{seed}" / transfer.name
    if is_complete(out, SPLITS) and not overwrite:
        print(f"skip iterative_soup seed {seed} {transfer.name}: finished run in {out}", flush=True)
        return

    commit = clean_commit()
    print(f"=== Iterative Soup (Deep CORAL + DANN + DSAN), seed {seed}: {transfer.name}", flush=True)
    (out / "config.json").unlink(missing_ok=True)
    started = datetime.now(timezone.utc)

    checkpoint = source_only_checkpoint(transfer, seed)
    checkpoint_hash = sha256(checkpoint)
    label_map = load_label_map(transfer)

    source_x, source_y, source_meta = load_labeled(transfer.family, transfer.source)
    if source_meta["label_map"] != label_map:
        raise ValueError(f"Source label map does not match the checkpoint: {transfer.name}")
    target_x = load_unlabeled(transfer.family, transfer.target)

    # Initialize separate models for each method starting from the same source-only checkpoint
    model_coral = load_source_only_model(transfer, checkpoint).to(device)
    model_dann = load_source_only_model(transfer, checkpoint).to(device)
    model_dsan = load_source_only_model(transfer, checkpoint).to(device)

    epochs = BASELINE.get("epochs", 30)
    batch_size = BASELINE.get("batch_size", 32)
    lr = BASELINE.get("lr", 3e-4)
    weight_decay = BASELINE.get("weight_decay", 1e-4)

    opt_coral = torch.optim.Adam(model_coral.parameters(), lr=lr, weight_decay=weight_decay)
    opt_dann = torch.optim.Adam(model_dann.parameters(), lr=lr, weight_decay=weight_decay)
    opt_dsan = torch.optim.Adam(model_dsan.parameters(), lr=lr, weight_decay=weight_decay)

    discriminator = DomainDiscriminator(model_dann.classifier[0].in_features).to(device).train()
    opt_disc = torch.optim.Adam(discriminator.parameters(), lr=lr, weight_decay=weight_decay, betas=DISCRIMINATOR_BETAS)

    source_gen_coral = torch.Generator().manual_seed(seed)
    source_gen_dann = torch.Generator().manual_seed(seed)
    source_gen_dsan = torch.Generator().manual_seed(seed)

    target_sampler_coral = CyclicTargetSampler(len(target_x["train"]), seed)
    target_sampler_dann = CyclicTargetSampler(len(target_x["train"]), seed)
    target_sampler_dsan = CyclicTargetSampler(len(target_x["train"]), seed)

    history = []
    tic = time.perf_counter()

    for epoch in range(epochs):
        # 1. Train each model for 1 epoch
        loss_coral = train_epoch_coral(
            model_coral, opt_coral, source_x["train"], source_y["train"], target_x["train"],
            CORAL_WEIGHT, CORAL_CE_WEIGHT, batch_size, source_gen_coral, target_sampler_coral, device
        )
        loss_dann = train_epoch_dann(
            model_dann, discriminator, opt_dann, opt_disc, source_x["train"], source_y["train"], target_x["train"],
            DANN_DOMAIN_WEIGHT, DANN_CE_WEIGHT, batch_size, source_gen_dann, target_sampler_dann, epoch, epochs, device
        )
        loss_dsan = train_epoch_dsan(
            model_dsan, opt_dsan, source_x["train"], source_y["train"], target_x["train"],
            DSAN_LMMD_WEIGHT, DSAN_CE_WEIGHT, batch_size, source_gen_dsan, target_sampler_dsan, device
        )

        # 2. Create soup of the 3 model state dictionaries
        states = [
            {k: v.cpu() for k, v in model_coral.state_dict().items()},
            {k: v.cpu() for k, v in model_dann.state_dict().items()},
            {k: v.cpu() for k, v in model_dsan.state_dict().items()}
        ]
        soup_state = average_state_dicts(states)

        # 3. Update all three models with the soup weights
        model_coral.load_state_dict(soup_state, strict=True)
        model_dann.load_state_dict(soup_state, strict=True)
        model_dsan.load_state_dict(soup_state, strict=True)

        history.append({
            "epoch": epoch + 1,
            "loss_coral": loss_coral,
            "loss_dann": loss_dann,
            "loss_dsan": loss_dsan,
            "steps": len(source_x["train"]) // batch_size
        })
        print(f"Epoch {epoch + 1}/{epochs} completed and merged via soup.", flush=True)

    train_seconds = time.perf_counter() - tic
    final_model = model_coral  # Shares identical weights with dann and dsan post-merge

    (out / "predictions").mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in final_model.state_dict().items()}, out / "model_last.pt")
    write_train_log(out / "train_log.csv", history)

    for split in SPLITS:
        domain, part = split.split("_")
        x = source_x[part] if domain == "source" else target_x[part]
        np.savez_compressed(out / "predictions" / f"{split}.npz", y_pred=predict(final_model, x, device))

    write_json(out / "config.json", {
        "method": "iterative_method_soup",
        "transfer": transfer.name, "family": transfer.family,
        "source": transfer.source, "target": transfer.target, "label_map": label_map, "seed": seed,
        "init_checkpoint": str(checkpoint.relative_to(ROOT)), "init_checkpoint_sha256": checkpoint_hash,
        "ingredients": ["deep_coral", "dann", "dsan"],
        "merging_frequency": "every epoch",
        "optimizer": "Adam (torch.optim.Adam, L2 weight decay)", **BASELINE,
        "model_selection": "last epoch", "adatime": ADATIME,
        **environment(device), "repository_commit": commit,
        "started_utc": started.isoformat(), "finished_utc": datetime.now(timezone.utc).isoformat(),
        "train_seconds": round(train_seconds, 1),
    })
    print(f"saved iterative soup {out} ({train_seconds:.0f}s training)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--transfer", nargs="+", choices=[t.name for t in TRANSFERS])
    which.add_argument("--all", action="store_true", help="all 12 transfers")
    parser.add_argument("--seeds", type=int, nargs="+", choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--output-dir", type=Path, help="default: results/")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--overwrite", action="store_true", help="retrain even if a finished run exists")
    args = parser.parse_args()

    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is not available")

    results_root = check_output_dir(args.output_dir or ROOT / "results")
    transfers = TRANSFERS if args.all else [get_transfer(name) for name in args.transfer]

    for seed in args.seeds:
        for transfer in transfers:
            run_iterative_soup_one(transfer, seed, results_root, torch.device(args.device), args.overwrite)


if __name__ == "__main__":
    main()