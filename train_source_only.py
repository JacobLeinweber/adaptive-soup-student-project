"""Source-only models for the seeds 100, 445, 545 and 881, following run_transfer step by step.

run_transfer (src/training/train_transfer.py) is never called, because it writes into models/.
The steps below are the same and in the same order, built from the supervisor's own parts; only the
seed, the device (MPS) and the output folder differ. The mean loss per epoch is recorded in addition,
which uses no random numbers. Seed 52 keeps the supplied checkpoint, so it needs --output-dir (test only).

Target data enter only as windows (features.pkl), for the predictions after training; this script
never opens a target evaluation_labels.pkl. Scoring happens afterwards in make_seed_tables.py.

Examples:
    python train_source_only.py --all
    python train_source_only.py --transfer PAMAP2_to_RealWorld --seeds 100
    python train_source_only.py --transfer PAMAP2_to_RealWorld --seeds 52 --output-dir /some/test/folder

Finished runs are skipped, so an interrupted call can simply be repeated.
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from src.training.experiments import TRANSFERS, Transfer
from src.training.model import SelfHARCNN
from src.training.train_transfer import _label_map
from src.training.utils import SEED, seed_everything, write_json
from uda.data import check_output_dir, get_transfer, load_label_map, load_labeled, load_unlabeled
from uda.evaluation import predict
from uda.runs import (BASELINE, SEEDS, SOURCE_ONLY_ROOT, SOURCE_ONLY_SPLITS, clean_commit, environment, is_complete,
                      write_train_log)


def train_one(transfer: Transfer, seed: int, output_root: Path, device: torch.device, overwrite: bool) -> None:
    out = output_root / f"seed_{seed}" / transfer.family / transfer.name
    if is_complete(out, SOURCE_ONLY_SPLITS) and not overwrite:
        print(f"skip seed {seed} {transfer.name}: finished run in {out}", flush=True)
        return
    commit = clean_commit()
    print(f"=== source only, seed {seed}: {transfer.name}", flush=True)
    (out / "config.json").unlink(missing_ok=True)
    started = datetime.now(timezone.utc)

    # Same order as run_transfer, because the random calls decide the initial weights: seed, data, model.
    seed_everything(seed)
    source_x, source_y, source_meta = load_labeled(transfer.family, transfer.source)
    target_x = load_unlabeled(transfer.family, transfer.target)  # windows only, no labels
    label_map = _label_map(source_meta)
    if label_map != load_label_map(transfer):
        raise ValueError(f"Source label map does not match the supplied checkpoint: {transfer.name}")
    x, y = source_x["train"].astype(np.float32), source_y["train"].astype(np.int64)
    model = SelfHARCNN((x.shape[1], x.shape[2]), max(len(label_map), int(y.max() + 1))).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=BASELINE["lr"], weight_decay=BASELINE["weight_decay"])
    loss_fn = torch.nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed)

    model.train()
    history = []
    tic = time.perf_counter()
    for epoch in range(1, BASELINE["epochs"] + 1):
        epoch_started = time.perf_counter()
        loss_sum = torch.zeros((), device=device)
        steps = 0
        for order in torch.randperm(len(x), generator=generator).split(BASELINE["batch_size"]):
            xb = torch.as_tensor(x[order.numpy()], dtype=torch.float32, device=device)
            yb = torch.as_tensor(y[order.numpy()], dtype=torch.long, device=device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            loss_sum += loss.detach()
            steps += 1
        history.append({"epoch": epoch, "source_ce": loss_sum.item() / steps, "steps": steps,
                        "seconds": time.perf_counter() - epoch_started})
        print(f"epoch {epoch:2d}  source_ce {history[-1]['source_ce']:.5f}  ({history[-1]['seconds']:.1f}s)", flush=True)
    train_seconds = time.perf_counter() - tic

    (out / "predictions").mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, out / "model_last.pt")
    write_train_log(out / "train_log.csv", history)
    for split in SOURCE_ONLY_SPLITS:
        domain, part = split.split("_")
        windows = (source_x if domain == "source" else target_x)[part]
        np.savez_compressed(out / "predictions" / f"{split}.npz", y_pred=predict(model, windows, device))
    write_json(out / "config.json", {
        "method": "source_only", "transfer": transfer.name, "family": transfer.family,
        "source": transfer.source, "target": transfer.target, "label_map": label_map, "seed": seed,
        "procedure": "steps of src/training/train_transfer.run_transfer, not called",
        "optimizer": "Adam (torch.optim.Adam, L2 weight decay)", **BASELINE,
        "batch_order": "new torch.randperm per epoch, generator seeded with the seed, as in run_transfer",
        "steps_per_epoch": history[0]["steps"], "n_source_train": len(x), "model_selection": "last epoch",
        "target_input": "target windows only (features.pkl), for predictions after training, no labels",
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
    parser.add_argument("--seeds", type=int, nargs="+", choices=SEEDS, default=[s for s in SEEDS if s != SEED])
    parser.add_argument("--output-dir", type=Path, help=f"default: {SOURCE_ONLY_ROOT}")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--overwrite", action="store_true", help="retrain even if a finished run exists")
    args = parser.parse_args()
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is not available")
    if SEED in args.seeds and args.output_dir is None:
        parser.error(f"Seed {SEED} uses the supplied checkpoint; a test run needs --output-dir")
    output_root = check_output_dir(args.output_dir or SOURCE_ONLY_ROOT)
    transfers = TRANSFERS if args.all else [get_transfer(args.transfer)]
    for seed in args.seeds:
        for transfer in transfers:
            train_one(transfer, seed, output_root, torch.device(args.device), args.overwrite)


if __name__ == "__main__":
    main()
