"""Stage 2 with a shared starting point (T11): every seed starts from the supplied seed-52 source-only checkpoint.

Runs seeds 100, 445, 545, 881 x 12 transfers x {Deep CORAL, DANN, DSAN}, each method with its own control
(lambda = 0, target windows pass through the model, the method's own source weight): 288 runs, 30 epochs,
model after the last epoch, everything else as in stage 2. The seed now only steers the adaptation
(source and target batch order, dropout, the initialization of the DANN discriminator). Seed 52 is not
rerun: with the shared start its runs are the stage-2 runs in results/<variant>/seed_52/.

Start detached, from the repository folder (Python of the project environment):
    mkdir -p logs/shared_init
    nohup caffeinate -is python run_shared_init.py --device mps >> logs/shared_init/run.log 2>&1 < /dev/null &

Results: results/shared_init/<variant>/seed_<seed>/<transfer>/ with the same files as stage 2. Finished runs
are skipped and every run is written to a temporary folder that is renamed at the end, so the same command
continues after an interruption. A failing run is logged in logs/shared_init/failures.log and skipped; after
three failures in a row the script stops. logs/shared_init/run.pid holds the PID while the script runs; a
second start is refused while that process is alive.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from datetime import datetime

import torch

import run_dann
import run_deep_coral
import run_dsan
from src.training.experiments import TRANSFERS
from src.training.train_transfer import ROOT
from src.training.utils import SEED
from uda.data import check_output_dir
from uda.runs import SEEDS, SHARED_INIT_ROOT

# Each method with its own control: lambda = 0, target windows pass through the model, the method's source weight.
PLAN = ((run_deep_coral, ("deep_coral", "control_target_pass")),
        (run_dann, ("dann", "control")),
        (run_dsan, ("dsan", "control")))
NEW_SEEDS = tuple(seed for seed in SEEDS if seed != SEED)
MAX_CONSECUTIVE_FAILURES = 3
LOG_DIR = ROOT / "logs" / "shared_init"


def other_instance(pid_file) -> int | None:
    """PID of a still running earlier start, if any."""
    try:
        pid = int(pid_file.read_text().strip())
    except (FileNotFoundError, ValueError):
        return None
    if pid == os.getpid():
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return None
    except PermissionError:
        return pid
    return pid


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", choices=NEW_SEEDS, default=list(NEW_SEEDS))
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is not available")
    device = torch.device(args.device)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    pid_file = LOG_DIR / "run.pid"
    running = other_instance(pid_file)
    if running is not None:
        print(f"run_shared_init.py is already running (PID {running}); not starting a second process.", flush=True)
        return 1
    pid_file.write_text(f"{os.getpid()}\n")
    results_root = check_output_dir(SHARED_INIT_ROOT)
    runs = [(module, key, seed, transfer) for seed in args.seeds for module, keys in PLAN for key in keys
            for transfer in TRANSFERS]
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} start: {len(runs)} runs, seeds {args.seeds}, device {device}, "
          f"PID {os.getpid()}, results {results_root}", flush=True)
    consecutive = 0
    failed = []
    try:
        for index, (module, key, seed, transfer) in enumerate(runs, 1):
            mode = module.MODES[key]
            label = f"{mode['folder']} seed {seed} {transfer.name}"
            print(f"[{index}/{len(runs)}] {datetime.now():%Y-%m-%d %H:%M:%S} {label}", flush=True)
            tic = time.perf_counter()
            try:
                module.run_one(transfer, mode, seed, results_root, device, overwrite=False, init="shared")
            except KeyboardInterrupt:
                raise
            except Exception:
                consecutive += 1
                failed.append(label)
                message = traceback.format_exc()
                print(f"FAILED {label}\n{message}", flush=True)
                with (LOG_DIR / "failures.log").open("a", encoding="utf-8") as handle:
                    handle.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} {label}\n{message}\n")
                if consecutive >= MAX_CONSECUTIVE_FAILURES:
                    print(f"STOPPED: {consecutive} runs failed in a row, see {LOG_DIR / 'failures.log'}", flush=True)
                    return 2
                continue
            consecutive = 0
            print(f"[{index}/{len(runs)}] done in {time.perf_counter() - tic:.0f}s wall time", flush=True)
    finally:
        if other_instance(pid_file) is None:
            pid_file.unlink(missing_ok=True)
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} finished: {len(runs)} runs planned, {len(failed)} failed in this "
          f"session{': ' + '; '.join(failed) if failed else ''}", flush=True)
    return 0 if not failed else 3


if __name__ == "__main__":
    sys.exit(main())
