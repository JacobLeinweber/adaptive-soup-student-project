# Hyperparameter search: start card

Cross-dataset transfers, seed 52, 100 combinations × 18 runs. No new dependencies (environment of `requirements.txt`, PyTorch with CUDA).

1. In your clone of the repository: `git fetch origin --tags`, then `git checkout hp-search-v1` (no uncommitted or untracked `.py` files, otherwise the start check refuses).
2. Put `hp_search_inputs_cross_dataset_seed52.zip` into the repository root and unpack it there, replacing existing files: `python -m zipfile -e hp_search_inputs_cross_dataset_seed52.zip .` (fills `data/prepared/cross_dataset/` and `models/cross_dataset/`).
3. Switch off sleep and hibernation for the night and postpone automatic restarts; the search stops when the computer sleeps.
4. From the repository root: `python run_hp_search.py --device cuda`. It first checks GPU, code, data, checkpoints and disk space (about 2.5 GB free) and stops with a list of problems if anything is missing. On Windows, do not click into the console window (a text selection pauses the program until Esc).
5. If it stops or the computer restarts: run the same command again. Finished runs are skipped.
6. In the morning: stop with Ctrl+C (finished runs are kept), run `python pack_hp_search.py results` and send back the ZIP it prints (the folder `results/hp_search/`, up to about 1.7 GB).

Background: [hp_search_protocol.md](hp_search_protocol.md).
