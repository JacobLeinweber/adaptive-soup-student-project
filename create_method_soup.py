"""Create uniform model soups by averaging weights from three specified methods
for each seed and domain shift (transfer) found in the results directory.

Examples:
    # 1. Build main UDA method soup (DANN, DSAN, Deep CORAL)
    python create_method_soup.py --methods dann dsan deep_coral --output-name uda_method_soup

    # 2. Build control method soup (Lambda = 0 target pass controls)
    python create_method_soup.py --methods dann_lambda0_target_pass dsan_lambda0_target_pass lambda0_target_pass --output-name control_method_soup
"""

from __future__ import annotations

import argparse
from pathlib import Path
import torch

from src.training.experiments import TRANSFERS
from uda.runs import SEEDS

DEFAULT_METHODS = ["dann", "dsan", "deep_coral"]


def create_soups(methods: list[str], results_root: Path, output_folder_name: str) -> None:
    print(f"=== Scanning Results Directory: {results_root.resolve()} ===")
    print(f"=== Target Methods: {methods} ===")
    print(f"=== Target Seeds: {SEEDS} ===")
    print(f"=== Total Transfers to Check: {len(TRANSFERS)} ==-\n")
    
    transfer_names = [t.name for t in TRANSFERS]
    success_count = 0
    
    for seed in SEEDS:
        seed_dir_name = f"seed_{seed}"
        
        for transfer_name in transfer_names:
            state_dicts = []
            missing = False
            
            # Check each method for this seed & transfer
            for method in methods:
                pt_path = results_root / method / seed_dir_name / transfer_name / "model_last.pt"
                
                if pt_path.is_file():
                    state_dicts.append(torch.load(pt_path, map_location="cpu"))
                else:
                    # Print why it's missing so you know what's not trained yet
                    print(f"[MISSING] {method} | seed_{seed} | {transfer_name}")
                    print(f"          Checked path: {pt_path}")
                    missing = True
                    break
            
            if missing or len(state_dicts) != len(methods):
                continue
            
            # 2. Average the weights (Uniform Simple Model Soup)
            soup_state_dict = {}
            first_sd = state_dicts[0]
            
            for key in first_sd.keys():
                soup_state_dict[key] = sum(sd[key].float() for sd in state_dicts) / len(state_dicts)
                
            # 3. Save the resulting method soup checkpoint
            soup_out_dir = results_root / output_folder_name / seed_dir_name / transfer_name
            soup_out_dir.mkdir(parents=True, exist_ok=True)
            
            output_file = soup_out_dir / "model_last.pt"
            torch.save(soup_state_dict, output_file)
            print(f"[+] Successfully created soup: {output_file}")
            success_count += 1

    print(f"\n=== Generation finished. Total soups created: {success_count} ===")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--methods", 
        nargs="+", 
        default=DEFAULT_METHODS, 
        help="List of 3 method folder names to combine"
    )
    parser.add_argument(
        "--results-dir", 
        type=Path, 
        default=Path("results"), 
        help="Path to the results directory"
    )
    parser.add_argument(
        "--output-name", 
        type=str, 
        default="uda_method_soup", 
        help="Name of the output subdirectory under results/ to store the soups"
    )
    args = parser.parse_args()
    
    create_soups(args.methods, args.results_dir, args.output_name)


if __name__ == "__main__":
    main()