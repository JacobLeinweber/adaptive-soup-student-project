"""Evaluate an included source-only checkpoint without retraining or replacing it.

Example: python evaluate_baseline.py --transfer PAMAP2_to_RealWorld
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score,
)

from src.data_access.load_data import load_experiment
from src.training.experiments import TRANSFERS
from src.training.model import SelfHARCNN

ROOT = Path(__file__).resolve().parent


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def scores(y, pred, label_map):
    ids = list(range(len(label_map)))
    names = [name for name, index in sorted(label_map.items(), key=lambda pair: pair[1])]
    return {
        'n_windows': len(y),
        'accuracy': float(accuracy_score(y, pred)),
        'macro_f1': float(f1_score(y, pred, labels=ids, average='macro', zero_division=0)),
        'balanced_accuracy': float(balanced_accuracy_score(y, pred)),
        'confusion_matrix': confusion_matrix(y, pred, labels=ids).tolist(),
        'per_class': classification_report(
            y, pred, labels=ids, target_names=names, output_dict=True, zero_division=0,
        ),
    }


def plot_confusions(report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    names = [name for name, index in sorted(report['label_map'].items(), key=lambda pair: pair[1])]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout='constrained')
    for ax, split in zip(axes, ('train', 'val', 'test')):
        result = report['results'][f'target_{split}']
        counts = np.array(result['confusion_matrix'])
        normalized = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
        display = ax.imshow(normalized, vmin=0, vmax=1, cmap='Blues')
        ax.set_xticks(range(len(names)), names, rotation=40, ha='right')
        ax.set_yticks(range(len(names)), names)
        ax.set_xlabel('Vorhergesagte Aktivität')
        ax.set_ylabel('Tatsächliche Aktivität')
        ax.set_title(f'Target {split}: Accuracy {result["accuracy"]:.1%}')
        for row in range(len(names)):
            for col in range(len(names)):
                value = normalized[row, col]
                ax.text(col, row, f'{value:.0%}', ha='center', va='center',
                        color='white' if value > .55 else 'black', fontsize=9)
    fig.colorbar(display, ax=axes, label='Anteil innerhalb der tatsächlichen Aktivität', shrink=.75)
    fig.suptitle(f'Source-only: {report["source"]} → {report["target"]}', fontsize=16)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--transfer', choices=[t.name for t in TRANSFERS], default='PAMAP2_to_RealWorld')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.batch_size < 1 or args.threads < 1:
        parser.error('batch-size and threads must be positive')
    transfer = next(t for t in TRANSFERS if t.name == args.transfer)
    output = (args.output_dir or ROOT / 'results' / 'baseline' / transfer.name).resolve()
    checkpoint_dir = ROOT / 'models' / transfer.family / transfer.name
    # Keep reports and generated predictions separate from the supplied assets.
    if any(output.is_relative_to(ROOT / folder) for folder in ('models', 'data', 'src', '.git', '.venv')):
        parser.error('output-dir must be outside the supplied assets, source and environment folders')
    checkpoint = checkpoint_dir / 'model_last.pt'
    checkpoint_hash = digest(checkpoint)
    config = json.loads((checkpoint_dir / 'config.json').read_text())
    reference = json.loads((checkpoint_dir / 'metrics.json').read_text())['final_epoch']
    label_map = config['label_map']
    data = load_experiment(transfer.family)
    torch.set_num_threads(args.threads)
    model = SelfHARCNN((150, 18), len(label_map))
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    model.eval()
    report = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'transfer': transfer.name, 'family': transfer.family,
        'source': transfer.source, 'target': transfer.target,
        'label_map': label_map, 'checkpoint': str(checkpoint),
        'checkpoint_sha256': checkpoint_hash,
        'repository_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'runtime': {'python': platform.python_version(), 'numpy': np.__version__,
                    'torch': torch.__version__, 'device': 'cpu', 'threads': args.threads},
        'protocol': 'Frozen supplied source-only checkpoint; no optimization; labels used only after predictions for scoring.',
        'original_training_config': config, 'data_metadata': {}, 'results': {},
    }
    output.mkdir(parents=True, exist_ok=True)
    for domain, name in (('source', transfer.source), ('target', transfer.target)):
        item = data[name]
        if item['labels']['label_map'] != label_map:
            raise ValueError(f'Label map does not match checkpoint: {name}')
        report['data_metadata'][domain] = item['metadata']
        for split in ('train', 'val', 'test'):
            truth, predictions, subject_ids = [], [], []
            per_subject = {}
            for subject, features in item['features'][split].items():
                y = np.asarray(item['labels']['labels'][split][subject], dtype=np.int64)
                if len(features) != len(y) or features.shape[1:] != (150, 18):
                    raise ValueError(f'Invalid alignment/shape: {name}/{split}/{subject}')
                batches = []
                with torch.inference_mode():
                    for start in range(0, len(features), args.batch_size):
                        batch = torch.from_numpy(np.asarray(features[start:start + args.batch_size], dtype=np.float32))
                        batches.append(model(batch).argmax(1).cpu().numpy())
                pred = np.concatenate(batches)
                per_subject[subject] = {
                    'n_windows': len(y), 'accuracy': float(accuracy_score(y, pred)),
                    'macro_f1': float(f1_score(y, pred, labels=range(len(label_map)), average='macro', zero_division=0)),
                }
                truth.append(y)
                predictions.append(pred)
                subject_ids.extend([subject] * len(y))
            y, pred = np.concatenate(truth), np.concatenate(predictions)
            key = f'{domain}_{split}'
            result = scores(y, pred, label_map)
            result['subjects'] = per_subject
            result['delta_from_supplied_metrics'] = {
                metric: result[metric] - reference[key][metric]
                for metric in ('accuracy', 'macro_f1', 'balanced_accuracy')
            }
            with np.load(checkpoint_dir / 'predictions' / f'{key}.npz') as stored:
                same_order = np.array_equal(y, stored['y_true'])
                result['stored_truth_order_matches'] = same_order
                result['agreement_with_supplied_predictions'] = (
                    float(np.mean(pred == stored['y_pred'])) if same_order else None
                )
            report['results'][key] = result
            np.savez_compressed(output / f'{key}.npz', y_true=y, y_pred=pred, subject=np.asarray(subject_ids))
            print(f'{key:13} n={len(y):5d} accuracy={result["accuracy"]:.6%} '
                  f'macro_f1={result["macro_f1"]:.6%} stored_agreement='
                  f'{result["agreement_with_supplied_predictions"]}', flush=True)
    if digest(checkpoint) != checkpoint_hash:
        raise RuntimeError('The checkpoint changed during evaluation')
    (output / 'baseline_results.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    plot_confusions(report, output / 'target_confusion_matrices.png')
    lines = [f'# Source-only-Baseline: {transfer.source} → {transfer.target}', '',
             'Das mitgelieferte Modell wurde auf der CPU neu ausgewertet; kein erneutes Training.', '',
             '| Split | Fenster | Accuracy | Macro-F1 | Balanced Accuracy |',
             '|---|---:|---:|---:|---:|']
    for key, result in report['results'].items():
        lines.append(f'| {key} | {result["n_windows"]} | {result["accuracy"]:.2%} | '
                     f'{result["macro_f1"]:.2%} | {result["balanced_accuracy"]:.2%} |')
    lines.extend(['', 'Die Ziel-Labels wurden ausschließlich zur Bewertung verwendet. '
                  'Es gab keine Modelloptimierung auf Ziel-Fenstern; die mitgelieferte '
                  'domänenspezifische Normalisierung wurde beibehalten.', '',
                  'Die JSON-Datei enthält Einzelmetriken pro Aktivität und Person sowie den Abgleich '
                  'mit den gespeicherten Vorhersagen und Metriken des Betreuers.', '',
                  f'Checkpoint SHA-256: `{checkpoint_hash}`',
                  f'Repository-Commit: `{report["repository_commit"]}`', ''])
    (output / 'Baseline_Kurzbericht.md').write_text('\n'.join(lines))
    print('Saved:', output, flush=True)


if __name__ == '__main__':
    main()
