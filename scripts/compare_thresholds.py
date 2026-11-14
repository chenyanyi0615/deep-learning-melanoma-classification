"""Compare frozen checkpoints using saved VALIDATION scores, without inference/training."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.metrics import binary_metrics

RUNS = {'BCE': 'bce_seed42', 'Weighted BCE': 'weighted_bce_seed42', 'Focal Loss': 'focal_seed42'}


def select_threshold(labels, scores, target_sensitivity=0.9):
    """Highest threshold satisfying empirical sensitivity, using score >= threshold."""
    y, p = np.asarray(labels), np.asarray(scores, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or set(np.unique(y)) != {0, 1}:
        raise ValueError('Expected matching one-dimensional scores and binary labels with both classes')
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Probabilities must be finite and between 0 and 1')
    if not 0 < target_sensitivity <= 1:
        raise ValueError('Target sensitivity must be in (0, 1]')
    positives = np.sort(p[y == 1])[::-1]
    required = math.ceil(target_sensitivity * len(positives) - 1e-12)
    return float(positives[required-1])


def threshold_sweep(y, p):
    order = np.argsort(p)
    scores, labels = p[order], y[order]
    thresholds = np.unique(np.r_[0., scores, .5, 1.])
    cut = np.searchsorted(scores, thresholds, side='left')
    positive_prefix = np.r_[0, np.cumsum(labels)]
    fn = positive_prefix[cut]
    tn = cut-fn
    tp = int(y.sum())-fn
    fp = int((y == 0).sum())-tn
    precision = np.divide(tp, tp+fp, out=np.zeros(len(tp), dtype=float), where=(tp+fp) > 0)
    f1 = np.divide(2*tp, 2*tp+fp+fn, out=np.zeros(len(tp), dtype=float), where=(2*tp+fp+fn) > 0)
    return pd.DataFrame(dict(threshold=thresholds, sensitivity=tp/y.sum(),
                             specificity=tn/(y == 0).sum(), precision=precision,
                             f1=f1, tn=tn, fp=fp, fn=fn, tp=tp))


def plot_figures(output, predictions, rows, target, dataset_label='Validation'):
    os.environ.setdefault('MPLCONFIGDIR', str(Path('.cache/matplotlib').resolve()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, roc_curve
    fig, ax = plt.subplots(figsize=(8, 6))
    for model, (y, p) in predictions.items():
        subset = [r for r in rows if r['model'] == model]
        fpr, tpr, _ = roc_curve(y, p)
        line, = ax.plot(fpr, tpr, label=f'{model} (AUROC={subset[0]["auroc"]:.4f})')
        for row, marker in zip(subset, ['x', 'o']):
            ax.scatter(1-row['specificity'], row['sensitivity'], marker=marker,
                       color=line.get_color(), s=70, zorder=3)
    ax.axhline(target, color='gray', linestyle='--', label=f'Target sensitivity {target:.0%}')
    ax.plot([], [], 'kx', label='Default threshold 0.5')
    ax.plot([], [], 'ko', label='Validation-selected threshold')
    ax.set(xlabel='False positive rate (1 - specificity)', ylabel='Sensitivity',
           title=f'{dataset_label} ROC and frozen operating thresholds', xlim=(-.02, 1.02), ylim=(-.02, 1.02))
    ax.legend(loc='lower right', fontsize=8)
    fig.tight_layout()
    fig.savefig(output/'roc_operating_points.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for column, model in enumerate(RUNS):
        for index, row in enumerate(r for r in rows if r['model'] == model):
            cm = [[row['tn'], row['fp']], [row['fn'], row['tp']]]
            ConfusionMatrixDisplay(np.asarray(cm), display_labels=['Benign', 'Melanoma']).plot(
                ax=axes[index, column], colorbar=False, values_format='d')
            axes[index, column].set_title(f'{model}: {row["threshold_policy"]}\nt={row["threshold"]:.6g}')
    fig.suptitle(f'{dataset_label.upper()} confusion matrices (rows: true; columns: predicted)')
    fig.tight_layout()
    fig.savefig(output/'confusion_matrices.png', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target-sensitivity', type=float, default=.90)
    parser.add_argument('--output', type=Path, default=Path('outputs/threshold_analysis'))
    args = parser.parse_args()
    manifest_path = Path('data/splits/val.csv')
    manifest = pd.read_csv(manifest_path)
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    rows, provenance, predictions, sweeps = [], {}, {}, {}
    for model, run_name in RUNS.items():
        run = Path('outputs/experiments')/run_name
        config = json.loads((run/'config.json').read_text())
        status = json.loads((run/'status.json').read_text())
        best = json.loads((run/'best_validation_metrics.json').read_text())
        path = run/'best_validation_predictions.csv'
        frame = pd.read_csv(path, float_precision='round_trip')
        if status['state'] != 'completed':
            raise ValueError(f'{model} is not complete')
        if config['split_sha256']['val'] != manifest_hash:
            raise ValueError(f'{model}: validation manifest changed')
        if frame.image_name.duplicated().any() or set(frame.image_name) != set(manifest.image_name):
            raise ValueError(f'{model}: predictions must cover the exact validation images once')
        merged = manifest[['image_name', 'target']].merge(frame, on='image_name', validate='one_to_one', suffixes=('_true', '_pred'))
        if not np.array_equal(merged.target_true, merged.target_pred):
            raise ValueError(f'{model}: prediction labels differ from manifest')
        y, p = merged.target_true.to_numpy(), merged.probability.to_numpy()
        threshold = select_threshold(y, p, args.target_sensitivity)
        predictions[model] = (y, p)
        sweep = threshold_sweep(y, p)
        sweeps[run_name] = sweep
        for policy, t in [('default_0.5', .5), ('sensitivity_target', threshold)]:
            metrics = binary_metrics(y, p, t)
            tn, fp = metrics['confusion_matrix'][0]
            fn, tp = metrics['confusion_matrix'][1]
            if policy == 'default_0.5':
                for key in ['auroc', 'sensitivity', 'specificity', 'precision', 'f1']:
                    if not math.isclose(metrics[key], best[key], abs_tol=1e-12):
                        raise ValueError(f'{model}: saved metrics mismatch: {key}')
            elif metrics['sensitivity'] < args.target_sensitivity:
                raise AssertionError('Sensitivity target not met')
            rows.append(dict(model=model, best_epoch=best['epoch'], threshold_policy=policy,
                             threshold=t, **{k:metrics[k] for k in ['auroc','sensitivity','specificity','precision','f1']},
                             tn=tn, fp=fp, fn=fn, tp=tp, accuracy=(tn+tp)/len(y)))
        valid = sweep[sweep.sensitivity >= args.target_sensitivity]
        if not math.isclose(threshold, valid.threshold.max(), abs_tol=1e-15):
            raise AssertionError('Selected threshold is not the highest feasible threshold')
        provenance[model] = dict(run=str(run), best_epoch=best['epoch'],
                                prediction_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    args.output.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    result.to_csv(args.output/'comparison.csv', index=False)
    for run_name, sweep in sweeps.items():
        sweep.to_csv(args.output/f'{run_name}_threshold_sweep.csv', index=False)
    selected = result[result.threshold_policy.eq('sensitivity_target')].sort_values('specificity', ascending=False)
    report = dict(dataset='validation only', test_evaluated=False, new_training_performed=False,
                  target_sensitivity=args.target_sensitivity, decision_rule='probability >= threshold',
                  selection_rule='Highest threshold achieving target sensitivity on validation data',
                  validation_images=len(manifest), positive_images=int(manifest.target.sum()),
                  validation_manifest_sha256=manifest_hash, checkpoints=provenance,
                  results=rows, ranking_by_specificity=selected.model.tolist(),
                  limitations=['Thresholds selected and measured on the same validation set; optimistic estimates.',
                               'No test performance or guaranteed test sensitivity is implied.',
                               'Only one training seed and one parameter setting per loss evaluated.',
                               'With 84 positives, sensitivity changes in steps of 1/84; ties can produce larger steps.'])
    (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    lines = ['# Validation threshold comparison', '',
             '**Validation only. No new training and no test-set evaluation.**', '',
             f'Select each checkpoint\'s highest threshold with sensitivity >= {args.target_sensitivity:.0%}.', '',
             '| Model | Policy | Threshold | AUROC | Sensitivity | Specificity | Precision | F1 | FP | FN |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f'| {r["model"]} | {r["threshold_policy"]} | {r["threshold"]:.8g} | {r["auroc"]:.4f} | '
                     f'{r["sensitivity"]:.2%} | {r["specificity"]:.2%} | {r["precision"]:.2%} | {r["f1"]:.4f} | {r["fp"]} | {r["fn"]} |')
    lines += ['', 'These thresholds are now recorded for subsequent frozen test evaluation.',
              'Validation-selected results are optimistic and do not demonstrate test-set generalization.',
              'ROC and confusion-matrix figures are in this directory.']
    (args.output/'comparison.md').write_text('\n'.join(lines)+'\n')
    plot_figures(args.output, predictions, rows, args.target_sensitivity)
    print(result.to_string(index=False), flush=True)
    print('Saved validation analysis to '+str(args.output.resolve()), flush=True)


if __name__ == '__main__':
    main()
