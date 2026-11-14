"""Evaluate three frozen checkpoints at default and validation-selected thresholds.

No training, parameter selection, or threshold search is performed on test data.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import platform
import time
import traceback

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from scripts.compare_thresholds import RUNS, plot_figures
from src.dataset import LesionDataset
from src.metrics import binary_metrics
from src.model import build_model
from src.train import verify_manifests, write_json


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def build_protocol():
    report_path = Path('outputs/threshold_analysis/report.json')
    report = json.loads(report_path.read_text())
    splits = verify_manifests()
    if splits['val'] != report['validation_manifest_sha256']:
        raise ValueError('Validation split differs from threshold-selection split')
    models = {}
    for model, run_name in RUNS.items():
        run = Path('outputs/experiments')/run_name
        config = json.loads((run/'config.json').read_text())
        status = json.loads((run/'status.json').read_text())
        best = json.loads((run/'best_validation_metrics.json').read_text())
        if status['state'] != 'completed' or config['split_sha256'] != splits:
            raise ValueError(f'{model}: incomplete training or changed data split')
        for source, digest in config['source_sha256'].items():
            if sha256(source) != digest:
                raise ValueError(f'{model}: source changed since training: {source}')
        if sha256(run/'best_validation_predictions.csv') != report['checkpoints'][model]['prediction_sha256']:
            raise ValueError(f'{model}: validation predictions changed since threshold selection')
        selected = [r for r in report['results'] if r['model'] == model and r['threshold_policy'] == 'sensitivity_target']
        if len(selected) != 1 or best['epoch'] != selected[0]['best_epoch']:
            raise ValueError(f'{model}: threshold and checkpoint do not match')
        threshold = float(selected[0]['threshold'])
        if not 0 <= threshold <= 1:
            raise ValueError('Invalid frozen threshold')
        weight = run/'checkpoints/best_weights.pt'
        models[model] = dict(run=str(run), checkpoint=str(weight), checkpoint_sha256=sha256(weight),
                             best_epoch=best['epoch'], validation_selected_threshold=threshold)
    return dict(dataset='held-out internal test', decision_rule='probability >= threshold',
                target_validation_sensitivity=report['target_sensitivity'],
                validation_preferred_model=report['ranking_by_specificity'][0],
                selection='All checkpoints, hyperparameters and thresholds frozen using validation data.',
                split_sha256=splits, threshold_report_sha256=sha256(report_path), models=models,
                new_training_performed=False, threshold_search_on_test=False)


def validate_predictions(frame, manifest):
    if frame.image_name.duplicated().any() or set(frame.image_name) != set(manifest.image_name):
        raise ValueError('Predictions do not cover test images exactly once')
    merged = manifest[['image_name', 'patient_id', 'target']].merge(
        frame[['image_name', 'target', 'probability']], on='image_name', validate='one_to_one', suffixes=('', '_prediction'))
    if not np.array_equal(merged.target, merged.target_prediction):
        raise ValueError('Saved prediction labels do not match test manifest')
    if not np.isfinite(merged.probability).all() or not merged.probability.between(0, 1).all():
        raise ValueError('Invalid predicted probabilities')
    return merged.drop(columns='target_prediction')


def evaluate(args):
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    lock = (output/'evaluation.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (output/'frozen_protocol.json').exists() and not args.resume:
        raise RuntimeError('Evaluation already exists. Use --resume to reuse the frozen protocol and saved scores.')
    status_path = output/'status.json'
    status = dict(state='running', phase='checking_integrity',
                  started_at=datetime.now(timezone.utc).isoformat())
    write_json(status_path, status)
    try:
        torch.set_num_threads(4)
        device = args.device
        if device == 'auto':
            device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
        protocol = build_protocol()
        protocol_path = output/'frozen_protocol.json'
        if protocol_path.exists():
            if json.loads(protocol_path.read_text())['protocol'] != protocol:
                raise ValueError('Protocol changed; cannot reuse this test evaluation')
        else:
            write_json(protocol_path, dict(frozen_at=datetime.now(timezone.utc).isoformat(), protocol=protocol))
        dataset = LesionDataset('data/splits/test.csv', training=False)
        batches = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=device == 'cuda',
                             **({'prefetch_factor': 2} if args.workers else {}))
        manifest = dataset.frame
        rows, all_predictions = [], {}
        status.update(device=device, test_images=len(dataset), models_total=len(RUNS), completed_models=0)
        print(f'Frozen test evaluation: {len(dataset)} images; device={device}; batch={args.batch_size}', flush=True)
        for model_name, run_name in RUNS.items():
            spec = protocol['models'][model_name]
            prediction_path = output/f'{run_name}_predictions.csv'
            status.update(phase='inference', current_model=model_name, batch=0, total_batches=len(batches))
            write_json(status_path, status)
            if prediction_path.exists() and args.resume:
                frame = validate_predictions(pd.read_csv(prediction_path, float_precision='round_trip'), manifest)
                print(f'Reusing complete saved test scores: {model_name}', flush=True)
            else:
                model = build_model(pretrained=False)
                model.load_state_dict(torch.load(spec['checkpoint'], map_location='cpu', weights_only=True), strict=True)
                model.to(device).eval()
                names, labels, scores = [], [], []
                started = time.monotonic()
                with torch.inference_mode():
                    for step, (images, target, image_names) in enumerate(batches, 1):
                        probability = model(images.to(device)).flatten().sigmoid().cpu()
                        if not torch.isfinite(probability).all():
                            raise RuntimeError('Non-finite test predictions')
                        names.extend(image_names)
                        labels.extend(target.tolist())
                        scores.extend(probability.tolist())
                        if step == 1 or step % 50 == 0 or step == len(batches):
                            elapsed = time.monotonic()-started
                            status.update(batch=step, images_per_second=len(scores)/max(elapsed, .001),
                                          updated_at=datetime.now(timezone.utc).isoformat())
                            write_json(status_path, status)
                            print(f'{model_name}: {step}/{len(batches)} batches; {len(scores)/max(elapsed,.001):.1f} images/s', flush=True)
                frame = validate_predictions(pd.DataFrame(dict(image_name=names, target=labels, probability=scores)), manifest)
                temporary = prediction_path.with_suffix('.tmp')
                frame.to_csv(temporary, index=False)
                temporary.replace(prediction_path)
                del model
                if device == 'mps':
                    torch.mps.empty_cache()
            y, p = frame.target.to_numpy(), frame.probability.to_numpy()
            all_predictions[model_name] = (y, p)
            for policy, threshold in [('default_0.5', .5), ('sensitivity_target', spec['validation_selected_threshold'])]:
                metrics = binary_metrics(y, p, threshold)
                (tn, fp), (fn, tp) = metrics['confusion_matrix']
                rows.append(dict(model=model_name, best_epoch=spec['best_epoch'], threshold_policy=policy,
                                 threshold=threshold, **{k:metrics[k] for k in ['auroc','sensitivity','specificity','precision','f1']},
                                 tn=tn, fp=fp, fn=fn, tp=tp, accuracy=(tn+tp)/len(y)))
            status.update(completed_models=len(all_predictions))
            write_json(status_path, status)
            pd.DataFrame(rows).to_csv(output/'comparison.csv', index=False)
            print(f'{model_name} test evaluation complete: '+json.dumps(rows[-1]), flush=True)
        status.update(phase='saving_figures')
        write_json(status_path, status)
        plot_figures(output, all_predictions, rows, protocol['target_validation_sensitivity'], dataset_label='Test')
        report = dict(dataset='held-out internal test', test_evaluated=True,
                      new_training_performed=False, threshold_search_on_test=False,
                      protocol_file='frozen_protocol.json', protocol_sha256=sha256(protocol_path),
                      test_images=len(manifest), test_patients=int(manifest.patient_id.nunique()),
                      test_positives=int(manifest.target.sum()), test_negatives=int((manifest.target == 0).sum()),
                      validation_preferred_model=protocol['validation_preferred_model'],
                      results=rows, device=device, torch_version=str(torch.__version__), platform=platform.platform(),
                      predictions_sha256={run:sha256(output/f'{run}_predictions.csv') for run in RUNS.values()},
                      limitations=['Internal patient-disjoint test, not external clinical validation.',
                                   'One training seed and one parameter combination per loss.',
                                   'Validation sensitivity target is not guaranteed on the test set.',
                                   'No confidence intervals or significance tests calculated.',
                                   'Test set is now used; future tuning requires a separate untouched evaluation set.'])
        write_json(output/'report.json', report)
        lines = ['# Frozen test-set evaluation', '',
                 'All thresholds and checkpoints were selected on validation data before test inference.',
                 f'Validation-preferred model: **{protocol["validation_preferred_model"]}**.',
                 f'Test images: {len(manifest)}; patients: {manifest.patient_id.nunique()}; '
                 f'melanoma: {int(manifest.target.sum())}.', '',
                 '| Model | Threshold policy | Threshold | AUROC | Sensitivity | Specificity | Precision | F1 | FP | FN |',
                 '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for r in rows:
            lines.append(f'| {r["model"]} | {r["threshold_policy"]} | {r["threshold"]:.8g} | {r["auroc"]:.4f} | '
                         f'{r["sensitivity"]:.2%} | {r["specificity"]:.2%} | {r["precision"]:.2%} | {r["f1"]:.4f} | {r["fp"]} | {r["fn"]} |')
        lines += ['', 'No thresholds were adjusted using test scores.',
                  'The validation sensitivity target is not a guaranteed test-set result.',
                  'Comparisons are descriptive; no statistical significance claim is supported.']
        (output/'comparison.md').write_text('\n'.join(lines)+'\n')
        status.update(state='completed', phase='finished', test_evaluated=True,
                      finished_at=datetime.now(timezone.utc).isoformat())
        write_json(status_path, status)
        print(pd.DataFrame(rows).to_string(index=False), flush=True)
        print('Test evaluation COMPLETE. Results: '+str(output.resolve()), flush=True)
    except BaseException as exc:
        status.update(state='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=str(exc),
                      updated_at=datetime.now(timezone.utc).isoformat())
        write_json(status_path, status)
        traceback.print_exc()
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['auto','mps','cuda','cpu'], default='auto')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--output', type=Path, default=Path('outputs/test_evaluation'))
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        parser.error('batch-size must be positive; workers must be nonnegative')
    evaluate(args)
