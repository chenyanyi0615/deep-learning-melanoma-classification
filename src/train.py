"""Train ResNet-50 and select checkpoints using validation AUROC only."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import sys
import time
import traceback

import numpy as np
import pandas as pd
import torch
import torchvision
from torch.utils.data import DataLoader

from .dataset import LesionDataset
from .losses import make_loss
from .metrics import binary_metrics
from .model import build_model


def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def save_checkpoint(path, value):
    temp = path.with_suffix('.tmp')
    torch.save(value, temp)
    temp.replace(path)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def seed_worker(_):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def loader(dataset, args, shuffle, seed):
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle,
                      num_workers=args.workers, worker_init_fn=seed_worker,
                      generator=torch.Generator().manual_seed(seed),
                      pin_memory=args.device == 'cuda',
                      **({'prefetch_factor': 2} if args.workers else {}))


def verify_manifests():
    audit = json.loads(Path('outputs/reports/dataset.json').read_text())
    if not audit.get('training_ready'):
        raise RuntimeError('Complete image verification before training')
    frames = {s: pd.read_csv(f'data/splits/{s}.csv') for s in ('train', 'val', 'test')}
    for name, frame in frames.items():
        if frame[['image_name', 'patient_id', 'target', 'image_path']].isna().any().any():
            raise ValueError(f'Missing required data: {name}')
        if frame.image_name.duplicated().any() or set(frame.target) != {0, 1}:
            raise ValueError(f'Invalid labels or duplicate images: {name}')
        if not all(Path(p).is_file() for p in frame.image_path):
            raise FileNotFoundError(f'Missing image file in {name}')
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        for column in ['patient_id', 'image_name']:
            if set(frames[a][column]) & set(frames[b][column]):
                raise ValueError(f'{column} leakage between {a} and {b}')
    return {s: hashlib.sha256(Path(f'data/splits/{s}.csv').read_bytes()).hexdigest() for s in frames}


@torch.inference_mode()
def validate(model, batches, criterion, device, status, report):
    model.eval()
    labels, probabilities, names = [], [], []
    total_loss = count = 0
    for step, (images, target, image_names) in enumerate(batches, 1):
        logits = model(images.to(device)).flatten()
        loss = criterion(logits, target.to(device))
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite validation loss')
        total_loss += loss.item() * len(target)
        count += len(target)
        labels.extend(target.tolist())
        probabilities.extend(logits.sigmoid().cpu().tolist())
        names.extend(image_names)
        if step % 50 == 0:
            status.update(phase='validation', batch=step, total_batches=len(batches),
                          updated_at=datetime.now(timezone.utc).isoformat())
            write_json(report, status)
            print(f'Validation {step}/{len(batches)}', flush=True)
    metrics = binary_metrics(labels, probabilities)
    metrics['loss'] = total_loss/count
    return metrics, pd.DataFrame(dict(image_name=names, target=labels, probability=probabilities))


def plot_results(run, history, prediction, metrics):
    os.environ.setdefault('MPLCONFIGDIR', str(Path('.cache/matplotlib').resolve()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history.epoch, history.train_loss, label='Train')
    axes[0].plot(history.epoch, history.val_loss, label='Validation')
    axes[0].set(xlabel='Epoch', ylabel='Loss')
    axes[0].legend()
    axes[1].plot(history.epoch, history.val_auroc)
    axes[1].set(xlabel='Epoch', ylabel='Validation AUROC', ylim=(0, 1))
    fig.tight_layout()
    fig.savefig(run/'figures/learning_curves.png', dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots()
    RocCurveDisplay.from_predictions(prediction.target, prediction.probability, ax=ax)
    ax.set_title('Best checkpoint: validation ROC')
    fig.savefig(run/'figures/val_roc.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(np.array(metrics['confusion_matrix']),
                           display_labels=['Benign (0)', 'Melanoma (1)']).plot(ax=ax)
    ax.set_title('Validation; threshold=0.5')
    fig.savefig(run/'figures/val_confusion_matrix.png', dpi=180, bbox_inches='tight')
    plt.close(fig)


def run_training(args):
    run = args.output
    run.mkdir(parents=True, exist_ok=True)
    lock = (run/'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (run/'config.json').exists() and not args.resume:
        raise RuntimeError('Run already exists: use --resume or a different --output')
    for folder in ['checkpoints', 'figures']:
        (run/folder).mkdir(exist_ok=True)
    status_path = run/'status.json'
    status = dict(state='running', phase='initializing', pid=os.getpid(),
                  started_at=datetime.now(timezone.utc).isoformat())
    write_json(status_path, status)
    try:
        torch.set_num_threads(4)
        seed_everything(args.seed)
        if args.device == 'auto':
            args.device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
        split_hashes = verify_manifests()
        train_data = LesionDataset('data/splits/train.csv', training=True)
        val_data = LesionDataset('data/splits/val.csv')
        config = vars(args).copy()
        config['output'] = str(run.resolve())
        config.pop('resume')
        config.update(split_sha256=split_hashes, pretrained_weights='IMAGENET1K_V2',
                      source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path('src').glob('*.py'))},
                      architecture='resnet50', image_size=224, preprocessing='JPEG draft to >=448; RGB; aspect-preserving contain/pad 224; ImageNet normalization',
                      train_augmentation='horizontal/vertical flip p=0.5; rotation +/-15; brightness/contrast/saturation 0.1',
                      train_images=len(train_data), val_images=len(val_data),
                      train_positive_weight=float((train_data.frame.target == 0).sum()/train_data.frame.target.sum()),
                      torch_version=str(torch.__version__), torchvision_version=str(torchvision.__version__),
                      python=sys.version, platform=platform.platform(), test_evaluated=False,
                      reproducibility='Fixed seeds; MPS bitwise reproducibility is not guaranteed')
        if args.resume:
            old = json.loads((run/'config.json').read_text())
            for key in ['loss', 'seed', 'batch_size', 'lr', 'weight_decay', 'epochs', 'patience', 'workers', 'device', 'gamma', 'alpha', 'split_sha256']:
                if old[key] != config[key]:
                    raise ValueError(f'Resume configuration mismatch: {key}')
        else:
            write_json(run/'config.json', config)
        torch.hub.set_dir(str(Path('data/pretrained').resolve()))
        print(f'Device={args.device}; train={len(train_data)}; val={len(val_data)}; loss={args.loss}; batch={args.batch_size}', flush=True)
        model = build_model(pretrained=not args.resume).to(args.device)
        criterion = make_loss(args.loss, train_data.frame.target, args.device, args.gamma, args.alpha)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        start_epoch, best_auc, bad_epochs, history = 1, -1.0, 0, []
        if args.resume:
            checkpoint = torch.load(run/'checkpoints/last.pt', map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']+1
            best_auc, bad_epochs, history = checkpoint['best_auroc'], checkpoint['bad_epochs'], checkpoint['history']
            print(f'Resuming after epoch {start_epoch-1}', flush=True)
        status.update(device=args.device, loss=args.loss, max_epochs=args.epochs)
        for epoch in range(start_epoch, args.epochs+1):
            if bad_epochs >= args.patience:
                break
            # Recreate loader and seed per epoch so epoch-boundary resume is stable.
            seed_everything(args.seed+epoch)
            batches = loader(train_data, args, True, args.seed+epoch)
            model.train()
            total_loss = count = 0
            epoch_start = time.monotonic()
            status.update(epoch=epoch, phase='training', batch=0, total_batches=len(batches))
            write_json(status_path, status)
            for step, (images, target, _) in enumerate(batches, 1):
                optimizer.zero_grad(set_to_none=True)
                logits = model(images.to(args.device)).flatten()
                loss = criterion(logits, target.to(args.device))
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite training loss')
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(target)
                count += len(target)
                if step == 1 or step % 10 == 0 or step == len(batches):
                    elapsed = time.monotonic()-epoch_start
                    status.update(batch=step, train_loss_so_far=total_loss/count,
                                  images_per_second=count/max(elapsed, 0.001),
                                  updated_at=datetime.now(timezone.utc).isoformat())
                    write_json(status_path, status)
                    print(f'Epoch {epoch}/{args.epochs} batch {step}/{len(batches)} '
                          f'loss={total_loss/count:.5f} speed={count/max(elapsed,0.001):.1f} images/s', flush=True)
            metrics, predictions = validate(model, loader(val_data, args, False, args.seed),
                                             criterion, args.device, status, status_path)
            row = dict(epoch=epoch, train_loss=total_loss/count, val_loss=metrics['loss'],
                       val_auroc=metrics['auroc'], val_sensitivity=metrics['sensitivity'],
                       val_specificity=metrics['specificity'], val_precision=metrics['precision'],
                       val_f1=metrics['f1'], seconds=time.monotonic()-epoch_start)
            history.append(row)
            if metrics['auroc'] > best_auc:
                best_auc, bad_epochs = metrics['auroc'], 0
                save_checkpoint(run/'checkpoints/best_weights.pt',
                                {k: v.detach().cpu() for k,v in model.state_dict().items()})
                write_json(run/'best_validation_metrics.json', dict(epoch=epoch, **metrics))
                predictions.to_csv(run/'best_validation_predictions.csv', index=False)
            else:
                bad_epochs += 1
            save_checkpoint(run/'checkpoints/last.pt', dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                            epoch=epoch, best_auroc=best_auc, bad_epochs=bad_epochs, history=history, config=config))
            pd.DataFrame(history).to_csv(run/'history.csv', index=False)
            status.update(completed_epochs=epoch, best_validation_auroc=best_auc, validation_metrics=metrics)
            write_json(status_path, status)
            print('Epoch completed: '+json.dumps(row), flush=True)
        best_metrics = json.loads((run/'best_validation_metrics.json').read_text())
        plot_results(run, pd.DataFrame(history), pd.read_csv(run/'best_validation_predictions.csv'), best_metrics)
        status.update(state='completed', phase='finished', test_evaluated=False,
                      stop_reason='early_stopping' if bad_epochs >= args.patience else 'max_epochs',
                      finished_at=datetime.now(timezone.utc).isoformat())
        write_json(status_path, status)
        print('Training completed. Best validation metrics: '+json.dumps(best_metrics), flush=True)
    except BaseException as exc:
        status.update(state='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                      error=str(exc), updated_at=datetime.now(timezone.utc).isoformat())
        write_json(status_path, status)
        traceback.print_exc()
        raise
    finally:
        lock.close()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--loss', choices=['bce', 'weighted_bce', 'focal'], default='bce')
    p.add_argument('--output', type=Path, default=Path('outputs/experiments/bce_seed42'))
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--patience', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    p.add_argument('--gamma', type=float, default=2.0)
    p.add_argument('--alpha', type=float, default=0.25)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if min(args.epochs, args.patience, args.batch_size) < 1 or args.workers < 0 or args.lr <= 0:
        p.error('Invalid training configuration')
    return args


if __name__ == '__main__':
    run_training(parse_args())
