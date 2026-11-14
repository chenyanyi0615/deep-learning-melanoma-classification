"""Budgeted second-round development: local execution, validation selection, no test inference."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from scripts.compare_thresholds import select_threshold
from src.dataset import LesionDataset
from src.losses import BinaryFocalLoss
from src.metrics import binary_metrics
from src.model import build_model
from src.train import seed_everything, seed_worker, save_checkpoint, write_json, verify_manifests

ROOT = Path('outputs/round2')


def now():
    return datetime.now(timezone.utc).isoformat()


def freeze_bn(model):
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)


def set_phase(model, warmup):
    model.eval()  # Early backbone remains in inference mode throughout.
    model.fc.train()
    model.layer4.train(not warmup)
    for parameter in model.layer4.parameters():
        parameter.requires_grad_(not warmup)
    freeze_bn(model)


def make_batches(dataset, config, epoch, training):
    seed = config['seed'] + epoch
    generator = torch.Generator().manual_seed(seed)
    sampler = None
    fraction = config.get('sample_positive_fraction') if training else None
    if fraction is not None:
        y = dataset.frame.target.to_numpy()
        weights = np.where(y == 1, fraction/(y == 1).sum(), (1-fraction)/(y == 0).sum())
        sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), len(y),
                                        replacement=True, generator=generator)
    return DataLoader(dataset, batch_size=8, shuffle=training and sampler is None, sampler=sampler,
                      num_workers=2, worker_init_fn=seed_worker,
                      generator=torch.Generator().manual_seed(seed), prefetch_factor=2)


def evaluate_validation(model, batches, loss_fn, deadline):
    model.eval()
    y, p, names, total_loss = [], [], [], 0.
    with torch.inference_mode():
        for images, target, image_names in batches:
            if time.time() >= deadline:
                return None
            logits = model(images.to('mps')).flatten()
            loss = loss_fn(logits, target.to('mps'))
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite validation loss')
            total_loss += loss.item()*len(target)
            y.extend(target.tolist())
            p.extend(logits.sigmoid().cpu().tolist())
            names.extend(image_names)
    threshold = select_threshold(y, p, .90)
    default, tuned = binary_metrics(y, p), binary_metrics(y, p, threshold)
    return default, tuned, total_loss/len(y), pd.DataFrame(dict(image_name=names, target=y, probability=p))


def checkpoint_score(tuned):
    return (tuned['specificity'], tuned['auroc'])


def smoke():
    torch.set_num_threads(4)
    torch.hub.set_dir(str(Path('data/pretrained').resolve()))
    seed_everything(42)
    model = build_model().to('mps')
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.fc.parameters():
        parameter.requires_grad_(True)
    dataset = LesionDataset('data/splits/train.csv', training=True)
    indices = list(dataset.frame[dataset.frame.target == 0].index[:2])+list(dataset.frame[dataset.frame.target == 1].index[:2])
    images = torch.stack([dataset[int(i)][0] for i in indices]).to('mps')
    target = torch.tensor([0.,0.,1.,1.], device='mps')
    results = []
    for warmup in [True, False]:
        set_phase(model, warmup)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        previous = model.fc.weight.detach().clone()
        model.zero_grad(set_to_none=True)
        loss = nn.BCEWithLogitsLoss()(model(images).flatten(), target)
        loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(model.fc.weight.grad).all()
        assert all(not m.training for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm))
        assert model.layer4[0].conv1.weight.grad is None if warmup else model.layer4[0].conv1.weight.grad is not None
        assert model.layer3[0].conv1.weight.grad is None
        optimizer.step()
        assert not torch.equal(previous, model.fc.weight)
        results.append(dict(warmup=warmup, finite_gradients=True, trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)))
    write_json(ROOT/'smoke.json', dict(purpose='execution checks only; not performance results', results=results))
    print('Round 2 GPU smoke checks passed', flush=True)


def worker(config_path):
    config = json.loads(Path(config_path).read_text())
    run = ROOT/config['name']
    run.mkdir(parents=True, exist_ok=True)
    status = dict(state='running', phase='initializing', started_at=now(), pid=os.getpid())
    write_json(run/'status.json', status)
    history, best_score, bad_epochs = [], (-1., -1.), 0
    try:
        torch.set_num_threads(4)
        if not torch.backends.mps.is_available():
            raise RuntimeError('MPS unavailable; refusing an unbudgeted CPU fallback')
        seed_everything(config['seed'])
        torch.hub.set_dir(str(Path('data/pretrained').resolve()))
        model = build_model().to('mps')
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in model.fc.parameters():
            parameter.requires_grad_(True)
        # Optimizer retains layer4 parameters even during the head-only first epoch.
        optimizer = torch.optim.AdamW([
            dict(params=list(model.layer4.parameters()), lr=3e-5),
            dict(params=list(model.fc.parameters()), lr=3e-4)], weight_decay=1e-4)
        loss_fn = BinaryFocalLoss(config.get('gamma', 2.), config.get('alpha', .9)) if config['loss'] == 'focal' else nn.BCEWithLogitsLoss()
        train = LesionDataset('data/splits/train.csv', training=True, image_size=config['image_size'])
        val = LesionDataset('data/splits/val.csv', image_size=config['image_size'])
        deadline = config['deadline_unix']
        stop_reason = 'max_epochs'
        for epoch in range(1, config['epochs']+1):
            if time.time() >= deadline-60:
                stop_reason = 'time_budget'
                break
            seed_everything(config['seed']+epoch)
            warmup = epoch == 1
            set_phase(model, warmup)
            factor = 1. if warmup else .05+.95*.5*(1+math.cos(math.pi*(epoch-2)/max(config['epochs']-2, 1)))
            optimizer.param_groups[0]['lr'] = 0. if warmup else 3e-5*factor
            optimizer.param_groups[1]['lr'] = 1e-3 if warmup else 3e-4*factor
            batches = make_batches(train, config, epoch, True)
            status.update(phase='training', epoch=epoch, batch=0, total_batches=len(batches), updated_at=now())
            write_json(run/'status.json', status)
            started, count, total_loss = time.monotonic(), 0, 0.
            for batch, (images, target, _) in enumerate(batches, 1):
                if time.time() >= deadline:
                    stop_reason = 'time_budget_mid_epoch'
                    break
                optimizer.zero_grad(set_to_none=True)
                logits = model(images.to('mps')).flatten()
                loss = loss_fn(logits, target.to('mps'))
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite training loss')
                loss.backward()
                optimizer.step()
                total_loss += loss.item()*len(target)
                count += len(target)
                if batch == 1 or batch % 100 == 0 or batch == len(batches):
                    status.update(batch=batch, train_loss=total_loss/count,
                                  images_per_second=count/max(time.monotonic()-started,.001), updated_at=now())
                    write_json(run/'status.json', status)
                    print(f'{config["name"]} epoch {epoch} batch {batch}/{len(batches)} loss={total_loss/count:.5f}', flush=True)
            if count != len(train):
                break  # Never report a partial epoch as a complete experiment result.
            status.update(phase='validation', updated_at=now())
            write_json(run/'status.json', status)
            result = evaluate_validation(model, make_batches(val, config, 0, False), loss_fn, deadline)
            if result is None:
                stop_reason = 'time_budget_during_validation'
                break
            default, tuned, val_loss, predictions = result
            row = dict(epoch=epoch, train_loss=total_loss/count, val_loss=val_loss,
                       auroc=tuned['auroc'], specificity_at_target=tuned['specificity'],
                       sensitivity_at_target=tuned['sensitivity'], threshold=tuned['threshold'],
                       precision=tuned['precision'], f1=tuned['f1'],
                       default_sensitivity=default['sensitivity'], default_specificity=default['specificity'],
                       seconds=time.monotonic()-started)
            history.append(row)
            if checkpoint_score(tuned) > best_score:
                best_score, bad_epochs = checkpoint_score(tuned), 0
                save_checkpoint(run/'best_weights.pt', {k:v.detach().cpu() for k,v in model.state_dict().items()})
                predictions.to_csv(run/'best_validation_predictions.csv', index=False)
                write_json(run/'best_metrics.json', dict(epoch=epoch, default=default, validation_selected=tuned))
            else:
                bad_epochs += 1
            save_checkpoint(run/'last.pt', dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                            epoch=epoch, config=config, history=history))
            pd.DataFrame(history).to_csv(run/'history.csv', index=False)
            status.update(completed_epochs=epoch, best_specificity_at_target=best_score[0], updated_at=now())
            write_json(run/'status.json', status)
            print('Completed epoch '+json.dumps(row), flush=True)
            if bad_epochs >= 3:
                stop_reason = 'early_stopping'
                break
        status.update(state='completed' if history else 'no_complete_epoch', phase='finished',
                      stop_reason=stop_reason, completed_epochs=len(history), finished_at=now())
        write_json(run/'status.json', status)
    except BaseException as exc:
        status.update(state='failed', error=str(exc), finished_at=now())
        write_json(run/'status.json', status)
        traceback.print_exc()
        raise


def metric_rows():
    rows = []
    for name, run_name in [('BCE (round 1)','bce_seed42'), ('Weighted BCE (round 1)','weighted_bce_seed42'), ('Focal (round 1)','focal_seed42')]:
        run = Path('outputs/experiments')/run_name
        f = pd.read_csv(run/'best_validation_predictions.csv', float_precision='round_trip')
        t = select_threshold(f.target, f.probability, .9)
        for policy, threshold in [('default', .5), ('validation_selected', t)]:
            rows.append(dict(name=name, round=1, policy=policy, **binary_metrics(f.target, f.probability, threshold)))
    for path in sorted(ROOT.glob('*/best_metrics.json')):
        best = json.loads(path.read_text())
        for policy, key in [('default','default'),('validation_selected','validation_selected')]:
            rows.append(dict(name=path.parent.name, round=2, epoch=best['epoch'], policy=policy, **best[key]))
    return rows


def summarize(final=False):
    rows = metric_rows()
    write_json(ROOT/'comparison.json', dict(validation_only=True, new_independent_test=False,
                 test_inference_performed=False, results=rows))
    table = pd.DataFrame([{k:v for k,v in r.items() if k != 'confusion_matrix'} for r in rows])
    table.to_csv(ROOT/'comparison.csv', index=False)
    selected = [r for r in rows if r['policy']=='validation_selected']
    baseline = next(r for r in selected if r['name']=='Weighted BCE (round 1)')
    candidates = [r for r in selected if r['round']==2]
    winner = max(candidates, key=lambda r:(r['specificity'],r['auroc'])) if candidates else None
    lines = ['# 第二轮改进实验：'+('已结束' if final else '进行中'), '',
             '这些是验证集上的探索结果，没有引入新数据，没有重新使用测试集来调参。',
             '第一轮测试集已使用过；本报告不把第二轮验证成绩称为独立测试成果。', '',
             '| 实验 | 阈值方案 | 阈值 | AUROC | 敏感度 | 特异度 | Precision | F1 | 误报 | 漏检 |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        (tn,fp),(fn,tp) = r['confusion_matrix']
        policy = '默认 0.5' if r['policy']=='default' else '验证敏感度 >=90%'
        lines.append(f'| {r["name"]} | {policy} | {r["threshold"]:.7g} | {r["auroc"]:.4f} | '
                     f'{r["sensitivity"]:.2%} | {r["specificity"]:.2%} | {r["precision"]:.2%} | {r["f1"]:.4f} | {fp} | {fn} |')
    if winner:
        delta = 100*(winner['specificity']-baseline['specificity'])
        lines += ['', f'第二轮验证集最高特异度来自 {winner["name"]}：{winner["specificity"]:.2%}。',
                  f'相对第一轮 Weighted BCE 的 {baseline["specificity"]:.2%}，变化为 {delta:+.2f} 个百分点。',
                  '这是对已反复使用的验证集的描述，不代表独立泛化提升，也不支持统计显著性结论。']
    lines += ['', '## 本轮方法',
              '保留 ResNet-50 和患者划分。冻结 BatchNorm 统计量及其仿射参数、冻结前面卷积层；',
              '先仅训练分类头一轮，再微调 layer4 与分类头，使用分组低学习率和余弦衰减。',
              '三组主实验只比较自然采样 BCE、期望阳性比例 25% 的重采样 BCE、alpha=0.9/gamma=2 的 Focal。',
              '阳性重采样组没有额外叠加加权损失。后续只根据验证结果选配置做不同种子重复及可选 320 分辨率实验。',
              '第一轮按 AUROC 选 checkpoint；第二轮按 >=90% 敏感度下的特异度选 checkpoint，两轮选择规则不同。',
              '因此不能把所有变化归因于损失函数，也不能证明单一技术改动有效。', '',
              '## 时间和失败记录',
              'plan.json 记录总时限及初始额度，status.json 记录队列状态，各实验目录保存配置、每轮日志、权重和最佳预测。',
              '到达时限的未完成轮次不计入指标；失败/跳过实验会在队列状态中列出。',
              '最后 30 分钟预留给报告。训练在本地运行，不调用语言模型或付费 API。']
    (ROOT/'中文实验报告.md').write_text('\n'.join(lines)+'\n')
    if final:
        os.environ.setdefault('MPLCONFIGDIR', str(Path('.cache/matplotlib').resolve()))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for path in sorted(ROOT.glob('*/history.csv')):
            frame = pd.read_csv(path)
            axes[0].plot(frame.epoch, frame.specificity_at_target, marker='o', label=path.parent.name)
            axes[1].plot(frame.epoch, frame.auroc, marker='o', label=path.parent.name)
        axes[0].axhline(baseline['specificity'], color='gray', linestyle='--', label='Round 1 weighted BCE')
        axes[0].set(xlabel='Epoch', ylabel='Validation specificity at sensitivity >=90%', ylim=(0,1))
        axes[1].set(xlabel='Epoch', ylabel='Validation AUROC', ylim=(0,1))
        axes[0].legend(fontsize=6)
        axes[1].legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(ROOT/'validation_curves.png', dpi=180)
        plt.close(fig)
    return candidates


def launch_one(config, plan, queue_status):
    remaining = plan['training_deadline_unix']-time.time()
    if remaining < 15*60:
        queue_status['skipped'].append(dict(name=config['name'], reason='global_time_budget'))
        return
    run = ROOT/config['name']
    run.mkdir(exist_ok=False)
    deadline = min(time.time()+config['minutes_cap']*60, plan['training_deadline_unix'])
    config.update(deadline_unix=deadline, train_manifest_sha256=queue_status['split_sha256']['train'],
                  val_manifest_sha256=queue_status['split_sha256']['val'],
                  checkpoint_selection='specificity at validation sensitivity >=90%, AUROC tie-breaker')
    write_json(run/'config.json', config)
    queue_status.update(current_run=config['name'], phase='training', updated_at=now())
    write_json(ROOT/'status.json', queue_status)
    with (run/'train.log').open('w') as log:
        process = subprocess.Popen([sys.executable,'-u','-m','scripts.round2_experiments',
                                    '--worker',str(run/'config.json')], stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
        worker_state = None
        while process.poll() is None:
            time.sleep(5)
            status_path = run/'status.json'
            if status_path.exists():
                worker_state = json.loads(status_path.read_text())
            finished = worker_state and worker_state.get('state') in ['completed','failed','no_complete_epoch']
            if finished or time.time() >= deadline+10:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                break
        state_path = run/'status.json'
        worker_state = json.loads(state_path.read_text()) if state_path.exists() else {'state':'failed','error':'no status written'}
        if worker_state['state'] == 'running':
            worker_state.update(state='budget_stopped', finished_at=now(), reason='supervisor_time_limit')
            write_json(state_path, worker_state)
    queue_status['runs'].append(dict(name=config['name'], status=worker_state,
                                    has_completed_epoch=(run/'best_metrics.json').exists()))
    summarize()
    write_json(ROOT/'status.json', queue_status)


def supervise():
    ROOT.mkdir(exist_ok=True)
    lock = (ROOT/'queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (ROOT/'status.json').exists():
        raise RuntimeError('Round 2 queue already exists; refusing to overwrite')
    plan = json.loads((ROOT/'plan.json').read_text())
    queue = dict(state='running', phase='integrity_checks', started_at=now(),
                 hard_deadline_unix=plan['absolute_deadline_unix'], runs=[], skipped=[])
    try:
        queue['split_sha256'] = verify_manifests()
        code = Path(__file__)
        queue['runner_sha256'] = hashlib.sha256(code.read_bytes()).hexdigest()
        write_json(ROOT/'status.json', queue)
        for config in plan['runs']:
            launch_one(config.copy(), plan, queue)
        candidates = summarize()
        if candidates:
            primary = [r for r in candidates if r['name'].endswith('seed42')]
            best = max(primary, key=lambda r:(r['specificity'],r['auroc']))
            chosen = json.loads((ROOT/best['name']/'config.json').read_text())
            if plan['training_deadline_unix']-time.time() >= 45*60:
                repeat = chosen.copy()
                repeat.update(name=chosen['name'].replace('seed42','seed43'), seed=43, minutes_cap=120)
                launch_one(repeat, plan, queue)
            if plan['training_deadline_unix']-time.time() >= 60*60:
                highres = chosen.copy()
                highres.update(name=chosen['name'].replace('_224_', '_320_'), image_size=320, epochs=6, minutes_cap=120)
                launch_one(highres, plan, queue)
        queue.update(phase='reporting', updated_at=now())
        write_json(ROOT/'status.json', queue)
        summarize(final=True)
        queue.update(state='completed', phase='finished', finished_at=now(), test_inference_performed=False)
        write_json(ROOT/'status.json', queue)
        print('ROUND 2 COMPLETE: '+str((ROOT/'中文实验报告.md').resolve()), flush=True)
    except BaseException as exc:
        queue.update(state='failed', error=str(exc), finished_at=now())
        write_json(ROOT/'status.json', queue)
        traceback.print_exc()
        try:
            summarize(final=True)
        except Exception:
            traceback.print_exc()
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.smoke:
        smoke()
    elif args.worker:
        worker(args.worker)
    else:
        supervise()
