"""Fixed validation thresholds for the final four-model internal test comparison."""
import json
import fcntl
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

from scripts.train_metadata_fusion import BASE, REPEAT, ROOT, SOURCE, FusionHead, metadata, sha
from scripts.compare_thresholds import select_threshold
from src.dataset import LesionDataset
from src.metrics import binary_metrics
from src.model import build_model
from src.train import verify_manifests, write_json

OUT = Path('outputs/final_internal_test')
META = ROOT / 'image_metadata'
BRANCHES = {'主模型：双 ResNet＋元数据': ['seed42', 'seed43', 'meta'],
            '填充增强 ResNet': ['seed42'], '纯双 ResNet': ['seed42', 'seed43'],
            '单 ResNet＋元数据': ['meta']}
WEIGHTS = [BASE / 'best_weights.pt', REPEAT / 'best_weights.pt', META / 'best_head.pt']


def combine(frame, cols):
    return frame[cols].to_numpy(dtype=float).mean(axis=1)


def metrics(target, scores, threshold):
    row = binary_metrics(target, scores, threshold)
    (tn, _), (_, tp) = row['confusion_matrix']
    return dict(row, accuracy=(tn + tp) / len(target),
                ap=float(average_precision_score(target, scores)))


def freeze():
    hashes = verify_manifests()
    original = json.loads(Path('outputs/test_evaluation/frozen_protocol.json').read_text())
    assert hashes == original['protocol']['split_sha256']
    val = pd.read_csv('data/splits/val.csv')[['image_name', 'target']]
    sources = {'seed42': BASE / 'best_validation_predictions.csv',
               'seed43': REPEAT / 'best_validation_predictions.csv',
               'meta': META / 'validation_predictions.csv'}
    for key, path in sources.items():
        frame = pd.read_csv(path, float_precision='round_trip')
        assert len(frame) == len(val)
        val = val.merge(frame.rename(columns={'probability': key}), on=['image_name','target'],
                        how='left', validate='one_to_one')
        assert val[key].between(0, 1).all()
    validation = {}
    for name, cols in BRANCHES.items():
        scores = combine(val, cols)
        threshold = select_threshold(val.target, scores, .90)
        if cols == ['meta']:
            threshold = json.loads((META / 'best_metrics.json').read_text())['threshold']
            scores = scores.astype(np.float32).astype(float)
        validation[name] = metrics(val.target, scores, threshold)
    assert [v['confusion_matrix'][0][1] for v in validation.values()] == [885, 943, 879, 1175]
    files = WEIGHTS + list(sources.values()) + [SOURCE, ROOT / 'config.json']
    protocol = dict(splits=hashes, file_sha256={str(p): sha(p) for p in files},
                    equal_weight_branches=BRANCHES, primary=next(iter(BRANCHES)),
                    main_comparator='纯双 ResNet', validation=validation,
                    decision='mean predicted probability >= frozen validation threshold')
    path = OUT / 'frozen_protocol.json'
    if path.exists():
        old = json.loads(path.read_text())
        old.pop('frozen_at')
        if old != protocol:
            raise ValueError('Inputs differ from frozen protocol')
    else:
        write_json(path, dict(protocol, frozen_at=datetime.now(timezone.utc).isoformat()))
    return validation


@torch.inference_mode()
def predict():
    datasets = {key: LesionDataset(f'data/splits/{key}.csv') for key in ('train','test')}
    enc, settings = metadata(datasets)
    cfg = json.loads((ROOT / 'config.json').read_text())
    for key in ('fields','age_median','age_scale','categories'):
        assert json.loads(json.dumps(settings[key])) == cfg['encoding'][key]
    assert sha(WEIGHTS[0]) == cfg['weights_sha256'] and sha(SOURCE) == cfg['metadata_sha256']
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    torch.set_num_threads(4)
    models = []
    for path in WEIGHTS[:2]:
        model = build_model(pretrained=False)
        model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
        models.append(model.eval().to(device))
    head = models[0].fc
    fusion = FusionHead(torch.nn.Linear(2048, 1), enc['test'].shape[1])
    fusion.load_state_dict(torch.load(WEIGHTS[2], map_location='cpu', weights_only=True))
    fusion.eval()
    models[0].fc = torch.nn.Identity()
    data = datasets['test']
    names, chunks = [], []
    print('Inference device:', device, flush=True)
    for images, _, ids in DataLoader(data, batch_size=24, shuffle=False, num_workers=0):
        start = len(names)
        images = images.to(device)
        feat = models[0](images)
        meta = enc['test'][start:start+len(ids)]
        values = [head(feat).flatten().sigmoid().cpu(), models[1](images).flatten().sigmoid().cpu(),
                  fusion(feat.cpu(), meta).sigmoid()]
        chunks.append(torch.stack(values, dim=1))
        names.extend(ids)
        if len(names) % 600 == 0:
            print(len(names), '/', len(data), flush=True)
    scores = torch.cat(chunks).numpy().astype(float)
    assert names == data.frame.image_name.tolist() and np.isfinite(scores).all()
    predictions = data.frame[['image_name','patient_id','target']].copy()
    predictions[['seed42','seed43','meta']] = scores
    return predictions


def run():
    val = freeze()
    print('Frozen validation thresholds:', [r['threshold'] for r in val.values()], flush=True)
    frame = predict()
    rows = []
    for name, cols in BRANCHES.items():
        frame[name] = combine(frame, cols)
        test = metrics(frame.target, frame[name], val[name]['threshold'])
        for split, result in [('验证集', val[name]), ('内部测试集', test)]:
            row = dict(result)
            (tn, fp), (fn, tp) = row.pop('confusion_matrix')
            rows.append(dict(model=name, split=split, tp=tp, fn=fn, fp=fp, tn=tn, **row))
    for filename, df in [('predictions.csv',frame), ('comparison.csv',pd.DataFrame(rows))]:
        path = OUT / filename
        temp = path.with_suffix('.tmp')
        df.to_csv(temp, index=False)
        temp.replace(path)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    with (OUT / 'evaluation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (OUT / 'comparison.csv').exists():
            raise SystemExit('Comparison already completed.')
        run()
