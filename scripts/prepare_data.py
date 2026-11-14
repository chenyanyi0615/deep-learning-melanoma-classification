"""Inspect official/Kaggle metadata and freeze patient-level splits. No training."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def prepare(metadata, images, output, report, seed=42, verify=False,
            duplicates=Path('data/raw/duplicates.csv')):
    df = pd.read_csv(metadata)
    required = {'image_name', 'patient_id', 'target'}
    if not required.issubset(df.columns):
        raise ValueError(f'Missing columns: {required - set(df.columns)}')
    if df.empty or df[list(required)].isna().any().any():
        raise ValueError('Empty metadata or missing image/patient/target: resolve before splitting.')
    if df.image_name.duplicated().any() or not df.target.isin([0, 1]).all():
        raise ValueError('Duplicate image names or non-binary labels.')
    raw_rows = len(df)
    pairs = pd.read_csv(duplicates)
    if not {'image_name_1', 'image_name_2'}.issubset(pairs.columns):
        raise ValueError('Invalid official duplicate-image list')
    # Remove every second endpoint: every listed pair loses at least one image.
    # Deterministic and independent of labels; original metadata is preserved.
    df = df[~df.image_name.isin(set(pairs.image_name_2))].copy()
    remaining = set(df.image_name)
    assert not any(a in remaining and b in remaining
                   for a, b in pairs[['image_name_1', 'image_name_2']].itertuples(index=False, name=None))
    # Stratify patients by whether at least one of their images is melanoma.
    patients = df.groupby('patient_id', sort=True).target.max()
    train, rest = train_test_split(patients.index, test_size=.30,
                                  stratify=patients, random_state=seed)
    val, test = train_test_split(rest, test_size=.5,
                                stratify=patients.loc[rest], random_state=seed)
    groups = {'train': set(train), 'val': set(val), 'test': set(test)}
    assert not (groups['train'] & groups['val'] or groups['train'] & groups['test']
                or groups['val'] & groups['test'])
    assert set.union(*groups.values()) == set(patients.index)
    paths = {}
    for path in sorted(images.rglob('*')):
        if path.is_file() and path.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
            if path.stem in paths:
                raise ValueError(f'Ambiguous image stem: {path.stem}')
            paths[path.stem] = str(path.resolve())
    df['image_path'] = df.image_name.map(paths)
    missing = df.loc[df.image_path.isna(), 'image_name'].tolist()
    if verify:
        if missing:
            raise ValueError(f'{len(missing)} images missing; first: {missing[:5]}')
        from PIL import Image
        for index, path in enumerate(df.image_path, 1):
            with Image.open(path) as img:
                img.verify()
            # JPEG verify() alone does not decode the compressed pixel stream.
            with Image.open(path) as img:
                img.load()
            if index % 1000 == 0:
                print(f'Decoded {index}/{len(df)} images', flush=True)
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        'seed': seed, 'metadata_sha256': hashlib.sha256(metadata.read_bytes()).hexdigest(),
        'split_unit': 'patient', 'requested_patient_ratios': [.70, .15, .15],
        'stratification': 'patient maximum target',
        'rows': len(df), 'patients': len(patients),
        'raw_rows': raw_rows, 'duplicate_rows_removed': raw_rows-len(df),
        'duplicates_sha256': hashlib.sha256(duplicates.read_bytes()).hexdigest(),
        'official_duplicate_pairs_remaining': 0,
        'columns': list(df.columns), 'missing_values': df.isna().sum().to_dict(),
        'target_counts': df.target.value_counts().to_dict(),
        'missing_images': len(missing), 'all_images_verified': verify,
        'training_ready': not missing and verify,
        'patient_overlap': 0, 'splits': {},
        'note': 'Ratios apply to patients; image ratios and positive rates can differ. '
                'Only officially listed duplicate pairs checked; no pixel hash audit yet.',
    }
    for name, ids in groups.items():
        subset = df[df.patient_id.isin(ids)].copy()
        if set(subset.target) != {0, 1}:
            raise ValueError(f'{name} lacks one class')
        subset.to_csv(output / f'{name}.csv', index=False)
        summary['splits'][name] = dict(images=len(subset), patients=len(ids),
            positives=int(subset.target.sum()), positive_rate=float(subset.target.mean()),
            patient_fraction=len(ids)/len(patients), image_fraction=len(subset)/len(df))
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metadata', type=Path, default=Path('data/raw/train.csv'))
    p.add_argument('--images', type=Path, default=Path('data/raw/jpeg/train'))
    p.add_argument('--output', type=Path, default=Path('data/splits'))
    p.add_argument('--report', type=Path, default=Path('outputs/reports/dataset.json'))
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--verify-images', action='store_true')
    p.add_argument('--duplicates', type=Path, default=Path('data/raw/duplicates.csv'))
    a = p.parse_args()
    prepare(a.metadata, a.images, a.output, a.report, a.seed, a.verify_images, a.duplicates)
