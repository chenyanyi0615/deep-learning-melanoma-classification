"""Wait for extraction, then decode images and update manifests automatically."""
import json
from pathlib import Path
import time

from prepare_data import prepare


def main():
    marker = Path('outputs/reports/download.json')
    print('Waiting for download and CRC-verified extraction...', flush=True)
    while not marker.exists():
        time.sleep(30)
    info = json.loads(marker.read_text())
    if not info.get('jpeg_crc_verified') or info.get('images') != 33126:
        raise RuntimeError('Extraction report is incomplete')
    summary = prepare(Path('data/raw/train.csv'), Path(info['extraction_directory']),
                      Path('data/splits'), Path('outputs/reports/dataset.json'), verify=True)
    before = json.loads(Path('outputs/reports/split_identity_before_images.json').read_text())
    import pandas as pd
    for split, expected in before.items():
        actual = pd.read_csv(f'data/splits/{split}.csv')
        if actual[['image_name', 'patient_id', 'target']].to_dict('records') != expected:
            raise RuntimeError(f'Split membership changed: {split}')
    env_path = Path('outputs/reports/environment.json')
    env = json.loads(env_path.read_text())
    env['images_downloaded'] = True
    env['all_retained_images_decoded'] = summary['all_images_verified']
    env_path.write_text(json.dumps(env, indent=2) + '\n')
    readme = Path('README.md')
    text = readme.read_text()
    text = text.replace('**图像尚未下载，未进行图像完整性验证，未训练模型，无实验指标。**',
                        '**官方 JPEG 已下载、CRC 校验并解压；保留图像均已完整解码验证。未训练模型，无实验指标。**')
    readme.write_text(text)
    print('COMPLETE: images ready; split membership unchanged; no training performed.', flush=True)


if __name__ == '__main__':
    main()
