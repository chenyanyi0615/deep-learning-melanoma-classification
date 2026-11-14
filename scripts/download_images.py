"""Resume official ISIC 2020 JPEG download, check ZIP CRC, and safely extract.

Run from project root. The checkpoint tracks completed byte ranges; archive size
alone is not evidence of completeness. Network access is required.
"""
import concurrent.futures
import json
import os
from pathlib import Path
import threading
import time
import urllib.request
import zipfile

URL = 'https://isic-archive.s3.amazonaws.com/challenges/2020/ISIC_2020_Training_JPEG.zip'
ARCHIVE = Path('data/raw/ISIC_2020_Training_JPEG.zip')
STATE = ARCHIVE.with_suffix('.download.json')
DEST = Path('data/raw/jpeg/train')
CHUNK = 64 * 1024 * 1024


def main():
    with urllib.request.urlopen(urllib.request.Request(URL, method='HEAD'), timeout=60) as r:
        size = int(r.headers['Content-Length'])
        etag = r.headers['ETag']
    if STATE.exists():
        state = json.loads(STATE.read_text())
        if state['size'] != size or state['etag'] != etag:
            raise RuntimeError('Remote archive changed; do not resume old ranges.')
        if not ARCHIVE.exists():
            raise RuntimeError('Resume state exists but archive is missing.')
    else:
        # Retain the sequential curl prefix downloaded before this script.
        prefix = ARCHIVE.stat().st_size if ARCHIVE.exists() else 0
        if prefix > size:
            raise RuntimeError('Local archive larger than remote object.')
        state = dict(url=URL, size=size, etag=etag, prefix=prefix, completed=[])
    lock = threading.Lock()

    def save():
        tmp = STATE.with_suffix('.tmp')
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(STATE)

    save()
    fd = os.open(ARCHIVE, os.O_RDWR | os.O_CREAT, 0o644)
    start_time = time.monotonic()
    initial = state['prefix'] + sum(min(CHUNK, size-s) for s in state['completed'])
    ranges = [(s, min(s+CHUNK, size)-1)
              for s in range(state['prefix'], size, CHUNK) if s not in state['completed']]

    def fetch(bounds):
        start, end = bounds
        for attempt in range(6):
            try:
                req = urllib.request.Request(URL, headers={'Range': f'bytes={start}-{end}'})
                with urllib.request.urlopen(req, timeout=90) as r:
                    if r.status != 206 or r.headers.get('Content-Range') != f'bytes {start}-{end}/{size}':
                        raise RuntimeError('Server did not return the requested byte range')
                    offset = start
                    while block := r.read(1024*1024):
                        if offset+len(block) > end+1:
                            raise RuntimeError('Range response too long')
                        view = memoryview(block)
                        while view:
                            written = os.pwrite(fd, view, offset)
                            offset += written
                            view = view[written:]
                    if offset != end+1:
                        raise RuntimeError('Incomplete range')
                with lock:
                    os.fsync(fd)
                    state['completed'].append(start)
                    save()
                    done = state['prefix'] + sum(min(CHUNK, size-s) for s in state['completed'])
                    speed = (done-initial)/max(time.monotonic()-start_time, 1)/1024**2
                    print(f'Download {done/size:.1%}: {done/1024**3:.2f}/{size/1024**3:.2f} GiB; '
                          f'{speed:.1f} MiB/s', flush=True)
                return
            except Exception as exc:
                if attempt == 5:
                    raise
                print(f'Retry range {start}, attempt {attempt+1}: {exc}', flush=True)
                time.sleep(min(2**attempt, 30))
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
            list(pool.map(fetch, ranges))
    finally:
        os.close(fd)
    print('Download complete. Checking ZIP CRCs and extracting JPEGs...', flush=True)
    DEST.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(ARCHIVE) as archive:
        members = [i for i in archive.infolist() if not i.is_dir() and i.filename.lower().endswith('.jpg')
                   and not i.filename.startswith('__MACOSX/')]
        names = [Path(i.filename).name for i in members]
        if len(names) != len(set(names)):
            raise RuntimeError('Ambiguous filenames in ZIP')
        if len(members) != 33126:
            raise RuntimeError(f'Unexpected image count: {len(members)}')
        for member in members:
            # Reading a full member checks its CRC. Flatten only JPEG filenames.
            data = archive.read(member)
            target = DEST / Path(member.filename).name
            tmp = target.with_suffix('.jpg.part')
            tmp.write_bytes(data)
            tmp.replace(target)
            count += 1
            if count % 1000 == 0:
                print(f'CRC checked and extracted {count}/{len(members)}', flush=True)
    Path('outputs/reports').mkdir(parents=True, exist_ok=True)
    Path('outputs/reports/download.json').write_text(json.dumps(dict(
        url=URL, bytes=size, etag=etag, images=count,
        jpeg_crc_verified=True, extraction_directory=str(DEST.resolve())), indent=2))
    print(f'Extracted {count} images. Run prepare_data.py --verify-images next.', flush=True)


if __name__ == '__main__':
    main()
