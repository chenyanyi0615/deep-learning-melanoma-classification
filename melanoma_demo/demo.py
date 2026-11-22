"""Offline CPU inference for the three selected ISIC 2020 cases."""
import argparse
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / 'assets'
CONFIG = json.loads((ASSETS / 'config.json').read_text(encoding='utf-8'))


def encode(case):
    settings = CONFIG['encoding']
    age = case['age_approx']
    median = settings['age_median']
    values = [((float(age) if age else median) - median) / settings['age_scale'], not age]
    for column in ('sex', 'anatom_site_general_challenge'):
        known = settings['categories'][column]
        values.extend(case[column] == item for item in known)
        values.append(case[column] not in known)
    return torch.tensor([values], dtype=torch.float32)


def image_tensor(key):
    with Image.open(ASSETS / (key + '.jpg')) as image:
        image.draft('RGB', (448, 448))
        image = ImageOps.exif_transpose(image).convert('RGB')
        image = ImageOps.contain(image, (224, 224), Image.Resampling.BILINEAR)
        canvas = Image.new('RGB', (224, 224), (124, 116, 104))
        canvas.paste(image, ((224-image.width)//2, (224-image.height)//2))
    arr = torch.from_numpy(np.array(canvas)).permute(2, 0, 1).float().div(255)
    mean = torch.tensor([.485,.456,.406])[:,None,None]
    std = torch.tensor([.229,.224,.225])[:,None,None]
    return ((arr - mean) / std).unsqueeze(0)


def load_model():
    torch.set_num_threads(min(4, torch.get_num_threads()))
    return torch.jit.load(str(ASSETS / 'model.pt'), map_location='cpu').eval()


@torch.inference_mode()
def predict(model, key):
    a,b,m = model(image_tensor(key), encode(CONFIG['cases'][key]))[0].tolist()
    scores = dict(fusion=(a+b+m)/3, single=a, double=(a+b)/2, metadata=m)
    results = {name: dict(score=scores[name], threshold=t, flagged=scores[name] >= t)
               for name, t in CONFIG['thresholds'].items()}
    results['branches'] = dict(seed42=a, seed43=b, metadata=m)
    return results


class Handler(BaseHTTPRequestHandler):
    def send(self, content, mime):
        self.send_response(200)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(len(content)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self):
        if self.path == '/':
            return self.send((ROOT/'index.html').read_bytes(),'text/html; charset=utf-8')
        parts = self.path.strip('/').split('/')
        if len(parts)!=2 or parts[1] not in CONFIG['cases']:
            return self.send_error(404)
        action,key = parts
        case = CONFIG['cases'][key]
        if action == 'image':
            return self.send((ASSETS/(key+'.jpg')).read_bytes(),'image/jpeg')
        if action == 'case':
            value = {k:v for k,v in case.items() if k!='label'}
        elif action == 'predict':
            with self.server.inference_lock:
                value = predict(self.server.model,key)
        elif action == 'label':
            value = case['label']
        else:
            return self.send_error(404)
        self.send(json.dumps(value).encode(),'application/json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-browser',action='store_true')
    args = parser.parse_args()
    model = load_model()
    with ThreadingHTTPServer(('127.0.0.1',0),Handler) as server:
        server.model = model
        server.inference_lock = threading.Lock()
        url = f'http://127.0.0.1:{server.server_port}/'
        print('Open',url,flush=True)
        if not args.no_browser:
            webbrowser.open(url)
        server.serve_forever()
