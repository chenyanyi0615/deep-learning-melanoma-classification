"""One real training batch; checks execution, not model performance."""
import json
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader, Subset

from src.dataset import LesionDataset
from src.losses import make_loss
from src.model import build_model


def main():
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.hub.set_dir(str(Path('data/pretrained').resolve()))
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    dataset = LesionDataset('data/splits/train.csv', training=True)
    indices = dataset.frame[dataset.frame.target.eq(0)].index[:4].tolist()
    indices += dataset.frame[dataset.frame.target.eq(1)].index[:4].tolist()
    images, targets, _ = next(iter(DataLoader(Subset(dataset, indices), batch_size=8)))
    assert images.shape == (8, 3, 224, 224) and torch.isfinite(images).all()
    print(f'Loading ImageNet pretrained weights; device={device}', flush=True)
    model = build_model().to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    old = model.fc.weight.detach().clone()
    started = time.monotonic()
    logits = model(images.to(device)).flatten()
    loss = make_loss('bce', dataset.frame.target, device)(logits, targets.to(device))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(model.fc.weight.grad).all()
    optimizer.step()
    assert not torch.equal(old, model.fc.weight)
    if device == 'mps':
        torch.mps.synchronize()
    report = dict(purpose='single-batch execution smoke test; not an experiment result',
                  device=device, images=8, input_shape=list(images.shape),
                  finite_loss_and_gradients=True, parameters_updated=True,
                  seconds=time.monotonic()-started)
    Path('outputs/reports/training_smoke.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
