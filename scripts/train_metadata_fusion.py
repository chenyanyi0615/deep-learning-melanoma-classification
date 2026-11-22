"""Compare two binary classifiers on identical frozen ResNet image features."""

import fcntl
import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from scripts.compare_thresholds import select_threshold
from src.dataset import LesionDataset
from src.metrics import binary_metrics
from src.model import build_model
from src.train import save_checkpoint, seed_everything, write_json


ROOT = Path("outputs/metadata_fusion")
BASE = Path("outputs/padding_augmentation/stable_bce_224_seed42_padding_mean25")
REPEAT = Path("outputs/padding_seed43_repeat/stable_bce_224_seed43_padding_mean25")
SOURCE = Path("data/raw/train.csv")
FIELDS = ("age_approx", "sex", "anatom_site_general_challenge")
SEED = 42
EPOCHS = 12
PATIENCE = 3


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metadata(datasets):
    raw = pd.read_csv(SOURCE, usecols=["image_name", *FIELDS]).set_index("image_name",
                                                                      verify_integrity=True)
    matched = {}
    for split, dataset in datasets.items():
        names = dataset.frame.image_name
        if not names.is_unique or not names.isin(raw.index).all():
            raise ValueError(f"Metadata alignment failed for {split}")
        matched[split] = raw.loc[names]
    training = matched["train"]
    median = float(training.age_approx.median())
    scale = float(training.age_approx.std(ddof=0))
    categories = {column: sorted(training[column].dropna().unique().tolist())
                  for column in FIELDS[1:]}
    encoded = {}
    for split, frame in matched.items():
        missing = frame.age_approx.isna()
        parts = [((frame.age_approx.fillna(median) - median) / scale).tolist(), missing.tolist()]
        for column, known in categories.items():
            parts.extend(frame[column].eq(value).tolist() for value in known)
            parts.append((~frame[column].isin(known)).tolist())
        encoded[split] = torch.tensor(parts, dtype=torch.float32).T.contiguous()
    settings = dict(fields=FIELDS, age_median=median, age_scale=scale, categories=categories,
                    missing_counts={split: frame.isna().sum().to_dict()
                                    for split, frame in matched.items()})
    return encoded, settings


def features(model, dataset, split):
    path = ROOT / f"{split}_features.pt"
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        values = saved["features"]
        names = saved["names"]
    else:
        loader = DataLoader(dataset, batch_size=24, num_workers=2, shuffle=False)
        chunks, names = [], []
        with torch.no_grad():
            for batch, (images, _, ids) in enumerate(loader, 1):
                result = model(images.to("mps")).cpu()
                if not torch.isfinite(result).all():
                    raise ValueError("Invalid image features")
                chunks.append(result)
                names.extend(ids)
                if batch == 1 or batch % 50 == 0:
                    write_json(ROOT / "status.json", dict(state="running",
                               phase=f"extract_{split}", images=len(names), total=len(dataset)))
                    print(f"{split}: extracted {len(names)}/{len(dataset)} images", flush=True)
        values = torch.cat(chunks)
        if names == dataset.frame.image_name.tolist() and values.shape == (len(dataset), 2048):
            save_checkpoint(path, dict(features=values, names=names))
    if names != dataset.frame.image_name.tolist() or values.shape != (len(dataset), 2048):
        raise ValueError(f"Image feature alignment failed for {split}")
    return values


class FusionHead(nn.Module):
    def __init__(self, common_head, metadata_width=0):
        super().__init__()
        if metadata_width:
            self.metadata = nn.Sequential(nn.Linear(metadata_width, 32), nn.SiLU(),
                                          nn.Linear(32, 8), nn.SiLU())
        else:
            self.metadata = None
        self.classifier = nn.Linear(2048 + (8 if metadata_width else 0), 1)
        with torch.no_grad():
            self.classifier.weight.zero_()
            self.classifier.weight[:, :2048].copy_(common_head.weight)
            self.classifier.bias.copy_(common_head.bias)

    def forward(self, image, meta):
        if self.metadata is not None:
            image = torch.cat([image, self.metadata(meta)], dim=1)
        return self.classifier(image).flatten()


def evaluate(target, scores):
    result = binary_metrics(target, scores, select_threshold(target, scores, .90))
    (tn, _), (_, tp) = result["confusion_matrix"]
    result.update(ap=float(average_precision_score(target, scores)),
                  accuracy=(tn + tp) / len(target))
    return result


def train_head(name, head, train, validation, val_frame):
    output = ROOT / name
    output.mkdir(exist_ok=True)
    optimizer = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=1e-4)
    batches = DataLoader(train, batch_size=128, shuffle=True,
                         generator=torch.Generator().manual_seed(SEED))
    loss_fn = nn.BCEWithLogitsLoss()
    best_key, bad_epochs = (-1, -1), 0
    history = []
    for epoch in range(1, EPOCHS + 1):
        head.train()
        total_loss = 0.
        for image, meta, target in batches:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(head(image, meta), target)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(target)
        head.eval()
        image, meta, target = validation
        with torch.no_grad():
            scores = head(image, meta).sigmoid().numpy()
        metrics = evaluate(target.numpy(), scores)
        history.append(dict(epoch=epoch, train_loss=total_loss / len(train), **metrics))
        key = (metrics["ap"], metrics["auroc"])
        if key > best_key:
            best_key, bad_epochs = key, 0
            best = dict(epoch=epoch, **metrics)
            save_checkpoint(output / "best_head.pt", head.state_dict())
            val_frame.assign(probability=scores).to_csv(output / "validation_predictions.csv",
                                                        index=False)
            write_json(output / "best_metrics.json", best)
        else:
            bad_epochs += 1
        pd.DataFrame(history).drop(columns="confusion_matrix").to_csv(output / "history.csv",
                                                                      index=False)
        print(f"{name} epoch {epoch}: AP={metrics['ap']:.5f}", flush=True)
        if bad_epochs >= PATIENCE:
            break
    return dict(best, completed_epochs=len(history))


def references(frame):
    scores = []
    for folder in (BASE, REPEAT):
        pred = pd.read_csv(folder / "best_validation_predictions.csv", float_precision="round_trip")
        matched = frame.merge(pred, on=["image_name", "target"], how="left", validate="one_to_one")
        if len(pred) != len(frame) or matched.probability.isna().any():
            raise ValueError("Historical validation predictions do not match")
        scores.append(matched.probability.to_numpy())
    return {"原单模型 seed42": evaluate(frame.target, scores[0]),
            "双 ResNet 等权集成": evaluate(frame.target, (scores[0] + scores[1]) / 2)}


def write_report(results, baseline_replay_error):
    write_json(ROOT / "comparison.json", dict(results=results, validation_only=True,
               baseline_replay_max_absolute_error=baseline_replay_error))
    lines = [
        "# 冻结 ResNet-50 的图像＋元数据对照",
        "",
        "两组共用 seed42 填充增强 ResNet-50 的冻结图像特征和固定的患者级划分。"
        "图像＋元数据组加入年龄、性别和病灶部位。两组用普通 BCE，按验证 AP 选最佳轮次，AUROC 用于并列时排序。",
        "阈值在同一验证集独立选择，目标敏感度至少 90%。此实验没有运行测试集。",
        "",
        "| 方案 | 最佳轮次 | AP | AUROC | 敏感度 | 特异度 | Precision | Accuracy | 误报 | 漏检 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, m in results.items():
        (_, fp), (fn, _) = m["confusion_matrix"]
        lines.append(f"| {name} | {m.get('epoch', '—')} | {m['ap']:.4f} | {m['auroc']:.4f} | "
                     f"{m['sensitivity']:.2%} | {m['specificity']:.2%} | {m['precision']:.2%} | "
                     f"{m['accuracy']:.2%} | {fp} | {fn} |")
    a, b = results["冻结骨干＋图像分类头"], results["冻结骨干＋图像和元数据"]
    delta_fp = b["confusion_matrix"][0][1] - a["confusion_matrix"][0][1]
    lines += ["", f"添加元数据后，AP 变化 {b['ap']-a['ap']:+.4f}，AUROC 变化 "
              f"{b['auroc']-a['auroc']:+.4f}，误报变化 {delta_fp:+d} 张。"]
    (ROOT / "中文实验报告.md").write_text("\n".join(lines) + "\n")


def run():
    status = ROOT / "status.json"
    if (ROOT / "comparison.json").exists():
        raise RuntimeError("This experiment already has a saved result")
    try:
        torch.set_num_threads(4)
        seed_everything(SEED)
        write_json(status, dict(state="running", phase="preparing"))
        manifests = {split: Path(f"data/splits/{split}.csv") for split in ("train", "val")}
        source_config = json.loads((BASE / "config.json").read_text())
        hashes = {split: sha(path) for split, path in manifests.items()}
        if any(hashes[split] != source_config[f"{split}_manifest_sha256"] for split in hashes):
            raise ValueError("The train/validation manifest differs from the frozen backbone run")
        datasets = {split: LesionDataset(path) for split, path in manifests.items()}
        if set(datasets["train"].frame.patient_id) & set(datasets["val"].frame.patient_id):
            raise ValueError("Train and validation patients overlap")
        encoded, encoding = metadata(datasets)
        weights = BASE / "best_weights.pt"
        config = dict(seed=SEED, manifest_sha256=hashes, weights_sha256=sha(weights),
                      metadata_sha256=sha(SOURCE), encoding=encoding, preprocessing="fixed_gray_224",
                      epochs=EPOCHS, patience=PATIENCE, head_batch_size=128, learning_rate=3e-4,
                      checkpoint_selection="validation AP, then AUROC", loss="BCE")
        config_path = ROOT / "config.json"
        serialized = json.loads(json.dumps(config, allow_nan=False))
        if config_path.exists() and json.loads(config_path.read_text()) != serialized:
            raise ValueError("Cached experiment inputs or settings changed")
        write_json(config_path, config)
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS unavailable for frozen feature extraction")
        model = build_model(pretrained=False)
        model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        original_head = model.fc
        model.fc = nn.Identity()
        model.eval().requires_grad_(False).to("mps")
        val_frame = datasets["val"].frame[["image_name", "target"]]
        historic = references(val_frame)
        tensors = {split: features(model, dataset, split) for split, dataset in datasets.items()}
        model.to("cpu")
        del model
        torch.mps.empty_cache()
        original_scores = pd.read_csv(BASE / "best_validation_predictions.csv",
                                      float_precision="round_trip")
        expected = val_frame.merge(original_scores, on=["image_name", "target"],
                                   validate="one_to_one").probability.to_numpy()
        with torch.no_grad():
            replay = original_head(tensors["val"]).flatten().sigmoid().numpy()
        replay_error = float(abs(expected - replay).max())
        if replay_error > 2e-4:
            raise ValueError(f"Frozen backbone does not reproduce saved validation scores: {replay_error}")
        data = {split: (tensors[split], encoded[split],
                        torch.tensor(dataset.frame.target.to_numpy(), dtype=torch.float32))
                for split, dataset in datasets.items()}
        common = nn.Linear(2048, 1)
        common.load_state_dict(original_head.state_dict())
        train = TensorDataset(*data["train"])
        results = {}
        for name, label, width in [
            ("image_only", "冻结骨干＋图像分类头", 0),
            ("image_metadata", "冻结骨干＋图像和元数据", encoded["train"].shape[1]),
        ]:
            write_json(status, dict(state="running", phase=name))
            results[label] = train_head(name, FusionHead(common, width), train, data["val"], val_frame)
        results.update(historic)
        write_report(results, replay_error)
        write_json(status, dict(state="completed", phase="finished"))
    except BaseException as exc:
        write_json(status, dict(state="failed", error=str(exc)))
        raise


if __name__ == "__main__":
    ROOT.mkdir(exist_ok=True)
    with (ROOT / "experiment.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run()
