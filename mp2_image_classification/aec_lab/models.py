"""Thin wrapper around a Hugging Face image classifier (ConvNeXt V2) that hides
all tensor handling from the notebooks."""
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval_transform(size: int = 224, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    return T.Compose([T.Resize(size), T.CenterCrop(size), T.ToTensor(), T.Normalize(mean, std)])


def train_transform(size: int = 224, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    return T.Compose([
        T.RandomResizedCrop(size, scale=(0.6, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.2, 0.2, 0.1, 0.02),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


class Classifier:
    """`classes` are display names in model-output order."""

    def __init__(self, model, classes: Sequence[str], mean=IMAGENET_MEAN, std=IMAGENET_STD,
                 size: int = 224, name: str = "model", device: Optional[torch.device] = None):
        self.device = device or get_device()
        self.model = model.float().eval().to(self.device)
        self.classes = list(classes)
        self.mean, self.std, self.size = list(mean), list(std), size
        self.name = name
        self.transform = eval_transform(size, mean, std)

    # ----- persistence
    @classmethod
    def load(cls, path, name: Optional[str] = None, device=None) -> "Classifier":
        import transformers
        from transformers import AutoModelForImageClassification
        path = Path(path)
        old = transformers.logging.get_verbosity()
        transformers.logging.set_verbosity_error()
        try:
            model = AutoModelForImageClassification.from_pretrained(str(path))
        finally:
            transformers.logging.set_verbosity(old)
        id2label = model.config.id2label
        classes = [id2label[i] for i in range(len(id2label))]
        mean, std, size = IMAGENET_MEAN, IMAGENET_STD, 224
        pp = path / "preprocessor_config.json"
        if pp.exists():
            d = json.loads(pp.read_text())
            mean, std = d.get("image_mean", mean), d.get("image_std", std)
            s = d.get("size", {})
            if isinstance(s, dict):
                size = s.get("shortest_edge") or s.get("height") or size
        return cls(model, classes, mean, std, size, name or path.name, device)

    def save(self, path, half: bool = True):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        m = self.model.to("cpu")
        m.config.id2label = {i: c for i, c in enumerate(self.classes)}
        m.config.label2id = {c: i for i, c in enumerate(self.classes)}
        (m.half() if half else m).save_pretrained(str(path), safe_serialization=True)
        (path / "preprocessor_config.json").write_text(json.dumps({
            "image_mean": self.mean, "image_std": self.std, "size": {"shortest_edge": self.size}}, indent=2))
        self.model = m.float().to(self.device)

    # ----- inference
    @torch.no_grad()
    def predict_proba(self, images: Sequence[Image.Image], batch_size: int = 64) -> np.ndarray:
        out = []
        use_amp = self.device.type == "cuda"
        for b in range(0, len(images), batch_size):
            x = torch.stack([self.transform(im.convert("RGB")) for im in images[b:b + batch_size]]).to(self.device)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                logits = self.model(pixel_values=x).logits
            out.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, len(self.classes)))

    def predict_paths(self, paths: Sequence[Path], batch_size: int = 64, progress: bool = False) -> np.ndarray:
        it = range(0, len(paths), batch_size)
        if progress:
            from tqdm.auto import tqdm
            it = tqdm(it, desc=f"{self.name}: classifying", unit="batch", leave=False)
        out = []
        for b in it:
            ims = [Image.open(p).convert("RGB") for p in paths[b:b + batch_size]]
            out.append(self.predict_proba(ims, batch_size))
        return np.concatenate(out) if out else np.zeros((0, len(self.classes)))

    def predict_one(self, image: Image.Image) -> Dict[str, float]:
        p = self.predict_proba([image])[0]
        return {c: float(v) for c, v in zip(self.classes, p)}

    def top(self, image: Image.Image):
        d = self.predict_one(image)
        label = max(d, key=d.get)
        return label, d[label], d
