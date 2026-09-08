"""A small, readable fine-tuning loop (no Trainer) so it behaves the same on
transformers 4.x and 5.x, on CPU and GPU, and can report progress to widgets."""
import random
import time
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .data import ImageSet
from .models import Classifier, get_device, train_transform

LEADERBOARD: List[dict] = []


class _DS(Dataset):
    def __init__(self, image_set: ImageSet, transform):
        self.items, self.transform = image_set.items, transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        return self.transform(Image.open(p).convert("RGB")), y


def _seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def build_model(base_model_dir, num_classes: int, pretrained: bool = True):
    import transformers
    from transformers import AutoConfig, AutoModelForImageClassification
    base_model_dir = str(base_model_dir)
    labels = {"id2label": {i: str(i) for i in range(num_classes)}, "label2id": {str(i): i for i in range(num_classes)}}
    old = transformers.logging.get_verbosity()
    transformers.logging.set_verbosity_error()  # hide the expected "classifier re-initialised" report
    try:
        if pretrained:
            return AutoModelForImageClassification.from_pretrained(
                base_model_dir, num_labels=num_classes, ignore_mismatched_sizes=True, **labels)
        cfg = AutoConfig.from_pretrained(base_model_dir, num_labels=num_classes, **labels)
        return AutoModelForImageClassification.from_config(cfg)
    finally:
        transformers.logging.set_verbosity(old)


def accuracy(clf: Classifier, test_set: ImageSet, max_images: Optional[int] = None, seed: int = 0):
    ts = test_set.stratified(total=max_images, seed=seed) if max_images and max_images < len(test_set) else test_set
    probs = clf.predict_paths(ts.paths())
    pred = probs.argmax(1)
    return float((pred == np.array(ts.labels())).mean()), len(ts)


def quick_train(train_set: ImageSet, test_set: ImageSet, base_model_dir, *, n_train: Optional[int] = None,
                epochs: int = 1, pretrained: bool = True, lr: Optional[float] = None, batch_size: Optional[int] = None,
                seed: int = 0, max_test: Optional[int] = 400, name: str = "my model",
                log: Callable[[str], None] = print, progress: bool = True, num_workers: int = 2,
                label_smoothing: float = 0.0, head_lr_mult: float = 10.0):
    """Fine-tune a copy of the base model. Returns (Classifier, history).

    Small runs get a smaller batch (more optimizer steps) and the freshly initialised classifier
    head learns faster than the pretrained body, so even 20 photos / 1 pass moves the needle."""
    _seed(seed)
    device = get_device()
    sub = train_set.stratified(total=n_train, seed=seed) if n_train and n_train < len(train_set) else train_set
    model = build_model(base_model_dir, len(train_set.classes), pretrained).to(device)
    lr = lr or (2e-4 if pretrained else 1e-3)
    batch_size = batch_size or int(min(32, max(8, len(sub) // 4)))
    head = [p for n, p in model.named_parameters() if "classifier" in n]
    body = [p for n, p in model.named_parameters() if "classifier" not in n]
    mult = head_lr_mult if pretrained else 1.0
    opt = torch.optim.AdamW([{"params": body, "lr": lr}, {"params": head, "lr": lr * mult}], weight_decay=0.05)
    loader = DataLoader(_DS(sub, train_transform()), batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, drop_last=False, pin_memory=device.type == "cuda")
    steps = max(1, epochs * len(loader))
    if steps >= 20:
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[lr, lr * mult], total_steps=steps, pct_start=0.15)
    else:  # too few steps for a warm-up/anneal cycle: constant rate after one warm-up step
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda st: 0.5 if st == 0 else 1.0)
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    clf = Classifier(model, train_set.pretty_classes, name=name, device=device)
    history = []
    log(f"Training '{name}' on {len(sub)} {('pretrained' if pretrained else 'randomly initialised')} "
        f"for {epochs} pass(es) over the data, on {'GPU' if device.type == 'cuda' else 'CPU'}.")
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        it = loader
        if progress:
            from tqdm.auto import tqdm
            it = tqdm(loader, desc=f"pass {ep}/{epochs}", unit="batch", leave=False)
        tot, n = 0.0, 0
        for x, y in it:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
                loss = loss_fn(model(pixel_values=x).logits, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item() * len(y); n += len(y)
        model.eval()
        acc, n_test = accuracy(clf, test_set, max_test, seed)
        history.append({"pass": ep, "train_loss": tot / max(1, n), "test_accuracy": acc})
        log(f"  pass {ep}: training loss {tot / max(1, n):.3f}  |  accuracy on {n_test} unseen test photos: {acc * 100:.1f}%")
    log(f"Done in {time.time() - t0:.0f} s.")
    return clf, history


def add_to_leaderboard(name: str, n_train: int, epochs: int, pretrained: bool, acc: float, seconds: float, note: str = ""):
    LEADERBOARD.append({"run": len(LEADERBOARD) + 1, "name": name, "training photos": n_train, "passes": epochs,
                        "start": "pretrained" if pretrained else "random", "test accuracy": round(acc * 100, 1),
                        "time (s)": round(seconds), "note": note})


def leaderboard_table():
    import pandas as pd
    if not LEADERBOARD:
        return pd.DataFrame(columns=["run", "name", "training photos", "passes", "start", "test accuracy", "time (s)"])
    return pd.DataFrame(LEADERBOARD).sort_values("test accuracy", ascending=False).reset_index(drop=True)
