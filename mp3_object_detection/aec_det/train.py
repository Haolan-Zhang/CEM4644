"""One-call fine-tuning of a YOLO detector on a small subset (Ultralytics under the hood)."""
import os
os.environ.setdefault("YOLO_VERBOSE", "False")

import time
from pathlib import Path
from typing import Callable, List, Optional

from .data import DetSet, write_data_yaml
from .models import Detector, get_device

LEADERBOARD: List[dict] = []


def quick_train(train_set: DetSet, val_set: DetSet, base_weights, *, n_train: Optional[int] = None, epochs: int = 3,
                imgsz: int = 640, batch: int = 16, pretrained: bool = True, name: str = "my model",
                workdir=Path("_runs"), log: Callable[[str], None] = print, seed: int = 0, arch_yaml: str = "yolo11n.yaml",
                extra: Optional[dict] = None):
    """Returns (Detector, history). history: one dict per pass with mAP50 / precision / recall on the validation photos."""
    from ultralytics import YOLO
    device = get_device()
    sub = train_set.subset(n_train, seed)
    run_dir = (Path(workdir) / f"{int(time.time())}_{''.join(ch if ch.isalnum() else '_' for ch in name)}").resolve()
    sub.write_yolo(run_dir, "train")
    val_set.write_yolo(run_dir, "valid")
    yaml = write_data_yaml(run_dir, train_set.classes)
    model = YOLO(str(base_weights) if pretrained else arch_yaml)
    history: List[dict] = []

    def on_epoch_end(trainer):
        if len(history) >= epochs:  # Ultralytics fires this once more after its final validation pass
            return
        m = trainer.metrics or {}
        rec = {"pass": int(trainer.epoch) + 1,
               "mAP50": float(m.get("metrics/mAP50(B)", float("nan"))),
               "precision": float(m.get("metrics/precision(B)", float("nan"))),
               "recall": float(m.get("metrics/recall(B)", float("nan")))}
        history.append(rec)
        log(f"  pass {rec['pass']}/{epochs}: quality score (mAP50) {rec['mAP50'] * 100:.1f}   "
            f"precision {rec['precision'] * 100:.0f}%   recall {rec['recall'] * 100:.0f}%   (on {len(val_set)} validation photos)")

    model.add_callback("on_fit_epoch_end", on_epoch_end)
    where = "GPU" if device != "cpu" else "CPU (slow: keep the number of photos and passes small)"
    log(f"Training '{name}' on {len(sub)} photos for {epochs} pass(es), {'pretrained' if pretrained else 'random'} start, on {where}.")
    t0 = time.time()
    kw = dict(data=str(yaml), epochs=epochs, imgsz=imgsz, batch=batch, device=device, project=str(run_dir), name="train",
              exist_ok=True, verbose=False, plots=False, workers=2, seed=seed, pretrained=pretrained, val=True,
              amp=device != "cpu", warmup_epochs=0.5, mosaic=0.0, close_mosaic=0, patience=100, deterministic=False)
    kw.update(extra or {})
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.train(**kw)
    save_dir = Path(getattr(model.trainer, "save_dir", run_dir / "train"))
    best = save_dir / "weights" / "best.pt"
    if not best.exists():
        best = save_dir / "weights" / "last.pt"
    if not best.exists():  # last resort: search below the working folder
        cands = sorted(Path(workdir).resolve().rglob("best.pt"), key=lambda q: q.stat().st_mtime)
        best = cands[-1] if cands else best
    log(f"Done in {time.time() - t0:.0f} s.")
    det = Detector(best, name=name, classes=train_set.pretty_classes)
    return det, history


def add_to_leaderboard(name: str, n_train: int, epochs: int, pretrained: bool, map50: float, recall: float, precision: float,
                       seconds: float):
    LEADERBOARD.append({"run": len(LEADERBOARD) + 1, "name": name, "training photos": n_train, "passes": epochs,
                        "start": "pretrained" if pretrained else "random", "quality (mAP50)": round(map50 * 100, 1),
                        "recall % @0.5": round(recall * 100), "precision % @0.5": round(precision * 100), "time (s)": round(seconds)})


def leaderboard_table():
    import pandas as pd
    cols = ["run", "name", "training photos", "passes", "start", "quality (mAP50)", "recall % @0.5", "precision % @0.5", "time (s)"]
    if not LEADERBOARD:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(LEADERBOARD)[cols].sort_values("quality (mAP50)", ascending=False).reset_index(drop=True)
