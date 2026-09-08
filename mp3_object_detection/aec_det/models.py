"""Thin wrapper around an Ultralytics YOLO detector."""
import os
os.environ.setdefault("YOLO_VERBOSE", "False")

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
from PIL import Image


def get_device() -> str:
    import torch
    return "0" if torch.cuda.is_available() else "cpu"


@dataclass
class Detections:
    xyxy: np.ndarray     # [n, 4] pixels
    conf: np.ndarray     # [n]
    cls: np.ndarray      # [n] int

    def at(self, conf: float) -> "Detections":
        m = self.conf >= conf
        return Detections(self.xyxy[m], self.conf[m], self.cls[m])

    def __len__(self):
        return len(self.conf)

    @staticmethod
    def empty():
        return Detections(np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int))


class Detector:
    def __init__(self, weights, name: str = "detector", classes: Optional[Sequence[str]] = None, imgsz: int = 640):
        from ultralytics import YOLO
        self.weights = str(weights)
        self.model = YOLO(self.weights)
        names = self.model.names
        self.classes = list(classes) if classes is not None else [names[i] for i in range(len(names))]
        self.name = name
        self.imgsz = imgsz
        self.device = get_device()
        self._cache = {}

    def predict(self, image: Image.Image, conf: float = 0.01, iou: float = 0.6, key=None) -> Detections:
        """Raw detections down to a very low confidence; filter later with .at(threshold)."""
        if key is not None and key in self._cache:
            return self._cache[key]
        r = self.model.predict(image.convert("RGB"), conf=conf, iou=iou, imgsz=self.imgsz, device=self.device,
                               verbose=False, max_det=300)[0]
        b = r.boxes
        det = Detections(b.xyxy.cpu().numpy() if len(b) else np.zeros((0, 4)),
                         b.conf.cpu().numpy() if len(b) else np.zeros(0),
                         b.cls.cpu().numpy().astype(int) if len(b) else np.zeros(0, dtype=int))
        if key is not None:
            self._cache[key] = det
        return det

    def predict_paths(self, paths: Sequence[Path], conf: float = 0.01, progress: bool = False, batch: int = 16) -> List[Detections]:
        out = []
        todo = [(i, p) for i, p in enumerate(paths)]
        it = range(0, len(todo), batch)
        if progress:
            from tqdm.auto import tqdm
            it = tqdm(it, desc=f"{self.name}: detecting", unit="batch", leave=False)
        res = [None] * len(paths)
        for b0 in it:
            chunk = todo[b0:b0 + batch]
            need = [(i, p) for i, p in chunk if str(p) not in self._cache]
            if need:
                rs = self.model.predict([str(p) for _, p in need], conf=conf, iou=0.6, imgsz=self.imgsz,
                                        device=self.device, verbose=False, max_det=300)
                for (i, p), r in zip(need, rs):
                    bx = r.boxes
                    self._cache[str(p)] = Detections(
                        bx.xyxy.cpu().numpy() if len(bx) else np.zeros((0, 4)),
                        bx.conf.cpu().numpy() if len(bx) else np.zeros(0),
                        bx.cls.cpu().numpy().astype(int) if len(bx) else np.zeros(0, dtype=int))
            for i, p in chunk:
                res[i] = self._cache[str(p)]
        return res

    def pretty(self, k: int) -> str:
        return self.classes[int(k)] if 0 <= int(k) < len(self.classes) else str(k)
