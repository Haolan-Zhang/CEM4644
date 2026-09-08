"""YOLO-format image sets: <root>/<split>/images/*.jpg + <root>/<split>/labels/*.txt"""
import json
import random
import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from PIL import Image

IMG_EXT = (".jpg", ".jpeg", ".png")


def unzip_dataset(zip_path: Path, dest_root: Path) -> Path:
    zip_path, dest_root = Path(zip_path), Path(dest_root)
    target = dest_root / zip_path.stem
    if not (target / "about.json").exists():
        dest_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dest_root)
    return target


@dataclass
class Item:
    path: Path
    boxes: np.ndarray            # [n, 4] normalised cx, cy, w, h  (YOLO)
    cls: np.ndarray              # [n] int
    size: Optional[tuple] = None  # (w, h) lazily read

    def wh(self):
        if self.size is None:
            with Image.open(self.path) as im:
                self.size = im.size
        return self.size

    def xyxy(self) -> np.ndarray:
        """Ground-truth boxes in pixels [n, 4]."""
        w, h = self.wh()
        if len(self.boxes) == 0:
            return np.zeros((0, 4))
        cx, cy, bw, bh = self.boxes.T
        return np.stack([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h], 1)

    def load(self) -> Image.Image:
        im = Image.open(self.path).convert("RGB")
        self.size = im.size
        return im


class DetSet:
    def __init__(self, items: Sequence[Item], classes: Sequence[str], display: Optional[Dict[str, str]] = None, name: str = ""):
        self.items = list(items)
        self.classes = list(classes)
        self.display = dict(display or {c: c for c in classes})
        self.name = name

    @classmethod
    def from_folder(cls, split_dir: Path, classes: Sequence[str], display=None, name: str = "") -> "DetSet":
        split_dir = Path(split_dir)
        items = []
        for p in sorted((split_dir / "images").iterdir()):
            if p.suffix.lower() not in IMG_EXT:
                continue
            lab = split_dir / "labels" / (p.stem + ".txt")
            rows = []
            if lab.exists():
                for line in lab.read_text().splitlines():
                    parts = line.split()
                    if len(parts) >= 5:
                        rows.append([float(x) for x in parts[:5]])
            arr = np.array(rows) if rows else np.zeros((0, 5))
            items.append(Item(p, arr[:, 1:5], arr[:, 0].astype(int)))
        return cls(items, classes, display, name or split_dir.name)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i) -> Item:
        return self.items[i]

    @property
    def pretty_classes(self) -> List[str]:
        return [self.display.get(c, c) for c in self.classes]

    def pretty(self, idx: int) -> str:
        c = self.classes[int(idx)]
        return self.display.get(c, c)

    def index_of(self, name: str) -> int:
        if name in self.classes:
            return self.classes.index(name)
        for i, c in enumerate(self.classes):
            if self.display.get(c, c) == name:
                return i
        raise KeyError(name)

    def box_counts(self) -> Dict[str, int]:
        out = {self.pretty(i): 0 for i in range(len(self.classes))}
        for it in self.items:
            for c in it.cls:
                out[self.pretty(c)] += 1
        return out

    def with_class(self, name: str) -> "DetSet":
        k = self.index_of(name)
        return DetSet([it for it in self.items if (it.cls == k).any()], self.classes, self.display, self.name)

    def sample(self, n: int, seed: Optional[int] = None) -> "DetSet":
        rng = random.Random(seed)
        items = list(self.items)
        rng.shuffle(items)
        return DetSet(items[:n], self.classes, self.display, self.name)

    def subset(self, n: Optional[int], seed: int = 0) -> "DetSet":
        if n is None or n >= len(self.items):
            return self
        return self.sample(n, seed)

    def count(self, i: int, class_names: Sequence[str]) -> int:
        ks = {self.index_of(c) for c in class_names}
        return int(sum(1 for c in self.items[i].cls if int(c) in ks))

    def write_yolo(self, out_dir: Path, split: str = "train"):
        """Materialise this set as <out_dir>/<split>/{images,labels} (symlinks when possible)."""
        out_dir = Path(out_dir)
        img_d, lab_d = out_dir / split / "images", out_dir / split / "labels"
        img_d.mkdir(parents=True, exist_ok=True); lab_d.mkdir(parents=True, exist_ok=True)
        for it in self.items:
            dst = img_d / it.path.name
            if not dst.exists():
                try:
                    dst.symlink_to(it.path.resolve())
                except OSError:
                    shutil.copy2(it.path, dst)
            rows = [f"{int(c)} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}" for c, b in zip(it.cls, it.boxes)]
            (lab_d / (it.path.stem + ".txt")).write_text("\n".join(rows) + ("\n" if rows else ""))
        return out_dir / split / "images"


def write_data_yaml(out_dir: Path, classes: Sequence[str], train="train/images", val="valid/images") -> Path:
    out_dir = Path(out_dir)
    txt = f"path: {out_dir.resolve()}\ntrain: {train}\nval: {val}\nnames:\n" + "".join(f"  {i}: '{c}'\n" for i, c in enumerate(classes))
    p = out_dir / "data.yaml"
    p.write_text(txt)
    return p
