"""Image sets stored as folders: <root>/<split>/<class>/<file>.jpg"""
import json
import random
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image

IMG_EXT = (".jpg", ".jpeg", ".png")


def unzip_dataset(zip_path: Path, dest_root: Path) -> Path:
    """Extract data/<name>.zip into dest_root/<name> once. Returns that folder."""
    zip_path, dest_root = Path(zip_path), Path(dest_root)
    name = zip_path.stem
    target = dest_root / name
    if not (target / "about.json").exists():
        dest_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dest_root)
    return target


def load_about(folder: Path) -> dict:
    p = Path(folder) / "about.json"
    return json.loads(p.read_text()) if p.exists() else {}


class ImageSet:
    """A list of (path, label_index) plus class names. Cheap to slice and relabel."""

    def __init__(self, items: Sequence[Tuple[Path, int]], classes: Sequence[str],
                 display: Optional[Dict[str, str]] = None, name: str = ""):
        self.items: List[Tuple[Path, int]] = [(Path(p), int(i)) for p, i in items]
        self.classes: List[str] = list(classes)
        self.display: Dict[str, str] = dict(display or {c: c for c in classes})
        self.name = name

    # ----- construction
    @classmethod
    def from_folder(cls, root: Path, classes: Optional[Sequence[str]] = None,
                    display: Optional[Dict[str, str]] = None, name: str = "") -> "ImageSet":
        root = Path(root)
        if classes is None:
            classes = sorted(d.name for d in root.iterdir() if d.is_dir())
        items = []
        for idx, c in enumerate(classes):
            d = root / c
            if not d.exists():
                continue
            for p in sorted(d.iterdir()):
                if p.suffix.lower() in IMG_EXT:
                    items.append((p, idx))
        return cls(items, classes, display, name or root.name)

    # ----- basics
    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]

    @property
    def pretty_classes(self) -> List[str]:
        return [self.display.get(c, c) for c in self.classes]

    def pretty(self, idx: int) -> str:
        c = self.classes[idx]
        return self.display.get(c, c)

    def counts(self) -> Dict[str, int]:
        out = {self.pretty(i): 0 for i in range(len(self.classes))}
        for _, i in self.items:
            out[self.pretty(i)] += 1
        return out

    def load(self, i: int) -> Image.Image:
        return Image.open(self.items[i][0]).convert("RGB")

    def index_of(self, class_name: str) -> int:
        """Accepts folder name or display name."""
        if class_name in self.classes:
            return self.classes.index(class_name)
        for i, c in enumerate(self.classes):
            if self.display.get(c, c) == class_name:
                return i
        raise KeyError(class_name)

    # ----- views
    def only(self, class_name: str) -> "ImageSet":
        k = self.index_of(class_name)
        return ImageSet([it for it in self.items if it[1] == k], self.classes, self.display, self.name)

    def sample(self, n: int, seed: Optional[int] = None, class_name: Optional[str] = None) -> "ImageSet":
        src = self.only(class_name) if class_name else self
        rng = random.Random(seed)
        items = list(src.items)
        rng.shuffle(items)
        return ImageSet(items[:n], self.classes, self.display, self.name)

    def stratified(self, per_class: Optional[int] = None, total: Optional[int] = None, seed: int = 0) -> "ImageSet":
        """Balanced subset: `per_class` images per class, or `total` spread evenly."""
        rng = random.Random(seed)
        by = {}
        for it in self.items:
            by.setdefault(it[1], []).append(it)
        k = len(by)
        if total is not None:
            per_class = max(1, total // k)
        out = []
        for idx in sorted(by):
            lst = list(by[idx])
            rng.shuffle(lst)
            out.extend(lst[: per_class] if per_class else lst)
        rng.shuffle(out)
        return ImageSet(out, self.classes, self.display, self.name)

    def relabel(self, mapping: Dict[str, str], new_classes: Sequence[str]) -> "ImageSet":
        """Collapse classes, e.g. 7 defect types -> defect / no defect. `mapping` uses folder names."""
        new_classes = list(new_classes)
        lut = {i: new_classes.index(mapping[c]) for i, c in enumerate(self.classes)}
        items = [(p, lut[i]) for p, i in self.items]
        return ImageSet(items, new_classes, {c: c for c in new_classes}, self.name)

    def labels(self) -> List[int]:
        return [i for _, i in self.items]

    def paths(self) -> List[Path]:
        return [p for p, _ in self.items]
