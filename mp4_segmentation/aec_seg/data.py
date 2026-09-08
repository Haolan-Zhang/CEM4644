"""Photo sets with credits, and the store of precomputed SAM 3 masks."""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

from .engine import SegResult


@dataclass
class Photo:
    id: str
    path: Path
    title: str
    author: str
    license: str
    source: str
    note: str = ""
    series: Optional[str] = None       # e.g. "A" for a time series
    order: Optional[int] = None        # position in the series
    date: Optional[str] = None

    def load(self) -> Image.Image:
        return Image.open(self.path).convert("RGB")

    @property
    def label(self) -> str:
        return f"{self.id}: {self.title}"


class PhotoSet:
    def __init__(self, folder: Path, name: str = ""):
        self.folder = Path(folder)
        self.name = name or self.folder.name
        credits = json.loads((self.folder / "credits.json").read_text())
        self.photos: List[Photo] = [Photo(c["id"], self.folder / c["file"], c["title"], c["author"], c["license"], c["source"], c.get("note", ""))
                                    for c in credits]

    def __len__(self):
        return len(self.photos)

    def __getitem__(self, key) -> Photo:
        if isinstance(key, int):
            return self.photos[key]
        for p in self.photos:
            if p.id == key or p.label == key:
                return p
        raise KeyError(key)

    def ids(self) -> List[str]:
        return [p.id for p in self.photos]

    def labels(self) -> List[str]:
        return [p.label for p in self.photos]

    def credits_text(self) -> str:
        return "\n".join(f"- {p.id}: \"{p.title}\" by {p.author}, {p.license}. {p.source}" for p in self.photos)


class MaskStore:
    """data/masks/<set>/<photo_id>/<prompt_key>.png (uint8 label map) + index.json with scores and boxes."""

    def __init__(self, folder: Path):
        self.folder = Path(folder)

    @staticmethod
    def key(prompt: str) -> str:
        return "".join(ch if ch.isalnum() else "_" for ch in prompt.strip().lower()).strip("_")

    def _index(self, photo_id: str) -> dict:
        p = self.folder / photo_id / "index.json"
        return json.loads(p.read_text()) if p.exists() else {}

    def has(self, photo_id: str, prompt: str) -> bool:
        return self.key(prompt) in self._index(photo_id)

    def prompts(self, photo_id: str) -> List[str]:
        return [v["prompt"] for v in self._index(photo_id).values()]

    def load(self, photo_id: str, prompt: str) -> Optional[SegResult]:
        idx = self._index(photo_id)
        k = self.key(prompt)
        if k not in idx:
            return None
        entry = idx[k]
        lab = np.array(Image.open(self.folder / photo_id / f"{k}.png"))
        return SegResult.from_label_map(entry["prompt"], lab, entry["scores"], entry["boxes"])

    def save(self, photo_id: str, res: SegResult):
        d = self.folder / photo_id
        d.mkdir(parents=True, exist_ok=True)
        k = self.key(res.prompt)
        Image.fromarray(res.label_map()).save(d / f"{k}.png", optimize=True)
        idx = self._index(photo_id)
        idx[k] = {"prompt": res.prompt, "scores": [round(float(s), 4) for s in res.scores],
                  "boxes": [[round(float(v), 1) for v in b] for b in res.boxes], "area_pct": round(res.area_pct(), 2),
                  "n": int(len(res))}
        (d / "index.json").write_text(json.dumps(idx, indent=1))
