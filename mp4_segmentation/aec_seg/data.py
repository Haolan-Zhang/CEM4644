"""Plan sets (image + answer key + credit) and the store of precomputed SAM 3 masks."""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

from .engine import SegResult


@dataclass
class Plan:
    id: str
    path: Path
    key: dict            # the answer key written by build/prepare_plans.py
    title: str
    author: str
    license: str
    source: str
    note: str = ""

    def load(self) -> Image.Image:
        return Image.open(self.path).convert("RGB")

    @property
    def label(self) -> str:
        return f"{self.id}: {self.title}"

    @property
    def size(self) -> Tuple[int, int]:
        return tuple(self.key["size"])

    @property
    def m_per_px(self) -> float:
        return float(self.key["scale_m_per_px"])

    def area_m2(self, pixels: float) -> float:
        return float(pixels) * self.m_per_px ** 2

    @property
    def floor_area_m2(self) -> float:
        return float(self.key["floor_area_m2"])

    @property
    def drawing_pixels(self) -> int:
        w, h = self.key["drawing_size"]
        return int(w * h)

    # ----- ground truth helpers
    def rooms(self, room_type: Optional[str] = None, indoor_only: bool = False) -> List[dict]:
        rs = self.key["rooms"]
        if room_type:
            rs = [r for r in rs if r["type"] == room_type]
        if indoor_only:
            rs = [r for r in rs if r["type"] not in ("balcony / terrace", "garage")]
        return rs

    def room_mask(self, room: dict) -> np.ndarray:
        m = Image.new("1", self.size, 0)
        ImageDraw.Draw(m).polygon([tuple(p) for p in room["poly"]], fill=1)
        return np.asarray(m, dtype=bool)

    def truth_boxes(self, what: str, sub: Optional[str] = None) -> List[List[float]]:
        """Boxes of doors / windows / fixtures (optionally one fixture type) / rooms (optionally one type)."""
        if what == "doors":
            return [list(b) for b in self.key["doors"]]
        if what == "windows":
            return [list(b) for b in self.key["windows"]]
        if what == "fixtures":
            return [list(f["box"]) for f in self.key["fixtures"] if sub is None or f["type"] == sub]
        if what == "rooms":
            return [list(r["box"]) for r in self.rooms(sub)]
        return []

    def truth_area_m2(self, what: str, sub: Optional[str] = None) -> Optional[float]:
        if what == "rooms":
            return round(sum(r["area_m2"] for r in self.rooms(sub)), 1)
        return None

    def facts(self) -> str:
        k = self.key
        types = {}
        for r in self.rooms(indoor_only=True):
            types[r["type"]] = types.get(r["type"], 0) + 1
        return (f"{k['n_rooms']} rooms ({', '.join(f'{n} {t}' for t, n in sorted(types.items(), key=lambda kv: -kv[1]))}), "
                f"floor area {k['floor_area_m2']} m², {k['n_doors']} doors, {k['n_windows']} windows; 1 px = {self.m_per_px * 100:.2f} cm")


class PlanSet:
    def __init__(self, folder: Path, name: str = ""):
        self.folder = Path(folder)
        self.name = name or self.folder.name
        credits = json.loads((self.folder / "credits.json").read_text())
        self.plans: List[Plan] = []
        for c in credits:
            key = json.loads((self.folder / f"{c['id']}.key.json").read_text())
            self.plans.append(Plan(c["id"], self.folder / c["file"], key, c["title"], c["author"], c["license"], c["source"], c.get("note", "")))

    def __len__(self):
        return len(self.plans)

    def __getitem__(self, key) -> Plan:
        if isinstance(key, int):
            return self.plans[key]
        for p in self.plans:
            if p.id == key or p.label == key or key.startswith(p.id + ":"):
                return p
        raise KeyError(key)

    def ids(self) -> List[str]:
        return [p.id for p in self.plans]

    def labels(self) -> List[str]:
        return [p.label for p in self.plans]

    def credits_text(self) -> str:
        return "\n".join(f"- {p.id}: {p.title}. {p.author}, {p.license}. {p.source}" for p in self.plans)


class MaskStore:
    """data/masks/<set>/<plan_id>/<prompt_key>.png (uint8 label map) + index.json with scores and boxes."""

    def __init__(self, folder: Path):
        self.folder = Path(folder)

    @staticmethod
    def key(prompt: str) -> str:
        return "".join(ch if ch.isalnum() else "_" for ch in prompt.strip().lower()).strip("_")

    def _index(self, plan_id: str) -> dict:
        p = self.folder / plan_id / "index.json"
        return json.loads(p.read_text()) if p.exists() else {}

    def has(self, plan_id: str, prompt: str) -> bool:
        return self.key(prompt) in self._index(plan_id)

    def prompts(self, plan_id: str) -> List[str]:
        return [v["prompt"] for v in self._index(plan_id).values()]

    def load(self, plan_id: str, prompt: str) -> Optional[SegResult]:
        idx = self._index(plan_id)
        k = self.key(prompt)
        if k not in idx:
            return None
        entry = idx[k]
        lab = np.array(Image.open(self.folder / plan_id / f"{k}.png"))
        return SegResult.from_label_map(entry["prompt"], lab, entry["scores"], entry["boxes"])

    def save(self, plan_id: str, res: SegResult):
        d = self.folder / plan_id
        d.mkdir(parents=True, exist_ok=True)
        k = self.key(res.prompt)
        Image.fromarray(res.label_map()).save(d / f"{k}.png", optimize=True)
        idx = self._index(plan_id)
        idx[k] = {"prompt": res.prompt, "scores": [round(float(s), 4) for s in res.scores],
                  "boxes": [[round(float(v), 1) for v in b] for b in res.boxes], "area_pct": round(res.area_pct(), 2),
                  "n": int(len(res))}
        (d / "index.json").write_text(json.dumps(idx, indent=1))
