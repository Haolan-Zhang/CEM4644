"""The example sets a variant works with: photos with a true class, site photos with every object boxed, floor plans with
answer keys; each also carries what the earlier lab's specialist model said."""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

from .client import file_key
from .config import LabSpec


@dataclass
class Photo:
    file: str
    path: Path
    truth: str
    specialist: dict = field(default_factory=dict)      # {"label", "confidence"}

    def load(self) -> Image.Image:
        return Image.open(self.path).convert("RGB")

    @property
    def cache_key(self) -> str:
        return file_key(self.path)

    @property
    def label(self) -> str:
        return f"{self.file.rsplit('.', 1)[0]}  (truth: {self.truth})"


@dataclass
class Site:
    file: str
    path: Path
    size: Tuple[int, int]
    truth: List[dict]                                  # [{"label", "box": [x1, y1, x2, y2]}]
    specialist: List[dict] = field(default_factory=list)   # [{"label", "box", "conf"}]

    def load(self) -> Image.Image:
        return Image.open(self.path).convert("RGB")

    @property
    def cache_key(self) -> str:
        return file_key(self.path)

    @property
    def label(self) -> str:
        return f"{self.file.rsplit('.', 1)[0]}  ({len(self.truth)} boxes in the answer key)"

    def counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for t in self.truth:
            out[t["label"]] = out.get(t["label"], 0) + 1
        return out


@dataclass
class Plan:
    """A floor plan with its answer key. Two key layouts are read: MP4's sheets as converted by build/prepare_data.py
    (units "ft", every room with "area" and "indoor") and the older metre-based copies of the Finnish scans
    ("scale_m_per_px", "area_m2"); both end up with the same fields."""
    id: str
    path: Path
    key: dict
    title: str
    specialist: dict = field(default_factory=dict)

    def __post_init__(self):
        k = self.key
        if "units" not in k:                                   # the older metre-based layout
            k["units"] = "m"
            k["scale_units_per_px"] = float(k["scale_m_per_px"])
            k["floor_area"] = float(k["floor_area_m2"])
            for r in k["rooms"]:
                r.setdefault("area", float(r["area_m2"]))
                r.setdefault("indoor", r.get("type") not in ("balcony / terrace", "garage"))
        for r in self.specialist.get("rooms", []):
            if "sam_area" not in r:
                r["sam_area"] = r.get("sam_m2"); r["truth_area"] = r.get("truth_m2")

    def load(self) -> Image.Image:
        return Image.open(self.path).convert("RGB")

    @property
    def cache_key(self) -> str:
        return file_key(self.path)

    @property
    def label(self) -> str:
        return f"{self.id}: {self.title}"

    @property
    def size(self) -> Tuple[int, int]:
        return tuple(self.key["size"])

    @property
    def units(self) -> str:
        return self.key["units"]

    @property
    def unit_label(self) -> str:
        return "sq ft" if self.units == "ft" else "m²"

    @property
    def scale(self) -> float:
        """Plan units (ft or m) per pixel."""
        return float(self.key["scale_units_per_px"])

    @property
    def scale_label(self) -> str:
        return f"1 ft = {1 / self.scale:.1f} px" if self.units == "ft" else f"1 px = {self.scale * 100:.2f} cm"

    def area(self, pixels: float) -> float:
        return float(pixels) * self.scale ** 2

    def rooms(self, indoor_only: bool = True) -> List[dict]:
        rs = self.key["rooms"]
        return [r for r in rs if r.get("indoor", True)] if indoor_only else list(rs)

    def room_mask(self, room: dict) -> np.ndarray:
        m = Image.new("1", self.size, 0)
        ImageDraw.Draw(m).polygon([tuple(p) for p in room["poly"]], fill=1)
        return np.asarray(m, dtype=bool)

    @property
    def floor_area(self) -> float:
        return float(self.key["floor_area"])


class Examples:
    def __init__(self, root: Path, spec: LabSpec):
        self.root = Path(root)
        self.spec = spec
        d = self.root / spec.folder
        ph = json.loads((d / "photos" / "index.json").read_text())
        self.photo_classes: List[str] = ph["classes"]
        self.photo_specialist: str = ph["specialist_name"]
        self.photos: List[Photo] = [Photo(it["file"], d / "photos" / it["file"], it["truth"], it.get("specialist", {})) for it in ph["items"]]
        st = json.loads((d / "sites" / "index.json").read_text())
        self.site_classes: List[str] = st["classes"]
        self.site_specialist: str = st["specialist_name"]
        self.sites: List[Site] = [Site(it["file"], d / "sites" / it["file"], tuple(it["size"]), it["truth"], it.get("specialist", [])) for it in st["items"]]
        pl = json.loads((d / "plans" / "index.json").read_text())
        self.plans: List[Plan] = [Plan(it["id"], d / "plans" / it["file"], json.loads((d / "plans" / it["key"]).read_text()), it["title"], it.get("specialist", {}))
                                  for it in pl["items"]]
        self.plan_credit: str = pl.get("credit", "")
        self.credits: dict = json.loads((d / "credits.json").read_text())

    def photo(self, which) -> Photo:
        if isinstance(which, int):
            return self.photos[which]
        for p in self.photos:
            if which in (p.file, p.label, p.file.rsplit(".", 1)[0]):
                return p
        raise KeyError(which)

    def site(self, which) -> Site:
        if isinstance(which, int):
            return self.sites[which]
        for s in self.sites:
            if which in (s.file, s.label, s.file.rsplit(".", 1)[0]):
                return s
        raise KeyError(which)

    def plan(self, which) -> Plan:
        if isinstance(which, int):
            return self.plans[which]
        for p in self.plans:
            if which in (p.id, p.label) or str(which).startswith(p.id + ":"):
                return p
        raise KeyError(which)
