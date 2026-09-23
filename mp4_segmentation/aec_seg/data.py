"""Reading a set of drawings: the image, the answer key, the credits - and the store of precomputed SAM 3 masks.

Answer key (`data/sheets/<set>/<id>.key.json`, next to `<id>.png`):

    {"id", "file", "title", "discipline", "units": "ft", "size": [W, H],
     "credit": {"title", "author", "license", "source"},
     "scale": {"px_per_ft": 20.3, "how": "..."},
     "scale_refs": [{"label", "feet", "box": [x1,y1,x2,y2], "axis": "x"|"y", "use": true, "note",
                     "from_category": "footing"   # optional: any box of that area category serves as this reference}],
     "areas":  [{"label", "type", "category", "indoor", "printed", "printed_sqft",
                 "box": [x1,y1,x2,y2], "poly": [[x,y], ...], "true_sqft", "note"}],
     "counts": {"footing": [[x1,y1,x2,y2], ...]},          # every one of them, for checking the model's count
     "count_hints": {"footing": {"threshold": 0.4, "size_range": [0.2, 5.0], "tip": ""}},
     "legend": null | "<file name>",
     "tasks": ["...", ...]}

The keys are written by hand, so `Sheets` VALIDATES every one of them and refuses to load a broken set
rather than quietly measuring against nonsense. `python -m aec_seg.data <folder>` prints the report.
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

from .config import ROOM_TYPES
from .engine import SegResult

REQUIRED = ["id", "file", "title", "discipline", "units", "size", "credit", "scale", "scale_refs", "areas",
            "counts", "count_hints", "tasks"]


class KeyError_(ValueError):
    """An answer key that does not follow the schema."""


def plural(n: int, word: str) -> str:
    """'1 pit', '2 pits', '5 lavatories' - the counting categories are plain English words."""
    if n == 1:
        return f"1 {word}"
    if word.endswith("y") and word[-2:-1] not in "aeiou":
        return f"{n} {word[:-1]}ies"
    return f"{n} {word}{'es' if word.endswith(('s', 'x', 'ch', 'sh')) else 's'}"


@dataclass
class Sheet:
    """One drawing plus its answer key."""
    id: str
    path: Path
    key: dict

    # ------------------------------------------------------------------ the drawing
    def load(self) -> Image.Image:
        return Image.open(self.path).convert("RGB")

    @property
    def title(self) -> str:
        return self.key["title"]

    @property
    def label(self) -> str:
        return f"{self.id}: {self.title}"

    @property
    def discipline(self) -> str:
        return self.key.get("discipline", "")

    @property
    def size(self) -> Tuple[int, int]:
        return tuple(self.key["size"])

    @property
    def credit(self) -> dict:
        return self.key.get("credit", {})

    @property
    def tasks(self) -> List[str]:
        return list(self.key.get("tasks", []))

    @property
    def legend_path(self) -> Optional[Path]:
        name = self.key.get("legend")
        return (self.path.parent / name) if name else None

    # ------------------------------------------------------------------ scale
    @property
    def px_per_ft(self) -> float:
        return float(self.key["scale"]["px_per_ft"])

    @property
    def scale_how(self) -> str:
        return self.key["scale"].get("how", "")

    @property
    def scale_refs(self) -> List[dict]:
        return list(self.key.get("scale_refs", []))

    def sqft(self, pixels: float, px_per_ft: Optional[float] = None) -> float:
        p = px_per_ft or self.px_per_ft
        return float(pixels) / (p * p)

    # ------------------------------------------------------------------ areas (rooms, footings, pits...)
    def areas(self, category: Optional[str] = None, type_: Optional[str] = None, indoor_only: bool = False,
              takeoff_only: bool = False) -> List[dict]:
        """takeoff_only: leave out the areas marked "takeoff": false (closets too small to box; they stay in the key
        for the ask-by-name steps but the take-off neither asks for them nor counts them)."""
        out = list(self.key.get("areas", []))
        if category:
            out = [a for a in out if a.get("category") == category]
        if type_:
            out = [a for a in out if a.get("type") == type_]
        if indoor_only:
            out = [a for a in out if a.get("indoor")]
        if takeoff_only:
            out = [a for a in out if a.get("takeoff", True)]
        return out

    @property
    def area_categories(self) -> List[str]:
        seen = []
        for a in self.key.get("areas", []):
            if a.get("category") not in seen:
                seen.append(a["category"])
        return seen

    def area_mask(self, area: dict) -> np.ndarray:
        m = Image.new("1", self.size, 0)
        ImageDraw.Draw(m).polygon([tuple(p) for p in area["poly"]], fill=1)
        return np.asarray(m, dtype=bool)

    # ------------------------------------------------------------------ counts (repeated symbols)
    @property
    def counts(self) -> Dict[str, List[List[float]]]:
        return {k: [list(b) for b in v] for k, v in self.key.get("counts", {}).items()}

    @property
    def count_categories(self) -> List[str]:
        return list(self.key.get("counts", {}))

    def hint(self, category: str) -> dict:
        h = dict(self.key.get("count_hints", {}).get(category, {}))
        h.setdefault("threshold", 0.4)
        h.setdefault("size_range", [0.2, 5.0])
        h.setdefault("tip", "")
        return h

    # ------------------------------------------------------------------ what the take-off cell offers
    def scale_label(self, ref: dict) -> str:
        return f"scale: {ref['label']}"

    def area_label(self, category: str) -> str:
        """The box label of an area category: the plain word ('room', 'footing', 'pit')."""
        return category

    def example_label(self, category: str) -> str:
        """The box label of a counting category: ONE example of it, which SAM 3 then finds everywhere."""
        return f"example: {category}"

    def box_labels(self) -> List[str]:
        """The labels of the box-drawing tool, in the order the student should draw them: the scale
        references, then the areas to measure, then one example of each thing to be counted."""
        labels = [self.scale_label(r) for r in self.scale_refs if r.get("from_category") not in self.area_categories]
        labels += [self.area_label(c) for c in self.area_categories]
        # a thing that is measured AND counted (a footing) needs no example label: the first area box is the example
        labels += [self.example_label(c) for c in self.count_categories if c not in self.area_categories]
        return labels

    # ------------------------------------------------------------------ truth lookups for the named things
    def truth_boxes(self, what: str, sub: Optional[str] = None) -> List[List[float]]:
        if what == "counts":
            return [list(b) for b in self.key.get("counts", {}).get(sub, [])]
        if what == "areas":
            return [list(a["box"]) for a in self.areas("room", sub)]
        return []

    # ------------------------------------------------------------------ text
    def facts(self) -> str:
        bits = []
        rooms = self.areas("room", takeoff_only=True)
        if rooms:
            indoor = self.areas("room", indoor_only=True, takeoff_only=True)
            bits.append(f"{plural(len(rooms), 'room')} ({sum(a['true_sqft'] for a in indoor):,.0f} sq ft indoors)")
        for c in self.area_categories:
            if c != "room":
                bits.append(f"{plural(len(self.areas(c)), c)} to measure")
        for c, boxes in self.counts.items():
            if c not in self.area_categories:                # a footing is listed once, under its areas
                bits.append(plural(len(boxes), c))
        return "; ".join(bits)

    def task_text(self) -> str:
        return "\n".join(f"  {i + 1}. {t}" for i, t in enumerate(self.tasks))

    def credit_line(self) -> str:
        c = self.credit
        return f"{c.get('title', self.title)}. {c.get('author', '')}, {c.get('license', '')}. {c.get('source', '')}".strip()


# --------------------------------------------------------------------------- validation
def _is_box(b, size) -> Optional[str]:
    if not (isinstance(b, (list, tuple)) and len(b) == 4 and all(isinstance(v, (int, float)) for v in b)):
        return "not four numbers"
    x1, y1, x2, y2 = b
    if x2 <= x1 or y2 <= y1:
        return f"not x1<x2, y1<y2: {list(b)}"
    if size and (x1 < -1 or y1 < -1 or x2 > size[0] + 1 or y2 > size[1] + 1):
        return f"outside the {size[0]}x{size[1]} image: {[round(v) for v in b]}"
    return None


def check_key(key: dict, folder: Path) -> List[str]:
    """Every way this answer key breaks the schema, in plain words. Empty list = fine."""
    bad: List[str] = []
    for k in REQUIRED:
        if k not in key:
            bad.append(f"missing '{k}'")
    if bad:
        return bad
    size = key.get("size")
    if not (isinstance(size, (list, tuple)) and len(size) == 2):
        bad.append(f"'size' is {size!r}, expected [width, height]"); size = None
    if key.get("units") != "ft":
        bad.append(f"'units' is {key.get('units')!r}, this lab is in feet ('ft')")
    img = folder / key["file"]
    if not img.exists():
        bad.append(f"the drawing '{key['file']}' is not in {folder}")
    elif size:
        with Image.open(img) as im:
            if tuple(im.size) != tuple(size):
                bad.append(f"'size' says {list(size)} but {key['file']} is {list(im.size)}")
    for f in ("title", "author", "license", "source"):
        if not key.get("credit", {}).get(f):
            bad.append(f"credit.{f} is empty")
    # scale
    if not (isinstance(key.get("scale"), dict) and isinstance(key["scale"].get("px_per_ft"), (int, float))
            and key["scale"]["px_per_ft"] > 0):
        bad.append("scale.px_per_ft is missing or not a positive number")
    refs = key.get("scale_refs") or []           # may be empty: the sheet's known px_per_ft is then used silently
    used = [r for r in refs if r.get("use")]
    if refs and len(used) != 1:
        bad.append(f"{len(used)} scale_refs have use=true, exactly one must")
    for i, r in enumerate(refs):
        for f in ("label", "feet", "box", "axis"):
            if f not in r:
                bad.append(f"scale_refs[{i}] has no '{f}'")
        if "feet" in r and not (isinstance(r["feet"], (int, float)) and r["feet"] > 0):
            bad.append(f"scale_refs[{i}] ('{r.get('label')}'): feet is {r.get('feet')!r}")
        if r.get("axis") not in ("x", "y"):
            bad.append(f"scale_refs[{i}] ('{r.get('label')}'): axis is {r.get('axis')!r}, expected 'x' or 'y'")
        if "box" in r:
            why = _is_box(r["box"], size)
            if why:
                bad.append(f"scale_refs[{i}] ('{r.get('label')}'): box {why}")
        if r.get("from_category") and r["from_category"] not in {a.get("category") for a in key.get("areas") or []}:
            bad.append(f"scale_refs[{i}] ('{r.get('label')}'): from_category '{r['from_category']}' is not an area category of this sheet")
    # areas
    for i, a in enumerate(key.get("areas") or []):
        who = f"areas[{i}] ('{a.get('label')}')"
        for f in ("label", "type", "category", "box", "poly", "true_sqft"):
            if f not in a:
                bad.append(f"{who} has no '{f}'")
        if "box" in a:
            why = _is_box(a["box"], size)
            if why:
                bad.append(f"{who}: box {why}")
        poly = a.get("poly")
        if not (isinstance(poly, (list, tuple)) and len(poly) >= 3 and all(len(p) == 2 for p in poly)):
            bad.append(f"{who}: poly needs at least 3 [x, y] points")
        if not isinstance(a.get("true_sqft"), (int, float)) or a.get("true_sqft", 0) <= 0:
            bad.append(f"{who}: true_sqft is {a.get('true_sqft')!r}")
        if a.get("category") == "room" and a.get("type") not in ROOM_TYPES:
            bad.append(f"{who}: type {a.get('type')!r} is not one of {', '.join(ROOM_TYPES)}")
        if a.get("printed_sqft") is not None and not isinstance(a["printed_sqft"], (int, float)):
            bad.append(f"{who}: printed_sqft is {a['printed_sqft']!r}")
    # counts
    counts = key.get("counts") or {}
    if not isinstance(counts, dict):
        bad.append("counts must be an object {category: [boxes]}")
    else:
        for cat, boxes in counts.items():
            if not isinstance(boxes, (list, tuple)):
                bad.append(f"counts['{cat}'] is not a list of boxes"); continue
            for j, b in enumerate(boxes):
                why = _is_box(b, size)
                if why:
                    bad.append(f"counts['{cat}'][{j}]: {why}")
    hints = key.get("count_hints") or {}
    for cat, h in hints.items():
        if cat not in counts:
            bad.append(f"count_hints['{cat}'] has no matching counts category")
        if not (isinstance(h.get("threshold", 0.4), (int, float)) and 0 < h.get("threshold", 0.4) <= 1):
            bad.append(f"count_hints['{cat}'].threshold is {h.get('threshold')!r}")
        sr = h.get("size_range", [0.2, 5.0])
        if not (isinstance(sr, (list, tuple)) and len(sr) == 2 and 0 < sr[0] < sr[1]):
            bad.append(f"count_hints['{cat}'].size_range is {sr!r}")
    if key.get("legend") and not (folder / key["legend"]).exists():
        bad.append(f"legend image '{key['legend']}' is not in {folder}")
    if not (key.get("tasks") and all(isinstance(t, str) and t.strip() for t in key["tasks"])):
        bad.append("tasks is empty")
    if not (key.get("areas") or key.get("counts")):
        bad.append("neither areas nor counts: there is nothing to take off on this sheet")
    return bad


class Sheets:
    """All the drawings of one set, in the order of credits.json (or alphabetical without it)."""

    def __init__(self, folder: Path, name: str = "", strict: bool = True):
        self.folder = Path(folder)
        self.name = name or self.folder.name
        self.problems: Dict[str, List[str]] = {}
        self.sheets: List[Sheet] = []
        if not self.folder.is_dir():
            raise FileNotFoundError(f"no drawings in {self.folder}")
        order = self._order()
        for sid in order:
            p = self.folder / f"{sid}.key.json"
            try:
                key = json.loads(p.read_text(encoding="utf-8"))
            except Exception as e:                                   # noqa: BLE001
                self.problems[sid] = [f"{p.name} is not readable JSON: {e}"]
                continue
            bad = check_key(key, self.folder)
            if bad:
                self.problems[sid] = bad
                continue
            self.sheets.append(Sheet(key.get("id", sid), self.folder / key["file"], key))
        if strict and self.problems:
            raise KeyError_(self.problem_text())
        if not self.sheets:
            raise FileNotFoundError(f"no usable answer keys in {self.folder}")

    def _order(self) -> List[str]:
        ids = sorted(p.name[:-len(".key.json")] for p in self.folder.glob("*.key.json"))
        cred = self.folder / "credits.json"
        if cred.exists():
            try:
                c = json.loads(cred.read_text(encoding="utf-8"))
                listed = [e.get("id") for e in c if isinstance(e, dict)] if isinstance(c, list) else list(c)
                first = [i for i in listed if i in ids]
                return first + [i for i in ids if i not in first]
            except Exception:                                        # noqa: BLE001
                pass
        return ids

    def problem_text(self) -> str:
        out = [f"{len(self.problems)} answer key(s) in {self.folder} do not follow the schema "
               f"(see the schema at the top of aec_seg/data.py):"]
        for sid, bad in self.problems.items():
            out.append(f"  {sid}.key.json:")
            out += [f"    - {b}" for b in bad]
        return "\n".join(out)

    def __len__(self):
        return len(self.sheets)

    def __iter__(self):
        return iter(self.sheets)

    def __getitem__(self, key) -> Sheet:
        if isinstance(key, int):
            return self.sheets[key]
        for s in self.sheets:
            if key in (s.id, s.label) or str(key).startswith(s.id + ":"):
                return s
        raise KeyError(f"no drawing '{key}' in {self.folder} (have: {', '.join(self.ids())})")

    def ids(self) -> List[str]:
        return [s.id for s in self.sheets]

    def labels(self) -> List[str]:
        return [s.label for s in self.sheets]



class MaskStore:
    """data/masks/<set>/<sheet_id>/<prompt_key>.png (uint8 label map) + index.json with scores and boxes."""

    def __init__(self, folder: Path):
        self.folder = Path(folder)

    @staticmethod
    def key(prompt: str) -> str:
        return "".join(ch if ch.isalnum() else "_" for ch in prompt.strip().lower()).strip("_")

    def _index(self, sheet_id: str) -> dict:
        p = self.folder / sheet_id / "index.json"
        return json.loads(p.read_text()) if p.exists() else {}

    def has(self, sheet_id: str, prompt: str) -> bool:
        return self.key(prompt) in self._index(sheet_id)

    def load(self, sheet_id: str, prompt: str) -> Optional[SegResult]:
        idx = self._index(sheet_id)
        k = self.key(prompt)
        if k not in idx:
            return None
        entry = idx[k]
        lab = np.array(Image.open(self.folder / sheet_id / f"{k}.png"))
        return SegResult.from_label_map(entry["prompt"], lab, entry["scores"], entry["boxes"])

    def save(self, sheet_id: str, res: SegResult):
        d = self.folder / sheet_id
        d.mkdir(parents=True, exist_ok=True)
        k = self.key(res.prompt)
        Image.fromarray(res.label_map()).save(d / f"{k}.png", optimize=True)
        idx = self._index(sheet_id)
        idx[k] = {"prompt": res.prompt, "scores": [round(float(s), 4) for s in res.scores],
                  "boxes": [[round(float(v), 1) for v in b] for b in res.boxes], "area_pct": round(res.area_pct(), 2),
                  "n": int(len(res))}
        (d / "index.json").write_text(json.dumps(idx, indent=1))


@dataclass
class IntroPhoto:
    """A photo for the first steps (what SAM 3 does before the drawings), with its credit."""
    id: str
    path: Path
    title: str
    author: str
    license: str
    source: str

    def load(self) -> Image.Image:
        return Image.open(self.path).convert("RGB")

    @property
    def label(self) -> str:
        return f"{self.id}: {self.title}"


class IntroPhotos:
    def __init__(self, folder: Path):
        self.folder = Path(folder)
        cred = json.loads((self.folder / "credits.json").read_text())
        self.photos: List[IntroPhoto] = [IntroPhoto(c["id"], self.folder / c["file"], c["title"], c["author"], c["license"], c["source"]) for c in cred]

    def __getitem__(self, key) -> IntroPhoto:
        for p in self.photos:
            if key in (p.id, p.label) or str(key).startswith(p.id + ":"):
                return p
        raise KeyError(key)

    def labels(self) -> List[str]:
        return [p.label for p in self.photos]



if __name__ == "__main__":                                           # python -m aec_seg.data data/sheets/workshop
    import sys
    for arg in sys.argv[1:] or ["data/sheets/workshop", "data/sheets/homework"]:
        f = Path(arg)
        print(f"== {f}")
        try:
            s = Sheets(f, strict=False)
        except Exception as e:                                       # noqa: BLE001
            print(f"   {e}"); continue
        for sh in s.sheets:
            print(f"   OK  {sh.id:16s} {sh.discipline:22s} {sh.facts()}")
        if s.problems:
            print(s.problem_text())
