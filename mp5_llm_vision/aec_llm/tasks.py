"""Turning the model's JSON into pictures and scores against the answer keys: classification, detection, counting,
and rooms on a plan (the model's own polygons, or its boxes handed to SAM 3)."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from . import config as C
from .client import GeminiClient, Reply
from .data import Photo, Plan, Site


# ----------------------------------------------------------------------------- geometry
def box_2d_to_xyxy(box_2d, size) -> Optional[List[float]]:
    """Gemini's [ymin, xmin, ymax, xmax] on a 0-1000 grid -> [x1, y1, x2, y2] in pixels (None if malformed)."""
    try:
        y1, x1, y2, x2 = [float(v) for v in box_2d][:4]
    except Exception:
        return None
    W, H = size
    x1, x2 = sorted((max(0.0, min(1000.0, x1)), max(0.0, min(1000.0, x2))))
    y1, y2 = sorted((max(0.0, min(1000.0, y1)), max(0.0, min(1000.0, y2))))
    return [x1 / 1000 * W, y1 / 1000 * H, x2 / 1000 * W, y2 / 1000 * H]


def points_to_px(points, size) -> Optional[List[List[float]]]:
    """A polygon given as [[x, y], ...] on the 0-1000 grid -> pixels (None if it is not a list of pairs)."""
    try:
        pts = [[float(p[0]) / 1000 * size[0], float(p[1]) / 1000 * size[1]] for p in points]
    except Exception:
        return None
    return pts if len(pts) >= 3 else None


def poly_area(pts: Sequence[Sequence[float]]) -> float:
    x = np.array([p[0] for p in pts]); y = np.array([p[1] for p in pts])
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def match(pred: Sequence[Tuple[str, Sequence[float]]], truth: Sequence[Tuple[str, Sequence[float]]], thr: float = 0.5,
          need_label: bool = True):
    """Greedy one-to-one matching by IoU. Returns (found: [bool per truth], extra: [bool per pred], pairs [(ti, pi, iou)])."""
    cand = []
    for ti, (tl, tb) in enumerate(truth):
        for pi, (pl, pb) in enumerate(pred):
            if need_label and pl != tl:
                continue
            v = iou(tb, pb)
            if v >= thr:
                cand.append((v, ti, pi))
    cand.sort(reverse=True)
    used_t, used_p, pairs = set(), set(), []
    for v, ti, pi in cand:
        if ti in used_t or pi in used_p:
            continue
        used_t.add(ti); used_p.add(pi); pairs.append((ti, pi, v))
    return [ti in used_t for ti in range(len(truth))], [pi not in used_p for pi in range(len(pred))], pairs


def normalise_label(raw, classes: Sequence[str]) -> str:
    """The model's label mapped onto the class list when it obviously means one of them (case, spaces, dashes)."""
    s = str(raw or "").strip()
    key = lambda t: t.lower().replace("-", " ").replace("_", " ").replace("  ", " ").strip()
    for c in classes:
        if key(c) == key(s):
            return c
    for c in classes:      # 'no helmet' -> 'no-helmet', 'excavators' -> 'excavator'
        if key(c).rstrip("s") == key(s).rstrip("s"):
            return c
    return s


# ----------------------------------------------------------------------------- classification
@dataclass
class ClsRow:
    photo: Photo
    label: str = ""
    confidence: Optional[float] = None
    reason: str = ""
    reply: Reply = field(default_factory=Reply)

    @property
    def ok(self) -> bool:
        return self.label == self.photo.truth

    @property
    def valid(self) -> bool:
        return self.reply.ok and isinstance(self.reply.data, dict) and "label" in self.reply.data


def classify(client: GeminiClient, photos: Sequence[Photo], prompt: str, classes: Sequence[str], schema: bool,
             run: int = 0, log=None, thinking: str = C.THINKING) -> List[ClsRow]:
    rows = []
    for i, ph in enumerate(photos):
        r = client.ask(ph.load(), prompt, schema=C.classify_schema(classes) if schema else None, run=run, thinking=thinking)
        row = ClsRow(ph, reply=r)
        if r.ok and isinstance(r.data, dict):
            row.label = normalise_label(r.data.get("label"), classes)
            try:
                row.confidence = float(r.data.get("confidence"))
            except Exception:
                row.confidence = None
            row.reason = str(r.data.get("reason", ""))[:200]
        rows.append(row)
        if log:
            log(f"  {i + 1}/{len(photos)} {ph.file}: {'✓' if row.ok else '✗'} {row.label or '(no label: ' + r.error + ')'}"
                + ("" if r.cached else f"  [{r.seconds:.1f} s live]"))
    return rows


def cls_summary(rows: Sequence[ClsRow], classes: Sequence[str]) -> dict:
    n = len(rows)
    ok = sum(r.ok for r in rows)
    invalid = sum(not r.valid for r in rows)
    off_list = sum(1 for r in rows if r.valid and r.label not in classes)
    return {"n": n, "correct": ok, "accuracy": ok / n * 100 if n else 0.0, "invalid": invalid, "off_list": off_list,
            "seconds": sum(r.reply.seconds for r in rows), "tokens": sum(r.reply.tokens_in + r.reply.tokens_out + r.reply.tokens_thought for r in rows)}


def specialist_cls_accuracy(photos: Sequence[Photo]) -> float:
    n = sum(1 for p in photos if p.specialist)
    return sum(1 for p in photos if p.specialist and p.specialist.get("label") == p.truth) / n * 100 if n else 0.0


# ----------------------------------------------------------------------------- detection
@dataclass
class DetRow:
    site: Site
    pred: List[Tuple[str, List[float]]] = field(default_factory=list)   # (label, xyxy px)
    found: List[bool] = field(default_factory=list)                     # per truth box
    extra: List[bool] = field(default_factory=list)                     # per prediction
    skipped: int = 0                                                    # entries of the reply that were not usable
    reply: Reply = field(default_factory=Reply)

    @property
    def n_found(self): return sum(self.found)
    @property
    def n_missed(self): return len(self.found) - sum(self.found)
    @property
    def n_extra(self): return sum(self.extra)


def boxes_from_reply(reply: Reply, size, classes: Sequence[str]) -> Tuple[List[Tuple[str, List[float]]], int]:
    """(label, xyxy) for every usable entry of a detection reply, and how many entries were unusable."""
    out, skipped = [], 0
    items = reply.data if isinstance(reply.data, list) else (reply.data.get("objects") if isinstance(reply.data, dict) and isinstance(reply.data.get("objects"), list) else [])
    for it in items or []:
        if not isinstance(it, dict):
            skipped += 1; continue
        b = box_2d_to_xyxy(it.get("box_2d"), size)
        if b is None or (b[2] - b[0]) < 1 or (b[3] - b[1]) < 1:
            skipped += 1; continue
        out.append((normalise_label(it.get("label"), classes), b))
    return out, skipped


def score_det(site: Site, pred: Sequence[Tuple[str, List[float]]], thr: float = 0.5) -> Tuple[List[bool], List[bool]]:
    truth = [(t["label"], t["box"]) for t in site.truth]
    found, extra, _ = match(pred, truth, thr=thr, need_label=True)
    return found, extra


def detect(client: GeminiClient, sites: Sequence[Site], prompt: str, classes: Sequence[str], schema: bool, run: int = 0, log=None,
           thinking: str = C.THINKING) -> List[DetRow]:
    rows = []
    for i, s in enumerate(sites):
        r = client.ask(s.load(), prompt, schema=C.detect_schema(classes) if schema else None, run=run, thinking=thinking)
        row = DetRow(s, reply=r)
        if r.ok:
            row.pred, row.skipped = boxes_from_reply(r, s.size, classes)
        row.found, row.extra = score_det(s, row.pred)
        rows.append(row)
        if log:
            log(f"  {i + 1}/{len(sites)} {s.file}: {len(row.pred)} boxes, found {row.n_found}/{len(s.truth)}, extra {row.n_extra}"
                + (f"  ({r.error})" if r.error else "") + ("" if r.cached else f"  [{r.seconds:.1f} s live]"))
    return rows


def specialist_det_rows(sites: Sequence[Site], conf: float = 0.25) -> List[DetRow]:
    rows = []
    for s in sites:
        pred = [(d["label"], d["box"]) for d in s.specialist if d.get("conf", 1.0) >= conf]
        row = DetRow(s, pred=pred)
        row.found, row.extra = score_det(s, pred)
        rows.append(row)
    return rows


def det_summary(rows: Sequence[DetRow], classes: Sequence[str]) -> dict:
    per = {c: {"truth": 0, "found": 0, "extra": 0} for c in classes}
    other = 0
    for row in rows:
        for t, f in zip(row.site.truth, row.found):
            per[t["label"]]["truth"] += 1; per[t["label"]]["found"] += int(f)
        for (l, _), e in zip(row.pred, row.extra):
            if e:
                if l in per: per[l]["extra"] += 1
                else: other += 1
    truth = sum(v["truth"] for v in per.values()); found = sum(v["found"] for v in per.values()); extra = sum(v["extra"] for v in per.values()) + other
    npred = sum(len(r.pred) for r in rows)
    return {"per_class": per, "truth": truth, "found": found, "missed": truth - found, "extra": extra, "other_labels": other,
            "recall": found / truth * 100 if truth else 0.0, "precision": (npred - extra) / npred * 100 if npred else 0.0,
            "invalid": sum(1 for r in rows if not r.reply.ok and r.reply.model), "seconds": sum(r.reply.seconds for r in rows),
            "tokens": sum(r.reply.tokens_in + r.reply.tokens_out + r.reply.tokens_thought for r in rows)}


# ----------------------------------------------------------------------------- counting
def count(client: GeminiClient, site: Site, prompt: str, count_field: str, schema: bool, run: int = 0, thinking: str = C.THINKING) -> Reply:
    return client.ask(site.load(), prompt, schema=C.count_schema(count_field) if schema else None, run=run, thinking=thinking)


# ----------------------------------------------------------------------------- rooms on a plan
@dataclass
class RoomRow:
    label: str
    box: List[float]                                  # xyxy px, from the model
    poly: Optional[List[List[float]]] = None          # px, from the model (LLM-only mode)
    mask: Optional[np.ndarray] = None                 # from SAM 3 (LLM + SAM 3 mode)
    m2: float = 0.0
    how: str = ""                                     # 'polygon', 'box', 'SAM 3'
    truth: Optional[dict] = None
    note: str = ""

    @property
    def truth_m2(self) -> Optional[float]:
        return self.truth["area_m2"] if self.truth else None

    @property
    def err_pct(self) -> Optional[float]:
        return (self.m2 - self.truth_m2) / self.truth_m2 * 100 if self.truth else None


def rooms_from_reply(reply: Reply, plan: Plan) -> Tuple[List[RoomRow], int]:
    rows, skipped = [], 0
    items = reply.data if isinstance(reply.data, list) else []
    for it in items:
        if not isinstance(it, dict):
            skipped += 1; continue
        b = box_2d_to_xyxy(it.get("box_2d"), plan.size)
        if b is None:
            skipped += 1; continue
        poly = points_to_px(it.get("mask"), plan.size) if it.get("mask") is not None else None
        rows.append(RoomRow(label=str(it.get("label", "?"))[:20], box=b, poly=poly))
    return rows, skipped


def attach_truth(rows: Sequence[RoomRow], plan: Plan, thr: float = 0.3):
    """Each predicted room gets the answer-key room whose box it overlaps most (one-to-one)."""
    truth = [(r["label"], r["box"]) for r in plan.rooms(indoor_only=False)]
    pred = [(r.label, r.box) for r in rows]
    _, _, pairs = match(pred, truth, thr=thr, need_label=False)
    rooms = plan.rooms(indoor_only=False)
    for ti, pi, v in pairs:
        rows[pi].truth = rooms[ti]


def wall_pixels(img: Image.Image) -> np.ndarray:
    from scipy import ndimage
    dark = np.asarray(img.convert("L")) < 100
    return ndimage.binary_opening(dark, structure=np.ones((7, 7)))


def measure_rooms(reply: Reply, plan: Plan, mode: str, sam=None) -> Tuple[List[RoomRow], int]:
    """From a reply (the model's or a pasted one) to measured rooms. mode 'llm': the reply's own polygons (its box when the
    polygon is unusable); mode 'llm+sam': the reply's boxes, each handed to SAM 3 as 'empty room' + box, the mask clipped
    to the box, holes filled, walls removed (as in MP4)."""
    rows, skipped = rooms_from_reply(reply, plan) if reply.ok else ([], 0)
    img = plan.load()
    walls = None
    for row in rows:
        if mode == "llm":
            if row.poly is not None:
                row.m2 = plan.area_m2(poly_area(row.poly)); row.how = "polygon"
            else:
                row.m2 = plan.area_m2((row.box[2] - row.box[0]) * (row.box[3] - row.box[1])); row.how = "box"
                row.note = "no usable polygon in the reply: the box's area is used"
        else:
            if sam is None:
                row.m2 = plan.area_m2((row.box[2] - row.box[0]) * (row.box[3] - row.box[1])); row.how = "box"
                row.note = "SAM 3 is not loaded: the box's area is used"
                continue
            from scipy import ndimage
            res = sam.segment_room(img, row.box)
            if len(res) == 0:
                row.m2 = 0.0; row.how = "SAM 3"; row.note = "SAM 3 found nothing in this box"; continue
            if walls is None:
                walls = wall_pixels(img)
            m = res.masks[0].copy()
            x1, y1, x2, y2 = [int(round(v)) for v in row.box]
            clip = np.zeros_like(m); clip[max(0, y1):y2 + 1, max(0, x1):x2 + 1] = True
            m = ndimage.binary_fill_holes(m & clip) & ~walls
            row.mask = m; row.m2 = plan.area_m2(m.sum()); row.how = "SAM 3"
            if m.sum() < 0.6 * (x2 - x1) * (y2 - y1):
                row.note = "the mask covers less than 60 % of the box (open plan?)"
    attach_truth(rows, plan)
    return rows, skipped


def segment_rooms(client: GeminiClient, plan: Plan, prompt: str, mode: str, schema: bool, sam=None, run: int = 0, log=None,
                  thinking: str = C.THINKING) -> Tuple[List[RoomRow], Reply, int]:
    """Ask the model for the rooms of a plan and measure them (see measure_rooms)."""
    with_mask = mode == "llm"
    r = client.ask(plan.load(), prompt, schema=C.rooms_schema(with_mask) if schema else None, run=run, thinking=thinking)
    rows, skipped = measure_rooms(r, plan, mode, sam)
    if log:
        log(f"  {plan.id}: {len(rows)} rooms in the reply" + (f", {skipped} unusable entries" if skipped else "") + (f"  ({r.error})" if r.error else ""))
    return rows, r, skipped


def seg_summary(rows: Sequence[RoomRow], plan: Plan) -> dict:
    truth_rooms = plan.rooms(indoor_only=True)
    matched = [r for r in rows if r.truth and r.truth["type"] not in ("balcony / terrace", "garage")]
    errs = [abs(r.err_pct) for r in matched if r.truth_m2]
    total_model = sum(r.m2 for r in rows if not (r.truth and r.truth["type"] in ("balcony / terrace", "garage")))
    return {"rooms_truth": len(truth_rooms), "rooms_found": len({id(r.truth) for r in matched}), "rooms_model": len(rows),
            "median_err": float(np.median(errs)) if errs else None, "max_err": float(max(errs)) if errs else None,
            "total_model_m2": total_model, "floor_area_m2": plan.floor_area_m2}
