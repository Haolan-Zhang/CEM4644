"""Matching predictions to ground truth, precision/recall per class at any threshold, AP50."""
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from .data import DetSet, Item
from .models import Detections, Detector


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-9)


def match(det: Detections, item: Item, iou_thr: float = 0.5):
    """Greedy class-aware matching, highest confidence first. Returns (matched_pred[bool], matched_gt[bool])."""
    gt = item.xyxy()
    mp = np.zeros(len(det), dtype=bool); mg = np.zeros(len(gt), dtype=bool)
    if len(det) == 0 or len(gt) == 0:
        return mp, mg
    ious = iou_matrix(det.xyxy, gt)
    order = np.argsort(-det.conf)
    for k in order:
        cand = np.where((item.cls == det.cls[k]) & (~mg) & (ious[k] >= iou_thr))[0]
        if len(cand):
            j = cand[np.argmax(ious[k, cand])]
            mp[k] = True; mg[j] = True
    return mp, mg


@dataclass
class EvalResult:
    classes: List[str]
    items: List[Item]
    dets: List[Detections]        # raw detections (conf >= 0.01)
    model_name: str = "detector"

    def counts_at(self, conf: float, iou_thr: float = 0.5) -> Dict[str, Dict[str, int]]:
        k = len(self.classes)
        tp = np.zeros(k, int); fp = np.zeros(k, int); fn = np.zeros(k, int)
        for it, det in zip(self.items, self.dets):
            d = det.at(conf)
            mp, mg = match(d, it, iou_thr)
            for c in range(k):
                pc = d.cls == c
                tp[c] += int((mp & pc).sum()); fp[c] += int((~mp & pc).sum())
                fn[c] += int((~mg & (it.cls == c)).sum())
        out = {}
        for c in range(k):
            p = tp[c] / max(1, tp[c] + fp[c]); r = tp[c] / max(1, tp[c] + fn[c])
            out[self.classes[c]] = {"TP": int(tp[c]), "FP": int(fp[c]), "FN": int(fn[c]), "precision": p, "recall": r,
                                    "n_gt": int(tp[c] + fn[c])}
        return out

    def ap50(self, c: int, iou_thr: float = 0.5) -> float:
        """Average precision (all-point interpolation) for one class."""
        scores, flags, n_gt = [], [], 0
        for it, det in zip(self.items, self.dets):
            n_gt += int((it.cls == c).sum())
            d = det.at(0.0)
            mp, _ = match(d, it, iou_thr)
            sel = d.cls == c
            scores.extend(d.conf[sel].tolist()); flags.extend(mp[sel].tolist())
        if n_gt == 0:
            return float("nan")
        if not scores:
            return 0.0
        order = np.argsort(-np.array(scores))
        tp = np.cumsum(np.array(flags)[order]); fp = np.cumsum(~np.array(flags)[order])
        rec = tp / n_gt; prec = tp / np.maximum(tp + fp, 1)
        mrec = np.concatenate([[0], rec, [1]]); mpre = np.concatenate([[1], prec, [0]])
        mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))

    def map50(self) -> float:
        aps = [self.ap50(c) for c in range(len(self.classes))]
        aps = [a for a in aps if not np.isnan(a)]
        return float(np.mean(aps)) if aps else float("nan")

    def image_errors(self, conf: float, iou_thr: float = 0.5):
        """Per image: (index, n_missed, n_false_alarm, n_correct, matched_pred, matched_gt)."""
        out = []
        for i, (it, det) in enumerate(zip(self.items, self.dets)):
            d = det.at(conf)
            mp, mg = match(d, it, iou_thr)
            out.append((i, int((~mg).sum()), int((~mp).sum()), int(mp.sum()), mp, mg))
        return out


def evaluate_set(detector: Detector, det_set: DetSet, max_images: Optional[int] = None, seed: int = 0,
                 progress: bool = True) -> EvalResult:
    ds = det_set.subset(max_images, seed) if max_images else det_set
    dets = detector.predict_paths([it.path for it in ds.items], conf=0.01, progress=progress)
    return EvalResult(list(ds.pretty_classes), list(ds.items), dets, detector.name)


def summary_text(res: EvalResult, conf: float = 0.5) -> str:
    c = res.counts_at(conf)
    lines = [f"At a confidence threshold of {conf * 100:.0f}% on {len(res.items)} unseen test photos "
             f"(a detection counts as correct when its box overlaps the true box by at least half):"]
    for name, v in c.items():
        lines.append(f"  - {name:12s}: found {v['TP']} of {v['n_gt']} (recall {v['recall'] * 100:.0f}%), "
                     f"false alarms {v['FP']} (precision {v['precision'] * 100:.0f}%), missed {v['FN']}")
    lines.append(f"Overall quality score (mAP50, 0-100): {res.map50() * 100:.1f}")
    return "\n".join(lines)
