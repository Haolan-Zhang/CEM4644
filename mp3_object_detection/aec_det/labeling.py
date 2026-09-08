"""Step 1c: students draw boxes themselves (jupyter_bbox_widget) and get scored against the dataset labels."""
import io
import time
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

from . import draw, ui
from .evaluate import iou_matrix


def score_boxes(student_xyxy: np.ndarray, student_cls: np.ndarray, gt_xyxy: np.ndarray, gt_cls: np.ndarray, iou_thr: float = 0.5) -> Dict:
    """Greedy matching of student boxes to labelled boxes. Returns counts and the mean IoU of the matched pairs."""
    n_s, n_g = len(student_xyxy), len(gt_xyxy)
    if n_s == 0 or n_g == 0:
        return {"matched": 0, "wrong_class": 0, "extra": n_s, "missed": n_g, "mean_iou": float("nan")}
    ious = iou_matrix(student_xyxy, gt_xyxy)
    used_g = np.zeros(n_g, bool); matched = wrong = 0; iou_vals = []
    order = np.argsort(-ious.max(1))
    for k in order:
        cand = np.where((~used_g) & (ious[k] >= iou_thr))[0]
        if not len(cand):
            continue
        j = cand[np.argmax(ious[k, cand])]
        used_g[j] = True
        iou_vals.append(float(ious[k, j]))
        if int(student_cls[k]) == int(gt_cls[j]):
            matched += 1
        else:
            wrong += 1
    return {"matched": matched, "wrong_class": wrong, "extra": n_s - matched - wrong, "missed": int((~used_g).sum()),
            "mean_iou": float(np.mean(iou_vals)) if iou_vals else float("nan")}


class LabelExercise:
    def __init__(self, det_set, spec, how_many: int = 3, seed: Optional[int] = None, est_train_photos: Optional[int] = None,
                 est_boxes_per_photo: Optional[float] = None, max_side: int = 640):
        rng = np.random.default_rng(seed if seed is not None else int(time.time()) % 100000)
        pool = [i for i in range(len(det_set)) if 2 <= len(det_set[i].cls) <= 8] or list(range(len(det_set)))
        self.order = [int(i) for i in rng.permutation(pool)[:how_many]]
        self.det_set, self.spec = det_set, spec
        self.classes = list(det_set.pretty_classes)
        self.colors = {spec.pretty(c): v for c, v in spec.colors.items()}
        self.est_train_photos, self.est_boxes_per_photo = est_train_photos, est_boxes_per_photo
        self.max_side = max_side
        self.results: List[Dict] = []
        self.i = 0
        self.t0 = None
        self.scale = 1.0

    # ------------------------------------------------------------------ UI
    def start(self):
        import ipywidgets as w
        from IPython.display import display
        from jupyter_bbox_widget import BBoxWidget
        self.widget = BBoxWidget(classes=self.classes)
        self.msg = w.HTML()
        self.out = w.Output()
        help_txt = w.HTML(
            "<b>How to label:</b> click and drag on the photo to draw a box, then click the class name under the photo "
            "(or press its number key). Click a box to select it; press <i>Delete</i> to remove it. "
            "Draw one box per object, for every class. Click <b>Submit</b> when the photo is done, or <b>Skip</b> to pass.")

        @self.widget.on_submit
        def _submit():
            self._on_submit()

        @self.widget.on_skip
        def _skip():
            self._on_skip()

        display(w.VBox([help_txt, self.msg, self.widget, self.out]))
        self._show(0)

    def _show(self, i: int):
        self.i = i
        item = self.det_set[self.order[i]]
        img = item.load()
        s = min(1.0, self.max_side / max(img.size))
        self.scale = s
        if s < 1.0:
            img = img.resize((round(img.width * s), round(img.height * s)), Image.BILINEAR)
        buf = io.BytesIO(); img.save(buf, "JPEG", quality=90)
        self.widget.bboxes = []
        self.widget.image_bytes = buf.getvalue()
        self.t0 = time.time()
        self.msg.value = f"<b>Photo {i + 1} of {len(self.order)}.</b> Draw a box around every object of these classes: {', '.join(self.classes)}."

    def _student_boxes(self):
        bb = list(self.widget.bboxes or [])
        xyxy = np.array([[b["x"], b["y"], b["x"] + b["width"], b["y"] + b["height"]] for b in bb], dtype=float) / self.scale if bb else np.zeros((0, 4))
        cls = np.array([self.classes.index(b["label"]) if b.get("label") in self.classes else -1 for b in bb], dtype=int)
        return xyxy, cls

    def _on_submit(self):
        elapsed = time.time() - self.t0
        item = self.det_set[self.order[self.i]]
        sx, sc = self._student_boxes()
        res = score_boxes(sx, sc, item.xyxy(), item.cls)
        res.update({"seconds": elapsed, "drawn": len(sx), "labelled": len(item.cls)})
        self.results.append(res)
        with self.out:
            self.out.clear_output(wait=True)
            img = item.load()
            im = draw.draw_ground_truth(img, item, self.classes, self.colors, label=True)
            labels = [self.classes[c] if c >= 0 else "?" for c in sc]
            im = draw.draw_boxes(im, sx, ["you: " + l for l in labels], [self.colors.get(l, "#ffffff") for l in labels])
            ui.show(ui.image_grid([im], [f"your boxes (solid) vs. the dataset labels (dashed)"], ncols=1, size=6.5))
            iou_txt = f", overlap of the matching pairs {res['mean_iou'] * 100:.0f}%" if not np.isnan(res["mean_iou"]) else ""
            print(f"Photo {self.i + 1}: {elapsed:.0f} s. You drew {res['drawn']} boxes; the dataset has {res['labelled']}. "
                  f"Matching box and class: {res['matched']}; right place, different class: {res['wrong_class']}; "
                  f"extra boxes: {res['extra']}; labelled objects you did not box: {res['missed']}{iou_txt}.")
        self._next()

    def _on_skip(self):
        self.results.append({"seconds": time.time() - self.t0, "drawn": 0, "labelled": len(self.det_set[self.order[self.i]].cls),
                             "matched": 0, "wrong_class": 0, "extra": 0, "missed": len(self.det_set[self.order[self.i]].cls),
                             "mean_iou": float("nan"), "skipped": True})
        self._next()

    def _next(self):
        if self.i + 1 < len(self.order):
            self._show(self.i + 1)
        else:
            self._finish()

    def _finish(self):
        self.widget.hide_buttons = True
        done = [r for r in self.results if not r.get("skipped")]
        secs = sum(r["seconds"] for r in self.results)
        drawn = sum(r["drawn"] for r in done); labelled = sum(r["labelled"] for r in self.results)
        matched = sum(r["matched"] for r in done); wrong = sum(r["wrong_class"] for r in done)
        extra = sum(r["extra"] for r in done); missed = sum(r["missed"] for r in self.results)
        lines = [f"<b>Done.</b> {len(self.results)} photos in {secs / 60:.1f} min: you drew {drawn} boxes for {labelled} labelled objects; "
                 f"{matched} agree with the labels, {wrong} have the right place but another class, {extra} are extra, {missed} were not boxed."]
        if drawn and secs > 0:
            per_min = drawn / (secs / 60)
            lines.append(f"Your speed: about {per_min:.0f} boxes per minute.")
            if self.est_train_photos and self.est_boxes_per_photo:
                total = self.est_train_photos * self.est_boxes_per_photo
                hours = total / per_min / 60
                lines.append(f"The course model was trained on about {self.est_train_photos:,} photos with roughly {total:,.0f} boxes: "
                             f"at your speed that is about <b>{hours:.0f} hours</b> of labelling for one person, before any checking.")
        self.msg.value = " ".join(lines)
        print("Write these numbers in your report.")
