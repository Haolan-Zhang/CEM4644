"""The one object the MP3 notebooks talk to."""
import os
os.environ.setdefault("YOLO_VERBOSE", "False")

import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from PIL import Image

from . import draw, evaluate as ev, tricky as tk, ui
from .config import DETSETS, DetSpec
from .data import DetSet, unzip_dataset
from .models import Detector, get_device
from .train import LEADERBOARD, add_to_leaderboard, leaderboard_table, quick_train

ULTRALYTICS_PIN = "ultralytics==8.4.143"
GRADIO_PIN = "gradio==6.26.0"
BBOX_PIN = "jupyter-bbox-widget>=0.7.0"


class DetLab:
    def __init__(self):
        self.root = Path(__file__).resolve().parents[1]
        self.ready = False
        self.spec: Optional[DetSpec] = None
        self.train_pool: Optional[DetSet] = None
        self.val_pool: Optional[DetSet] = None
        self.test: Optional[DetSet] = None
        self.det: Optional[Detector] = None
        self.base: Optional[Detector] = None
        self.other: Optional[Detector] = None
        self.other_spec: Optional[DetSpec] = None
        self.other_test: Optional[DetSet] = None
        self.my: Optional[Detector] = None
        self.results: Dict[str, ev.EvalResult] = {}

    # ------------------------------------------------------------------ setup
    def setup(self, dataset: str = "construction_safety", install: bool = True):
        t0 = time.time()
        if install:
            self._install_missing({"ultralytics": ULTRALYTICS_PIN, "gradio": GRADIO_PIN, "jupyter_bbox_widget": BBOX_PIN})
        try:  # Colab needs this for third-party widgets such as the box-drawing tool in Step 1c
            from google.colab import output as _colab_output
            _colab_output.enable_custom_widget_manager()
        except Exception:
            pass
        self.spec = spec = DETSETS[dataset]
        unz = self.root / "_unzipped"
        folder = unzip_dataset(self.root / "data" / spec.zip_name, unz)
        pool = DetSet.from_folder(folder / "train", spec.classes, spec.display, "training pool")
        # a fixed slice of the pool is kept aside as validation photos for the training exercise
        rng = np.random.default_rng(4644)
        idx = rng.permutation(len(pool))
        n_val = min(60, len(pool) // 5)
        self.val_pool = DetSet([pool[int(i)] for i in idx[:n_val]], spec.classes, spec.display, "validation")
        self.train_pool = DetSet([pool[int(i)] for i in idx[n_val:]], spec.classes, spec.display, "training pool")
        self.test = DetSet.from_folder(folder / "test", spec.classes, spec.display, "test")
        self.det = Detector(self.root / spec.course_model, name="course model", classes=[spec.pretty(c) for c in spec.classes])
        base = self.root / "models" / "yolo11n.pt"
        if base.exists():
            self.base = Detector(base, name="YOLO trained on everyday photos (COCO)")
            if not Path("yolo11n.pt").exists():  # lets Ultralytics skip a download when it checks mixed precision
                try:
                    shutil.copy2(base, "yolo11n.pt")
                except Exception:
                    pass
        if spec.other_domain and DETSETS[spec.other_domain].key in DETSETS:
            o = DETSETS[spec.other_domain]
            if (self.root / o.course_model).exists() and (self.root / "data" / o.zip_name).exists():
                self.other_spec = o
                of = unzip_dataset(self.root / "data" / o.zip_name, unz)
                self.other_test = DetSet.from_folder(of / "test", o.classes, o.display, "other test")
                self.other = Detector(self.root / o.course_model, name=f"model trained on {o.key.replace('_', ' ')}",
                                      classes=[o.pretty(c) for c in o.classes])
        self.ready = True
        dev = "GPU" if get_device() != "cpu" else "CPU (fine; training will be slower)"
        print(f"✅ Ready in {time.time() - t0:.0f} s. Running on: {dev}.")
        self.intro()

    def _install_missing(self, pkgs: Dict[str, str]):
        for mod, req in pkgs.items():
            try:
                __import__(mod)
            except ImportError:
                print(f"Installing {mod} (about a minute)...")
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", req], check=False)

    def _need(self):
        if not self.ready:
            raise RuntimeError("Run the 'Run me first' cell at the top of the notebook first.")

    @property
    def _pretty_colors(self):
        return {self.spec.pretty(c): v for c, v in self.spec.colors.items()}

    # ------------------------------------------------------------------ Part 1
    def intro(self):
        self._need()
        s = self.spec
        print(f"\n■ Dataset: {s.title}")
        print(f"  {s.description}")
        print(f"  Classes ({len(s.classes)}): " + ", ".join(s.pretty(c) for c in s.classes))
        print(f"  Photos for training: {len(self.train_pool)} (+{len(self.val_pool)} kept for checking)   |   unseen test photos: {len(self.test)}")
        print("  Labelled boxes in the test photos: " + ", ".join(f"{k}: {v}" for k, v in self.test.box_counts().items()))

    def show_gallery(self, category="all", how_many=6, with_boxes=True):
        self._need()
        ui.gallery(self.train_pool, self.spec, category, int(how_many), with_boxes=bool(with_boxes))

    def count_game(self, rounds=5):
        self._need()
        ui.count_game(self.test, self.spec, int(rounds))

    def label_yourself(self, how_many=3):
        """Students draw the boxes themselves on a few test photos and are scored against the dataset labels."""
        self._need()
        from .labeling import LabelExercise
        import json
        n_train, per_photo = None, None
        log = self.root / "models" / f"{self.spec.key}_training_log.json"
        if log.exists():
            n_train = json.loads(log.read_text()).get("n_train")
        if len(self.train_pool):
            per_photo = sum(len(it.cls) for it in self.train_pool.items) / len(self.train_pool)
        self._label = LabelExercise(self.test, self.spec, int(how_many), est_train_photos=n_train, est_boxes_per_photo=per_photo)
        self._label.start()

    # ------------------------------------------------------------------ Part 2
    def pick_and_detect(self):
        self._need()
        ui.pick_and_detect(self.det, self.test, self.spec)

    def evaluate(self, how_many="all", threshold=0.5):
        self._need()
        n = None if str(how_many).lower() in ("all", "") else int(how_many)
        res = ev.evaluate_set(self.det, self.test, n)
        self.results["course"] = res
        print(ev.summary_text(res, float(threshold)))

    def _res(self):
        if "course" not in self.results:
            self.evaluate()
        return self.results["course"]

    def threshold_explorer(self):
        self._need()
        ui.threshold_explorer(self._res(), self.spec)

    def error_explorer(self):
        self._need()
        ui.error_explorer(self._res(), self.spec)

    # ------------------------------------------------------------------ Part 3
    def tricky(self, group="all", threshold=0.5, include_my_model=False):
        self._need()
        dets = [self.det] + ([self.my] if include_my_model and self.my is not None else [])
        tk.show_tricky(dets, self.root / self.spec.tricky_dir, self.spec, group, float(threshold))

    def playground(self):
        self._need()
        tk.playground(self.det, self.test, self.spec)

    def compare_pretrained(self, how_many=3, threshold=0.4):
        """The same photos through the everyday-object YOLO (80 COCO classes) and through the course model."""
        self._need()
        if self.base is None:
            print("Base model not available."); return
        rng = np.random.default_rng(int(time.time()) % 100000)
        idx = rng.choice(len(self.test), size=min(int(how_many), len(self.test)), replace=False)
        print("Left: YOLO as downloaded (80 everyday classes such as person, truck, car). Right: the same network after fine-tuning on our photos.")
        for i in idx:
            it = self.test[int(i)]
            ui.compare_detectors([self.base, self.det], it.load(), self._pretty_colors, float(threshold), caption=f"labelled truth: {ui.gt_summary(it, self.test.pretty_classes)}")

    def domain_shift(self, how_many=3, threshold=0.4):
        self._need()
        if self.other is None:
            print("No other-domain model available."); return
        rng = np.random.default_rng(7)
        colors = dict(self._pretty_colors); colors.update({self.other_spec.pretty(c): v for c, v in self.other_spec.colors.items()})
        print(f"Photos from '{self.other_spec.title}' given to BOTH models. Each model can only answer with the classes it was trained on.")
        for i in rng.choice(len(self.other_test), size=min(int(how_many), len(self.other_test)), replace=False):
            it = self.other_test[int(i)]
            ui.compare_detectors([self.det, self.other], it.load(), colors, float(threshold), caption=f"labelled truth: {ui.gt_summary(it, self.other_test.pretty_classes)}")

    def upload_app(self):
        self._need()
        if os.environ.get("AEC_LAB_NO_APP"):
            print("(upload app skipped: AEC_LAB_NO_APP is set)"); return
        from . import app
        dets = {"course model": self.det}
        if self.base is not None:
            dets["YOLO (everyday objects)"] = self.base
        if self.my is not None:
            dets["my model"] = self.my
        colors = dict(self._pretty_colors)
        return app.launch(dets, colors, title=f"Try the detector on your own photo: {self.spec.title}",
                          description="Upload a site photo (or any photo) and move the confidence slider. "
                                      "Boxes are drawn for every detection above the threshold.")

    # ------------------------------------------------------------------ Part 4: from boxes to decisions
    def dashboard(self, threshold=0.5):
        self._need()
        import matplotlib.pyplot as plt
        res = self._res()
        t = float(threshold)
        s = self.spec
        names = res.classes
        d = s.dashboard or {"kind": "count", "label": "objects counted per photo"}
        if d["kind"] == "ppe":
            gi, bi = names.index(s.pretty(d["good"])), names.index(s.pretty(d["bad"]))
            pred_g = np.array([int((det.at(t).cls == gi).sum()) for det in res.dets]); pred_b = np.array([int((det.at(t).cls == bi).sum()) for det in res.dets])
            true_g = np.array([int((it.cls == s.classes.index(d["good"])).sum()) for it in res.items]); true_b = np.array([int((it.cls == s.classes.index(d["bad"])).sum()) for it in res.items])
            pc = 100 * pred_g.sum() / max(1, pred_g.sum() + pred_b.sum()); tc = 100 * true_g.sum() / max(1, true_g.sum() + true_b.sum())
            print(f"PPE compliance over {len(res.items)} site photos at threshold {t * 100:.0f}%:")
            print(f"  AI:      {pred_g.sum()} {d['label']} vs {pred_b.sum()} without  ->  {pc:.0f}% compliant")
            print(f"  labels:  {true_g.sum()} {d['label']} vs {true_b.sum()} without  ->  {tc:.0f}% compliant")
            flag_pred = pred_b > 0; flag_true = true_b > 0
            print(f"  Photos flagged for follow-up (at least one worker without a helmet): AI {int(flag_pred.sum())}, labels {int(flag_true.sum())}; "
                  f"agree on {int((flag_pred == flag_true).sum())} of {len(res.items)} photos "
                  f"(missed flags {int((flag_true & ~flag_pred).sum())}, false flags {int((flag_pred & ~flag_true).sum())}).")
            fig, ax = plt.subplots(figsize=(10, 3.2))
            order = np.argsort(-(true_b + pred_b))[:40]
            x = np.arange(len(order))
            ax.bar(x - 0.2, true_b[order], 0.4, label="labelled: workers without helmet", color=ui.GREY)
            ax.bar(x + 0.2, pred_b[order], 0.4, label="AI: workers without helmet", color=ui.RED)
            ax.set_xlabel("test photos (the 40 with most bare heads)"); ax.set_ylabel("count"); ax.legend(fontsize=8); ax.set_xticks([])
            fig.tight_layout(); ui.show(fig)
        else:
            pred = np.array([len(det.at(t)) for det in res.dets]); true = np.array([len(it.cls) for it in res.items])
            print(f"Equipment count over {len(res.items)} photos at threshold {t * 100:.0f}%: AI counted {pred.sum()} machines, labels say {true.sum()}.")
            print(f"  Exactly right on {int((pred == true).sum())} photos; AI counted too few on {int((pred < true).sum())}, too many on {int((pred > true).sum())}.")
            for c in range(len(names)):
                pc_ = sum(int((det.at(t).cls == c).sum()) for det in res.dets); tc_ = sum(int((it.cls == c).sum()) for it in res.items)
                print(f"  {names[c]}: AI {pc_}, labels {tc_}")
            fig, ax = plt.subplots(figsize=(4.5, 4))
            m = max(pred.max(), true.max()) + 1
            ax.scatter(true + np.random.default_rng(0).uniform(-0.15, 0.15, len(true)), pred + np.random.default_rng(1).uniform(-0.15, 0.15, len(pred)), alpha=0.5, color=ui.BLUE)
            ax.plot([0, m], [0, m], "k--", lw=1); ax.set_xlabel("machines in the labels"); ax.set_ylabel("machines counted by the AI"); ax.set_title("one dot per photo", fontsize=10)
            fig.tight_layout(); ui.show(fig)

    # ------------------------------------------------------------------ Part 5: training
    def train_my_model(self, training_photos=120, passes=10, start="pretrained", name="my model", imgsz=None, max_test=None):
        self._need()
        pretrained = str(start).lower().startswith("pre")
        n = min(int(training_photos), len(self.train_pool))
        device = get_device()
        imgsz = int(imgsz) if imgsz else (640 if device != "cpu" else 416)
        if device == "cpu" and (n > 60 or int(passes) > 10):
            print("No GPU: this run may take more than ten minutes. 60 photos and 10 passes is the sensible maximum on CPU.")
        # small runs need many optimizer steps: small batches, and a frozen backbone so only the detection layers adapt
        extra = {"freeze": 10} if pretrained else {}
        t0 = time.time()
        det, hist = quick_train(self.train_pool, self.val_pool, self.root / "models" / "yolo11n.pt", n_train=n, epochs=int(passes),
                                pretrained=pretrained, name=name, workdir=self.root / "_runs", imgsz=imgsz,
                                batch=4 if n <= 60 else 8, extra=extra)
        self.my = det
        res = ev.evaluate_set(det, self.test, max_test, progress=False)
        c = res.counts_at(0.5)
        tp = sum(v["TP"] for v in c.values()); fp = sum(v["FP"] for v in c.values()); fn = sum(v["FN"] for v in c.values())
        recall = tp / max(1, tp + fn); precision = tp / max(1, tp + fp)
        add_to_leaderboard(name, n, int(passes), pretrained, res.map50(), recall, precision, time.time() - t0)
        print(f"On the {len(res.items)} unseen test photos: quality score (mAP50) {res.map50() * 100:.1f}, "
              f"recall {recall * 100:.0f}%, precision {precision * 100:.0f}% at threshold 50%.")
        base = self._res()
        print(f"For comparison, the course model scores mAP50 {base.map50() * 100:.1f} on the same photos.")

    def leaderboard(self):
        df = leaderboard_table()
        if len(df) == 0:
            print("No runs yet. Train a model first."); return
        ui.table(df)

    def compare_my_model(self, group="all", threshold=0.5):
        self._need()
        if self.my is None:
            print("Train your own model first."); return
        print("Left: course model. Right: your model.")
        self.tricky(group, threshold, include_my_model=True)

    # ------------------------------------------------------------------ wrap-up
    def report_summary(self):
        self._need()
        print("Copy these numbers into your report:")
        if "course" in self.results:
            r = self.results["course"]
            print(f"  course model on {len(r.items)} test photos: mAP50 {r.map50() * 100:.1f}")
            for k, v in r.counts_at(0.5).items():
                print(f"    {k}: recall {v['recall'] * 100:.0f}%, precision {v['precision'] * 100:.0f}% (found {v['TP']}/{v['n_gt']}, false alarms {v['FP']})")
        if LEADERBOARD:
            print("  Your training runs:")
            for r in LEADERBOARD:
                print(f"    run {r['run']}: {r['name']} — {r['training photos']} photos, {r['passes']} pass(es), {r['start']} start -> mAP50 {r['quality (mAP50)']}")


lab = DetLab()
