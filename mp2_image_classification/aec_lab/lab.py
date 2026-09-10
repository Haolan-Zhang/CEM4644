"""The one object the notebooks talk to. Every public method prints or displays
something and needs no return value, so the notebook cells stay one-liners."""
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from PIL import Image

from . import evaluate as ev
from . import tricky as tk
from . import ui
from .config import DATASETS, DatasetSpec
from .data import ImageSet, load_about, unzip_dataset
from .models import Classifier, get_device
from .train import LEADERBOARD, add_to_leaderboard, leaderboard_table, quick_train

TASK_NAMES = {"binary": "defect vs. no defect", "multiclass": "defect type"}


class Lab:
    def __init__(self):
        self.root = Path(__file__).resolve().parents[1]
        self.ready = False
        self.task_names = dict(TASK_NAMES)
        self.specs: Dict[str, DatasetSpec] = {}
        self.sets: Dict[str, Dict[str, ImageSet]] = {}      # task -> {"train": ..., "test": ...}
        self.clf: Dict[str, Classifier] = {}                 # task -> course model
        self.my: Dict[str, Classifier] = {}                  # task -> student model
        self.results: Dict[str, ev.EvalResult] = {}
        self.other: Optional[Classifier] = None
        self.other_spec: Optional[DatasetSpec] = None
        self._zs = None
        self.notes = {}
        self.apps = {}                                       # running Gradio apps of the *_gradio notebooks

    # ------------------------------------------------------------------ setup
    def _task(self, name: str) -> str:
        """Accepts 'binary' / 'multiclass' or the friendly task name shown in the notebook."""
        n = (name or "").strip().lower()
        if n in ("binary", "multiclass"):
            return n
        for k, v in self.task_names.items():
            if v.lower() == n:
                return k
        return "binary" if ("vs" in n or "no " in n) else "multiclass"

    def setup(self, binary: str = "facade_defects", multiclass: str = "facade_defects", install: bool = True,
              task_names: Optional[Dict[str, str]] = None, gradio: Optional[str] = None):
        """gradio: optional exact version to install (the *_gradio notebooks pin it); None keeps the old
        behaviour (install whatever pip picks, only if gradio is missing)."""
        t0 = time.time()
        if task_names:
            self.task_names.update(task_names)
        if install and gradio:
            self._ensure_version("gradio", gradio)
        elif install:
            self._install_missing(["gradio"])
        self.specs = {"binary": DATASETS[binary], "multiclass": DATASETS[multiclass]}
        unz = self.root / "_unzipped"
        for task, spec in self.specs.items():
            folder = unzip_dataset(self.root / "data" / spec.zip_name, unz)
            tr = ImageSet.from_folder(folder / "train", spec.classes, spec.display, "training pool")
            te = ImageSet.from_folder(folder / "test", spec.classes, spec.display, "test")
            if task == "binary":
                tr, te = tr.relabel(spec.binary.mapping, spec.binary.classes), te.relabel(spec.binary.mapping, spec.binary.classes)
            self.sets[task] = {"train": tr, "test": te}
            mdir = spec.binary_model if task == "binary" else spec.multiclass_model
            if mdir:
                clf = Classifier.load(self.root / mdir, name=f"course model ({self.task_names[task]})")
                if clf.classes != tr.pretty_classes:
                    raise RuntimeError(f"class order mismatch for {task}: {clf.classes} vs {tr.pretty_classes}")
                self.clf[task] = clf
        spec_b = self.specs["binary"]
        if spec_b.other_domain and DATASETS[spec_b.other_domain].binary_model:
            self.other_spec = DATASETS[spec_b.other_domain]
            self.other = Classifier.load(self.root / self.other_spec.binary_model,
                                         name=f"model trained on {self.other_spec.key.replace('_', ' ')}")
        self.ready = True
        dev = "GPU" if get_device().type == "cuda" else "CPU (fine, just a bit slower)"
        print(f"✅ Ready in {time.time() - t0:.0f} s. Running on: {dev}.")
        self.intro()

    def _install_missing(self, pkgs):
        for p in pkgs:
            try:
                __import__(p)
            except ImportError:
                print(f"Installing {p} (about a minute)...")
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", p], check=False)

    def _ensure_version(self, pkg: str, version: str):
        """Install pkg==version unless exactly that version is already importable. Only pip's own
        dependency rules apply; torch, numpy and pillow are left alone (constraints file)."""
        import importlib
        import tempfile
        try:
            have = importlib.import_module(pkg).__version__
        except Exception:
            have = None
        if have == version:
            return
        print(f"Installing {pkg} {version} (about a minute)...")
        keep = []
        for name in ("torch", "torchvision", "numpy", "pillow", "pandas"):
            try:
                keep.append(f"{name}=={importlib.import_module('PIL' if name == 'pillow' else name).__version__}")
            except Exception:
                pass
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("\n".join(keep) + "\n")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-c", f.name, f"{pkg}=={version}"], check=False)
        try:
            importlib.invalidate_caches()
            importlib.import_module(pkg)
        except Exception:
            print(f"⚠️ {pkg} could not be installed without changing core packages. The app steps (3b, 5a) will not work "
                  f"in this notebook; use the non-Gradio version of the notebook instead. Everything else works.")

    def _need(self, task=None):
        if not self.ready:
            raise RuntimeError("Run the 'Run me first' cell at the top of the notebook first.")
        if task and task not in self.clf:
            raise RuntimeError(f"No course model for '{task}' in this notebook.")

    # ------------------------------------------------------------------ data
    def intro(self):
        self._need()
        for task in ("binary", "multiclass"):
            spec, s = self.specs[task], self.sets[task]
            print(f"\n■ {self.task_names[task].upper()}  —  dataset: {spec.title}")
            print(f"  {spec.description}")
            print(f"  Classes ({len(s['test'].classes)}): " + ", ".join(s["test"].pretty_classes))
            print(f"  Photos available for training: {len(s['train'])}   |   unseen test photos: {len(s['test'])}")
            c = s["test"].counts()
            print("  Test photos per class: " + ", ".join(f"{k}: {v}" for k, v in c.items()))

    def show_gallery(self, task="binary", category="all", how_many=12):
        task = self._task(task); self._need()
        s = self.sets[task]["train"]
        if category not in ("all", "any", None) and category not in s.pretty_classes:
            print(f"'{category}' is not one of the classes of '{self.task_names[task]}' "
                  f"({', '.join(s.pretty_classes)}). Showing all classes instead.")
            category = "all"
        ui.gallery(s, category, how_many)

    def guess_game(self, task="binary", rounds=5):
        task = self._task(task); self._need()
        ui.guess_game(self.sets[task]["test"], rounds)

    # ------------------------------------------------------------------ run the course model
    def pick_and_predict(self, task="binary"):
        task = self._task(task); self._need(task)
        ui.pick_and_predict(self.clf[task], self.sets[task]["test"])

    def evaluate(self, task="binary", how_many="all"):
        task = self._task(task); self._need(task)
        n = None if str(how_many).lower() in ("all", "") else int(how_many)
        res = ev.evaluate_set(self.clf[task], self.sets[task]["test"], n)
        self.results[task] = res
        print(ev.summary_text(res))
        ui.show(ev.plot_confusion(res))

    def error_explorer(self, task="binary"):
        task = self._task(task); self._need(task)
        if task not in self.results:
            self.evaluate(task)
        ev.error_explorer(self.results[task])

    def threshold_explorer(self):
        self._need("binary")
        if "binary" not in self.results:
            self.evaluate("binary")
        ev.threshold_explorer(self.results["binary"], self.specs["binary"].binary.positive)

    # ------------------------------------------------------------------ limitations
    def tricky(self, task="binary", group="all", include_my_model=False):
        task = self._task(task); self._need(task)
        folder = self.root / self.specs[task].tricky_dir
        clfs = [self.clf[task]] + ([self.my[task]] if include_my_model and task in self.my else [])
        tk.show_tricky(clfs, folder, task, group)

    def tricky_groups(self):
        self._need()
        folder = self.root / self.specs["binary"].tricky_dir
        return sorted({m.get("group", "other") for m in tk.load_tricky(folder)})

    def playground(self, task="binary"):
        task = self._task(task); self._need(task)
        tk.playground(self.clf[task], self.sets[task]["test"])

    def domain_shift(self, how_many=200):
        """Apply the binary model trained on the OTHER domain to this domain's test photos."""
        self._need("binary")
        if self.other is None:
            print("No other-domain model available in this notebook."); return
        test = self.sets["binary"]["test"]
        sub = test.stratified(total=how_many, seed=1)
        res_home = ev.evaluate_set(self.clf["binary"], sub, progress=False)
        # the other model has its own class names; map by position (negative, positive)
        other_probs = self.other.predict_paths(sub.paths())
        acc_other = float((other_probs.argmax(1) == np.array(sub.labels())).mean())
        print(f"Same {len(sub)} test photos of '{self.specs['binary'].title}':")
        print(f"  model trained on THIS kind of photo      : {res_home.accuracy * 100:.1f}% correct")
        print(f"  model trained on '{self.other_spec.title}': {acc_other * 100:.1f}% correct"
              f"   (its classes: {self.other.classes[0]} / {self.other.classes[1]})")
        idx = np.random.default_rng(0).choice(len(sub), size=min(8, len(sub)), replace=False)
        images = [sub.load(int(i)) for i in idx]
        captions = [f"really: {sub.pretty(sub[int(i)][1])}" for i in idx]
        ui.compare_models([self.clf["binary"], self.other], images, captions, ncols=4)

    def upload_app(self, task="both"):
        self._need()
        import os
        if os.environ.get("AEC_LAB_NO_APP"):
            print("(upload app skipped: AEC_LAB_NO_APP is set)"); return
        from . import app
        clfs = {}
        if task in ("both", "binary") and "binary" in self.clf:
            clfs[self.task_names["binary"]] = self.clf["binary"]
        if task in ("both", "multiclass") and "multiclass" in self.clf:
            clfs[self.task_names["multiclass"]] = self.clf["multiclass"]
        for t, m in self.my.items():
            clfs[f"my model ({self.task_names[t]})"] = m
        spec = self.specs["binary"]
        return app.launch(clfs, title=f"Try the {spec.title} classifier on your own photo",
                          description="Take a photo of a wall, floor, pavement or anything else and see what the model says. "
                                      "Remember: the model has to pick one of its classes, whatever you show it.")

    # ------------------------------------------------------------------ zero-shot
    def zero_shot(self, class_names: str, how_many: int = 8, source: str = "test photos", task="multiclass"):
        self._need()
        from .zeroshot import ZeroShot, parse_classes, zero_shot_grid
        names = parse_classes(class_names)
        if len(names) < 2:
            print("Please type at least two class names, one per line."); return
        if self._zs is None:
            print("Loading CLIP (a model that understands both images and text) — about a minute the first time...")
            self._zs = ZeroShot()
        task = self._task(task)
        if source.startswith("tricky"):
            items = tk.load_tricky(self.root / self.specs[task].tricky_dir)[:how_many]
            images = [Image.open(m["path"]).convert("RGB") for m in items]
            captions = [m["caption"] for m in items]
        else:
            s = self.sets[task]["test"].sample(how_many, seed=int(time.time()) % 10000)
            images = [s.load(i) for i in range(len(s))]
            captions = [f"dataset label: {s.pretty(s[i][1])}" for i in range(len(s))]
        zero_shot_grid(self._zs, images, names, captions)

    # ------------------------------------------------------------------ training
    def train_my_model(self, task="binary", training_photos=100, passes=1, start="pretrained", name="my model",
                       max_test=400):
        task = self._task(task); self._need()
        s = self.sets[task]
        pretrained = str(start).lower().startswith("pre")
        n = min(int(training_photos), len(s["train"]))
        t0 = time.time()
        clf, hist = quick_train(s["train"], s["test"], self.root / "models" / "base_convnextv2_femto",
                                n_train=n, epochs=int(passes), pretrained=pretrained,
                                name=f"{name} ({self.task_names[task]})", max_test=max_test)
        self.my[task] = clf
        acc = hist[-1]["test_accuracy"]
        add_to_leaderboard(name, n, int(passes), pretrained, acc, time.time() - t0, note=self.task_names[task])
        if task in self.clf:
            base_acc, _ = self._course_acc(task, max_test)
            print(f"For comparison, the course model scores {base_acc * 100:.1f}% on the same test photos.")
        res = ev.evaluate_set(clf, s["test"], max_test, progress=False)
        ui.show(ev.plot_confusion(res, title=f"{clf.name}: {res.accuracy * 100:.1f}% correct on {len(res.y_true)} test photos"))

    def _course_acc(self, task, max_test):
        from .train import accuracy
        return accuracy(self.clf[task], self.sets[task]["test"], max_test)

    def leaderboard(self):
        df = leaderboard_table()
        if len(df) == 0:
            print("No runs yet. Train a model first."); return
        ui.table(df)

    def compare_my_model(self, task="binary"):
        task = self._task(task); self._need(task)
        if task not in self.my:
            print("Train your own model first."); return
        print("Course model vs. your model on the tricky photos:")
        self.tricky(task, include_my_model=True)

    # ------------------------------------------------------------------ report helper
    def report_summary(self):
        self._need()
        print("Copy these numbers into your report:")
        for task, res in self.results.items():
            print(f"  {self.task_names[task]} — course model accuracy on {len(res.y_true)} test photos: {res.accuracy * 100:.1f}%")
        if LEADERBOARD:
            print("  Your training runs:")
            for r in LEADERBOARD:
                print(f"    run {r['run']}: {r['name']} — {r['training photos']} photos, {r['passes']} pass(es), "
                      f"{r['start']} start -> {r['test accuracy']}% ({r['note']})")

    # ------------------------------------------------------------------ Gradio variants (used by the *_gradio notebooks only)
    def playground_app(self, task="binary", share=None):
        """Step 3b as an inline Gradio app: live sliders plus a drawing pad."""
        task = self._task(task); self._need(task)
        from . import apps
        self.apps["playground"] = apps.playground_app(self.clf[task], self.sets[task]["test"], share=share)

    def zero_shot_app(self, class_names: str = "", how_many: int = 8, task="multiclass", share=None):
        """Step 5a as an inline Gradio app: same photos, new class names, one click."""
        self._need()
        from . import apps
        from .zeroshot import ZeroShot
        task = self._task(task)

        def get_zs():
            if self._zs is None:
                print("Loading CLIP (a model that understands both images and text) — about a minute the first time...")
                self._zs = ZeroShot()
            return self._zs

        tricky_items = tk.load_tricky(self.root / self.specs[task].tricky_dir) if self.specs[task].tricky_dir else []
        self.apps["zero_shot"] = apps.zero_shot_app(get_zs, self.sets[task]["test"], tricky_items, class_names,
                                                    how_many=how_many, share=share)


lab = Lab()
