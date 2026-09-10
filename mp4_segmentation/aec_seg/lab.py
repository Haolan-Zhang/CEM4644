"""The one object the MP4 notebooks talk to."""
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional

from . import ui, viz
from .config import SETS, Material, SetSpec
from .data import MaskStore, PhotoSet
from .engine import SegResult, Sam3Engine

TRANSFORMERS_MIN = "4.57.2"
GRADIO_PIN = "gradio==6.26.0"
BBOX_PIN = "jupyter-bbox-widget>=0.7.0"


class SegLab:
    def __init__(self):
        self.root = Path(__file__).resolve().parents[1]
        self.ready = False
        self.spec: Optional[SetSpec] = None
        self.photos: Optional[PhotoSet] = None
        self.store: Optional[MaskStore] = None
        self.engine: Optional[Sam3Engine] = None
        self.plans: Optional[PhotoSet] = None
        self.plan_specs = []
        self._cache: Dict[tuple, SegResult] = {}
        self.notes = {}

    # ------------------------------------------------------------------ setup
    def setup(self, dataset: str = "site", load_model: bool = True, plans=False, install: bool = True):
        """plans: False, True (all plan drawings) or a list of plan ids."""
        t0 = time.time()
        if install and not self._ensure_packages():
            return
        try:
            from google.colab import output as _o
            _o.enable_custom_widget_manager()
        except Exception:
            pass
        self.spec = SETS[dataset]
        self.photos = PhotoSet(self.root / self.spec.folder, dataset)
        self.store = MaskStore(self.root / self.spec.masks)
        if plans:
            self.plans = PhotoSet(self.root / SETS["plans"].folder, "plans")
            ids = [p["id"] for p in SETS["plans"].plans] if plans is True else [str(x) for x in plans]
            self.plan_specs = [p for p in SETS["plans"].plans if p["id"] in ids]
        if load_model:
            try:
                self.engine = Sam3Engine(log=print)
            except Exception as e:
                print(f"Could not load SAM 3 ({str(e)[:120]}). The precomputed steps still work; live steps are disabled.")
                self.engine = None
        self.ready = True
        print(f"✅ Ready in {time.time() - t0:.0f} s.")
        self.intro()

    def _ensure_packages(self) -> bool:
        """Install what is missing. Returns False when the runtime must be restarted before continuing."""
        need, restart = [], False
        try:
            from importlib.metadata import version
            from packaging.version import Version
            if Version(version("transformers")) < Version(TRANSFORMERS_MIN):
                need.append(f"transformers>={TRANSFORMERS_MIN}"); restart = True
        except Exception:
            need.append(f"transformers>={TRANSFORMERS_MIN}"); restart = True
        for mod, req in (("gradio", GRADIO_PIN), ("jupyter_bbox_widget", BBOX_PIN)):
            try:
                __import__(mod)
            except ImportError:
                need.append(req)
        if need:
            print(f"Installing {', '.join(need)} (one to two minutes)...")
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", *need], check=False)
        if restart:
            print("\n⚠️  A newer 'transformers' was installed. Please restart the runtime now "
                  "(menu Runtime → Restart session), then run this Step 0 cell again.")
            return False
        return True

    def _need(self):
        if not self.ready:
            raise RuntimeError("Run the 'Run me first' cell at the top of the notebook first.")

    def intro(self):
        self._need()
        s = self.spec
        print(f"\n■ Photos: {s.title}")
        print(f"  {s.description}")
        print(f"  {len(self.photos)} photos. Materials you can ask for: " + ", ".join(m.name for m in s.materials))
        if s.series:
            for k, v in s.series.items():
                print(f"  Time series {k}: {v['title']} ({len(v['photos'])} photos)")
        if self.plans is not None:
            print("  Plans for quantity take-off (real drawings, public domain):")
            for p in self.plan_specs:
                print(f"    {p['id']}: {p['short']}")
        print(f"  SAM 3 live model: {'loaded' if self.engine else 'not loaded (precomputed results only)'}")

    # ------------------------------------------------------------------ results
    def get_phrase(self, photo_id: str, phrase: str, threshold: float = 0.2) -> SegResult:
        key = (photo_id, phrase.strip().lower())
        if key in self._cache:
            return self._cache[key]
        res = self.store.load(photo_id, phrase) if self.store else None
        if res is None:
            if self.engine is None:
                raise RuntimeError(f"No precomputed result for '{phrase}' on {photo_id} and SAM 3 is not loaded.")
            res = self.engine.segment(self.photos[photo_id].load(), phrase, threshold=threshold)
        self._cache[key] = res
        return res

    def get(self, photo_id: str, material: Material) -> SegResult:
        return self.get_phrase(photo_id, material.prompt)

    def _pid(self, label: str) -> str:
        return self.photos[label].id

    # ------------------------------------------------------------------ notebook steps
    def show_photos(self, which="all"):
        self._need()
        if which in ("all", None):
            ui.gallery(self.photos, ncols=4, size=3.4, title=self.spec.title)
        elif which in self.spec.series:
            s = self.spec.series[which]
            ui.gallery(self.photos, s["photos"], ncols=5, size=3.0, title=s["title"])
        else:
            ui.gallery(self.photos, [self._pid(which)], ncols=1, size=7)

    def segment(self, photo, material, threshold=0.5):
        self._need()
        ui.segment_view(self, self._pid(photo), material, float(threshold))

    def material_mix(self, photo, threshold=0.5):
        self._need()
        ui.material_mix(self, self._pid(photo), float(threshold))

    def guess_game(self, rounds=4):
        self._need()
        ui.guess_game(self, int(rounds))

    def compare(self, photo_a, photo_b, threshold=0.5):
        self._need()
        ui.compare(self, self._pid(photo_a), self._pid(photo_b), float(threshold))

    def series(self, name, threshold=0.5, materials=None):
        self._need()
        ui.series_view(self, name, float(threshold), materials)

    def phrase_lab(self, photo, material, threshold=0.5):
        self._need()
        pid = self._pid(photo)
        if pid not in self.spec.phrase_lab_photos and self.engine is None:
            print(f"Alternative wordings are precomputed for {', '.join(self.spec.phrase_lab_photos)}; pick one of those (or load SAM 3).")
            return
        ui.phrase_lab(self, pid, material, float(threshold))

    def inspect(self, photo, material):
        self._need()
        ui.inspector(self, self._pid(photo), material)

    def fix(self, photo, material, threshold=0.5):
        self._need()
        ui.fix_with_box(self, self._pid(photo), material, float(threshold))

    def your_phrase(self, photo, phrase, threshold=0.5):
        self._need()
        ui.live_phrase(self, self._pid(photo), phrase, float(threshold))

    def plan_phrase(self, plan, phrase, confidence=0.4):
        """Step 5a: a phrase on a drawing (live model)."""
        self._need()
        if self.plans is None:
            print("This notebook has no plan drawings."); return
        ui.plan_phrase(self, self.plans[plan].id, str(phrase), float(confidence))

    def takeoff(self, plan, find_all=True, confidence=0.3):
        """Step 5b: boxes -> areas in sq ft, and optionally everything like the first footing box."""
        self._need()
        if self.plans is None:
            print("This notebook has no plan drawings."); return
        ui.takeoff(self, self.plans[plan].id, bool(find_all), float(confidence))

    def upload_app(self):
        self._need()
        if os.environ.get("AEC_LAB_NO_APP"):
            print("(upload app skipped: AEC_LAB_NO_APP is set)"); return
        if self.engine is None:
            print("SAM 3 is not loaded; the upload app needs the live model."); return
        from . import app
        return app.launch(self)

    def report_summary(self):
        self._need()
        print("Numbers for your report: the notebook prints every measurement under its step; copy the ones you used.")
        print(f"Photos: {self.spec.title}. Materials: {', '.join(m.name for m in self.spec.materials)}.")


lab = SegLab()
