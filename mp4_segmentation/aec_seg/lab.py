"""The one object the MP4 notebooks talk to."""
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional

from . import ui, viz
from .config import SETS, PlanSet as PlanSpec, Thing
from .data import IntroPhotos, MaskStore, PlanSet
from .engine import SegResult, Sam3Engine

TRANSFORMERS_MIN = "4.57.2"
GRADIO_PIN = "gradio==6.26.0"
BBOX_PIN = "jupyter-bbox-widget>=0.7.0"


class SegLab:
    def __init__(self):
        self.root = Path(__file__).resolve().parents[1]
        self.ready = False
        self.spec: Optional[PlanSpec] = None
        self.plans: Optional[PlanSet] = None
        self.store: Optional[MaskStore] = None
        self.engine: Optional[Sam3Engine] = None
        self._cache: Dict[tuple, SegResult] = {}
        self.apps = {}

    # ------------------------------------------------------------------ setup
    def setup(self, dataset: str = "homes_a", load_model: bool = True, install: bool = True):
        t0 = time.time()
        if install and not self._ensure_packages():
            return
        try:
            from google.colab import output as _o
            _o.enable_custom_widget_manager()
        except Exception:
            pass
        self.spec = SETS[dataset]
        self.plans = PlanSet(self.root / self.spec.folder, dataset)
        self.intro_photos = IntroPhotos(self.root / "data" / "intro")
        self.store = MaskStore(self.root / self.spec.masks)
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
        print(f"\n■ Plans: {s.title}")
        print(f"  {s.description}")
        for p in self.plans.plans:
            print(f"  {p.id}: {p.title} — {p.facts()}")
        print("  Things you can ask for by name: " + ", ".join(t.name for t in s.things))
        print(f"  SAM 3 live model: {'loaded' if self.engine else 'not loaded (precomputed results only)'}")

    # ------------------------------------------------------------------ results
    def get_phrase(self, plan_id: str, phrase: str, threshold: float = 0.1) -> SegResult:
        key = (plan_id, phrase.strip().lower())
        if key in self._cache:
            return self._cache[key]
        res = self.store.load(plan_id, phrase) if self.store else None
        if res is None:
            if self.engine is None:
                raise RuntimeError(f"No precomputed result for '{phrase}' on {plan_id} and SAM 3 is not loaded.")
            res = self.engine.segment(self.plans[plan_id].load(), phrase, threshold=threshold)
        self._cache[key] = res
        return res

    def get(self, plan_id: str, thing: Thing) -> SegResult:
        return self.get_phrase(plan_id, thing.prompt)

    def _pid(self, label: str) -> str:
        return self.plans[str(label)].id

    # ------------------------------------------------------------------ notebook steps
    # ------------------------------------------------------------------ Part 1: SAM 3 on a photo
    def intro_phrase(self, photo: str, phrase: str = "person", own_phrase: str = "", confidence: float = 0.3):
        self._need()
        ui.intro_phrase(self, photo, own_phrase.strip() or phrase, float(confidence))

    def intro_draw(self, photo: str):
        self._need()
        ui.intro_draw(self, photo)

    def show_plans(self, which="all"):
        self._need()
        if which in ("all", None, "all plans"):
            ui.gallery(self, ncols=2, size=5.2, title=self.spec.title)
        else:
            ui.gallery(self, [self._pid(which)], ncols=1, size=9)

    def legend(self):
        ui.legend()

    def segment(self, plan, thing, confidence=0.4):
        self._need()
        ui.segment_view(self, self._pid(plan), thing, float(confidence))

    def count(self, plan, thing, confidence=0.4):
        self._need()
        ui.count_view(self, self._pid(plan), thing, float(confidence))

    def mix(self, plan, confidence=0.4):
        self._need()
        ui.mix(self, self._pid(plan), float(confidence))

    def guess_game(self, rounds=4):
        self._need()
        ui.guess_game(self, int(rounds))

    def compare(self, plan_a, plan_b, confidence=0.4):
        self._need()
        ui.compare(self, self._pid(plan_a), self._pid(plan_b), float(confidence))

    def phrase_lab(self, plan, thing, confidence=0.4):
        self._need()
        ui.phrase_lab(self, self._pid(plan), thing, float(confidence))

    def inspect(self, plan, thing):
        self._need()
        ui.inspector(self, self._pid(plan), thing)

    def fix(self, plan, thing, confidence=0.4):
        self._need()
        ui.fix_with_box(self, self._pid(plan), thing, float(confidence))

    def your_phrase(self, plan, phrase, confidence=0.4):
        self._need()
        ui.live_phrase(self, self._pid(plan), str(phrase), float(confidence))

    def scale_check(self, plan):
        self._need()
        ui.scale_check(self, self._pid(plan))

    def takeoff(self, plan, find_all=True, confidence=0.3):
        self._need()
        ui.takeoff(self, self._pid(plan), bool(find_all), float(confidence))

    def upload_app(self):
        self._need()
        if os.environ.get("AEC_LAB_NO_APP"):
            print("(upload app skipped: AEC_LAB_NO_APP is set)"); return
        if self.engine is None:
            print("SAM 3 is not loaded; the upload app needs the live model."); return
        from . import app
        self.apps["upload"] = app.launch(self)

    def report_summary(self):
        self._need()
        print("Numbers for your report: the notebook prints every measurement under its step; copy the ones you used.")
        print(f"Plans: {self.spec.title}.")
        for p in self.plans.plans:
            print(f"  {p.id}: {p.title} — {p.facts()}")


lab = SegLab()
