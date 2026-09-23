"""The one object the MP4 notebooks talk to."""
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional

from . import ui
from .config import SETS, SheetSet as SheetSpec, Thing
from .data import IntroPhotos, KeyError_, MaskStore, Sheets
from .engine import SegResult, Sam3Engine

TRANSFORMERS_MIN = "4.57.2"
GRADIO_PIN = "gradio==6.26.0"
BBOX_PIN = "jupyter-bbox-widget>=0.7.0"


class SegLab:
    def __init__(self):
        self.root = Path(__file__).resolve().parents[1]
        self.ready = False
        self.spec: Optional[SheetSpec] = None
        self.sheets: Optional[Sheets] = None
        self.store: Optional[MaskStore] = None
        self.engine: Optional[Sam3Engine] = None
        self._cache: Dict[tuple, SegResult] = {}
        self.apps = {}
        self.takeoffs: Dict[str, dict] = {}     # the students' take-off of each drawing, for the summary
        self.phrases: Dict[tuple, dict] = {}    # (sheet, phrase) -> what the phrase cell found, for the summary
        self.found: Dict[str, dict] = {}        # sheet -> what one example box found (workshop Step 3d), for the summary

    # ------------------------------------------------------------------ setup
    def setup(self, dataset: str = "workshop", load_model: bool = True, install: bool = True):
        t0 = time.time()
        if install and not self._ensure_packages():
            return
        try:
            from google.colab import output as _o
            _o.enable_custom_widget_manager()
        except Exception:
            pass
        self.spec = SETS[dataset]
        try:
            self.sheets = Sheets(self.root / self.spec.folder, dataset)
        except KeyError_ as e:
            print("The answer keys of this set are broken, so the notebook cannot check anything:\n" + str(e))
            raise
        self.intro_photos = IntroPhotos(self.root / "data" / "intro")
        self.store = MaskStore(self.root / self.spec.masks) if self.spec.masks else None
        if load_model:
            try:
                self.engine = Sam3Engine(log=print)
            except Exception as e:
                print(f"Could not load SAM 3 ({str(e)[:120]}). The precomputed steps still work; live steps are disabled.")
                self.engine = None
        self.ready = True
        print(f"Ready in {time.time() - t0:.0f} s.")
        if getattr(self.spec, "guided", True):       # the workshop's Step 0 stops at the 'Ready' line
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
        if need:                                  # quietly: the notebook says to wait for the 'Ready' line
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", *need], check=False, capture_output=True)
        if restart:
            print("\nA newer 'transformers' was installed. Please restart the runtime now "
                  "(menu Runtime -> Restart session), then run this Step 0 cell again.")
            return False
        return True

    def _need(self):
        if not self.ready:
            raise RuntimeError("Run the 'Run me first' cell at the top of the notebook first.")

    def intro(self):
        self._need()
        s = self.spec
        print(f"\nDrawings: {s.title}")
        print(f"  {s.description}")
        for sh in self.sheets:
            print(f"  {sh.id}: {sh.title} ({sh.discipline}) - {sh.facts()}")
        if self.store:
            print("  Things you can ask for by name: " + ", ".join(t.name for t in s.things))
        print(f"  SAM 3 live model: {'loaded' if self.engine else 'not loaded (precomputed results only)'}")

    # ------------------------------------------------------------------ results
    def get_phrase(self, sheet_id: str, phrase: str, threshold: float = 0.1) -> SegResult:
        key = (sheet_id, phrase.strip().lower())
        if key in self._cache:
            return self._cache[key]
        res = self.store.load(sheet_id, phrase) if self.store else None
        if res is None:
            if self.engine is None:
                raise RuntimeError(f"No precomputed result for '{phrase}' on {sheet_id} and SAM 3 is not loaded.")
            res = self.engine.segment(self.sheets[sheet_id].load(), phrase, threshold=threshold)
        self._cache[key] = res
        return res

    def get(self, sheet_id: str, thing: Thing) -> SegResult:
        return self.get_phrase(sheet_id, thing.prompt)

    def _sid(self, label) -> str:
        return self.sheets[str(label)].id

    # ------------------------------------------------------------------ notebook steps
    def intro_phrase(self, photo: str, phrase: str = "person", own_phrase: str = "", confidence: float = 0.3):
        self._need()
        ui.intro_phrase(self, photo, own_phrase.strip() or phrase, float(confidence))

    def intro_draw(self, photo: str):
        self._need()
        ui.intro_draw(self, photo)

    def show_sheets(self, which="all"):
        self._need()
        if which in ("all", None, "all drawings"):
            ui.gallery(self, ncols=2, size=5.2, title=self.spec.title, tasks=True)
        else:
            ui.gallery(self, [self._sid(which)], ncols=1, size=9, tasks=True)

    def show_legend(self, sheet=None):
        self._need()
        ui.show_legend(self, self._sid(sheet) if sheet and sheet != "all" else None)

    def segment(self, sheet, thing, confidence=0.3):
        self._need()
        ui.segment_view(self, self._sid(sheet), thing, float(confidence))

    def count(self, sheet, thing, confidence=0.3):
        self._need()
        ui.count_view(self, self._sid(sheet), thing, float(confidence))

    def phrase_lab(self, sheet, thing, confidence=0.3):
        self._need()
        ui.phrase_lab(self, self._sid(sheet), thing, float(confidence))

    def inspect(self, sheet, thing):
        self._need()
        ui.inspector(self, self._sid(sheet), thing)

    def your_phrase(self, sheet, phrase, confidence=0.3):
        self._need()
        ui.live_phrase(self, self._sid(sheet), str(phrase), float(confidence))

    def ask(self, sheet, phrase, confidence=0.3):
        """Homework: a phrase straight to SAM 3 on a drawing, scored against the drawing's answer key."""
        self._need()
        ui.phrase_check(self, self._sid(sheet), phrase, float(confidence))

    def find_like(self, sheet):
        """Workshop: one example box, and SAM 3 finds every other one of the same thing (checked against the key)."""
        self._need()
        ui.find_like(self, self._sid(sheet))

    def takeoff(self, sheet):
        self._need()
        ui.takeoff(self, self._sid(sheet))

    def upload_app(self):
        self._need()
        if os.environ.get("AEC_LAB_NO_APP"):
            print("(upload app skipped: AEC_LAB_NO_APP is set)"); return
        if self.engine is None:
            print("SAM 3 is not loaded; the upload app needs the live model."); return
        from . import app
        self.apps["upload"] = app.launch(self)

    # ------------------------------------------------------------------ wrap-up
    def report_summary(self):
        self._need()
        print(f"Drawings: {self.spec.title}.")
        for sh in self.sheets:
            print(f"  {sh.id}: {sh.title} - {sh.facts()}")
        if not self.takeoffs:
            print("\nYou have not submitted a take-off yet. Go back to the take-off step, draw the boxes and click Submit; "
                  "every submitted drawing is summarised here.")
        else:
            print(f"\nYour take-offs ({len(self.takeoffs)} drawing(s)) - these are the numbers for your report:")
            for sid, rep in self.takeoffs.items():
                print(f"\n=== {sid}: {rep['title']}")
                ui.print_takeoff(self.sheets[sid], rep, guided=getattr(self.spec, "guided", True), scale_check=getattr(self.spec, "scale_check", True))
            missing = [sh.id for sh in self.sheets if sh.id not in self.takeoffs]
            if missing:
                print(f"\nNot done yet: {', '.join(missing)}.")
        if self.found:
            print("\nWhat SAM 3 found from ONE example box:")
            for sid, counts in self.found.items():
                for cat, c in counts.items():
                    if "error" in c:
                        continue
                    print(f"  {sid}: '{cat}' at confidence >= {c['threshold']:.2f}: {c['matched']} of the drawing's "
                          f"{c['truth']} found, {len(c['missed'])} missed, {len(c['extra'])} extra region(s)")
        if self.phrases:
            print(f"\nYour phrases ({len(self.phrases)}) - what SAM 3 found from words alone:")
            for (sid, ph), r in self.phrases.items():
                line = f"  {sid}: '{ph}' at confidence >= {r['threshold']:.2f}: {r['regions']} region(s)"
                if r.get("compare"):
                    line += f"; vs the key's '{r['compare']}': found {r['found']}/{r['truth']}, {r['extra']} extra"
                if r.get("median_err") is not None:
                    line += f", areas median {r['median_err']:.0f} % off (worst {r['worst_err']:.0f} %)"
                print(line)


lab = SegLab()
