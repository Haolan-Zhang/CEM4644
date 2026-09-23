"""The one object the MP5 notebooks talk to."""
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from . import config as C
from . import ui
from .client import GeminiClient, find_api_key
from .data import Examples

GENAI_MIN = "1.0.0"
GRADIO_PIN = "gradio==6.26.0"
TRANSFORMERS_MIN = "4.57.2"


class LLMLab:
    def __init__(self):
        self.root = Path(__file__).resolve().parents[1]
        self.ready = False
        self.spec: Optional[C.LabSpec] = None
        self.examples: Optional[Examples] = None
        self.client: Optional[GeminiClient] = None
        self.sam = None
        self.apps = {}
        self.results = {}          # what the steps computed, for the summary

    # ------------------------------------------------------------------ setup
    def setup(self, dataset: str = "workshop", api_key: str = "", model: str = C.DEFAULT_MODEL, load_sam: bool = True, install: bool = True):
        t0 = time.time()
        if install and not self._ensure_packages(load_sam):
            return
        self.spec = C.SPECS[dataset]
        self.examples = Examples(self.root, self.spec)
        key = find_api_key(api_key)
        self.client = GeminiClient(key, model=model, cache_dir=self.root / "data" / "cache" / dataset, log=print)
        if load_sam:
            self._load_sam()
        self.ready = True
        print(f"✅ Ready in {time.time() - t0:.0f} s.")

    def _ensure_packages(self, load_sam: bool) -> bool:
        need, restart = [], False
        try:
            import google.genai  # noqa: F401
        except ImportError:
            need.append("google-genai")
        try:
            import gradio  # noqa: F401
        except ImportError:
            need.append(GRADIO_PIN)
        if load_sam:
            try:
                from importlib.metadata import version
                from packaging.version import Version
                if Version(version("transformers")) < Version(TRANSFORMERS_MIN):
                    need.append(f"transformers>={TRANSFORMERS_MIN}"); restart = True
            except Exception:
                need.append(f"transformers>={TRANSFORMERS_MIN}"); restart = True
        if need:                                  # quietly: the notebook says to wait for the ✅ line
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", *need], check=False, capture_output=True)
        if restart:
            print("\n⚠️  A newer 'transformers' was installed for SAM 3. Please restart the runtime now "
                  "(menu Runtime → Restart session), then run this Step 0 cell again.")
            return False
        return True

    def _load_sam(self):
        """SAM 3 from the MP4 folder of the same repository (only Step 4b needs it)."""
        mp4 = self.root.parent / "mp4_segmentation"
        if not mp4.exists():
            print("MP4 folder not found next to this one: the 'LLM + SAM 3' step will use the model's boxes only."); return
        try:
            sys.path.insert(0, str(mp4))
            from aec_seg.engine import Sam3Engine
            self.sam = Sam3Engine(log=print)
        except Exception as e:  # noqa: BLE001
            print(f"Could not load SAM 3 ({str(e)[:120]}). The 'LLM + SAM 3' step will use the model's boxes only.")
            self.sam = None

    def _need(self):
        if not self.ready:
            raise RuntimeError("Run the 'Run me first' cell at the top of the notebook first.")

    def intro(self):
        ex, sp = self.examples, self.spec
        print(f"■ Examples: {sp.title}")
        print(f"  {len(ex.photos)} photos to classify ({ex.spec.photos.title}; {len(ex.photo_classes)} classes)")
        print(f"  {len(ex.sites)} site photos to detect in ({ex.spec.sites.title}; {sum(len(s.truth) for s in ex.sites)} boxes in the answer key)")
        print(f"  {len(ex.plans)} floor plans ({sp.plans.title})")
        if self.client.live:
            print(f"■ Gemini: key found, model {self.client.model}. Precomputed answers are used where they exist; everything else runs live.")
        else:
            print("■ Gemini: no API key. The precomputed answers of the built-in examples work; live requests (your own prompts and images) do not.\n"
                  "  To go live: create a free key at https://aistudio.google.com/apikey, add it as a Colab secret named GEMINI_API_KEY\n"
                  "  (the key icon in the left bar, notebook access on), then run Step 0 again.")
        print(f"■ SAM 3 (for the 'LLM + SAM 3' step): {'loaded' if self.sam else 'not loaded'}")

    # ------------------------------------------------------------------ steps (Part 1: talk to the model)
    def show_image(self, which, width=900):
        """The picture of one built-in example (a photo, a site photo or a plan), nothing else."""
        self._need(); ui.show_one(self, which, int(width))

    def show_examples(self, which: str = "photos"):
        self._need(); ui.show_examples(self, which)

    def describe(self, photo: str, question: str):
        self._need(); ui.describe(self, photo, question)

    def json_lab(self, photo: str, repeats: int = 3):
        self._need(); ui.json_lab(self, photo, repeats)

    # ------------------------------------------------------------------ Part 2: classification
    def classify(self, prompt: str = "basic", schema: bool = True, show_mistakes: bool = True):
        self._need(); ui.classify(self, prompt, schema, show_mistakes)

    def classify_own(self, prompt_text: str, schema: bool = True):
        self._need(); ui.classify_own(self, prompt_text, schema)

    # ------------------------------------------------------------------ Part 3: detection
    def detect(self, site: str, schema: bool = True):
        self._need(); ui.detect(self, site, schema)

    def detect_all(self, schema: bool = True):
        self._need(); ui.detect_all(self, schema)

    def count(self, site: str, schema: bool = True):
        self._need(); ui.count(self, site, schema)

    # ------------------------------------------------------------------ Part 4: rooms on a plan
    def segment(self, plan: str, mode: str = "LLM only (polygons)", schema: bool = False):
        self._need(); ui.segment(self, plan, mode, schema)

    def segment_all(self, schema: bool = True):
        self._need(); ui.segment_all(self, schema)

    def segment_compare(self, plan: str):
        self._need(); ui.segment_compare(self, plan)

    # ------------------------------------------------------------------ the chat-window route (paste-back steps)
    def chat_describe(self, photo: str, question: str):
        self._need(); from . import chat; chat.chat_describe(self, photo, question)

    def chat_classify(self, photo: str):
        self._need(); from . import chat; chat.chat_classify(self, photo)

    def chat_detect(self, site: str):
        self._need(); from . import chat; chat.chat_detect(self, site)

    def chat_count(self, site: str):
        self._need(); from . import chat; chat.chat_count(self, site)

    def chat_rooms(self, plan: str):
        self._need(); from . import chat; chat.chat_rooms(self, plan)

    # ------------------------------------------------------------------ Part 5 / wrap-up
    def prompt_app(self):
        self._need()
        from .app import launch
        self.apps["prompt"] = launch(self)

    def summary(self, chat: bool = False):
        self._need(); ui.summary(self, chat)

    def report_summary(self):
        self._need(); ui.report_summary(self)


lab = LLMLab()
