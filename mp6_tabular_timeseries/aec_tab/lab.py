"""The one object the MP6 notebooks talk to."""
import contextlib
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from . import config as C
from .data import Meter, credits, load_meters, load_table

GRADIO_PIN = "gradio==6.26.0"
CHRONOS_PIN = "chronos-forecasting>=2.0"


@contextlib.contextmanager
def quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            yield


class TabLab:
    def __init__(self):
        self.root = Path(__file__).resolve().parents[1]
        self.ready = False
        self.spec: Optional[C.LabSpec] = None
        self.df = None
        self.meters: Dict[str, Meter] = {}
        self.forecaster = None
        self.models: dict = {}
        self.results: dict = {}
        self.guesses: Optional[dict] = None
        self.game_order = [2, 0, 3, 1]
        self.apps: dict = {}
        self.part = "both"
        self.chat_runs: dict = {}                       # every reply pasted back, per task, so two chats can be compared
        self.chat_cache: dict = {}                      # the files handed to the chat

    @property
    def table_spec(self) -> C.TableSpec:
        return self.spec.table

    # ------------------------------------------------------------------ setup
    def setup(self, dataset: str = "workshop", part: str = "both", load_forecaster: bool = True, install: bool = True):
        """part: "tabular" (the table only), "series" (the meters only) or "both"."""
        t0 = time.time()
        self.part = part
        load_forecaster = load_forecaster and part != "tabular"
        if install:
            self._install_missing({"gradio": GRADIO_PIN, "chronos": CHRONOS_PIN} if load_forecaster else {"gradio": GRADIO_PIN})
        self.spec = C.SPECS[dataset]
        self.models, self.results, self.guesses, self.chat_runs, self.chat_cache = {}, {}, None, {}, {}
        self.forecaster = None
        with quiet():
            self.df = load_table(self.root, self.spec.table) if part != "series" else None
            self.meters = load_meters(self.root, self.spec.series.meters) if part != "tabular" else {}
        if load_forecaster:
            try:
                from .series import Forecaster
                with quiet():
                    self.forecaster = Forecaster()
            except Exception as e:  # noqa: BLE001
                self.forecaster = None
                print(f"The pretrained forecaster could not be loaded ({str(e)[:80]}); the other two forecasts still work.")
        self.ready = True
        print(f"✅ Ready in {time.time() - t0:.0f} s.")

    def _install_missing(self, pkgs: Dict[str, str]):
        for mod, req in pkgs.items():
            try:
                with quiet():
                    __import__(mod)
            except ImportError:
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", req], check=False, capture_output=True)

    def _need(self):
        if not self.ready:
            raise RuntimeError("Run the 'Run me first' cell at the top of the notebook first.")

    def meter(self, which) -> Meter:
        for m in self.meters.values():
            if which in (m.id, m.label) or str(which).startswith(m.id):
                return m
        raise KeyError(which)

    # ------------------------------------------------------------------ steps
    def show_table(self, rows=10):
        self._need(); from . import ui; ui.show_table(self, int(rows))

    def guess(self):
        self._need(); from . import ui; ui.guess_game(self)

    def regression(self, model="decision trees (gradient boosting)"):
        self._need(); from . import ui; ui.regression_view(self, model)

    def classification(self, model="decision trees (gradient boosting)", task="grades", threshold=None):
        self._need(); from . import ui
        ui.classification_view(self, model, task, float(threshold if threshold is not None else self.table_spec.spec_default))

    def importance(self):
        self._need(); from . import ui; ui.importance_view(self)

    def whatif(self, start_from="a typical row"):
        self._need(); from . import ui; ui.whatif(self, start_from)

    def buildings(self):
        self._need(); from . import ui; ui.buildings_game(self)

    def buildings_answer(self, a="?", b="?", c="?", d="?"):
        self._need(); from . import ui; ui.buildings_check(self, a=a, b=b, c=c, d=d)

    def anatomy(self, building):
        self._need(); from . import ui; ui.anatomy(self, building)

    def forecast(self, building, method="all three"):
        self._need(); from . import ui; ui.forecast_view(self, building, method)

    def odd_days(self, building, threshold=3.5):
        self._need(); from . import ui; ui.oddday_view(self, building, float(threshold))

    # ------------------------------------------------------------------ the chat (hokie.ai) steps
    def chat_table(self, give="attach the files"):
        self._need(); from . import chat; chat.chat_table(self, give)

    def chat_table_tool(self, model="gradient-boosted trees"):
        self._need(); from . import chat; chat.chat_table_tool(self, model)

    def chat_forecast(self, building, give="attach the files"):
        self._need(); from . import chat; chat.chat_forecast(self, building, give)

    def chat_forecast_tool(self, building, model="gradient-boosted trees"):
        self._need(); from . import chat; chat.chat_forecast_tool(self, building, model)

    def chat_odd_days(self, building, give="attach the files"):
        self._need(); from . import chat; chat.chat_odd_days(self, building, give)

    def upload_app(self):
        self._need()
        if os.environ.get("AEC_LAB_NO_APP"):
            print("(upload app skipped: AEC_LAB_NO_APP is set)"); return
        from . import app
        self.apps["upload"] = app.launch(kind={"tabular": "table", "series": "time series"}.get(self.part))

    def report_summary(self):
        self._need(); from . import ui; ui.report_summary(self)

    def credits(self) -> dict:
        return credits(self.root)


lab = TabLab()
