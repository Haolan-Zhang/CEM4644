"""The one object the MP7 notebook talks to."""
import json
import time
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from . import score, ui
from .texts import FOUNDATION_CONVENTIONS, ROOF_CONVENTIONS, fill, fill_prompt, say, text

INCIDENT_VERSIONS = {"v1 · task only": "v1", "v2 · role and definitions": "v2", "v3 · definitions and worked examples": "v3", "my own prompt": "own"}
TAKEOFF_VERSIONS = {"v1 · task only": "v1", "v2 · step-by-step reasoning": "v2", "v3 · estimating conventions": "v3", "my own prompt": "own"}
SCHEDULE_VERSIONS = {"v1 · whole sheet": "v1", "v2 · cropped schedule": "v2", "v3 · cropped schedule and column names": "v3", "my own prompt": "own"}
SUBMITTAL_VERSIONS = {"v1 · task only": "v1", "v2 · requirements first, then compare": "v2", "v3 · comparison table": "v3", "my own prompt": "own"}
CONTRACT_VERSIONS = {"v1 · questions only": "v1", "v2 · answers with quotes": "v2", "v3 · only from the text, with quotes": "v3", "my own prompt": "own"}
FAMILY_NAME = {"foundation": "Foundation plan", "roof": "Gable roof", "schedule": "Door schedule sheet"}


def submittal_label(p: dict) -> str:
    n = p["id"].split("_")[-1]
    kind = "Concrete mix design" if p["family"] == "concrete" else "Mortar and grout"
    return f"{kind} {n} · {p['short']}"


def drawing_label(p: dict) -> str:
    n = p["id"].split("_")[-1]
    return f"{FAMILY_NAME[p['family']]} {n}" + (f" (unit type {p['unit']})" if p["family"] == "schedule" else "")


class PromptLab:
    def __init__(self):
        self.root = Path(__file__).resolve().parents[1]
        self.ready = False
        self.runs = []                         # every scored output: task, drawing, version, correct, of

    def setup(self):
        t0 = time.time()
        d = self.root / "data"
        self.incidents = pd.read_csv(d / "incidents.csv")
        self.examples = pd.read_csv(d / "incident_examples.csv")
        self.problems = json.loads((d / "practice" / "problems.json").read_text())
        self.drawings = d / "practice" / "drawings"
        self.by_label = {drawing_label(p): p for p in self.problems}
        self.submittals = json.loads((d / "submittals" / "problems.json").read_text())
        self.specs = {k: (d / "submittals" / f"spec_{k}.txt").read_text() for k in ("concrete", "masonry")}
        self.by_submittal = {submittal_label(p): p for p in self.submittals}
        self.clauses = (d / "contract" / "clauses.txt").read_text()
        self.questions = json.loads((d / "contract" / "questions.json").read_text())
        self.runs = []
        self.ready = True
        print(f"✅ Ready in {time.time() - t0:.0f} s.")

    def _need(self):
        if not self.ready:
            raise RuntimeError("Run Step 0 at the top of the notebook first.")

    def _problem(self, label: str) -> dict:
        p = self.by_label.get(label)
        if p is None:
            raise KeyError(f"No drawing called {label!r}")
        return p

    def _record(self, task, drawing, version, right, n):
        self.runs.append({"task": task, "example": drawing, "version": version, "correct": int(right), "of": int(n)})

    @staticmethod
    def _named(df: pd.DataFrame, texts: dict, group: str) -> pd.DataFrame:
        """The table with the column names of the cell's columns_text (names separated by |), if their number matches."""
        names = [c.strip() for c in text(texts, group, "columns_text").split("|")]
        return df.set_axis(names, axis=1) if len(names) == df.shape[1] else df

    @staticmethod
    def _template(texts: dict, group: str, versions: dict, version: str) -> str:
        """The cell's own prompt (prompt_text, in the one-prompt notebook), else the chosen prompt version's."""
        if texts.get("prompt_text"):
            return texts["prompt_text"]
        v = versions.get(version, "v3")
        return text(texts, group, f"{'v3' if v == 'own' else v}_prompt_text")

    @staticmethod
    def _version_label(choice: str, versions: dict, original: str, sent: str, texts: Optional[dict] = None) -> str:
        edited = sent.strip() != original.strip()
        if (texts or {}).get("prompt_text"):
            return "edited prompt" if edited else "given prompt"
        if versions.get(choice) == "own":
            return "my own prompt"
        return f"my own prompt (edited {choice.split(' ·')[0]})" if edited else choice

    # ------------------------------------------------------------------ Part 1: incident reports
    def show_reports(self, rows: int = 10, **texts):
        self._need()
        ui.table(self.incidents[["id", "report"]].head(int(rows)))

    def _incident_prompt(self, version: str, texts: dict) -> str:
        tmpl = self._template(texts, "incidents", INCIDENT_VERSIONS, version)
        reports = "\n".join(f"{r.id}: {r.report}" for r in self.incidents.itertuples())
        examples = "\n".join(f"Report: {r.report}\nCategory: {r.focus_four}" for r in self.examples.itertuples())
        return fill_prompt(tmpl, reports=reports, examples=examples)

    def classify_incidents(self, version: str = "v1 · task only", **texts):
        self._need()
        prompt = self._incident_prompt(version, texts)
        G = "incidents"

        def on_score(reply, sent):
            label = self._version_label(version, INCIDENT_VERSIONS, prompt, sent, texts)
            r = score.incidents(reply, self.incidents)
            self._record("incidents", "30 reports", label, r["right"], r["n"])
            if r["got"] < r["n"]:
                say(texts, G, "missing_text", got=r["got"], n=r["n"])
            say(texts, G, "accuracy_text", version=label, right=r["right"], n=r["n"], accuracy=round(r["right"] / r["n"] * 100))
            say(texts, G, "matrix_text"); ui.matrix(r["matrix"])
            if len(r["disagree"]):
                say(texts, G, "disagree_text"); ui.table(self._named(r["disagree"], texts, G))
        ui.hokieai_box([], prompt, on_score, texts, G)

    # ------------------------------------------------------------------ Part 2: submittal review
    def review_submittal(self, submittal: str, version: str = "v1 · task only", **texts):
        self._need(); G = "submittals"
        p = self.by_submittal.get(submittal)
        if p is None:
            raise KeyError(f"No submittal called {submittal!r}")
        tmpl = self._template(texts, G, SUBMITTAL_VERSIONS, version)
        prompt = fill_prompt(tmpl, final=text(texts, G, "final_text"), spec=self.specs[p["spec"]], submittal=p["text"])

        def on_score(reply, sent):
            label = self._version_label(version, SUBMITTAL_VERSIONS, prompt, sent, texts)
            r = score.submittal(reply, p)
            self._record("submittals", submittal, label, r["right"], r["n"])
            say(texts, G, "result_text", version=label, submittal=submittal, right=r["right"], n=r["n"], found=r["found"], planted=r["planted"], false=r["false"])
            say(texts, G, "table_text"); ui.table(self._named(r["table"], texts, G))
        ui.hokieai_box([], prompt, on_score, texts, G)

    # ------------------------------------------------------------------ Part 3: contract questions
    def contract_questions(self, version: str = "v1 · questions only", check_quotes: bool = True, **texts):
        self._need(); G = "contract"
        tmpl = self._template(texts, G, CONTRACT_VERSIONS, version)
        qs = "\n".join(f"{q['id']}. {q['question']}" for q in self.questions)
        prompt = fill_prompt(tmpl, questions=qs, clauses=self.clauses)

        def on_score(reply, sent):
            label = self._version_label(version, CONTRACT_VERSIONS, prompt, sent, texts)
            r = score.contract(reply, self.questions, self.clauses)
            self._record("contract", f"{r['n']} questions", label, r["right"], r["n"])
            say(texts, G, "result_text", version=label, right=r["right"], n=r["n"], made_up=r["made_up"], unanswerable=r["unanswerable"],
                verified=r["verified"], quotes=r["quotes"])
            table = r["table"] if check_quotes else r["table"].drop(columns="quote found")
            say(texts, G, "table_text"); ui.table(self._named(table, texts, G))
        ui.hokieai_box([], prompt, on_score, texts, G)

    # ------------------------------------------------------------------ Part 4: takeoff from drawings
    def takeoff(self, drawing: str, version: str = "v1 · task only", **texts):
        self._need()
        p = self._problem(drawing); G = "takeoff"
        tmpl = self._template(texts, G, TAKEOFF_VERSIONS, version)
        lines = "\n".join(f"FINAL | {k['q']} | <number> | <unit>" for k in p["key"])
        conv = texts.get("conventions_text") or (FOUNDATION_CONVENTIONS if p["family"] == "foundation" else ROOF_CONVENTIONS)
        prompt = fill_prompt(tmpl, problem=p["text"], conventions=conv, finals=fill_prompt(text(texts, G, "finals_text"), lines=lines))
        img = self.drawings / p["image"]
        ui.show_image(img, 760)

        def on_score(reply, sent):
            label = self._version_label(version, TAKEOFF_VERSIONS, prompt, sent, texts)
            r = score.takeoff(reply, p["key"])
            self._record(p["family"], drawing, label, r["right"], r["n"])
            say(texts, G, "result_text", version=label, drawing=drawing, right=r["right"], n=r["n"])
            say(texts, G, "table_text"); ui.table(self._named(r["table"], texts, G))
        ui.hokieai_box([img], prompt, on_score, texts, G)

    # ------------------------------------------------------------------ Part 5: door schedules
    def schedule(self, sheet: str, version: str = "v1 · whole sheet", **texts):
        self._need()
        p = self._problem(sheet); G = "schedule"
        tmpl = self._template(texts, G, SCHEDULE_VERSIONS, version)
        v = SCHEDULE_VERSIONS.get(version, "v3")
        prompt = fill_prompt(tmpl, unit=p["unit"], format=text(texts, G, "format_text"))
        full, crop = self.drawings / p["image"], self.drawings / p["crop"]
        files = [full] if v == "v1" else ([crop] if v in ("v2", "v3") else [full, crop])
        ui.show_image(files[0], 900)

        def on_score(reply, sent):
            label = self._version_label(version, SCHEDULE_VERSIONS, prompt, sent, texts)
            r = score.schedule(reply, p["key"])
            self._record("schedule", sheet, label, r["right"], r["n"])
            say(texts, G, "result_text", version=label, sheet=sheet, right=r["right"], n=r["n"], fields_right=r["fields_right"], fields=r["fields"])
            say(texts, G, "table_text"); ui.table(self._named(r["table"], texts, G))
        ui.hokieai_box(files, prompt, on_score, texts, G)

    # ------------------------------------------------------------------ comparisons
    def compare(self, task: str, **texts):
        """task: "incidents", "submittals", "contract", "takeoff" (foundation plans and roofs) or "schedule"."""
        self._need()
        tasks = {"takeoff": ("foundation", "roof")}.get(task, (task,))
        runs = pd.DataFrame(self.runs, columns=["task", "example", "version", "correct", "of"])
        ui.compare(runs[runs.task.isin(tasks)].drop(columns="task"), texts)

    @staticmethod
    def drawings_of(family: str, problems) -> list:
        return [drawing_label(p) for p in problems if p["family"] == family]


lab = PromptLab()
