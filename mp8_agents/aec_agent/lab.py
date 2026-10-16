"""The one object the MP8 notebook talks to."""
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from . import score, ui
from .client import Gemini, NoKey, find_api_key
from .texts import FINALS, MODEL, SKILL, fill, fill_prompt, say, text
from .tools import PARAMETERS, Project, run_tool

TOOL_NAMES = ["search_spec", "calculator", "lookup_log"]
DOCUMENTS = {"specification (Section 03 30 00)": "spec_03_30_00.txt", "submittal log": "submittal_log.csv", "schedule": "schedule.csv"}


class AgentLab:
    def __init__(self):
        self.root = Path(__file__).resolve().parents[1]
        self.ready = False

    def setup(self, api_key: str = "", model: str = MODEL):
        t0 = time.time()
        d = self.root / "data"
        self.gemini = Gemini(find_api_key(api_key), d / "cache", model)
        self.project = Project(d / "project")
        self.questions = json.loads((d / "project" / "questions.json").read_text(encoding="utf-8"))
        self.takeoffs = json.loads((d / "project" / "takeoff_questions.json").read_text(encoding="utf-8"))
        self.drawings = json.loads((d / "drawings" / "problems.json").read_text(encoding="utf-8"))
        self.by_drawing = {p["label"]: p for p in self.drawings}
        self.workflow_runs: Dict[str, dict] = {}
        self.tool_runs: List[dict] = []
        self.skill_runs: List[dict] = []
        self.ready = True
        print(f"✅ Ready in {time.time() - t0:.0f} s.")

    def _need(self):
        if not self.ready:
            raise RuntimeError("Run Step 0 at the top of the notebook first.")

    def _image(self, p: dict) -> Path:
        return self.root / "data" / "drawings" / p["image"]

    @staticmethod
    def _finals(key: List[dict]) -> str:
        return "\n".join(f"FINAL: {k['q']} = <number> {k['unit']}" for k in key)

    # ================================================================== Part 1: workflows
    def show_drawing(self, drawing: str, **texts):
        self._need()
        p = self.by_drawing[drawing]
        ui.show_image(self._image(p))
        ui.show_text(p["text"])

    def one_prompt(self, **texts):
        """Each drawing with one prompt: the AI model reads the drawing and computes every quantity."""
        self._need(); G = "one_prompt"
        tmpl = text(texts, G, "prompt_text")
        rows, t_all, prog = [], 0.0, ui.Progress()
        try:
            for i, p in enumerate(self.drawings):
                prog.say(f"{p['label']} ({i + 1} of {len(self.drawings)})")
                prompt = fill_prompt(tmpl, problem=p["text"], lines=self._finals(p["key"]))
                r = self.gemini.ask(prompt, self._image(p)); t_all += r["seconds"]
                rows.append(score.takeoff(r["text"], p["key"]).assign(drawing=p["label"]))
        except NoKey as e:
            prog.done(); print(e); return
        prog.done()
        df = pd.concat(rows)[["drawing", "question", "quantity", "ground truth", "output", "unit", "correct"]]
        right = int((df.correct == "✓").sum())
        self.workflow_runs["one prompt"] = {"right": right, "n": len(df), "calls": len(self.drawings), "seconds": round(t_all, 1)}
        say(texts, G, "result_text", right=right, n=len(df))
        say(texts, G, "table_text"); ui.table(ui.named(df, text(texts, G, "columns_text")))

    def workflow(self, **texts):
        """A fixed workflow: step 1, the AI model reads the side lengths; step 2, code applies the estimating conventions."""
        self._need(); G = "workflow"
        tmpl = text(texts, G, "read_prompt_text")
        steps, rows, t_all, prog = [], [], 0.0, ui.Progress()
        try:
            for i, p in enumerate(self.drawings):
                prog.say(f"{p['label']} ({i + 1} of {len(self.drawings)})")
                r = self.gemini.ask(tmpl, self._image(p)); t_all += r["seconds"]
                s = score.sides(r["text"]); length = sum(s)
                q = score.compute(length, p) if s else {}
                steps.append({"drawing": p["label"], "sides": ", ".join(f"{v:g}" for v in s) or "—", "length": f"{length:g}" if s else "—",
                              "truth": f"{p['wall_length_ft']:g}"})
                lines = "\n".join(f"{k} = {v}" for k, v in q.items())
                rows.append(score.takeoff(lines, p["key"]).assign(drawing=p["label"]))
        except NoKey as e:
            prog.done(); print(e); return
        prog.done()
        df = pd.concat(rows)[["drawing", "question", "quantity", "ground truth", "output", "unit", "correct"]]
        right = int((df.correct == "✓").sum())
        self.workflow_runs["workflow"] = {"right": right, "n": len(df), "calls": len(self.drawings), "seconds": round(t_all, 1)}
        say(texts, G, "result_text", right=right, n=len(df))
        say(texts, G, "steps_text"); ui.table(ui.named(pd.DataFrame(steps), text(texts, G, "steps_columns_text")))
        say(texts, G, "table_text"); ui.table(ui.named(df, text(texts, G, "columns_text")))

    def compare_workflow(self, **texts):
        self._need(); G = "compare_workflow"
        if not self.workflow_runs:
            say(texts, G, "empty_text"); return
        names = list(self.workflow_runs)
        ui.bars(names, [self.workflow_runs[n]["right"] / self.workflow_runs[n]["n"] * 100 for n in names], text(texts, G, "title_text"), "%")
        df = pd.DataFrame([{"method": n, "correct": f"{v['right']} of {v['n']}", "calls": v["calls"]}
                           for n, v in self.workflow_runs.items()])
        ui.table(ui.named(df, text(texts, G, "columns_text")))

    # ================================================================== Part 2: an AI agent and its tools
    def show_document(self, document: str, **texts):
        self._need()
        p = self.root / "data" / "project" / DOCUMENTS[document]
        if p.suffix == ".csv":
            ui.table(pd.read_csv(p), wrap=False)
        else:
            ui.show_text(p.read_text(encoding="utf-8"))

    def _tools(self, names: List[str], texts: dict, group: str) -> Dict[str, dict]:
        return {n: {"description": text(texts, group, f"{n}_text"), "parameters": PARAMETERS[n]} for n in names}

    def _run_agent(self, system, question, tools, skills=None, extra=None):
        return self.gemini.agent(system, question, tools, lambda name, args: run_tool(self.project, skills or {}, name, args), extra)

    def agent_tools(self, search_spec: bool = True, calculator: bool = True, lookup_log: bool = True, **texts):
        """The six project questions, answered by the agent with the tools ticked."""
        self._need(); G = "tools"
        names = [n for n, on in zip(TOOL_NAMES, (search_spec, calculator, lookup_log)) if on]
        tools = self._tools(names, texts, G)
        system = text(texts, G, "system_text")
        results, prog = [], ui.Progress()
        try:
            for i, q in enumerate(self.questions):
                prog.say(f"{q['id']} ({i + 1} of {len(self.questions)})")
                r = self._run_agent(system, q["question"], tools)
                ans = score.final_answer(r["answer"])
                results.append({"q": q, "run": r, "answer": ans, "result": score.judge(ans, q)})
        except NoKey as e:
            prog.done(); print(e); return
        prog.done()
        label = ", ".join(names) or text(texts, G, "none_text")
        run = {"run": len(self.tool_runs) + 1, "tools": label, "results": results,
               "right": sum(r["result"] == "correct" for r in results), "unknown": sum(r["result"] == "UNKNOWN" for r in results),
               "wrong": sum(r["result"] == "wrong" for r in results), "calls": sum(len(r["run"]["steps"]) for r in results),
               "model_calls": sum(r["run"].get("model_calls", 1) for r in results)}
        self.tool_runs.append(run)
        say(texts, G, "result_text", run=run["run"], tools=label, right=run["right"], n=len(results), unknown=run["unknown"],
            wrong=run["wrong"], calls=run["calls"], model_calls=run["model_calls"])
        df = pd.DataFrame([{"q": f"{r['q']['id']}. {r['q']['question']}", "truth": score.truth_text(r["q"]), "answer": r["answer"],
                            "result": r["result"], "tools": ", ".join(s["tool"] for s in r["run"]["steps"]) or "—"} for r in results])
        say(texts, G, "table_text"); ui.table(ui.named(df, text(texts, G, "columns_text")))
        say(texts, G, "runs_text"); ui.table(ui.named(self._runs_table(), text(texts, G, "runs_columns_text")))

    def _runs_table(self) -> pd.DataFrame:
        return pd.DataFrame([{"run": r["run"], "tools": r["tools"], "correct": f"{r['right']} of {len(r['results'])}", "unknown": r["unknown"],
                              "wrong": r["wrong"], "calls": r["calls"], "model_calls": r["model_calls"]} for r in self.tool_runs])

    def show_trace(self, question: str, **texts):
        """Every run's tool calls for one question."""
        self._need(); G = "trace"
        if not self.tool_runs:
            say(texts, G, "empty_text"); return
        qid = question.split(".")[0].strip()
        q = next(x for x in self.questions if x["id"] == qid)
        runs = []
        for run in self.tool_runs:
            r = next(x for x in run["results"] if x["q"]["id"] == qid)
            runs.append({"title": fill(text(texts, G, "run_text"), run=run["run"], tools=run["tools"], result=r["result"]),
                         "steps": r["run"]["steps"], "answer": r["answer"], "full": r["run"]["answer"], "result": r["result"]})
        labels = {k: text(texts, G, f"{k}_label_text") for k in ("question", "call", "output", "answer")}
        ui.trace_flow(q["question"], runs, labels)

    def question_labels(self) -> List[str]:
        return [f"{q['id']}. {q['question']}" for q in self.questions]

    # ================================================================== Part 3: skills
    def _skill_questions(self):
        """The two takeoff questions (the skill applies) and one project question (it does not)."""
        tq = [{"id": t["id"], "question": t["text"] + "\n\n" + FINALS.replace("{lines}", self._finals(t["key"])), "key": t["key"]} for t in self.takeoffs]
        p3 = next(q for q in self.questions if q["id"] == "P3")
        return tq + [{"id": p3["id"], "question": p3["question"], "q": p3}]

    def agent_skill(self, use_skill: bool = True, **texts):
        self._need(); G = "skills"
        name = text(texts, G, "skill_name_text").strip(); desc = text(texts, G, "skill_description_text").strip()
        body = text(texts, G, "skill_body_text").strip()
        tools = self._tools(["calculator", "lookup_log"], texts, G)
        system = text(texts, G, "system_text")
        skills = {}
        if use_skill:
            tools["load_skill"] = {"description": text(texts, G, "load_skill_text"), "parameters": PARAMETERS["load_skill"]}
            system += "\n\n" + fill_prompt(text(texts, G, "skills_list_text"), name=name, description=desc)
            skills = {name: body}
        if use_skill:                                       # the skill itself, so it is seen without opening the code
            say(texts, G, "card_text", name=name, description=desc); ui.show_text(body)
        rows, loaded, calls, model_calls, prog = [], [], 0, 0, ui.Progress()
        try:
            for i, q in enumerate(self._skill_questions()):
                prog.say(f"{q['id']} ({i + 1} of 3)")
                r = self._run_agent(system, q["question"], tools, skills, extra=[name, desc, body] if use_skill else None)
                used = any(s["tool"] == "load_skill" for s in r["steps"])
                if used:
                    loaded.append(q["id"])
                calls += len(r["steps"]); model_calls += r.get("model_calls", 1)
                if "key" in q:
                    t = score.takeoff("\n".join(score.final_lines(r["answer"])), q["key"])
                    rows.append(t.assign(question=[f"{q['id']} {x}" for x in t.question], loaded="yes" if used else "no"))
                else:
                    ans = score.final_answer(r["answer"])
                    rows.append(pd.DataFrame([{"question": q["id"], "quantity": "review status", "ground truth": score.truth_text(q["q"]),
                                               "output": ans, "unit": "", "correct": "✓" if score.judge(ans, q["q"]) == "correct" else "✗",
                                               "loaded": "yes" if used else "no"}]))
        except NoKey as e:
            prog.done(); print(e); return
        prog.done()
        df = pd.concat(rows)
        takeoff_rows = df[df.question.str.startswith("T")]
        right = int((takeoff_rows.correct == "✓").sum())
        skill_label = name if use_skill else text(texts, G, "no_skill_text")
        loaded_label = ", ".join(loaded) or text(texts, G, "none_loaded_text")
        run = {"run": len(self.skill_runs) + 1, "skill": skill_label, "right": right, "n": len(takeoff_rows), "loaded": loaded_label,
               "calls": calls, "model_calls": model_calls}
        self.skill_runs.append(run)
        say(texts, G, "result_text", run=run["run"], skill=skill_label, right=right, n=len(takeoff_rows), loaded=loaded_label,
            calls=calls, model_calls=model_calls)
        say(texts, G, "table_text"); ui.table(ui.named(df, text(texts, G, "columns_text")))
        say(texts, G, "runs_text")
        ui.table(ui.named(pd.DataFrame([{**{k: v for k, v in r.items() if k not in ("right", "n")}, "correct": f"{r['right']} of {r['n']}"}
                                        for r in self.skill_runs])[["run", "skill", "correct", "loaded", "calls", "model_calls"]],
                          text(texts, G, "runs_columns_text")))

    def hokieai_skill(self, skill: str = "with the skill", **texts):
        self._need(); G = "hokieai_skill"
        problems = "\n\n".join(f"{t['id']}. " + t["text"].replace("\nQ", f"\n{t['id']} Q") for t in self.takeoffs)
        lines = "\n".join(f"FINAL: {t['id']} {k['q']} = <number> {k['unit']}" for t in self.takeoffs for k in t["key"])
        prompt = fill_prompt(text(texts, G, "prompt_text"), problems=problems, lines=lines)

        def on_score(reply):
            df = pd.concat([score.takeoff(reply, t["key"], prefix=t["id"]).assign(question=lambda d, t=t: [f"{t['id']} {x}" for x in d.question])
                            for t in self.takeoffs])
            right = int((df.correct == "✓").sum())
            say(texts, G, "result_text", skill=skill, right=right, n=len(df))
            say(texts, G, "table_text"); ui.table(ui.named(df, text(texts, G, "columns_text")))
        ui.hokieai_box(prompt, on_score, texts, G)

    # ================================================================== Part 4: your own HokieAI agent
    def project_files(self) -> List[Path]:
        return [self.root / "data" / "project" / f for f in DOCUMENTS.values()]

    def create_agent(self, **texts):
        """The steps to create the agent, the three files to upload, and the instructions to paste."""
        self._need(); G = "hokieai_agent"
        import ipywidgets as w
        from IPython.display import display
        from .texts import CHAT_URL, fill
        how = ui.md_html(fill(text(texts, G, "create_steps_text"), url=CHAT_URL))
        box = w.Textarea(value=text(texts, G, "instructions_text"), layout=w.Layout(width="100%", height="120px"))
        display(w.VBox([w.HTML(how), *ui.download_buttons(self.project_files()), box]))

    def hokieai_agent(self, **texts):
        self._need(); G = "hokieai_agent"
        qs = "\n".join(f"{q['id']}. {q['question']}" for q in self.questions)
        prompt = fill_prompt(text(texts, G, "prompt_text"), questions=qs)
        ids = [q["id"] for q in self.questions]

        def on_score(reply):
            got = score.answers_by_id(reply, ids)
            res = [(q, got.get(q["id"], ""), score.judge(got.get(q["id"], ""), q) if q["id"] in got else "wrong") for q in self.questions]
            right = sum(r == "correct" for *_, r in res); unknown = sum(r == "UNKNOWN" for *_, r in res); wrong = len(res) - right - unknown
            say(texts, G, "result_text", right=right, unknown=unknown, wrong=wrong, n=len(res))
            df = pd.DataFrame([{"q": f"{q['id']}. {q['question']}", "truth": score.truth_text(q), "answer": a or "—", "result": r} for q, a, r in res])
            say(texts, G, "table_text"); ui.table(ui.named(df, text(texts, G, "columns_text")))
            comp = [{"agent": "HokieAI agent", "correct": f"{right} of {len(res)}", "unknown": unknown, "wrong": wrong}]
            comp += [{"agent": f"Gemini agent, run {r['run']} ({r['tools']})", "correct": f"{r['right']} of {len(r['results'])}",
                      "unknown": r["unknown"], "wrong": r["wrong"]} for r in self.tool_runs]
            say(texts, G, "compare_text"); ui.table(ui.named(pd.DataFrame(comp), text(texts, G, "compare_columns_text")))
        ui.hokieai_box(prompt, on_score, texts, G)


lab = AgentLab()
