"""What each step shows: tables, drawings, the agent's trace, the comparison chart and the HokieAI box."""
import html
import io
import re
from pathlib import Path
from typing import Callable, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import HTML, Image, display

from .texts import CHAT_URL, fill, text

BLUE, GREEN, ORANGE, GREY = "#457b9d", "#2a9d8f", "#f4a261", "#8d99ae"


def named(df: pd.DataFrame, columns_text: str) -> pd.DataFrame:
    """The table with the column names of a columns_text wording (names separated by |), if their number matches."""
    names = [c.strip() for c in columns_text.split("|")]
    return df.set_axis(names, axis=1) if len(names) == df.shape[1] else df


def table(df: pd.DataFrame, max_rows: int = 80, wrap: bool = True):
    s = df.head(max_rows).to_html(index=False, border=0, escape=True).replace("<th>", "<th style='text-align:left'>")
    if wrap:
        s = s.replace("<td>", "<td style='text-align:left; white-space:pre-wrap; vertical-align:top'>")
    display(HTML(s))


def show_image(path: Path, width: int = 820):
    display(Image(filename=str(path), width=width))


def show_text(s: str):
    display(HTML(f"<pre style='white-space:pre-wrap; font-size:12px'>{html.escape(s)}</pre>"))


def bars(labels: List[str], values: List[float], title: str, xlabel: str, highlight: Optional[List[bool]] = None):
    fig, ax = plt.subplots(figsize=(7.5, 0.5 * len(labels) + 1.2))
    colors = [ORANGE if h else BLUE for h in (highlight or [False] * len(labels))]
    ax.barh(labels[::-1], values[::-1], color=colors[::-1])
    for y, v in enumerate(values[::-1]):
        ax.text(v + 1, y, f"{v:.0f} %", va="center", fontsize=9)
    ax.set_xlim(0, 110); ax.set_xlabel(xlabel); ax.set_title(title, fontsize=11)
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=100, bbox_inches="tight"); plt.close(fig)
    display(Image(buf.getvalue()))


def call_text(step: dict) -> str:
    args = ", ".join(f"{k}={v!r}" for k, v in step["args"].items())
    return f"{step['tool']}({args})"


# ----------------------------------------------------------------------------- the agent's trace, drawn as cards
CARD = {"question": ("#eef1f5", "#8d99ae"), "call": ("#dbe9f6", "#457b9d"), "output": ("#d8f0ec", "#2a9d8f"),
        "correct": ("#d8f0ec", "#2a9d8f"), "wrong": ("#fde2e4", "#e63946"), "UNKNOWN": ("#f3f3f3", "#999999")}


def readable(output: str) -> str:
    """A tool's output as short lines: each table row as 'column: value' pairs, other lines as they are."""
    lines = [l for l in output.split("\n") if l.strip()]
    out, head = [], None
    for l in lines:
        cells = [c.strip() for c in l.split("|")]
        if len(cells) > 2 and head is None and (cells[0].lower() in ("submittal", "activity", "element")):
            head = cells; continue
        if len(cells) > 2 and head and len(cells) == len(head):
            out.append(" · ".join(f"{h}: {c}" for h, c in zip(head, cells) if c))
        else:
            out.append(l.strip())
    return "\n".join(out)


def _box(kind: str, label: str, body: str, title: str = "") -> str:
    face, edge = CARD[kind]
    return (f"<div title='{html.escape(title)}' style='background:{face}; border:2px solid {edge}; border-radius:8px; padding:5px 9px; "
            f"max-width:260px; color:#111'><div style='font-size:11px; color:{edge}; font-weight:bold'>{html.escape(label)}</div>"
            f"<div style='font-size:12.5px; white-space:pre-wrap; max-height:90px; overflow:auto'>{html.escape(body)}</div></div>")


def _step(k: int, step: dict, labels: dict) -> str:
    """One tool call and its output in one card: the call (blue) above the output (green)."""
    (cf, ce), (of, oe) = CARD["call"], CARD["output"]
    args = ", ".join(f"{a}={v!r}" for a, v in step["args"].items())
    return (f"<div style='border:2px solid {ce}; border-radius:8px; overflow:hidden; width:250px; color:#111'>"
            f"<div style='background:{cf}; padding:5px 9px'><div style='font-size:11px; color:{ce}; font-weight:bold'>{k} · {html.escape(labels['call'])}</div>"
            f"<div style='font-family:monospace; font-size:12px; font-weight:bold'>{html.escape(step['tool'])}</div>"
            f"<div style='font-family:monospace; font-size:11px; white-space:pre-wrap'>{html.escape(args)}</div></div>"
            f"<div style='background:{of}; border-top:2px solid {oe}; padding:5px 9px'><div style='font-size:11px; color:{oe}; font-weight:bold'>"
            f"{html.escape(labels['output'])}</div><div style='font-size:11.5px; white-space:pre-wrap; max-height:80px; overflow:auto'>"
            f"{html.escape(readable(step['output']))}</div></div></div>")


ARROW = "<div style='font-size:20px; color:#555; padding:0 2px; align-self:center'>→</div>"


def trace_flow(question: str, runs: List[dict], labels: dict):
    """The question once, then one row per run: each tool call with its output, and the answer coloured by its result."""
    parts = [f"<div style='margin-bottom:12px'>{_box('question', labels['question'], question)}</div>"]
    for r in runs:
        cards = [_step(k, st, labels) for k, st in enumerate(r["steps"], 1)]
        cards.append(_box(r["result"] if r["result"] in CARD else "UNKNOWN", f"{labels['answer']} · {r['result']}", r["answer"], title=r["full"]))
        parts.append("<div style='display:flex; gap:10px; align-items:flex-start; margin-bottom:14px'>"
                     f"<div style='width:150px; flex:none; color:#111; font-size:13px'>{md_html(r['title'])}</div>"
                     "<div style='display:flex; flex-wrap:wrap; gap:4px; align-items:center'>" + ARROW.join(cards) + "</div></div>")
    display(HTML("<div style='background:white; padding:6px'>" + "".join(parts) + "</div>"))


def gantt(rows: List[dict], title: str, axis: str):
    """The look-ahead schedule: one bar per activity, red where the mix design is not approved."""
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(rows) + 1.2))
    for y, r in enumerate(rows[::-1]):
        ax.barh(y, r["finish"] - r["start"] + 1, left=r["start"] - 0.5, color="#e63946" if r["flag"] == "YES" else GREY, height=0.6)
        ax.text(r["finish"] + 0.7, y, f"{r['start']}–{r['finish']}", va="center", fontsize=8)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([f"{r['activity']} {r['description']}" for r in rows[::-1]], fontsize=9)
    last = max(r["finish"] for r in rows)
    ax.set_xlim(0.5, last + 3); ax.set_xticks(range(1, last + 1)); ax.tick_params(axis="x", labelsize=8)
    ax.set_xlabel(axis); ax.set_title(title, fontsize=10); ax.grid(axis="x", alpha=0.3)
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=100, bbox_inches="tight"); plt.close(fig)
    display(Image(buf.getvalue()))


def outline_figure(corners, pts, title: str):
    """The drawing's outline (gray) and the one drawn from the model's sides (orange), with the end point marked."""
    import numpy as np
    P = np.array(corners + [corners[0]], float)
    fig, ax = plt.subplots(figsize=(6, 4.6))
    ax.plot(P[:, 0], P[:, 1], color=GREY, lw=6, alpha=0.5, label="the drawing")
    if len(pts) > 1:
        ax.plot(pts[:, 0], pts[:, 1], color=ORANGE, lw=2, marker="o", ms=3, label="from the model's sides")
        ax.plot(*pts[-1], marker="x", color="#e63946", ms=10, mew=2)
    ax.set_aspect("equal"); ax.legend(fontsize=8, loc="best"); ax.set_title(title, fontsize=10); ax.set_xlabel("ft"); ax.set_ylabel("ft")
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=100, bbox_inches="tight"); plt.close(fig)
    display(Image(buf.getvalue()))


class Progress:
    """One line that says which question is running (live runs take a few seconds each), cleared at the end."""

    def __init__(self):
        self.h = display(HTML(""), display_id=True)

    def say(self, s: str):
        if self.h:
            self.h.update(HTML(f"<i>{html.escape(s)}</i>"))

    def done(self):
        if self.h:
            self.h.update(HTML(""))


def md_html(s: str) -> str:
    t = html.escape(s.strip())
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", t)
    t = re.sub(r"(https?://[^\s<]+[^\s<.,;)])", r"<a href='\1' target='_blank'>\1</a>", t)
    return "<br>".join(t.split("\n"))


def download_buttons(files: List[Path]):
    import ipywidgets as w
    rows = []
    for p in files:
        b = w.Button(description=f"Download {p.name}", icon="download", layout=w.Layout(width="auto")); o = w.Output()

        def _dl(_, p=p, o=o):
            with o:
                o.clear_output()
                try:
                    from google.colab import files as colab_files
                    colab_files.download(str(p))
                except Exception:  # noqa: BLE001
                    print(f"Not in Colab: the file is at {p}")
        b.on_click(_dl); rows.append(w.HBox([b, o]))
    return rows


def hokieai_box(prompt: str, on_score: Callable[[str], None], texts: Optional[dict], group: str, files: List[Path] = ()):
    """The steps, download buttons, the prompt to copy, a box for HokieAI's output, and a Score button."""
    import ipywidgets as w
    names = " and ".join(f"*{p.name}*" for p in files)
    how = md_html(fill(text(texts, group, "steps_text"), url=CHAT_URL, files=names))
    lw = {"description_width": "60px"}
    prompt_box = w.Textarea(value=prompt, layout=w.Layout(width="100%", height="200px"), description=text(texts, group, "prompt_label_text"), style=lw)
    reply_box = w.Textarea(placeholder=text(texts, group, "reply_text"), layout=w.Layout(width="100%", height="160px"),
                           description=text(texts, group, "reply_label_text"), style=lw)
    btn = w.Button(description=text(texts, group, "button_text"), button_style="primary", layout=w.Layout(width="auto"))
    out = w.Output()

    def _go(_):
        with out:
            out.clear_output(wait=True)
            reply = reply_box.value.strip()
            if not reply:
                print("Paste HokieAI's output first."); return
            try:
                on_score(reply)
            except Exception as e:  # noqa: BLE001
                print(f"Could not read that output: {e}")
    btn.on_click(_go)
    display(w.VBox([w.HTML(how), *download_buttons(list(files)), prompt_box, reply_box, btn, out]))
