"""What each step shows: the HokieAI box (files, prompt, output, Score), the drawing, result tables and the comparison chart."""
import html
import io
import re
from pathlib import Path
from typing import Callable, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import HTML, Image, display

from .texts import CHAT_URL, fill, say, text

BLUE, GREEN, ORANGE, GREY = "#457b9d", "#2a9d8f", "#f4a261", "#8d99ae"


def table(df: pd.DataFrame, max_rows: int = 60):
    display(HTML(df.head(max_rows).to_html(index=False, border=0, escape=True).replace("<th>", "<th style='text-align:left'>")))


def matrix(df: pd.DataFrame):
    display(HTML(df.to_html(border=0).replace("<th>", "<th style='text-align:center'>")))


def show_fig(fig):
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=100, bbox_inches="tight"); plt.close(fig)
    display(Image(buf.getvalue()))


def show_image(path: Path, width: int = 900):
    display(Image(filename=str(path), width=width))


def md_html(s: str) -> str:
    """The little markdown the instructions use: **bold**, *italic*, links, one line per line."""
    t = html.escape(s.strip())
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", t)
    t = re.sub(r"(https?://[^\s<]+[^\s<.,;)])", r"<a href='\1' target='_blank'>\1</a>", t)
    return "<br>".join(t.split("\n"))


def hokieai_box(files: List[Path], prompt: str, on_score: Callable[[str, str], None], texts: Optional[dict], group: str):
    """Download buttons, the prompt to copy (editable), a box for HokieAI's output, and a Score button.
    on_score(output, prompt_as_sent) is called with the prompt as it is in the box when Score is clicked."""
    import ipywidgets as w
    names = " and ".join(f"*{p.name}*" for p in files)
    how = md_html(fill(text(texts, group, "steps_text"), files=names, url=CHAT_URL))
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
    lab_w = {"description_width": "60px"}
    prompt_box = w.Textarea(value=prompt, layout=w.Layout(width="100%", height="200px"), description=text(texts, group, "prompt_label_text"), style=lab_w)
    reply_box = w.Textarea(placeholder=text(texts, group, "reply_text"), layout=w.Layout(width="100%", height="160px"),
                           description=text(texts, group, "reply_label_text"), style=lab_w)
    btn = w.Button(description=text(texts, group, "button_text"), button_style="primary", layout=w.Layout(width="auto"))
    out = w.Output()

    def _go(_):
        with out:
            out.clear_output(wait=True)
            reply = reply_box.value.strip()
            if not reply:
                print("Paste HokieAI's output first."); return
            try:
                on_score(reply, prompt_box.value)
            except Exception as e:  # noqa: BLE001
                print(f"Could not read that output: {e}")
    btn.on_click(_go)
    display(w.VBox([w.HTML(how), *rows, prompt_box, reply_box, btn, out]))


def compare(runs: pd.DataFrame, texts: Optional[dict]):
    """Every scored output of one task, and the share correct by prompt version."""
    G = "compare"
    if runs.empty:
        say(texts, G, "empty_text"); return
    s = runs.groupby("version").agg(outputs=("correct", "size"), correct=("correct", "sum"), of=("of", "sum")).reset_index()
    s["correct (%)"] = (s.correct / s.of * 100).round(0).astype(int)
    order = sorted(s.version, key=lambda v: (v.startswith("my own"), v))
    s = s.set_index("version").loc[order].reset_index()
    fig, ax = plt.subplots(figsize=(7.5, 0.55 * len(s) + 1.2))
    ax.barh(s.version[::-1], s["correct (%)"][::-1], color=[ORANGE if v.startswith("my own") else BLUE for v in s.version[::-1]])
    for y, v in enumerate(s["correct (%)"][::-1]):
        ax.text(v + 1, y, f"{v} %", va="center", fontsize=9)
    ax.set_xlim(0, 108); ax.set_xlabel(fill(text(texts, G, "axis_text"))); ax.set_title(fill(text(texts, G, "title_text")), fontsize=11)
    show_fig(fig)
    say(texts, G, "table_text")
    table(runs.assign(**{"correct (%)": (runs.correct / runs.of * 100).round(0).astype(int)}))
