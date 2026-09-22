"""Instructor-side: write docs/Homework_Answer_Keys.md - every homework sheet with its answer key drawn on it, and
what SAM 3 measures on it from boxes taken from the key (moved by a few percent, like a student's) and from phrases.

    AEC_SEG_MODEL_ID=... HF_HUB_OFFLINE=1 python build/answer_key_report.py [--set homework]

Needs the live model (GPU). Writes docs/keys/<id>.key.jpg (the key) and docs/keys/<id>.result.jpg (the measurement).
"""
import argparse
import contextlib
import io
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from aec_seg import lab, ui, viz  # noqa: E402
from aec_seg.data import plural  # noqa: E402

DOCS = REPO / "docs"
KEYS = DOCS / "keys"
MAX_SIDE = 1200

# the phrases the homework suggests; the second value is the key category the phrase is expected to hit (None = nothing)
PHRASES = {
    "usda_5542": [("room", "room"), ("bedroom", "room"), ("door", None), ("window", None), ("curved line", None)],
    "va_floor": [("room", "room"), ("door", None), ("curved line", None)],
    "test_fp": [("footing", "footing"), ("foundation", "footing"), ("square", "footing"), ("hatched square", "footing"), ("rectangle", "footing")],
    "test_fp_2": [("footing", "footing"), ("square", "footing"), ("hatched square", "footing"), ("rectangle", "footing")],
    "uscg_motorpool": [("footing", "footing"), ("square", "footing"), ("rectangle", "footing"), ("circle", "grid bubble")],
    "uscg_pile": [("footing", "pile footing"), ("square", "pile footing"), ("rectangle", "pile footing")],
    "va_ceiling": [("light fixture", "2x4 light fixture"), ("diffuser", None), ("rectangle", "2x4 light fixture"),
                   ("rectangle with a diagonal line", "2x4 light fixture"), ("small circle", "recessed light"), ("square", "2x2 light fixture")],
}
CAT_COLORS = {"room": "#2a9d8f", "footing": "#e63946", "pit": "#f4a261"}


def jitter(b, f=0.015):
    """A student's box: every side moved outward by a random few percent of the box."""
    w, h = b[2] - b[0], b[3] - b[1]
    return [b[0] - abs(random.gauss(0, f)) * w, b[1] - abs(random.gauss(0, f)) * h,
            b[2] + abs(random.gauss(0, f)) * w, b[3] + abs(random.gauss(0, f)) * h]


def save(img: Image.Image, path: Path):
    im = img.convert("RGB")
    if max(im.size) > MAX_SIDE:
        s = MAX_SIDE / max(im.size)
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    im.save(path, "JPEG", quality=78, optimize=True)


def key_overlay(sheet) -> Image.Image:
    """The answer key drawn on the sheet: areas as filled polygons with their true sq ft, scale references in blue,
    every counted symbol as a thin purple box."""
    im = sheet.load().convert("RGBA")
    fill = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(fill)
    font = viz._font(max(12, round(max(im.size) / 90)))
    for a in sheet.areas():
        col = viz.hex_rgb(CAT_COLORS.get(a["category"], "#7b2cbf"))
        if a.get("poly"):
            d.polygon([tuple(p) for p in a["poly"]], fill=col + (70,), outline=col + (255,))
        else:
            d.rectangle(a["box"], outline=col + (255,), width=3)
    out = Image.alpha_composite(im, fill).convert("RGB")
    d = ImageDraw.Draw(out)
    w = max(2, round(max(out.size) / 500))
    for a in sheet.areas():
        col = CAT_COLORS.get(a["category"], "#7b2cbf")
        tag = (a.get("label") or a["type"])[:18] + f" {a['true_sqft']:,.0f}" + ("" if a.get("takeoff", True) else " (not in take-off)")
        d.text((a["box"][0] + 3, a["box"][1] + 3), tag, fill=col, font=font)
    for cat, boxes in sheet.counts.items():
        for b in boxes:
            d.rectangle(b, outline=ui.PURPLE, width=w)
    for r in sheet.scale_refs:
        d.rectangle(r["box"], outline=ui.BLUE, width=w + 1)
        d.text((r["box"][0], max(0, r["box"][1] - font.size - 4)), f"scale: {r['label']} = {r['feet']:g} ft" + ("" if r.get("use") else " (check only)"),
               fill=ui.BLUE, font=font)
    return out


def measure(sheet):
    boxes = [(sheet.scale_label(r), jitter(r["box"], 0.01)) for r in sheet.scale_refs]
    boxes += [(sheet.area_label(a["category"]), jitter(a["box"])) for a in sheet.areas(takeoff_only=True)]
    for cat, truth in sheet.counts.items():
        if cat not in sheet.area_categories:                 # a footing's first area box is its own example
            boxes.append((sheet.example_label(cat), jitter(truth[0], 0.03)))
    over, rep = ui.takeoff_compute(lab, sheet.id, boxes, threshold=None)
    return over, rep


def phrase_rows(sheet):
    rows = []
    for phrase, cat in PHRASES.get(sheet.id, []):
        with contextlib.redirect_stdout(io.StringIO()):
            rec = ui.phrase_check(lab, sheet.id, phrase, 0.3) or {}
        if rec.get("compare"):
            score = f"{rec['found']}/{rec['truth']} found, {rec['extra']} extra"
            if rec.get("median_err") is not None:
                score += f"; areas median {rec['median_err']:.0f} %, worst {rec['worst_err']:.0f} %"
        else:
            score = "not scored (nothing in the key to score against)"
        rows.append((phrase, rec.get("regions", 0), rec.get("compare") or "-", score))
    return rows


def table(rows, header):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def sheet_section(sheet) -> tuple:
    over, rep = measure(sheet)
    save(key_overlay(sheet), KEYS / f"{sheet.id}.key.jpg")
    save(over, KEYS / f"{sheet.id}.result.jpg")
    md = [f"## {sheet.id} - {sheet.title}", "",
          f"*{sheet.discipline}.* {sheet.credit_line()}", "",
          f"**The answer key drawn on the sheet** (areas filled with their true sq ft, scale references in blue, every counted symbol in purple):", "",
          f"![{sheet.id} key](keys/{sheet.id}.key.jpg)", ""]
    # the key
    md += ["### Answer key", "", f"Scale: {sheet.px_per_ft:.2f} px/ft. {sheet.scale_how}", "",
           table([(r["label"], f"{r['feet']:g}", f"{ui._span(r['box'], r['axis']):.0f}", f"{ui._span(r['box'], r['axis']) / r['feet']:.2f}",
                   "yes" if r.get("use") else "check only", r.get("note", "")) for r in sheet.scale_refs],
                 ["scale reference", "feet", "px in the key", "px/ft", "used for the take-off", "note"]), ""]
    areas = sheet.areas()
    if areas:
        md += [table([(i, a.get("label") or "", a["type"], a["category"], a.get("printed", ""), f"{a['true_sqft']:,.1f}",
                       "yes" if a.get("takeoff", True) else "no", a.get("note", "")[:120]) for i, a in enumerate(areas, 1)],
                     ["#", "label", "type", "category", "printed size", "true sq ft", "in the take-off", "note"]), ""]
        for cat in sheet.area_categories:
            items = sheet.areas(cat, takeoff_only=True)
            md.append(f"- {plural(len(items), cat)} in the take-off, {sum(a['true_sqft'] for a in items):,.0f} sq ft together"
                      + (f" ({sum(a['true_sqft'] for a in sheet.areas(cat, indoor_only=True, takeoff_only=True)):,.0f} sq ft indoors)" if cat == "room" else "") + ".")
        md.append("")
    if sheet.counts:
        md += [table([(cat, len(boxes), f"{sheet.hint(cat)['threshold']:.2f}", sheet.hint(cat).get("tip", "")[:160]) for cat, boxes in sheet.counts.items()],
                     ["counted from one example", "how many on the sheet", "confidence the key suggests", "tip"]), ""]
    md += ["Tasks in the key:", ""] + [f"{i}. {t}" for i, t in enumerate(sheet.tasks, 1)] + [""]
    # the measurement
    md += ["### What SAM 3 measures (boxes taken from the key, each side moved by a few percent)", "",
           f"![{sheet.id} result](keys/{sheet.id}.result.jpg)", ""]
    sc = rep["scale"]
    md += [table([(r["label"], f"{r['px']:.0f}" if r["drawn"] else "-", f"{r['px_per_ft']:.2f}" if r["drawn"] else "-",
                   f"{r['err_pct']:+.1f} %" if r["drawn"] else "-") for r in sc["refs"]],
                 ["scale box", "px", "px/ft", "vs the key"]),
           f"Scale used: {sc['used_px_per_ft']:.2f} px/ft.", ""]
    summary = {"areas": {}, "counts": {}, "phrases": []}
    for cat, rows in rep["areas"].items():
        errs = [abs(r["err_pct"]) for r in rows if r.get("err_pct") is not None]
        md += [table([(r["n"], r["label"] or "-", f"{r['your_sqft']:,.1f}", f"{r['true_sqft']:,.0f}" if r["true_sqft"] else "-",
                       f"{r['err_pct']:+.0f} %" if r["err_pct"] is not None else "-", (r["note"] or "")[:110]) for r in rows],
                     ["#", f"{cat} (key)", "measured sq ft", "true sq ft", "error", "note"]),
               f"{plural(len(rows), cat)}: median error {np.median(errs):.1f} %, worst {max(errs):.0f} %." if errs else "", ""]
        if errs:
            summary["areas"][cat] = (len(rows), float(np.median(errs)), float(max(errs)))
    if rep["counts"]:
        md += [table([(cat, f"{c['threshold']:.2f}", c["found"], f"{c['matched']}/{c['truth']}", len(c["missed"]), len(c["extra"]))
                      for cat, c in rep["counts"].items()],
                     ["one example box of", "confidence", "regions returned", "found / on the sheet", "missed", "extra"]), ""]
        for cat, c in rep["counts"].items():
            summary["counts"][cat] = (c["matched"], c["truth"], len(c["extra"]))
    rows = phrase_rows(sheet)
    if rows:
        md += ["### What phrases find (the phrase cells 2b, 3b and 4b, confidence 0.3)", "",
               table(rows, ["phrase", "regions", "scored against (the key category the regions match best)", "result"]), ""]
        summary["phrases"] = rows
    return "\n".join(md), summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="homework")
    a = ap.parse_args()
    random.seed(20260922)
    KEYS.mkdir(parents=True, exist_ok=True)
    with contextlib.redirect_stdout(io.StringIO()):
        lab.setup(dataset=a.set, load_model=True, install=False)
    sections, overview = [], []
    for sheet in lab.sheets:
        print("measuring", sheet.id, flush=True)
        text, s = sheet_section(sheet)
        sections.append(text)
        areas = "; ".join(f"{cat} n={n}, median {m:.1f} %, worst {w:.0f} %" for cat, (n, m, w) in s["areas"].items()) or "-"
        counts = "; ".join(f"{cat} {f}/{t} (+{e})" for cat, (f, t, e) in s["counts"].items()) or "-"
        best = [r for r in s["phrases"] if r[2] != "-" and r[3].startswith(tuple("123456789")) and not r[3].startswith("0/")]
        overview.append((sheet.id, sheet.discipline, areas, counts, "; ".join(f"*{r[0]}*: {r[3].split(';')[0]}" for r in best) or "nothing works by phrase"))
    head = [f"# MP4 {a.set}: the answer keys and what SAM 3 does with them", "",
            "Generated by `build/answer_key_report.py` from `data/sheets/" + a.set + "/*.key.json` with the live model; every number here is",
            "reproducible from the keys. Boxes are the key's own boxes moved by a few percent, like a student's; counts come from one",
            "example box (the key's first one) through `segment_like`; phrases are scored by `lab.ask`. The raw feasibility runs that",
            "chose these sheets live in `_candidates/sam3_eval/` (not in the repository); the guide's section 2 holds the same numbers in prose.", "",
            "## Overview", "",
            table(overview, ["sheet", "discipline", "areas (measured)", "counts from one example (found / on the sheet, extras)", "phrases that work"]), ""]
    (DOCS / f"{a.set.capitalize()}_Answer_Keys.md").write_text("\n".join(head + sections), encoding="utf-8")
    print("wrote", DOCS / f"{a.set.capitalize()}_Answer_Keys.md", "and", len(list(KEYS.glob("*.jpg"))), "images in", KEYS)


if __name__ == "__main__":
    main()
