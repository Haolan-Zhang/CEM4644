"""The notebook steps: each prints what the student needs and shows a picture or a table."""
import json
from typing import List, Optional

import numpy as np

from . import config as C
from . import tasks, viz

PALETTE = ["#e63946", "#2a9d8f", "#f4a261", "#7b2cbf", "#457b9d", "#e9c46a", "#06d6a0", "#8d6e63", "#4cc9f0", "#ffd166", "#9aa5b1", "#ff70a6"]
MODES = {"LLM only (polygons)": "llm", "LLM boxes + SAM 3": "llm+sam"}


def _snippet(text: str, n: int = 500) -> str:
    t = (text or "").strip().replace("\n", " ")
    return t[:n] + (" …" if len(t) > n else "")


def _cost_line(r) -> str:
    src = "precomputed" if r.cached else "live"
    return f"{r.seconds:.1f} s, tokens in {r.tokens_in} / out {r.tokens_out} / thinking {r.tokens_thought} ({src}, {r.model})"


# ----------------------------------------------------------------------------- Part 1
def show_examples(lab, which: str):
    ex = lab.examples
    if which.startswith("photo"):
        ims = [p.load() for p in ex.photos]
        titles = [f"{p.file.rsplit('.', 1)[0]}\ntruth: {p.truth}" for p in ex.photos]
        viz.show(viz.image_grid(ims, titles, ncols=7, size=2.2, suptitle=f"{ex.spec.photos.title}: {len(ims)} photos, {len(ex.photo_classes)} classes"))
        print("Classes:", "; ".join(ex.photo_classes))
        print(f"Answer key: the folder each photo came from. Specialist for comparison: {ex.photo_specialist} "
              f"({tasks.specialist_cls_accuracy(ex.photos):.0f} % correct on these {len(ims)}).")
        print("Source:", ex.spec.photos.source)
    elif which.startswith("site"):
        cols = ex.spec.sites.colors
        ims = [viz.draw_boxes(s.load(), [(t["label"], t["box"], cols.get(t["label"], viz.GREY), 2) for t in s.truth]) for s in ex.sites]
        titles = [f"{s.file.rsplit('.', 1)[0]}: " + ", ".join(f"{n} {ex.spec.sites.display.get(l, l)}" for l, n in s.counts().items()) for s in ex.sites]
        viz.show(viz.image_grid(ims, titles, ncols=3, size=3.6, suptitle=f"{ex.spec.sites.title}: the answer key drawn on each photo"))
        print("Labels:", ", ".join(ex.site_classes))
        print(f"Answer key: the boxes people drew for the dataset. Specialist for comparison: {ex.site_specialist}.")
        print("Source:", ex.spec.sites.source)
    else:
        ims = [p.load() for p in ex.plans]
        titles = [f"{p.id}: {p.title}\n{len(p.rooms())} rooms, {p.floor_area_m2} m², 1 px = {p.m_per_px * 100:.2f} cm" for p in ex.plans]
        viz.show(viz.image_grid(ims, titles, ncols=3, size=4.2, suptitle=lab.spec.plans.title))
        print("Answer key: every room's real area from the plans' vector annotations (see MP4). Specialist for comparison: SAM 3 asked for 'room' (MP4).")
        print(ex.plan_credit)


def describe(lab, photo: str, question: str):
    ph = lab.examples.photo(photo)
    prompt = C.fill("describe", lab.spec, question=question)
    r = lab.client.ask(ph.load(), prompt)
    viz.show_image(ph.load(), 320)
    print(f"Prompt: {prompt}\n")
    if r.error and not r.text:
        print("⚠️", r.error); return
    print("Reply:\n" + (r.text or "").strip())
    print(f"\n{_cost_line(r)}   (truth for this photo: {ph.truth})")


def json_lab(lab, photo: str, repeats: int = 3):
    ph = lab.examples.photo(photo)
    img = ph.load()
    classes = lab.examples.photo_classes
    prompt = C.fill("classify_basic", lab.spec)
    schema = C.classify_schema(classes)
    viz.show_image(img, 260)
    print(f"Prompt (both ways use exactly this text):\n{prompt}\n")
    results = {}
    for mode, use_schema in (("A · JSON asked for in the prompt, nothing enforces it", False), ("B · the same prompt, JSON schema enforced by the API", True)):
        print(f"\n{'=' * 100}\n{mode}\n{'=' * 100}")
        labels, valid = [], 0
        for run in range(repeats):
            r = lab.client.ask(img, prompt, schema=schema if use_schema else None, run=run)
            raw = (r.text or "").strip()
            fenced = raw.startswith("```")
            status = "valid JSON" if r.ok else f"⚠️ {r.error}"
            label = tasks.normalise_label(r.data.get("label"), classes) if r.ok and isinstance(r.data, dict) else "—"
            in_list = label in classes
            labels.append(label); valid += r.ok
            print(f"run {run + 1}: {status}{'; wrapped in ``` fences' if fenced else ''}{'' if in_list or label == '—' else '; label is NOT one of the categories'}"
                  f" → label: {label}   [{_cost_line(r)}]")
            print("   raw reply: " + _snippet(raw, 260))
        results[mode] = (valid, labels)
    print(f"\nThe schema used in B (the API rejects any reply that does not fit it):\n{json.dumps(schema, indent=2)}")
    a, b = results.values()
    print(f"\nSummary: A valid {a[0]}/{repeats}, labels {sorted(set(a[1]))};  B valid {b[0]}/{repeats}, labels {sorted(set(b[1]))};  truth: {ph.truth}")


# ----------------------------------------------------------------------------- Part 2
def _classify_report(lab, rows: List[tasks.ClsRow], title: str, show_mistakes: bool):
    ex = lab.examples
    classes = ex.photo_classes
    summ = tasks.cls_summary(rows, classes)
    table_rows = [(r.photo.file.rsplit(".", 1)[0], r.photo.truth, r.label or "(no label)", f"{r.confidence:.2f}" if r.confidence is not None else "",
                   "✓" if r.ok else ("⚠ " + r.reply.error[:28] if not r.valid else "✗")) for r in rows]
    print(viz.table(table_rows, ["photo", "truth", "model says", "conf.", "ok"], [22, 26, 26, 6, 32]))
    spec_acc = tasks.specialist_cls_accuracy(ex.photos)
    print(f"\n{title}: {summ['correct']}/{summ['n']} correct = {summ['accuracy']:.0f} %"
          + (f"; {summ['invalid']} reply(ies) unusable" if summ["invalid"] else "") + (f"; {summ['off_list']} label(s) not in the list" if summ["off_list"] else "")
          + f".  {ex.photo_specialist}: {spec_acc:.0f} %.")
    print(f"Time {summ['seconds']:.0f} s and {summ['tokens']} tokens for {summ['n']} photos "
          f"({'all precomputed' if all(r.reply.cached for r in rows) else 'some live'}).\n")
    print(viz.confusion_text([r.photo.truth for r in rows], [r.label for r in rows], classes))
    viz.show(viz.bar_compare({"Gemini": summ["accuracy"], ex.photo_specialist.split(" (")[0]: spec_acc}, title="accuracy on these photos", ylabel="% correct"))
    wrong = [r for r in rows if not r.ok][:8]
    if show_mistakes and wrong:
        viz.show(viz.image_grid([r.photo.load() for r in wrong], [f"truth: {r.photo.truth}\nmodel: {r.label or '?'}\n{r.reason[:60]}" for r in wrong],
                                ncols=4, size=2.6, suptitle="the mistakes (with the model's reason)", title_size=7))
    return summ


def classify(lab, prompt_choice: str, schema: bool, show_mistakes: bool = True):
    ex = lab.examples
    key = C.CLASSIFY_PROMPTS.get(prompt_choice, prompt_choice)
    text = C.fill(key, lab.spec)
    print(f"Prompt '{prompt_choice}'" + (" with the JSON schema enforced" if schema else " (JSON only asked for)") + f":\n{text}\n")
    rows = tasks.classify(lab.client, ex.photos, text, ex.photo_classes, schema, log=print)
    print()
    summ = _classify_report(lab, rows, f"Gemini, prompt '{prompt_choice}'{' + schema' if schema else ''}", show_mistakes)
    lab.results[("classify", prompt_choice, schema)] = summ


def classify_own(lab, prompt_text: str, schema: bool):
    ex = lab.examples
    text = prompt_text.replace("{classes}", "; ".join(ex.photo_classes)).replace("{intro}", lab.spec.photos.intro)
    if "json" not in text.lower() and not schema:
        print("Note: your prompt does not mention JSON and the schema is off: the reply will probably be prose and cannot be scored.\n")
    print(f"Your prompt{' + schema' if schema else ''}:\n{text}\n")
    rows = tasks.classify(lab.client, ex.photos, text, ex.photo_classes, schema, log=print)
    print()
    summ = _classify_report(lab, rows, "Gemini, your prompt", True)
    lab.results[("classify", "own", schema)] = summ


# ----------------------------------------------------------------------------- Part 3
def _det_image(site, pred, found, extra, colors, title_pred: str = ""):
    boxes = []
    for t, f in zip(site.truth, found):
        boxes.append(("", t["box"], viz.GREEN if f else viz.RED, 2))
    for (l, b), e in zip(pred, extra):
        boxes.append((l + (" ?" if e else ""), b, colors.get(l, viz.BLUE), 3))
    return viz.draw_boxes(site.load(), boxes)


def detect(lab, site: str, schema: bool):
    ex = lab.examples
    s = ex.site(site)
    prompt = C.fill("detect", lab.spec)
    rows = tasks.detect(lab.client, [s], prompt, ex.site_classes, schema)
    row = rows[0]
    sp = tasks.specialist_det_rows([s])[0]
    cols = lab.spec.sites.colors
    ims = [_det_image(s, row.pred, row.found, row.extra, cols), _det_image(s, sp.pred, sp.found, sp.extra, cols)]
    titles = [f"Gemini: {len(row.pred)} boxes; found {row.n_found}/{len(s.truth)}, missed {row.n_missed}, extra {row.n_extra}",
              f"{ex.site_specialist.split(' (')[0]}: {len(sp.pred)} boxes; found {sp.n_found}/{len(s.truth)}, missed {sp.n_missed}, extra {sp.n_extra}"]
    viz.show(viz.image_grid(ims, titles, ncols=2, size=5.2, suptitle="thin green = answer key box found, thin red = missed; thick = the model's boxes (? = extra)"))
    per = {}
    for t, f in zip(s.truth, row.found):
        d = per.setdefault(t["label"], [0, 0]); d[0] += 1; d[1] += int(f)
    print("Per label (found / in the answer key): " + ", ".join(f"{lab.spec.sites.display.get(l, l)} {v[1]}/{v[0]}" for l, v in per.items())
          + (f"; {row.skipped} unusable entries in the reply" if row.skipped else ""))
    if row.reply.error:
        print("⚠️", row.reply.error)
    print(f"{_cost_line(row.reply)}\nRaw reply: {_snippet(row.reply.text, 600)}")


def detect_all(lab, schema: bool):
    ex = lab.examples
    prompt = C.fill("detect", lab.spec)
    print(f"Prompt{' + schema' if schema else ''}:\n{prompt}\n")
    rows = tasks.detect(lab.client, ex.sites, prompt, ex.site_classes, schema, log=print)
    sp_rows = tasks.specialist_det_rows(ex.sites)
    g, sp = tasks.det_summary(rows, ex.site_classes), tasks.det_summary(sp_rows, ex.site_classes)
    print()
    table_rows = [(ex.spec.sites.display.get(c, c), g["per_class"][c]["truth"], g["per_class"][c]["found"], g["per_class"][c]["extra"],
                   sp["per_class"][c]["found"], sp["per_class"][c]["extra"]) for c in ex.site_classes]
    print(viz.table(table_rows, ["label", "in key", "Gemini found", "Gemini extra", "specialist found", "specialist extra"], [16, 7, 13, 13, 17, 17]))
    print(f"\nGemini: found {g['found']}/{g['truth']} (recall {g['recall']:.0f} %), {g['extra']} extra (precision {g['precision']:.0f} %)"
          + (f", {g['other_labels']} with a label outside the list" if g["other_labels"] else "") + (f", {g['invalid']} unusable replies" if g["invalid"] else "")
          + f"; {g['seconds']:.0f} s, {g['tokens']} tokens.")
    print(f"{ex.site_specialist}: found {sp['found']}/{sp['truth']} (recall {sp['recall']:.0f} %), {sp['extra']} extra (precision {sp['precision']:.0f} %) at confidence 0.25.")
    print("A box counts as found when it has the right label and overlaps the answer key's box by at least half (IoU ≥ 0.5).")
    viz.show(viz.bar_compare({"Gemini recall": g["recall"], "Gemini precision": g["precision"], "specialist recall": sp["recall"], "specialist precision": sp["precision"]},
                             title="all photos", ylabel="%", colors=["#457b9d", "#457b9d", "#f4a261", "#f4a261"], figsize=(6.5, 3)))
    lab.results[("detect", schema)] = g
    lab.results[("detect", "specialist")] = sp


def count(lab, site: str, schema: bool):
    ex = lab.examples
    s = ex.site(site)
    sp = lab.spec.sites
    prompt = C.fill("count", lab.spec)
    r = tasks.count(lab.client, s, prompt, sp.count_field, schema)
    counts = s.counts()
    truth_n = counts.get(sp.count_class, 0)
    truth_total = counts.get("person", sum(counts.values()))
    cols = sp.colors
    viz.show_image(viz.draw_boxes(s.load(), [(sp.display.get(t["label"], t["label"]), t["box"], cols.get(t["label"], viz.GREY), 3 if t["label"] == sp.count_class else 1)
                                             for t in s.truth]), 640)
    print(f"Question: {sp.count_question}\n")
    if r.ok and isinstance(r.data, dict):
        print(f"Model: {r.data.get(sp.count_field)} of {r.data.get('total')}   ({r.data.get('reason', '')})")
    else:
        print(f"Model reply could not be used: {r.error}\n   raw: {_snippet(r.text, 300)}")
    print(f"Answer key: {truth_n} {sp.display.get(sp.count_class, sp.count_class)} of {truth_total} in total.")
    det_rows = tasks.detect(lab.client, [s], C.fill("detect", lab.spec), ex.site_classes, True)
    n_boxes = sum(1 for l, _ in det_rows[0].pred if l == sp.count_class)
    print(f"Counting the model's own boxes from the detection step instead: {n_boxes} {sp.display.get(sp.count_class, sp.count_class)}.")
    print(_cost_line(r))


# ----------------------------------------------------------------------------- Part 4
def _seg_image(plan, rows: List[tasks.RoomRow], mode: str):
    img = plan.load()
    if mode == "llm":
        polys = [(f"{r.label} {r.m2:.1f}", r.poly, PALETTE[i % len(PALETTE)]) for i, r in enumerate(rows) if r.poly]
        out = viz.draw_polygons(img, polys) if polys else img
        boxes = [(f"{r.label} {r.m2:.1f} (box)", r.box, PALETTE[i % len(PALETTE)], 3) for i, r in enumerate(rows) if not r.poly]
        return viz.draw_boxes(out, boxes) if boxes else out
    out = img
    for i, r in enumerate(rows):
        if r.mask is not None:
            out = viz.mask_overlay(out, r.mask, PALETTE[i % len(PALETTE)], 0.5)
    return viz.draw_boxes(out, [(f"{r.label} {r.m2:.1f}", r.box, PALETTE[i % len(PALETTE)], 2) for i, r in enumerate(rows)])


def _specialist_seg(plan):
    rooms = plan.specialist.get("rooms", [])
    errs = [abs(r["sam_m2"] - r["truth_m2"]) / r["truth_m2"] * 100 for r in rooms if r["sam_m2"] and r["truth_m2"]]
    return sum(1 for r in rooms if r["sam_m2"] is not None), len(rooms), (float(np.median(errs)) if errs else None)


def segment(lab, plan: str, mode_name: str, schema: bool):
    ex = lab.examples
    p = ex.plan(plan)
    mode = MODES.get(mode_name, mode_name)
    prompt = C.PROMPTS["rooms_masks" if mode == "llm" else "rooms_boxes"]
    rows, r, skipped = tasks.segment_rooms(lab.client, p, prompt, mode, schema, sam=lab.sam)
    viz.show_image(_seg_image(p, rows, mode), 820)
    print(f"Prompt{' + schema' if schema else ''}: {prompt}\n")
    if r.error:
        print("⚠️", r.error, "\n   raw:", _snippet(r.text, 300), "\n")
    table_rows = [(rw.label, rw.how, f"{rw.m2:.1f}", (rw.truth["label"] or rw.truth["type"]) if rw.truth else "—", f"{rw.truth_m2:.1f}" if rw.truth else "—",
                   f"{rw.err_pct:+.0f} %" if rw.truth else "no room of the drawing fits", rw.note) for rw in rows]
    print(viz.table(table_rows, ["model label", "area from", "model m²", "drawing room", "drawing m²", "error", "note"], [12, 9, 9, 13, 10, 24, 46]))
    summ = tasks.seg_summary(rows, p)
    sf, st, se = _specialist_seg(p)
    print(f"\n{mode_name}: {summ['rooms_found']}/{summ['rooms_truth']} rooms of the drawing found, "
          + (f"median error {summ['median_err']:.0f} % (worst {summ['max_err']:.0f} %); " if summ["median_err"] is not None else "")
          + f"model total {summ['total_model_m2']:.1f} m² vs floor area {summ['floor_area_m2']:.1f} m²."
          + (f" {skipped} unusable entries in the reply." if skipped else ""))
    print(f"MP4's specialist (SAM 3 asked for 'room'): {sf}/{st} rooms found" + (f", median error {se:.0f} %." if se is not None else "."))
    print(_cost_line(r))
    lab.results[("segment", p.id, mode)] = summ


def segment_compare(lab, plan: str):
    ex = lab.examples
    p = ex.plan(plan)
    rows_llm, r1, _ = tasks.segment_rooms(lab.client, p, C.PROMPTS["rooms_masks"], "llm", True, sam=lab.sam)
    rows_sam, r2, _ = tasks.segment_rooms(lab.client, p, C.PROMPTS["rooms_boxes"], "llm+sam", True, sam=lab.sam)
    by_truth = {}
    for rows, key in ((rows_llm, "llm"), (rows_sam, "sam")):
        for rw in rows:
            if rw.truth:
                by_truth.setdefault(id(rw.truth), {})[key] = rw.m2
    spec_rooms = {(r["label"], r["truth_m2"]): r["sam_m2"] for r in p.specialist.get("rooms", [])}
    table_rows = []
    errs = {"llm": [], "sam": [], "spec": []}
    for room in p.rooms(indoor_only=True):
        got = by_truth.get(id(room), {})
        sp = spec_rooms.get((room["label"], room["area_m2"]))
        cells = [room["label"] or room["type"], f"{room['area_m2']:.1f}"]
        for key, val in (("llm", got.get("llm")), ("sam", got.get("sam")), ("spec", sp)):
            if val is None:
                cells.append("—")
            else:
                e = (val - room["area_m2"]) / room["area_m2"] * 100; errs[key].append(abs(e)); cells.append(f"{val:.1f} ({e:+.0f} %)")
        table_rows.append(cells)
    print(f"Plan {p.id}: {p.title}\n")
    print(viz.table(table_rows, ["room", "drawing m²", "LLM polygons", "LLM boxes + SAM 3", "MP4: SAM 3 'room'"], [12, 10, 18, 18, 18]))
    med = lambda k: f"median error {np.median(errs[k]):.0f} % on {len(errs[k])} rooms" if errs[k] else "no room matched"
    print(f"\nLLM polygons: {med('llm')}.   LLM boxes + SAM 3: {med('sam')}.   MP4's SAM 3 by phrase: {med('spec')}.")
    viz.show(viz.image_grid([_seg_image(p, rows_llm, "llm"), _seg_image(p, rows_sam, "llm+sam")], ["LLM polygons", "LLM boxes + SAM 3"], ncols=2, size=5.2))


# ----------------------------------------------------------------------------- wrap-up
def summary(lab, chat: bool = False):
    ex = lab.examples
    cls_rows = tasks.classify(lab.client, ex.photos, C.fill("classify_basic", lab.spec), ex.photo_classes, True)
    cls = tasks.cls_summary(cls_rows, ex.photo_classes)
    det_rows = tasks.detect(lab.client, ex.sites, C.fill("detect", lab.spec), ex.site_classes, True)
    det = tasks.det_summary(det_rows, ex.site_classes)
    sp_det = tasks.det_summary(tasks.specialist_det_rows(ex.sites), ex.site_classes)
    if chat:
        # the rooms were done in the chat window: the student's own pasted results, not the API again; no SAM 3 route
        seg_spec = [se for p in ex.plans for se in [_specialist_seg(p)[2]] if se is not None]
        mine = lab.results.get(("chat", "rooms"), {})
        med = [v["median_err"] for v in mine.values() if v.get("median_err") is not None]
        rooms_cell = (f"{np.mean(med):.0f} % on {len(mine)} plan(s) (your pasted replies)" if med else "do Step 4a first (your pasted replies)")
        m = lambda v: f"{np.mean(v):.0f} %" if v else "—"   # noqa: E731
        rows = [
            ("classification (API, batch)", f"accuracy on {cls['n']} photos", f"{cls['accuracy']:.0f} %", f"{tasks.specialist_cls_accuracy(ex.photos):.0f} %", f"{cls['seconds']:.0f} s, {cls['tokens']} tokens"),
            ("detection (API, batch)", f"recall / precision, {det['truth']} boxes", f"{det['recall']:.0f} % / {det['precision']:.0f} %", f"{sp_det['recall']:.0f} % / {sp_det['precision']:.0f} %", f"{det['seconds']:.0f} s, {det['tokens']} tokens"),
            ("rooms (chat, polygons)", "median area error per plan", rooms_cell, m(seg_spec) + " (SAM 3 'room')", "your time in the chat window"),
        ]
        print(viz.table(rows, ["task", "measure", "generalist", "specialist from MP2/MP3/MP4", "cost"], [28, 30, 44, 28, 32]))
        print("\nSpecialists: " + ex.photo_specialist + "; " + ex.site_specialist + "; MP4's SAM 3 asked for 'room'.")
        print("The generalist needed no training data and no training; the specialists needed hundreds of labelled photos each. "
              "The batch rows are the API; the single examples of Parts 1, 3a, 3c and 4 were the chat window, once each.")
        return
    seg_llm, seg_sam, seg_spec, secs, toks = [], [], [], 0.0, 0
    for p in ex.plans:
        rows, r1, _ = tasks.segment_rooms(lab.client, p, C.PROMPTS["rooms_masks"], "llm", True, sam=lab.sam)
        s1 = tasks.seg_summary(rows, p)
        rows, r2, _ = tasks.segment_rooms(lab.client, p, C.PROMPTS["rooms_boxes"], "llm+sam", True, sam=lab.sam)
        s2 = tasks.seg_summary(rows, p)
        if s1["median_err"] is not None: seg_llm.append(s1["median_err"])
        if s2["median_err"] is not None: seg_sam.append(s2["median_err"])
        _, _, se = _specialist_seg(p)
        if se is not None: seg_spec.append(se)
        secs += r1.seconds + r2.seconds; toks += r1.tokens_in + r1.tokens_out + r1.tokens_thought + r2.tokens_in + r2.tokens_out + r2.tokens_thought
    m = lambda v: f"{np.mean(v):.0f} %" if v else "—"
    rows = [
        ("classification", f"accuracy on {cls['n']} photos", f"{cls['accuracy']:.0f} %", f"{tasks.specialist_cls_accuracy(ex.photos):.0f} %", f"{cls['seconds']:.0f} s, {cls['tokens']} tokens"),
        ("detection", f"recall / precision, {det['truth']} boxes", f"{det['recall']:.0f} % / {det['precision']:.0f} %", f"{sp_det['recall']:.0f} % / {sp_det['precision']:.0f} %", f"{det['seconds']:.0f} s, {det['tokens']} tokens"),
        ("rooms: LLM polygons", "median area error per plan", m(seg_llm), m(seg_spec) + " (SAM 3 'room')", f"{secs:.0f} s, {toks} tokens (both modes)"),
        ("rooms: LLM boxes + SAM 3", "median area error per plan", m(seg_sam), "", ""),
    ]
    print(viz.table(rows, ["task", "measure", "Gemini (one prompt each)", "specialist from MP2/MP3/MP4", "Gemini time and tokens"], [24, 30, 24, 28, 30]))
    print("\nSpecialists: " + ex.photo_specialist + "; " + ex.site_specialist + "; MP4's SAM 3 asked for 'room'.")
    print("The generalist needed no training data and no training; the specialists needed hundreds of labelled photos each. "
          "Times are per request to a remote service; the specialists answer in milliseconds on a GPU.")


def report_summary(lab):
    ex = lab.examples
    c = lab.client
    print("Numbers for your report: every step prints its own; copy the ones you used.")
    print(f"Model: {c.model}. Live requests in this session: {c.calls} ({c.seconds:.0f} s, {c.tokens} tokens). "
          f"Precomputed answers on disk: {len(c.cache) if c.cache else 0}.")
    print(f"Examples: {ex.spec.photos.title} ({len(ex.photos)} photos, {ex.spec.photos.source}); "
          f"{ex.spec.sites.title} ({len(ex.sites)} photos, {ex.spec.sites.source}); {lab.spec.plans.title} ({', '.join(p.id for p in ex.plans)}).")
    print(ex.plan_credit)
