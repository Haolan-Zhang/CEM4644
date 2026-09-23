"""The chat-window route: the student downloads an example image, pastes the prompt into a chat model (hokie.ai at
Virginia Tech, or any chat), attaches the image, and pastes the reply back here. The notebook then parses, draws and
scores the reply exactly as it does for the API model, so the two routes can be compared."""
import json
import re
from typing import List, Optional, Sequence

from . import ui
from . import config as C
from . import tasks, viz
from .client import Reply, parse_json

CHAT_URL = "https://hokie.ai.vt.edu/"
RAW_BASE = "https://raw.githubusercontent.com/Haolan-Zhang/CEM4644/master/mp5_llm_vision/"
ORDERS = ["ymin, xmin, ymax, xmax (as the prompt asks)", "x1, y1, x2, y2"]
SCALES = ["detect automatically", "0-1000 grid (as the prompt asks)", "pixels", "fractions 0-1"]
BOX_KEYS = ("box_2d", "box", "bbox", "bounding_box", "box2d", "coordinates")
LABEL_KEYS = ("label", "name", "class", "object", "category")
POLY_KEYS = ("mask", "polygon", "points", "outline", "segmentation")


# ----------------------------------------------------------------------------- making sense of a pasted reply
def _scale_of(values: Sequence[float], size, scale: str) -> str:
    if scale.startswith("0-1000"):
        return "grid"
    if scale.startswith("pixel"):
        return "pixels"
    if scale.startswith("fraction"):
        return "fractions"
    m = max(values) if values else 0
    if m <= 1.0:
        return "fractions"
    if m > 1000 or (m > max(size) and max(size) <= 1000 and False):
        return "pixels"
    return "grid"


def _to_grid_box(box, size, order: str, scale: str) -> Optional[List[float]]:
    """Any reasonable box -> [ymin, xmin, ymax, xmax] on the 0-1000 grid the scoring code expects."""
    try:
        if isinstance(box, dict):
            keys = [k.lower() for k in box]
            if all(k in keys for k in ("x1", "y1", "x2", "y2")):
                v = [float(box[k]) for k in ("x1", "y1", "x2", "y2")]; order = "x1"
            elif all(k in keys for k in ("xmin", "ymin", "xmax", "ymax")):
                v = [float(box[k]) for k in ("xmin", "ymin", "xmax", "ymax")]; order = "x1"
            elif all(k in keys for k in ("x", "y", "width", "height")):
                v = [float(box["x"]), float(box["y"]), float(box["x"]) + float(box["width"]), float(box["y"]) + float(box["height"])]; order = "x1"
            else:
                return None
        else:
            v = [float(x) for x in box][:4]
        if len(v) != 4:
            return None
    except Exception:
        return None
    s = _scale_of(v, size, scale)
    W, H = size
    if order.startswith("x1"):
        x1, y1, x2, y2 = v
    else:
        y1, x1, y2, x2 = v
    if s == "pixels":
        x1, x2, y1, y2 = x1 / W * 1000, x2 / W * 1000, y1 / H * 1000, y2 / H * 1000
    elif s == "fractions":
        x1, x2, y1, y2 = x1 * 1000, x2 * 1000, y1 * 1000, y2 * 1000
    return [y1, x1, y2, x2]


def _to_grid_poly(pts, size, scale: str) -> Optional[List[List[float]]]:
    try:
        pairs = []
        for p in pts:
            if isinstance(p, dict):
                pairs.append([float(p.get("x")), float(p.get("y"))])
            else:
                pairs.append([float(p[0]), float(p[1])])
    except Exception:
        return None
    if len(pairs) < 3:
        return None
    flat = [v for p in pairs for v in p]
    s = _scale_of(flat, size, scale)
    W, H = size
    if s == "pixels":
        return [[x / W * 1000, y / H * 1000] for x, y in pairs]
    if s == "fractions":
        return [[x * 1000, y * 1000] for x, y in pairs]
    return pairs


def _items(data):
    """The list of objects in a reply, wherever the model put it."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for k in ("objects", "detections", "boxes", "rooms", "items", "results", "masks"):
            if isinstance(data.get(k), list):
                return data[k]
        if any(k in data for k in BOX_KEYS):
            return [data]
    return []


def _get(d: dict, keys):
    for k in keys:
        for kk in d:
            if kk.lower() == k:
                return d[kk]
    return None


def normalise_reply(text: str, size, order: str = ORDERS[0], scale: str = SCALES[0]) -> Reply:
    """The pasted text as a Reply whose data follows the API convention (labels, box_2d on the 0-1000 grid, mask polygons)."""
    data, err = parse_json(text)
    if data is None:
        return Reply(text=text, data=None, error=err, model="chat")
    out = []
    items = _items(data)
    if not items and isinstance(data, dict):
        return Reply(text=text, data=data, error="", model="chat")        # a classification-style object
    for it in items:
        if not isinstance(it, dict):
            continue
        entry = {"label": str(_get(it, LABEL_KEYS) or "")}
        b = _get(it, BOX_KEYS)
        if b is not None:
            g = _to_grid_box(b, size, order, scale)
            if g is not None:
                entry["box_2d"] = g
        p = _get(it, POLY_KEYS)
        if p is not None:
            g = _to_grid_poly(p, size, scale)
            if g is not None:
                entry["mask"] = g
        if "box_2d" not in entry and "mask" in entry:                     # a polygon without a box: its bounding box
            ys = [q[1] for q in entry["mask"]]; xs = [q[0] for q in entry["mask"]]
            entry["box_2d"] = [min(ys), min(xs), max(ys), max(xs)]
        out.append(entry)
    return Reply(text=text, data=out, error="", model="chat")


# ----------------------------------------------------------------------------- the paste-back widget
def _raw_url(lab, kind: str, file: str) -> str:
    return RAW_BASE + f"data/{lab.spec.key}/{kind}/{file}"


def paste_step(lab, image, path, kind: str, prompt: str, on_score, what: str, extra_controls=None):
    """Shows the image with a download button, the prompt to copy, a box to paste the reply into and a Score button."""
    import ipywidgets as w
    from IPython.display import display
    viz.show_image(image, 560)
    url = _raw_url(lab, kind, path.name)
    steps = w.HTML(
        f"<b>1.</b> Get the image: click <i>Download</i> (or right-click the picture above, <i>Save image as</i>; or open "
        f"<a href='{url}' target='_blank'>this link</a>). "
        f"<b>2.</b> Open <a href='{CHAT_URL}' target='_blank'>{CHAT_URL}</a> (sign in with your VT account), start a new chat, attach the image, "
        f"paste the prompt below, send. <b>3.</b> Copy the whole reply and paste it into the second box. <b>4.</b> Click <i>Score</i>.")
    dl = w.Button(description="Download the image", icon="download")
    dl_out = w.Output()

    def _dl(_):
        with dl_out:
            dl_out.clear_output()
            try:
                from google.colab import files
                files.download(str(path))
            except Exception:
                print(f"Not in Colab: save the picture from the browser, or download it here: {url}")
    dl.on_click(_dl)
    prompt_box = w.Textarea(value=ui.student_prompt(prompt), layout=w.Layout(width="100%", height="120px"), description="prompt", style={"description_width": "60px"})
    reply_box = w.Textarea(placeholder="paste the chat model's reply here (all of it; fences and extra words are fine)",
                           layout=w.Layout(width="100%", height="160px"), description="reply", style={"description_width": "60px"})
    btn = w.Button(description=f"Score the reply ({what})", button_style="primary")
    out = w.Output()

    def _go(_):
        with out:
            out.clear_output(wait=True)
            text = reply_box.value.strip()
            if not text:
                print("Paste the reply first."); return
            try:
                on_score(text, *(c.value for c in (extra_controls or [])))
            except Exception as e:  # noqa: BLE001
                print(f"Could not use that reply: {e}")
    btn.on_click(_go)
    controls = [w.HBox([dl, dl_out]), prompt_box, reply_box]
    if extra_controls:
        controls.append(w.HBox(extra_controls))
    controls += [btn, out]
    display(w.VBox([steps] + controls))


# ----------------------------------------------------------------------------- the steps
def chat_describe(lab, photo: str, question: str):
    ph = lab.examples.photo(photo)
    prompt = C.fill("describe", lab.spec, question=question)

    def score(text):
        print("The chat model wrote:\n" + text.strip())
        print(f"\nTruth for this photo (its folder in the dataset): {ph.truth}. Nothing to score here: read the answer against the picture.")
    paste_step(lab, ph.load(), ph.path, "photos", prompt, score, "free text")


def chat_classify(lab, photo: str):
    ph = lab.examples.photo(photo)
    classes = lab.examples.photo_classes
    prompt = C.fill("classify_basic", lab.spec)

    def score(text):
        r = normalise_reply(text, ph.load().size)
        fenced = text.strip().startswith("```")
        if not r.ok or not isinstance(r.data, dict):
            print(f"⚠️ The reply is not a usable JSON object ({r.error or 'a list, not an object'}). A program reading it would stop here."
                  + (" It is wrapped in ``` fences." if fenced else "")); return
        label = tasks.normalise_label(_get(r.data, LABEL_KEYS), classes)
        print(("valid JSON" + (", wrapped in ``` fences (a naive parser would choke)" if fenced else "")) + f"; label: {label}"
              + ("" if label in classes else " (NOT one of the categories)") + f"; confidence: {r.data.get('confidence')}; reason: {r.data.get('reason', '')}")
        print(f"Truth: {ph.truth} → {'✓ correct' if label == ph.truth else '✗ wrong'}.")
        if ph.specialist:
            print(f"{lab.examples.photo_specialist.split(' (')[0]} on this photo: {ph.specialist.get('label')} (confidence {ph.specialist.get('confidence')}).")
        lab.results.setdefault(("chat", "classify"), {})[ph.file] = {"label": label, "ok": label == ph.truth, "valid": True}
    paste_step(lab, ph.load(), ph.path, "photos", prompt, score, "classification")


def chat_detect(lab, site: str):
    import ipywidgets as w
    ex = lab.examples
    s = ex.site(site)
    prompt = C.fill("detect", lab.spec)
    order = w.Dropdown(options=ORDERS, value=ORDERS[0], description="box order", style={"description_width": "80px"}, layout=w.Layout(width="360px"))
    scale = w.Dropdown(options=SCALES, value=SCALES[0], description="numbers are", style={"description_width": "90px"}, layout=w.Layout(width="320px"))

    def score(text, order_v, scale_v):
        r = normalise_reply(text, s.size, order_v, scale_v)
        if not r.ok or not isinstance(r.data, list):
            print(f"⚠️ The reply could not be read as a list of boxes ({r.error or 'no list found'}). Raw: {text[:300]}"); return
        pred, skipped = tasks.boxes_from_reply(r, s.size, ex.site_classes)
        found, extra = tasks.score_det(s, pred)
        cols = lab.spec.sites.colors
        boxes = [("", t["box"], viz.GREEN if f else viz.RED, 2) for t, f in zip(s.truth, found)]
        boxes += [(l + (" ?" if e else ""), b, cols.get(l, viz.BLUE), 3) for (l, b), e in zip(pred, extra)]
        viz.show_image(viz.draw_boxes(s.load(), boxes), 760)
        print(f"Chat model: {len(pred)} boxes; found {sum(found)}/{len(s.truth)}, missed {len(found) - sum(found)}, extra {sum(extra)}"
              + (f"; {skipped} entries unusable" if skipped else "") + ". Thin green = answer key found, thin red = missed, thick = the model's boxes (? = extra).")
        if pred and sum(found) == 0:
            print("Nothing matched: if the boxes sit in the wrong place, try the other box order or scale above and score again.")
        sp = tasks.specialist_det_rows([s])[0]
        print(f"{ex.site_specialist.split(' (')[0]} on this photo: found {sp.n_found}/{len(s.truth)}, extra {sp.n_extra}.")
        lab.results.setdefault(("chat", "detect"), {})[s.file] = {"found": int(sum(found)), "truth": len(s.truth), "extra": int(sum(extra)), "boxes": len(pred)}
    paste_step(lab, s.load(), s.path, "sites", prompt, score, "boxes", extra_controls=[order, scale])


def chat_count(lab, site: str):
    ex = lab.examples
    s = ex.site(site)
    sp = lab.spec.sites
    prompt = C.fill("count", lab.spec)

    def score(text):
        data, err = parse_json(text)
        counts = s.counts()
        truth_n = counts.get(sp.count_class, 0); truth_total = counts.get("person", sum(counts.values()))
        if isinstance(data, dict):
            n = _get(data, (sp.count_field, "count", "number")); tot = _get(data, ("total", "workers", "machines"))
            print(f"Chat model: {n} of {tot}   ({data.get('reason', '')})")
        else:
            nums = re.findall(r"\d+", text)
            print(f"⚠️ Not JSON ({err}); numbers found in the text: {nums[:6]}")
        print(f"Answer key: {truth_n} {sp.display.get(sp.count_class, sp.count_class)} of {truth_total} in total.")
        det = lab.results.get(("chat", "detect"), {}).get(s.file)
        if det:
            print(f"Your Step 3a boxes on this photo gave {det['boxes']} objects in total; counting from boxes and asking for a number are two different questions to the model.")
    paste_step(lab, s.load(), s.path, "sites", prompt, score, "count")


def chat_rooms(lab, plan: str):
    """Rooms on one plan from the chat model's own polygons (the box's area when a polygon is missing), each checked against the drawing."""
    import ipywidgets as w
    from . import ui
    ex = lab.examples
    p = ex.plan(plan)
    prompt = C.fill("rooms_masks", lab.spec)
    order = w.Dropdown(options=ORDERS, value=ORDERS[0], description="box order", style={"description_width": "80px"}, layout=w.Layout(width="360px"))
    scale = w.Dropdown(options=SCALES, value=SCALES[0], description="numbers are", style={"description_width": "90px"}, layout=w.Layout(width="320px"))

    def score(text, order_v, scale_v):
        r = normalise_reply(text, p.size, order_v, scale_v)
        if not r.ok or not isinstance(r.data, list) or not r.data:
            print(f"⚠️ The reply could not be read as a list of rooms ({r.error or 'no list found'}). Raw: {text[:300]}"); return
        rows, skipped = tasks.measure_rooms(r, p, "llm", sam=None)
        viz.show_image(ui._seg_image(p, rows, "llm"), 820)
        table_rows = [(rw.label, rw.how, f"{rw.area:.1f}", (rw.truth["label"] or rw.truth["type"]) if rw.truth else "—", f"{rw.truth_area:.1f}" if rw.truth else "—",
                       f"{rw.err_pct:+.0f} %" if rw.truth else "no room of the drawing fits", rw.note) for rw in rows]
        print(viz.table(table_rows, ["model label", "area from", f"model {p.unit_label}", "drawing room", f"drawing {p.unit_label}", "error", "note"], [12, 9, 9, 13, 10, 24, 46]))
        summ = tasks.seg_summary(rows, p)
        n_poly = sum(1 for rw in rows if rw.poly)
        print(f"\nChat model: {summ['rooms_found']}/{summ['rooms_truth']} rooms of the drawing found ({n_poly} of {len(rows)} entries had a usable polygon)"
              + (f", median error {summ['median_err']:.0f} % (worst {summ['max_err']:.0f} %)" if summ["median_err"] is not None else "")
              + f"; model total {summ['total_model']:.1f} vs floor area {summ['floor_area']:.1f} {p.unit_label}." + (f" {skipped} unusable entries." if skipped else ""))
        if rows and summ["rooms_found"] == 0:
            print("Nothing matched: if the shapes sit in the wrong place, try the other box order or scale above and score again.")
        sf, st, se = ui._specialist_seg(p)
        print(f"SAM 3 asked for 'room' by phrase (previous class): {sf}/{st} rooms found" + (f", median error {se:.0f} %." if se is not None else "."))
        lab.results.setdefault(("chat", "rooms"), {})[p.id] = summ
    paste_step(lab, p.load(), p.path, "plans", prompt, score, "rooms", extra_controls=[order, scale])
