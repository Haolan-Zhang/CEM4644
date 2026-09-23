"""Generates the MP5 Workshop and Homework notebooks (+ report templates) from one template."""
import json
import re
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from aec_llm import config as C  # noqa: E402
from aec_llm.data import Examples  # noqa: E402

GITHUB_URL = "https://github.com/Haolan-Zhang/CEM4644.git"
CHAT_URL = "https://hokie.ai.vt.edu/"
CHAT_INTRO_OLD = f"""**Two routes, nothing done twice.** For the *single-example* steps (a photo described, a photo classified, boxes on one site photo, a count, the rooms of one plan) you use a plain chat window: **hokie.ai** ({CHAT_URL}, Virginia Tech's free access to GPT models, sign in with your VT account). You download the example image, paste the prompt the notebook gives you, attach the image, and paste the reply back into the notebook, which draws and scores it against the answer key and next to the earlier labs' specialist models. The *batch* steps (every photo scored at once, prompts compared) use the Gemini API, whose answers are precomputed. No GPU is needed in this notebook.

"""

CHAT_INTRO = (f"- **hokie.ai:** the single-example steps use a chat window, {CHAT_URL} (VT login): download the picture, "
              "paste the prompt with the picture attached, paste the reply back into the cell.\n")

VARIANTS = {
    "workshop": dict(
        file="MP5_Workshop_LLM_Vision.ipynb", label="Workshop (in class)", minutes=90, dataset="workshop", own_experiments=2,
        q_classify=("From Step 2a: the accuracy of the three prompts (basic / with descriptions / with descriptions and rules) and of the MP2 model on the same 14 photos. "
                    "Which classes does Gemini confuse (use the confusion table), and what did the descriptions and the rules change?"),
        q_detect=("From Step 3b: Gemini's recall and precision against the MP3 YOLO model's. Which label is hardest for Gemini (helmet, NO helmet, vest, NO vest, person) "
                  "and why might that be? Paste one overlay from Step 3a and explain the extras (thick boxes marked '?')."),
        q_segment_chat=("From Step 4a on two plans: copy the per-room tables from your pasted replies. Which rooms did the chat model outline well and which not (missing polygons, merged rooms, shapes in the wrong place)? "
                        "Then from Step 4b: the table for all three plans from the API, with and without the schema. How do the two routes compare on the plan you did by hand, how does either compare with MP4's SAM 3 by phrase, "
                        "and what does the batch table tell you that one plan could not?"),
        q_segment=("From Step 4c on one plan: copy the per-room table. Which way is closer to the drawing, the model's own polygons or its boxes handed to SAM 3, "
                   "and on which rooms do they differ most? How does this compare with drawing the boxes yourself in MP4?"),
    ),
    "homework": dict(
        file="MP5_Homework_LLM_Vision.ipynb", label="Homework (individual)", minutes=120, dataset="homework", own_experiments=3,
        q_classify=("From Step 2a: the accuracy of the three prompts and of the MP2 model on the 20 style images. The images are computer-generated and the MP2 model was "
                    "trained on the same kind of images: is that a fair comparison? Which styles does Gemini confuse with each other, and does the rules prompt help?"),
        q_detect=("From Step 3b: Gemini's recall and precision against the MP3 YOLO model's on the machinery photos. Which machine is hardest and why? "
                  "From Step 3c: does the model's count of excavators agree with its own boxes and with the answer key?"),
        q_segment_chat=("From Step 4a on plan 8138 and on plan 11615 (scanned drawings with furniture and dimension strings): copy the per-room tables from your pasted replies. "
                        "What does the scan's clutter do to the chat model's polygons? Then from Step 4b: the table for all four plans from the API, including the two-storey sheet 5018. "
                        "Where is the API's batch result better or worse than your chat replies, and how does either compare with MP4's SAM 3 by phrase?"),
        q_segment=("From Step 4c on plan 8138 and on plan 11615 (scanned drawings with furniture and dimension strings, unlike the clean drawings of the workshop): "
                   "copy the per-room tables. Which way holds up better on a scan, and what does the scan's clutter do to the polygons? Compare with the MP4 numbers for the same plans."),
    ),
}


KEEP_TEXT = True          # --fresh-text: rebuild every text from this file instead of keeping the notebook's
EXISTING = {}             # text already in the notebook being rebuilt: first line -> markdown source, title -> notes


def load_existing(path: Path):
    """The notebook is where the text is edited (in Colab, then saved to GitHub). A rebuild keeps every markdown
    cell and every cell's #@markdown notes that it finds there, matched by the cell's first line / title."""
    EXISTING.clear()
    if not (KEEP_TEXT and path.exists()):
        return
    for c in nbformat.read(str(path), as_version=4).cells:
        first = c.source.split("\n", 1)[0].strip()
        if c.cell_type == "markdown":
            EXISTING[first] = c.source
        elif first.startswith("#@title"):
            EXISTING[first] = [l[len("#@markdown "):] for l in c.source.split("\n") if l.startswith("#@markdown ")]


def form(title, body, notes=(), params=()):
    head = f'#@title {title} {{ display-mode: "form" }}'
    if head in EXISTING:
        notes = EXISTING[head]
    src = head + "\n"
    src += "".join(f"#@markdown {n}\n" for n in notes)
    src += "".join(p + "\n" for p in params)
    src += body.rstrip() + "\n"
    c = new_code_cell(src)
    c.metadata["cellView"] = "form"
    return c


def md(text):
    text = text.strip("\n")
    first = text.split("\n", 1)[0].strip()
    return new_markdown_cell(EXISTING.get(first, text))


def q(n, text):
    return md(f"> ### 📝 Report question {n}\n> {text}")


def jlist(items):
    return json.dumps(list(items), ensure_ascii=False)


def build(variant, chat=False):
    """chat=True: the single-example steps go through a chat window (hokie.ai) with paste-back cells; the batch and
    schema steps stay on the API. Written as separate *_chat notebooks; the originals are untouched."""
    v = VARIANTS[variant]
    spec = C.SPECS[v["dataset"]]
    ex = Examples(REPO, spec)
    load_existing(REPO / (v["file"].replace(".ipynb", "_chat.ipynb") if chat else v["file"]))
    photos = [p.label for p in ex.photos]
    sites = [s.label for s in ex.sites]
    plans = [p.label for p in ex.plans]
    files = [p.file for p in ex.photos]; site_files = [s.file for s in ex.sites]; plan_ids = [p.id for p in ex.plans]
    # the chat notebooks get a picture-picking cell before each chat task, so their later steps move one letter down
    CHAT_STEPS = {"1b": "1c", "1c": "1d", "3a": "3b", "3b": "3c", "3c": "3d", "4a": "4b", "4b": "4c"}
    renum = (lambda s: re.sub(r"Step (1b|1c|3a|3b|3c|4a|4b)\b", lambda m: "Step " + CHAT_STEPS[m.group(1)], s)) if chat else (lambda s: s)
    questions, cells = [], []
    own_template = C.PROMPTS["classify_basic"].replace("{intro}", "{intro}").replace("{{", "{").replace("}}", "}")

    cells.append(md(f"""
# 🤖 CEM4644 · MP5 — One model for everything?
## {v['label']}: *{spec.title}*

**No coding needed.** Each grey box is one step: click ▶, wait, read the result, answer the report question. Run from top to bottom.

One **generalist** model (Gemini) does the MP2, MP3 and MP4 tasks from words alone, and every answer is scored against the same answer keys as before. About {v['minutes']} minutes.

**Before you start**
- **Gemini API key (free):** https://aistudio.google.com/apikey → in Colab, the key icon (*Secrets*) → name `GEMINI_API_KEY`, *Notebook access* on. Never paste the key into a cell.
{CHAT_INTRO if chat else "- **GPU (optional):** *Runtime → Change runtime type → T4 GPU*, only for Step 4b."}
"""))
    # the chat notebooks never load SAM 3, so they do not need the MP4 folder next to this one
    folders = ["mp5_llm_vision"] if chat else ["mp5_llm_vision", "mp4_segmentation"]
    folders_note = "only this lab folder is" if chat else "only these two lab folders are"
    cells.append(form(
        "▶ Step 0 · Run me first (1–3 minutes)",
        f"""import importlib, os, shutil, subprocess, sys
REPO, FOLDER, PKG = "CEM4644", "mp5_llm_vision", "aec_llm"
FOLDERS = {json.dumps(folders)}  # {folders_note} downloaded, not the whole course repository

def _git(*args):
    return subprocess.run(["git", "-C", REPO, *args], capture_output=True, text=True).returncode == 0

if os.path.isdir(REPO):                      # a copy is already here: pull the newest course code over it
    if not (_git("sparse-checkout", "set", *FOLDERS)     # also trims a full copy left by an earlier run
            and _git("fetch", "-q", "--depth", "1", "origin", "master")
            and _git("reset", "-q", "--hard", "FETCH_HEAD") and _git("clean", "-qfd")):
        shutil.rmtree(REPO, ignore_errors=True)          # broken copy: start again from scratch
if not os.path.isdir(REPO):
    subprocess.run(["git", "clone", "-q", "--depth", "1", "--filter=blob:none", "--sparse", "{GITHUB_URL}", REPO], check=True)
    subprocess.run(["git", "-C", REPO, "sparse-checkout", "set", *FOLDERS], check=True)
for _m in [m for m in list(sys.modules) if m.split(".")[0] in (PKG, "aec_seg")]:
    del sys.modules[_m]                      # Python caches imported code: drop it, or this cell keeps the old version
importlib.invalidate_caches()
sys.path.insert(0, os.path.abspath(os.path.join(REPO, FOLDER)))
from aec_llm import lab
lab.setup(dataset="{spec.key}", api_key=api_key, model=model, load_sam={"False" if chat else "load_sam"})""",
        notes=["Click ▶ and wait for the ✅ line. Leave *api_key* empty to use the Colab secret."
               + ("" if chat else " Untick *load_sam* if you have no GPU.")],
        params=['api_key = "" #@param {type:"string"}', f'model = "{C.DEFAULT_MODEL}" #@param {jlist(C.MODELS)}'] + ([] if chat else ['load_sam = True #@param {type:"boolean"}']),
    ))

    # ------------------------------------------------------------------ Part 1
    cells.append(md("""
## Part 1 · Talk to the model

A vision-language model reads an image and text and answers in text. Whatever structure you want back, you ask for it in words. Watch the reply and its cost (seconds, tokens).
"""))
    cells.append(form("▶ Step 1a · The examples and their answer keys", "lab.show_examples(which)",
                      params=[f'which = "photos" #@param {jlist(["photos", "site photos", "plans"])}']))
    if chat:
        cells.append(form("▶ Step 1b · Pick a photo and look at it", "lab.show_image(photo)",
                          notes=["The picture and nothing else: save it or screenshot it for the chat window."],
                          params=[f'photo = "{files[0]}" #@param {jlist(files)}']))
        cells.append(form("▶ Step 1c · Ask the chat anything about a photo", "lab.chat_describe(photo, question)",
                          params=[f'photo = "{photos[0]}" #@param {jlist(photos)}', f'question = "{C.DESCRIBE_QUESTIONS[0]}" #@param {jlist(C.DESCRIBE_QUESTIONS)} {{allow-input: true}}']))
        cells.append(form("▶ Step 1d · Ask the chat for the category", "lab.chat_classify(photo)",
                          notes=["Paste the reply back. Do it two or three times (a new chat each time)."],
                          params=[f'photo = "{photos[0]}" #@param {jlist(photos)}']))
        questions.append((1, "From Step 1c: across your tries, how many of the chat replies were valid JSON, was the label always one of the categories, and did it stay the same? "
                             "In Step 2a, run the *basic* prompt with the schema off and on: what does the schema change in the replies, and what does it not change (the label can still be wrong)? "
                             "Why does a program that has to read the reply (to fill a table, to count, to draw a box) need the schema rather than a polite request?"))
    else:
        cells.append(form("▶ Step 1b · Ask anything about a photo", "lab.describe(photo, question)",
                          params=[f'photo = "{photos[0]}" #@param {jlist(photos)}', f'question = "{C.DESCRIBE_QUESTIONS[0]}" #@param {jlist(C.DESCRIBE_QUESTIONS)} {{allow-input: true}}']))
        cells.append(form("▶ Step 1c · Getting JSON, two ways", "lab.json_lab(photo, repeats)",
                          notes=["**A:** JSON asked for in the prompt. **B:** a JSON schema enforced by the API. Each run several times."],
                          params=[f'photo = "{photos[0]}" #@param {jlist(photos)}', 'repeats = 3 #@param {type:"slider", min:1, max:3, step:1}']))
        questions.append((1, "From Step 1c: how many of the plain replies (A) were valid JSON, and did the label stay the same across the runs? What did the schema (B) change, and what did it not change? "
                             "Why does a program that has to read the reply (to fill a table, to count, to draw a box) need B rather than A?"))
    cells.append(q(*questions[-1]))

    # ------------------------------------------------------------------ Part 2
    cells.append(md(f"""
## Part 2 · Classification by prompt

The MP2 photos ({len(ex.photo_classes)} classes), three prompts: the class names, the names with descriptions, the descriptions with rules. Scored against the answer key and the MP2 model.
"""))
    cells.append(form("▶ Step 2a · Classify all the photos", "lab.classify(prompt, schema, show_mistakes)",
                      params=[f'prompt = "basic" #@param {jlist(C.CLASSIFY_PROMPTS)}', 'schema = True #@param {type:"boolean"}', 'show_mistakes = True #@param {type:"boolean"}']))
    cells.append(form("▶ Step 2b · Your own prompt", "lab.classify_own(prompt_text, schema)",
                      notes=["Keep `{classes}` and `{intro}` in the text. Runs live on every photo (one to three minutes)."],
                      params=[f'prompt_text = {json.dumps(own_template, ensure_ascii=False)} #@param {{type:"string"}}', 'schema = True #@param {type:"boolean"}']))
    questions.append((2, v["q_classify"]))
    cells.append(q(*questions[-1]))
    questions.append((3, "From Step 2b: what did you change in the prompt and what accuracy did you get? If it went up, what is the risk of tuning a prompt on the same photos you score it on (think of MP2's training / test split)?"))
    cells.append(q(*questions[-1]))

    # ------------------------------------------------------------------ Part 3
    cells.append(md(f"""
## Part 3 · Detection and counting by prompt

The MP3 photos. Boxes come back as `[ymin, xmin, ymax, xmax]` on a 0–1000 grid and are scored like MP3: right label and an overlap of at least half (IoU ≥ 0.5).
"""))
    if chat:
        cells.append(form("▶ Step 3a · Pick a site photo", "lab.show_image(site)",
                          params=[f'site = "{site_files[0]}" #@param {jlist(site_files)}']))
        cells.append(form("▶ Step 3b · Boxes on one photo, from the chat", "lab.chat_detect(site)",
                          notes=["If the boxes land in the wrong place, change *box order* or *numbers are* and score again."],
                          params=[f'site = "{sites[0]}" #@param {jlist(sites)}']))
    else:
        cells.append(form("▶ Step 3a · Boxes on one photo", "lab.detect(site, schema)",
                          params=[f'site = "{sites[0]}" #@param {jlist(sites)}', 'schema = True #@param {type:"boolean"}']))
    cells.append(form(renum("▶ Step 3b · All the photos, scored"), "lab.detect_all(schema)",
                      params=['schema = True #@param {type:"boolean"}']))
    if chat:
        cells.append(form("▶ Step 3d · Just ask the chat for the number", "lab.chat_count(site)",
                          params=[f'site = "{sites[0]}" #@param {jlist(sites)}']))
    else:
        cells.append(form("▶ Step 3c · Just ask for the number", "lab.count(site, schema)",
                          params=[f'site = "{sites[0]}" #@param {jlist(sites)}', 'schema = True #@param {type:"boolean"}']))
    questions.append((4, v["q_detect"] + (" Also: on the photo you gave the chat in Step 3a, how did its boxes compare with the MP3 model's on the same photo, and what coordinate convention did the chat use?" if chat else "")))
    cells.append(q(*questions[-1]))
    questions.append((5, ("From Step 3c on two photos: the chat's count, the number of boxes it gave you in Step 3a, and the answer key. " if chat else
                          "From Step 3c on two photos: the model's count, the count of its own boxes and the answer key. ")
                      + "When they disagree, which one is wrong and how would you know on a site where there is no answer key? Which of the two ways of counting would you trust on a site camera, and why?"))
    cells.append(q(*questions[-1]))

    # ------------------------------------------------------------------ Part 4
    unit = "square feet" if spec.plans.units == "ft" else "square metres"
    cells.append(md(f"""
## Part 4 · Rooms on a floor plan

The MP4 plans, in {unit}. {"One plan from the chat's polygons, then every plan at once through the API." if chat else "Two routes: the model's own polygons, or its boxes handed to SAM 3."} Every room is scored against the drawing, next to MP4's SAM 3 by phrase.
"""))
    if chat:
        cells.append(form("▶ Step 4a · Pick a plan", "lab.show_image(plan)",
                          params=[f'plan = "{plan_ids[0]}" #@param {jlist(plan_ids)}']))
        cells.append(form("▶ Step 4b · Rooms from the chat's polygons (one plan)", "lab.chat_rooms(plan)",
                          notes=["If the shapes sit in the wrong place, change *box order* or *numbers are* and score again."],
                          params=[f'plan = "{plans[0]}" #@param {jlist(plans)}']))
        cells.append(form(f"▶ Step 4c · All {len(plans)} plans at once (the API)", "lab.segment_all(schema)",
                          notes=["Untick *schema* to see the same prompt without the enforced structure."],
                          params=['schema = True #@param {type:"boolean"}']))
    else:
        cells.append(form("▶ Step 4a · The model's own polygons", "lab.segment(plan, mode, schema)",
                          params=[f'plan = "{plans[0]}" #@param {jlist(plans)}', f'mode = "LLM only (polygons)" #@param {jlist(["LLM only (polygons)"])}', 'schema = True #@param {type:"boolean"}']))
        cells.append(form("▶ Step 4b · The model's boxes, SAM 3's pixels", "lab.segment(plan, mode, schema)",
                          params=[f'plan = "{plans[0]}" #@param {jlist(plans)}', f'mode = "LLM boxes + SAM 3" #@param {jlist(["LLM boxes + SAM 3"])}', 'schema = True #@param {type:"boolean"}']))
        cells.append(form("▶ Step 4c · Room by room, three ways", "lab.segment_compare(plan)",
                          params=[f'plan = "{plans[0]}" #@param {jlist(plans)}']))
    questions.append((6, v["q_segment_chat"] if chat else v["q_segment"]))
    cells.append(q(*questions[-1]))

    # ------------------------------------------------------------------ Part 5
    cells.append(md("""
## Part 5 · Your image, your words

A small app for your own image and your own prompt. Needs your key.
"""))
    cells.append(form("▶ Step 5 · Prompt lab", "lab.prompt_app()",
                      notes=["Open the printed link in a new tab."]))
    questions.append((7, f"Run at least {v['own_experiments']} experiments of your own in Step 5 (a photo from a site or from the internet, a plan, a changed prompt, the schema on and off). "
                         "For each: the image, the prompt, the raw reply, and whether it was right. What kind of request broke the model, and how did it break (wrong answer, invented objects, unreadable reply)?"))
    cells.append(q(*questions[-1]))

    # ------------------------------------------------------------------ wrap-up
    cells.append(md("## Wrap-up · Generalist or specialist?"))
    cells.append(form("▶ Step 6 · All tasks side by side", "lab.summary(chat=True)" if chat else "lab.summary()"))
    questions.append((8, "From Step 6: for each task, would you use the generalist, the specialist, or both together" + (" (as in Step 4b)" if not chat else "") + "? Argue with the numbers you got and with what each needs: labelled data, training, a GPU, a network connection, money per request, and someone who checks. "
                         "What does structured output guarantee about a reply, and what does it not guarantee?"))
    cells.append(q(*questions[-1]))
    cells.append(form("▶ Numbers for your report", "lab.report_summary()"))
    cells.append(md("### Credits\n"
                    f"- Photos: {spec.photos.source}.\n- Site photos: {spec.sites.source}.\n- Floor plans: {spec.plans.source} (plans {', '.join(spec.plans.ids)}).\n"
                    + ("- Models: the chat model behind hokie.ai (Virginia Tech) for the single examples; Gemini (Google) through the Gemini API, free tier, for the batch steps.\n" if chat else
                       "- Model: Gemini (Google) through the Gemini API, free tier; SAM 3 (Meta, SAM License) from the MP4 folder.\n")
                    + "- Lab code: https://github.com/Haolan-Zhang/CEM4644 (folder `mp5_llm_vision`).\n"))

    questions = []                                             # from the cells: an edited question is kept
    for c in cells:
        if c.cell_type == "markdown" and c.source.startswith("> ### 📝 Report question "):
            c.source = renum(c.source)
            head, _, body = c.source.partition("\n")
            questions.append((int(head.rsplit(" ", 1)[1]), body[2:].strip() if body.startswith("> ") else body.strip()))
    nb = new_notebook(cells=cells)
    nb.metadata.update({"colab": {"provenance": [], "gpuType": "T4", "toc_visible": True}, "accelerator": "GPU",
                        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                        "language_info": {"name": "python"}})
    out = REPO / (v["file"].replace(".ipynb", "_chat.ipynb") if chat else v["file"])
    nbformat.write(nb, str(out))
    print("wrote", out)
    lines = [f"# CEM4644 · MP5 report — {v['label']}", "", "Name: ______________________    Date: ____________", "",
             f"Notebook: `{v['file']}` — examples: *{spec.title}*.", "",
             "Answer every question in a few sentences. Paste screenshots where the question asks for overlays, tables or raw replies. "
             "Numbers must come from **your** run of the notebook.", ""]
    for n, text in questions:
        lines += [f"## Question {n}", "", text, "", "*Your answer:*", "", "", ""]
    p = REPO / "docs" / f"MP5_{variant.capitalize()}_Report_Template{'_chat' if chat else ''}.md"
    p.parent.mkdir(exist_ok=True)
    p.write_text("\n".join(lines))
    print("wrote", p)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    KEEP_TEXT = "--fresh-text" not in sys.argv
    chat = "--chat" in sys.argv or "--all" in sys.argv
    for variant in (args or VARIANTS):
        if "--all" in sys.argv or not chat:
            build(variant)
        if chat:
            build(variant, chat=True)
