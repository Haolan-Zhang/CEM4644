"""Generates the MP5 Workshop and Homework notebooks (+ report templates) from one template."""
import json
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
CHAT_INTRO = f"""**Two routes, nothing done twice.** For the *single-example* steps (a photo described, a photo classified, boxes on one site photo, a count, the rooms of one plan) you use a plain chat window: **hokie.ai** ({CHAT_URL}, Virginia Tech's free access to GPT models, sign in with your VT account). You download the example image, paste the prompt the notebook gives you, attach the image, and paste the reply back into the notebook, which draws and scores it against the answer key and next to the earlier labs' specialist models. The *batch* steps (every photo scored at once, prompts compared) use the Gemini API, whose answers are precomputed. No GPU is needed in this notebook.

"""

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


def form(title, body, notes=(), params=()):
    src = f'#@title {title} {{ display-mode: "form" }}\n'
    src += "".join(f"#@markdown {n}\n" for n in notes)
    src += "".join(p + "\n" for p in params)
    src += body.rstrip() + "\n"
    c = new_code_cell(src)
    c.metadata["cellView"] = "form"
    return c


def md(text):
    return new_markdown_cell(text.strip("\n"))


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
    photos = [p.label for p in ex.photos]
    sites = [s.label for s in ex.sites]
    plans = [p.label for p in ex.plans]
    questions, cells = [], []
    own_template = C.PROMPTS["classify_basic"].replace("{intro}", "{intro}").replace("{{", "{").replace("}}", "}")

    cells.append(md(f"""
# 🤖 CEM4644 · MP5 — One model for everything? A vision-language model for classification, detection and take-off
## {v['label']}: *{spec.title}*

**No coding needed.** Each grey box below is one *step*: click the ▶ (play) button at its left, wait until it finishes, look at the result, then answer the report question that follows. Run the steps **from top to bottom**.

In MP2, MP3 and MP4 you trained or used one **specialist** model per task: a classifier, a detector, a segmentation model. This lab gives all three tasks to one **generalist**: a large multimodal language model (Gemini), which has never seen our photos and is steered only by the words you send it. The lab is built around two questions: *can a prompt replace a trained model?* and *how do you get an answer a program can read?*

**What you will do (about {v['minutes']} minutes)**
1. Talk to the model about a photo, and get the answer as JSON in two ways: by asking nicely, and by enforcing a schema.
2. Classify the MP2 photos with three prompts and compare with the MP2 model.
3. Detect and count on the MP3 photos, compare with the MP3 model.
4. {"Measure rooms from the model's own polygons: one plan through the chat, then every plan at once through the API." if chat else "Measure rooms on the MP4 plans: the model's own polygons, or its boxes handed to SAM 3."}
5. Your own image and your own words in a small app.

{CHAT_INTRO if chat else ""}**Before you start**
- **Gemini API key (free).** Open https://aistudio.google.com/apikey, sign in with your Google account, create a key. In Colab, click the **key icon** in the left bar (*Secrets*), add a secret named `GEMINI_API_KEY` with the key as its value, and switch on *Notebook access*. Without a key the precomputed answers of the built-in examples still work; your own prompts and images do not.
{"" if chat else "- **GPU (optional):** *Runtime → Change runtime type → T4 GPU* is only needed for Step 4b (SAM 3). Everything else runs on a remote service and needs no GPU."}
- Never paste your key into a cell you might share: use the secret.
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
        notes=(["Click ▶ and wait for the green ✅ line. This downloads the examples with their precomputed answers.",
                "Leave *api_key* empty to use the Colab secret GEMINI_API_KEY (recommended)."] if chat else
               ["Click ▶ and wait for the green ✅ line. This downloads the examples with their precomputed answers, and (if *load_sam* is ticked) SAM 3 for Step 4b (about 3 GB).",
                "Leave *api_key* empty to use the Colab secret GEMINI_API_KEY (recommended). Untick *load_sam* if you have no GPU and want to skip Step 4b's live part."]),
        params=['api_key = "" #@param {type:"string"}', f'model = "{C.DEFAULT_MODEL}" #@param {jlist(C.MODELS)}'] + ([] if chat else ['load_sam = True #@param {type:"boolean"}']),
    ))

    # ------------------------------------------------------------------ Part 1
    cells.append(md("""
## Part 1 · Talk to the model

A **vision-language model** reads an image and text together and answers in text. It has no fixed list of classes and no output layer for boxes: whatever structure you want back, you must **ask for it in words**, and the reply is a piece of text that a program then has to read. That is the whole difference from the specialists: the prompt is the program.

Two things to watch in every step: the **reply itself** (is it what you asked for, is it right?) and its **cost**: seconds per request and **tokens** (the units the service bills; an image costs a few hundred tokens, the model's private *thinking* costs more).
"""))
    cells.append(form("▶ Step 1a · The examples and their answer keys", "lab.show_examples(which)",
                      notes=["The same kind of material as in MP2, MP3 and MP4, with the answer keys and with what the earlier labs' specialist models said about them."],
                      params=[f'which = "photos" #@param {jlist(["photos", "site photos", "plans"])}']))
    if chat:
        cells.append(form("▶ Step 1b · Ask the chat anything about a photo", "lab.chat_describe(photo, question)",
                          notes=["Free text in, free text out, through the chat window: download the photo, paste the prompt with the photo attached, paste the reply back."],
                          params=[f'photo = "{photos[0]}" #@param {jlist(photos)}', f'question = "{C.DESCRIBE_QUESTIONS[0]}" #@param {jlist(C.DESCRIBE_QUESTIONS)} {{allow-input: true}}']))
        cells.append(form("▶ Step 1c · JSON, way A: ask the chat nicely", "lab.chat_classify(photo)",
                          notes=["The classification prompt asks for JSON; nothing enforces it. Paste the reply back: is it valid JSON, is the label one of the categories, is it right? "
                                 "Do it two or three times (new chat each time) and watch whether the format and the label stay the same."],
                          params=[f'photo = "{photos[0]}" #@param {jlist(photos)}']))
        questions.append((1, "From Step 1c: across your tries, how many of the chat replies were valid JSON, was the label always one of the categories, and did it stay the same? "
                             "In Step 2a, run the *basic* prompt with the schema off and on: what does the schema change in the replies, and what does it not change (the label can still be wrong)? "
                             "Why does a program that has to read the reply (to fill a table, to count, to draw a box) need the schema rather than a polite request?"))
    else:
        cells.append(form("▶ Step 1b · Ask anything about a photo", "lab.describe(photo, question)",
                          notes=["Free text in, free text out. The three questions below are precomputed for the first photo; any other photo or question runs live (needs your key)."],
                          params=[f'photo = "{photos[0]}" #@param {jlist(photos)}', f'question = "{C.DESCRIBE_QUESTIONS[0]}" #@param {jlist(C.DESCRIBE_QUESTIONS)} {{allow-input: true}}']))
        cells.append(form("▶ Step 1c · Getting JSON, two ways", "lab.json_lab(photo, repeats)",
                          notes=["**A:** the prompt asks for JSON, nothing enforces it. **B:** the same prompt with a **JSON schema** handed to the API, which rejects any reply that does not fit. "
                                 "Each way is run several times: watch whether the format and the label stay the same."],
                          params=[f'photo = "{photos[0]}" #@param {jlist(photos)}', 'repeats = 3 #@param {type:"slider", min:1, max:3, step:1}']))
        questions.append((1, "From Step 1c: how many of the plain replies (A) were valid JSON, and did the label stay the same across the runs? What did the schema (B) change, and what did it not change? "
                             "Why does a program that has to read the reply (to fill a table, to count, to draw a box) need B rather than A?"))
    cells.append(q(*questions[-1]))

    # ------------------------------------------------------------------ Part 2
    cells.append(md(f"""
## Part 2 · Classification by prompt

The MP2 task again: {spec.photos.title.lower()}, {len(ex.photo_classes)} classes, and a model that was never trained on them. Three prompts of increasing length are prepared: the class names only, the names with a one-line description each, and the descriptions plus decision rules. The reply is scored against the answer key exactly as in MP2, and the MP2 course model's accuracy on the same photos is shown next to it.
"""))
    cells.append(form("▶ Step 2a · Classify all the photos", "lab.classify(prompt, schema, show_mistakes)",
                      notes=["All three prompts are precomputed with the schema on; *basic* is also precomputed with the schema off. Other combinations run live (a few minutes on the free tier)."],
                      params=[f'prompt = "basic" #@param {jlist(C.CLASSIFY_PROMPTS)}', 'schema = True #@param {type:"boolean"}', 'show_mistakes = True #@param {type:"boolean"}']))
    cells.append(form("▶ Step 2b · Your own prompt", "lab.classify_own(prompt_text, schema)",
                      notes=["Edit the text (keep `{classes}` where the list of categories should go; `{intro}` is the first sentence). Runs live on every photo: needs your key; the free tier allows only a few requests per minute, so this takes one to three minutes."],
                      params=[f'prompt_text = {json.dumps(own_template, ensure_ascii=False)} #@param {{type:"string"}}', 'schema = True #@param {type:"boolean"}']))
    questions.append((2, v["q_classify"]))
    cells.append(q(*questions[-1]))
    questions.append((3, "From Step 2b: what did you change in the prompt and what accuracy did you get? If it went up, what is the risk of tuning a prompt on the same photos you score it on (think of MP2's training / test split)?"))
    cells.append(q(*questions[-1]))

    # ------------------------------------------------------------------ Part 3
    cells.append(md(f"""
## Part 3 · Detection and counting by prompt

The MP3 task: {spec.sites.title[0].lower() + spec.sites.title[1:]}. The prompt asks for a list of boxes as `[ymin, xmin, ymax, xmax]` on a 0–1000 grid (the convention this model was trained with), the notebook converts them to pixels and scores them like MP3 did: a box is *found* when its label is right and it overlaps the answer key's box by at least half. The MP3 YOLO model's boxes on the same photos are shown next to Gemini's.
"""))
    if chat:
        cells.append(form("▶ Step 3a · Boxes on one photo, from the chat", "lab.chat_detect(site)",
                          notes=["Paste the reply back and the notebook draws the boxes on the photo and scores them against the answer key, next to the MP3 model's numbers on the same photo. "
                                 "If the boxes land in the wrong place, the chat used another coordinate convention: change *box order* or *numbers are* and score again."],
                          params=[f'site = "{sites[0]}" #@param {jlist(sites)}']))
    else:
        cells.append(form("▶ Step 3a · Boxes on one photo", "lab.detect(site, schema)",
                          params=[f'site = "{sites[0]}" #@param {jlist(sites)}', 'schema = True #@param {type:"boolean"}']))
    cells.append(form("▶ Step 3b · All the photos, scored", "lab.detect_all(schema)",
                      params=['schema = True #@param {type:"boolean"}']))
    if chat:
        cells.append(form("▶ Step 3c · Just ask the chat for the number", "lab.chat_count(site)",
                          notes=["Instead of boxes, the chat is asked for a count. Compared with the answer key and with the boxes it gave you in Step 3a."],
                          params=[f'site = "{sites[0]}" #@param {jlist(sites)}']))
    else:
        cells.append(form("▶ Step 3c · Just ask for the number", "lab.count(site, schema)",
                          notes=["Instead of boxes, the model is asked for a count. Compared with the answer key and with counting the model's own boxes from Step 3a."],
                          params=[f'site = "{sites[0]}" #@param {jlist(sites)}', 'schema = True #@param {type:"boolean"}']))
    questions.append((4, v["q_detect"] + (" Also: on the photo you gave the chat in Step 3a, how did its boxes compare with the MP3 model's on the same photo, and what coordinate convention did the chat use?" if chat else "")))
    cells.append(q(*questions[-1]))
    questions.append((5, ("From Step 3c on two photos: the chat's count, the number of boxes it gave you in Step 3a, and the answer key. " if chat else
                          "From Step 3c on two photos: the model's count, the count of its own boxes and the answer key. ")
                      + "When they disagree, which one is wrong and how would you know on a site where there is no answer key? Which of the two ways of counting would you trust on a site camera, and why?"))
    cells.append(q(*questions[-1]))

    # ------------------------------------------------------------------ Part 4
    cells.append(md(f"""
## Part 4 · Rooms on a floor plan

The MP4 task on {spec.plans.description}. {"The prompt asks for each room's outline as a polygon (a list of points on the 0–1000 grid) and its label; the notebook converts the polygon to pixels and to m² with the plan's scale, and checks every room against the drawing's answer key, next to MP4's result (SAM 3 asked for *room* by phrase). First one plan by hand in the chat window, then all of them at once through the API." if chat else "Two ways to get square metres out of a language model:"}
{"" if chat else """
- **LLM only:** the prompt asks for each room's outline as a polygon (a list of points on the 0–1000 grid) and its label; the notebook converts the polygon to pixels and to m² with the plan's scale.
- **LLM boxes + SAM 3:** the prompt asks only for a box per room; each box is handed to SAM 3 exactly as your own boxes were in MP4 (*empty room* + box, holes filled, walls removed). The language model does the *finding and naming*, the segmentation model does the *pixels*.

Both are scored against the drawing's answer key, and MP4's result (SAM 3 asked for *room* by phrase) is shown for comparison."""}
"""))
    if chat:
        cells.append(form("▶ Step 4a · Rooms from the chat's polygons (one plan)", "lab.chat_rooms(plan)",
                          notes=["Paste the reply back: the notebook converts the polygons, measures every room in m² and checks each against the drawing. "
                                 "Long lists of coordinates are where chat replies break: look at what you get, and try a second plan."],
                          params=[f'plan = "{plans[0]}" #@param {jlist(plans)}']))
        cells.append(form(f"▶ Step 4b · All {len(plans)} plans at once (the API)", "lab.segment_all(schema)",
                          notes=[f"The same prompt sent to the API for all {len(plans)} plans, which would take you {len(plans)} rounds of copying by hand. One picture and one line per plan: "
                                 "rooms found, median error against the drawing, the model's total against the floor area, and MP4's SAM 3 for comparison.",
                                 "Untick *schema* to see what the same prompt returns when nothing enforces the structure (precomputed)."],
                          params=['schema = True #@param {type:"boolean"}']))
    else:
        cells.append(form("▶ Step 4a · The model's own polygons", "lab.segment(plan, mode, schema)",
                          notes=["Precomputed for every plan with the schema on and off. Try both: without the schema this model tends to skip the polygons and only give boxes, "
                                 "and long lists of coordinates are where the plain JSON breaks."],
                          params=[f'plan = "{plans[0]}" #@param {jlist(plans)}', f'mode = "LLM only (polygons)" #@param {jlist(["LLM only (polygons)"])}', 'schema = True #@param {type:"boolean"}']))
        cells.append(form("▶ Step 4b · The model's boxes, SAM 3's pixels", "lab.segment(plan, mode, schema)",
                          notes=["The model's boxes are precomputed; SAM 3 runs live on them (GPU: seconds; CPU: about a minute per plan). Without SAM 3 loaded, the box areas are used."],
                          params=[f'plan = "{plans[0]}" #@param {jlist(plans)}', f'mode = "LLM boxes + SAM 3" #@param {jlist(["LLM boxes + SAM 3"])}', 'schema = True #@param {type:"boolean"}']))
        cells.append(form("▶ Step 4c · Room by room, three ways", "lab.segment_compare(plan)",
                          params=[f'plan = "{plans[0]}" #@param {jlist(plans)}']))
    questions.append((6, v["q_segment_chat"] if chat else v["q_segment"]))
    cells.append(q(*questions[-1]))

    # ------------------------------------------------------------------ Part 5
    cells.append(md("""
## Part 5 · Your image, your words

A small app, opened from a link: pick a built-in image or upload your own, choose a task (it fills in a starting prompt), edit the words, switch the schema on or off, and read the raw reply next to what the notebook draws from it. Needs your key: everything here is live.
"""))
    cells.append(form("▶ Step 5 · Prompt lab", "lab.prompt_app()",
                      notes=["This cell prints a **link**: open it in a new tab (or on your phone). The app stays alive while this notebook is running; if no link appears, run the cell again."]))
    questions.append((7, f"Run at least {v['own_experiments']} experiments of your own in Step 5 (a photo from a site or from the internet, a plan, a changed prompt, the schema on and off). "
                         "For each: the image, the prompt, the raw reply, and whether it was right. What kind of request broke the model, and how did it break (wrong answer, invented objects, unreadable reply)?"))
    cells.append(q(*questions[-1]))

    # ------------------------------------------------------------------ wrap-up
    cells.append(md("## Wrap-up · Generalist or specialist?"))
    cells.append(form("▶ Step 6 · All tasks side by side", "lab.summary(chat=True)" if chat else "lab.summary()",
                      notes=["One table: the generalist with one prompt per task against the three specialists from MP2, MP3 and MP4, with time and tokens."]))
    questions.append((8, "From Step 6: for each task, would you use the generalist, the specialist, or both together" + (" (as in Step 4b)" if not chat else "") + "? Argue with the numbers you got and with what each needs: labelled data, training, a GPU, a network connection, money per request, and someone who checks. "
                         "What does structured output guarantee about a reply, and what does it not guarantee?"))
    cells.append(q(*questions[-1]))
    cells.append(form("▶ Numbers for your report", "lab.report_summary()"))
    cells.append(md("### Credits\n"
                    f"- Photos: {spec.photos.source}.\n- Site photos: {spec.sites.source}.\n- Floor plans: CubiCasa5K (CC BY-NC-SA 4.0), prepared for MP4 (plans {', '.join(spec.plans.ids)}).\n"
                    + ("- Models: the chat model behind hokie.ai (Virginia Tech) for the single examples; Gemini (Google) through the Gemini API, free tier, for the batch steps.\n" if chat else
                       "- Model: Gemini (Google) through the Gemini API, free tier; SAM 3 (Meta, SAM License) from the MP4 folder.\n")
                    + "- Lab code: https://github.com/Haolan-Zhang/CEM4644 (folder `mp5_llm_vision`).\n"))

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
    chat = "--chat" in sys.argv or "--all" in sys.argv
    for variant in (args or VARIANTS):
        if "--all" in sys.argv or not chat:
            build(variant)
        if chat:
            build(variant, chat=True)
