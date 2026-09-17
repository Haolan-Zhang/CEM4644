# MP5 · Instructor guide

## 1. What the students get

Two Colab notebooks built from one template. Students never write code: each cell is a form with a ▶ button.
Every built-in example × prompt the notebooks use is **precomputed** (`data/cache`), so the notebooks are instant and
work without a key; a student's own prompt or image runs live and needs their free Gemini key (Colab secret
`GEMINI_API_KEY`, set-up in the notebook's header). Step 4b (SAM 3) is the only step that wants a GPU.

| | Workshop | Homework |
|---|---|---|
| Classification | 14 façade-defect photos, 7 classes (MP2 test split) | 20 computer-generated façade-style images, 10 classes (MP2) |
| Detection | 6 site-safety photos, 52 boxes (MP3 test split) | 6 machinery photos, 9 boxes (MP3) |
| Plans | 1293, 2536, 2090 (clean renderings, MP4 set A) | 8138, 10715, 11615 (scanned drawings, MP4 set B) |
| Own experiments | 2 | 3 |
| Time | about 90 min | about 2 h |

The specialists the generalist is compared with are the earlier labs' own course models, run once by
`build/prepare_data.py`: MP2's ConvNeXt V2 femto (93 % on the 14 workshop photos, 100 % on the 20 style images), MP3's
YOLO11n at confidence 0.25, and MP4's SAM 3 asked for *room* by phrase (whole instances, so its areas include the
furniture holes: 19–21 % median error).

## 1b. The chat-window variants (`*_chat.ipynb`)

The `_chat` notebooks send the single-example steps through **hokie.ai** (https://hokie.ai.vt.edu/, VT login), so
students meet the ordinary chat interface and see that a general chat model does these tasks too: Step 1b (describe),
1c (JSON asked nicely; 1d is the API with the schema for the contrast), 3a (boxes on one photo), 3c (the count) and
4a (rooms on one plan; the API's version becomes 4a′). Each paste-back cell shows the image with a download button,
the prompt to copy, a box for the reply and a *Score* button; the reply is parsed with the same tolerant parser as the
API replies, and drawn and scored the same way, next to the API model's numbers for the same image. Two dropdowns
handle the chat model's habits: box order (`ymin, xmin, ymax, xmax` as asked, or `x1, y1, x2, y2`) and whether the
numbers are on the 0–1000 grid, in pixels or fractions (detected automatically when the range makes it obvious).
Everything else in those notebooks (batch scoring, prompt comparison, Step 4b, the prompt lab) is the API path.

What to expect from a chat model: prose around the JSON and ``` fences (the parser copes, a naive one would not);
boxes given in pixels of a resized image or in the other order (hence the dropdowns); rougher boxes than the API's
grounded output; polygons that are coarse or missing. Those are the lesson, not bugs. The notebook cannot call hokie.ai
itself, so nothing there is precomputed; each student's replies are their own. The chat replies are not stored.

## 2. The Gemini API as the students meet it

- **Model:** `gemini-3.5-flash-lite` by default; Step 0 also offers `gemini-3.5-flash`, `gemini-3.8-flash` and `gemini-3.6-flash`. Thinking level *low*.
- **Boxes:** `box_2d = [ymin, xmin, ymax, xmax]` on a 0–1000 grid (the convention the model was trained with);
  the notebook converts to pixels. **Polygons:** `mask = [[x, y], ...]` on the same grid.
- **Structured output:** `response_json_schema` on the request. With it the API refuses to return anything that does not
  fit the schema (enum labels, four integers in a box); without it the model may wrap the JSON in ``` fences, break a
  long list of coordinates, or invent a label. Step 1c shows both on the same prompt, three runs each.
- **Free tier (measured on 2026-09-15 with the course key):** every `gemini-3.x-flash` model is capped at **20
  requests per day** (and 5 per minute on 3.5-flash); `gemini-3.5-flash-lite` allowed 30+ requests in three minutes with
  no daily cap in sight, at 1–5 s per request. That is why the lite model is the default: a student's live steps
  (own prompt over 14 photos, six detections, a few app experiments) fit in a session. The stronger models are there
  for a handful of comparison requests. The client waits 5 → 90 s on per-minute limits and 503s and says so; a
  daily-cap error is reported at once with the model's name; a reply with an error is never cached.

## 3. What the model actually does (measured)

All numbers below: `gemini-3.5-flash-lite`, thinking *low*, the replies stored in `data/cache` (what the students see
when they keep the defaults). Live runs vary by a photo or two.

**Workshop**

| step | Gemini | specialist | notes |
|---|---|---|---|
| Classification, *basic* prompt + schema | **93 %** (13/14; spalling → algae once) | MP2 model 93 % | 13 s, 17.5k tokens for 14 photos |
| *basic* prompt, JSON only asked for | 71 % (all replies parse, four wrong) | | the same prompt without the schema drifts: major → minor crack, peeling → spalling, stain → spalling |
| *with descriptions* + schema | 86 % | | the descriptions do not help this model on these photos |
| *with descriptions and rules* + schema | 86 % | | |
| JSON lab, photo 1, three runs | A: 3/3 valid but every reply wrapped in ``` fences; B: 3/3 valid, no fences; the label stayed the same in all six | | the fences are what a naive `json.loads` chokes on |
| Detection, all six photos, schema | recall **90 %** (47/52), precision 82 %, 10 extra | MP3 YOLO: recall 87 %, precision 90 % | per label: person 17/19, helmet 12/12, NO helmet 1/3 (+3 extra), vest 7/7, NO vest 10/11 |
| Detection, JSON only asked for | recall 88 %, precision 92 % | | all six replies parse; NO helmet 0/3 |
| Counting (asked for the number) | site 1: 1 NO helmet of 4 workers (key 3 of 4; its own boxes say 3); site 3: 1 of 3 (key 0; boxes 0); the other four photos right | | the direct count and the box count disagree on two photos: the lesson of Step 3c |
| Rooms, LLM polygons + schema | 1293: 5/5 rooms, median error 0 %; 2536: 8/9, 16 %; 2090: 5/5, 16 % | MP4 SAM 3 'room': 5/5, 8/9, 4/5; 19–21 % | without the schema the model drops the polygons (1293) or breaks the JSON on the long coordinate lists (2536, 2090) |
| Rooms, LLM boxes + SAM 3 | 1293: 5/5, 9 %; 2536: 8/9, 16 %; 2090: 5/5, 20 % | | the lite model's boxes are a little loose, so SAM 3 does not gain over the polygons here; `gemini-3.8-flash` gave boxes within a few pixels of the answer key in the first probe (but it allows 20 requests a day) |

Worst rooms in both modes are the open-plan hall (1293 ET, +65 to +80 %: the model extends it into the kitchen) and
the walk-in closet of 2536 (missed).

**Homework**

| step | Gemini | specialist | notes |
|---|---|---|---|
| Classification (20 style images), *basic* + schema | **90 %** (both Georgian images called Victorian Terrace) | MP2 model 100 % | the same 90 % with every prompt and without the schema: the descriptions and rules change nothing here |
| JSON lab | A: 3/3 valid, all fenced; B: 3/3 valid, no fences; same label throughout | | |
| Detection (machinery, 9 boxes) | recall **100 %**, precision 100 %, with and without the schema | MP3 YOLO: recall 100 %, precision 90 % | large, distinct objects: the easy case for a generalist |
| Counting excavators | right on three photos; on site 3 it says 1 of 2 where the key has 0 (its own boxes: 0); on site 6 it says 0 of 1 where the key has 2 machines | | again the direct count is the less reliable of the two |
| Rooms, LLM polygons + schema (scans) | 8138: 6/6 rooms, median 10 %; 10715: 6/7, 20 %; 11615: 4/6, 7 % | MP4 SAM 3 'room': 6/6, 4/7, 5/6 | on the scans the model returns extra "rooms" (9 for 6 on 8138) and the totals overshoot the floor area |
| Rooms, LLM boxes + SAM 3 | 8138: 5/6, 10 %; 10715: 6/7, 12 %; 11615: 3/6, 26 % | | on 10715 SAM 3 tightens the loose boxes (20 → 12 %); on the small flat 11615 it loses rooms |
| Rooms without the schema | 8138 and 10715: the reply is not valid JSON (long coordinate lists); 11615: 5/6, 15 % | | |

The scans (set B) versus the clean drawings (set A) is the point of homework question 6: polygons drift into
furniture and dimension strings, extra regions appear, and the boxes-to-SAM 3 route holds up better on the
cluttered plan but not on the small one.

## 4. Suggested workshop timing (90 min)

| Min | Part | What to say / do |
|---|---|---|
| 0–8 | Set-up | Keys created before class if possible (aistudio.google.com/apikey → Colab secret `GEMINI_API_KEY`). Step 0 while you explain what a vision-language model is: image and text in, text out, no class list, no box head; the prompt is the program. |
| 8–22 | Part 1 | Step 1b: a free question. Step 1c: the same prompt run three times without and with a schema; count the valid replies, the fences, the label drift. The point: a person can read prose, a program cannot. |
| 22–40 | Part 2 | Step 2a with the three prompts: read the confusion table, look at the mistakes with the model's reasons. Step 2b: change one thing in the prompt (a rule, a description) and rerun live. Discuss overfitting a prompt to 14 photos. |
| 40–58 | Part 3 | Step 3a on one photo, Gemini next to YOLO. Step 3b: recall and precision. Step 3c: ask for the number vs count the boxes; when they disagree, which one is wrong? |
| 58–75 | Part 4 | Step 4a: polygons (watch the JSON break on long coordinate lists without a schema). Step 4b: the boxes to SAM 3. Step 4c: room by room, three ways. |
| 75–90 | Part 5 + wrap-up | The prompt lab with a phone photo. Step 6: the table. Report template: `docs/MP5_Workshop_Report_Template.md`. |

## 5. Answer key and marking notes (100 points)

Points are suggestions; the model's replies vary between runs (temperature), so accept numbers within a few
percent of the ones in section 3.

1. **JSON two ways (10).** A: some runs fenced, occasionally invalid or with a label outside the list; the label can
   change between runs. B: always valid, labels from the enum; the label can still be wrong or change. A program needs
   B because it must parse the reply every time; correctness is a separate question.
2. **Classification (14).** Accuracy per prompt vs the MP2 model (section 3). Confusions between neighbouring classes
   (minor/major crack, stain/algae, peeling/spalling); descriptions and rules move a few photos. Homework: the style
   images are synthetic and the MP2 model was trained on the same kind, so the comparison favours the specialist.
3. **Own prompt (10).** Any documented change and its accuracy; the risk is the same as tuning on the test split in
   MP2: a prompt tuned on the 14 photos is not proven on the next 14.
4. **Detection (14).** Recall/precision vs YOLO (section 3); hardest labels are the head/torso states (NO helmet, NO
   vest) and small distant workers; extras are duplicated boxes and labels the model adds on its own.
5. **Counting (10).** The direct count, the box count and the answer key on two photos; without an answer key the only
   check is consistency between the two ways and a person looking; boxes are checkable, a number is not.
6. **Rooms (14).** The per-room table; polygons vs boxes + SAM 3 vs MP4's phrase result. Homework: on scans the
   polygons drift into furniture and dimension strings while the boxes stay usable.
7. **Own experiments (12; 16 in the homework).** The image, the prompt, the raw reply, right or wrong, and the failure
   mode named (wrong answer, invented objects, unreadable reply).
8. **Reflection (16; 12 in the homework).** Per task a defensible choice; the generalist needs no data and no training
   but a network, a key, money per request and a check; the specialist needs labelled data and a GPU but is fast,
   offline and consistent; together (Step 4b) each does what it is good at. Structured output guarantees the
   *shape* of the reply, not its truth.

## 6. Known failure modes

- **"Gemini is busy ... waiting"** for a minute or more: the free tier's per-minute limit. It resolves itself; a
  whole class on one key would not, hence one key per student.
- **"request failed: You exceeded your current quota"**: the daily cap of that model. Pick another model in Step 0
  (the precomputed answers are per model, so switching only affects live requests) or come back tomorrow.
- **No key, live step**: a clear message, nothing else happens. The precomputed steps still work.
- **Invalid JSON in Step 4a without a schema**: expected and part of the lesson; the schema version is one tick away.
- **SAM 3 not loaded** (no GPU, or Step 0 asked for a restart after installing transformers): Step 4b falls back to
  the box areas and says so.
- **The prompt lab does not appear**: run the cell again; it needs a live key.
- **Licences**: the photos and plans keep the licences of the earlier labs (see the README); the model's replies are
  the students' to use.

## 7. Rebuilding or changing the material

See `README.md`. Prompts live in `aec_llm/config.py` (`PROMPTS`, filled with each set's `intro`, class list, `hints`,
`rules`). After changing a prompt, rerun `build/precompute.py` with a key in `GEMINI_API_KEY` (only the missing replies
are requested; delete the stale files in `data/cache/<variant>` if you want the folder tidy) and
`build/make_notebooks.py`. To change the examples, edit the specs in `config.py` and rerun `build/prepare_data.py`.
