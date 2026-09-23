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
| Plans | `usda_5544`, `usda_5540`, `usda_5539`: MP4's three 1940 USDA farmhouse plans (public domain), answer keys in square feet | 8138, 10715, 11615, 5018 (the earlier MP4 set of CubiCasa scans, metres; 5018 has both storeys on one sheet) |
| Own experiments | 2 | 3 |
| Time | about 90 min | about 2 h |

The specialists the generalist is compared with are the earlier labs' own course models, run once by
`build/prepare_data.py`: MP2's ConvNeXt V2 femto (93 % on the 14 workshop photos, 100 % on the 20 style images), MP3's
YOLO11n at confidence 0.25, and MP4's SAM 3 asked for *room* by phrase (whole instances, so its areas include the
furniture holes: 7–14 % median error on the USDA plans, 19–21 % on the scans).

## 1b. The chat-window variants (`*_chat.ipynb`)

The `_chat` notebooks send the single-example steps through **hokie.ai** (https://hokie.ai.vt.edu/, VT login), so
students meet the ordinary chat interface and see that a general chat model does these tasks too: Step 1c (describe),
1d (the category asked for), 3b (boxes on one photo), 3d (the count) and 4b (the rooms of one plan from the model's
own polygons); Steps 1b, 3a and 4a just show the chosen picture for saving (2026-09-23). Nothing done in the chat is
repeated on the API: the API keeps the batch steps (2a, 2b, 3c and **4c**, which
segments every plan at once with `lab.segment_all` and is the point of the batch route: three or four plans that would
each cost a round of copying by hand) and the prompt lab, and the schema contrast comes from Step 2a's *schema* switch
and Step 4c's (question 1 and question 6 point there). There is no SAM 3 route in
these notebooks and Step 0 does not load it, so they need no GPU. Each paste-back cell shows the image with a download
button, the prompt to copy, a box for the reply and a *Score* button; the reply is parsed with the same tolerant parser
as the API replies, drawn and scored the same way, and shown next to the earlier lab's specialist for the same image
(the MP2 model's label, the MP3 model's boxes, MP4's SAM 3 by phrase). Two dropdowns handle the chat model's habits: box
order (`ymin, xmin, ymax, xmax` as asked, or `x1, y1, x2, y2`) and whether the numbers are on the 0–1000 grid, in pixels or
fractions (detected automatically when the range makes it obvious). Step 6 shows the API's batch rows and the student's
own pasted room results.

What to expect from a chat model: prose around the JSON and ``` fences (the parser copes, a naive one would not);
boxes given in pixels of a resized image or in the other order (hence the dropdowns); rougher boxes than the API's
grounded output; polygons that are coarse or missing. Those are the lesson, not bugs. The notebook cannot call hokie.ai
itself, so nothing there is precomputed; each student's replies are their own and are not stored.

## 2. The Gemini API as the students meet it

- **Model:** `gemini-3.5-flash-lite` by default; Step 0 also offers `gemini-3.5-flash`, `gemini-3.8-flash` and `gemini-3.6-flash`. Thinking level *low*.
- **Boxes:** `box_2d = [ymin, xmin, ymax, xmax]` on a 0–1000 grid (the convention the model was trained with);
  the notebook converts to pixels. **Polygons:** `mask = [[x, y], ...]` on the same grid, except that the model sometimes
  writes them as `[y, x]` (the same order as its boxes); the notebook reads each polygon in whichever orientation fits
  the entry's own box, so the outlines land on the rooms either way.
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
| Rooms, LLM polygons + schema | usda_5544: 6/14 rooms, median error 7 % (worst 33 %); usda_5540: 2/11, 28 %; usda_5539: 5/12, 13 % | MP4 SAM 3 'room': 5/14 at 8 %, 4/11 at 7 %, 6/12 at 14 % | the lite model names the big rooms right and skips the closets and halls (14 rooms in the key of usda_5544, 6 in the reply); on these clean line drawings its "mask" is just the box's four corners (usda_5544, usda_5539) or two corners (usda_5540, unusable, so the box's area is used), so the polygon route is a box route in disguise; over all 13 matched rooms the median is 13 % and 3 rooms are off by more than 25 % |
| Rooms, LLM boxes + SAM 3 (plain notebooks only) | usda_5544: 12/14, median 6 % (worst 74 %, the hall); usda_5540: 10/11, **37 %** (worst 72 %, the kitchen); usda_5539: 5/12, 30 % | | the box prompt returns nearly every room (14 and 11 entries), so SAM 3 gets to measure many more of them than the polygon route finds, and on usda_5544 it does so well; on usda_5540 the lite model's boxes sit a room off and SAM 3 faithfully measures the wrong rectangles — the lesson of Step 4c is that the finding and the measuring are separate jobs and the first one failed. `gemini-3.8-flash` gave boxes within a few pixels of the answer key in the first probe (but it allows 20 requests a day) |
| Rooms without the schema | usda_5544: 9 entries, no usable polygon, box areas 6/14 at 1 %; usda_5540: 5/11 at 21 %; usda_5539: 6/12 at 10 % | | all three replies parse: with masks this short there is no long coordinate list to break the JSON, so the schema contrast on these plans is about the *shape* of the reply, not its validity (the homework scans still break without the schema) |
| Step 4b, all 3 plans at once (chat notebooks) | 13 matched rooms, median 13 %, 3 above 25 % | | 7 s and about 4,800 tokens for the three plans, against three rounds of copying by hand |

Worst rooms: the L-shaped hall of usda_5544 (+74 % through SAM 3: the box spans the whole L and the mask fills it),
the kitchen of usda_5540 (−72 %, a misplaced box), the bath of usda_5544 (+33 % from the polygon route). The five
closets and the halls are missed by every route: the generalist stops at the rooms with a printed name.

**Homework**

| step | Gemini | specialist | notes |
|---|---|---|---|
| Classification (20 style images), *basic* + schema | **90 %** (both Georgian images called Victorian Terrace) | MP2 model 100 % | the same 90 % with every prompt and without the schema: the descriptions and rules change nothing here |
| JSON lab | A: 3/3 valid, all fenced; B: 3/3 valid, no fences; same label throughout | | |
| Detection (machinery, 9 boxes) | recall **100 %**, precision 100 %, with and without the schema | MP3 YOLO: recall 100 %, precision 90 % | large, distinct objects: the easy case for a generalist |
| Counting excavators | right on three photos; on site 3 it says 1 of 2 where the key has 0 (its own boxes: 0); on site 6 it says 0 of 1 where the key has 2 machines | | again the direct count is the less reliable of the two |
| Rooms, LLM polygons + schema (scans) | 8138: 6/6 rooms, median 10 %; 10715: 6/7, 20 %; 11615: 4/6, 7 %; 5018 (two storeys): 13/17, 15 % | MP4 SAM 3 'room': 6/6 at 3 %, 4/7 at 24 %, 5/6 at 56 %, 13/17 at 8 % | on the scans the model returns extra "rooms" (9 for 6 on 8138, where it also gives no usable polygon and the areas come from the boxes) and the totals overshoot the floor area |
| Rooms, LLM boxes + SAM 3 (plain notebooks only) | 8138: 5/6, 10 %; 10715: 6/7, 12 %; 11615: 3/6, 26 %; 5018: 13/17, 45 % | | on 10715 SAM 3 tightens the loose boxes (20 → 12 %); on the small flat 11615 and the two-storey sheet it loses area |
| Rooms without the schema | 8138 and 10715: the reply is not valid JSON (long coordinate lists); 11615: 5/6 at 15 %; 5018: 15 entries but only 1 room matched, 60 % off | | |
| Step 4b, all 4 plans at once (chat notebooks) | 30 matched rooms, median 11 %, 9 above 25 % | | 12 s and about 7,400 tokens; with the schema off, two of the four replies are unusable, which is the switch worth showing in class |

The scans versus the workshop's clean USDA drawings is the point of homework question 6: polygons drift into
furniture and dimension strings, extra regions appear, and the boxes-to-SAM 3 route holds up better on the
cluttered plan but not on the small one.

## 4. Suggested workshop timing (90 min)

| Min | Part | What to say / do |
|---|---|---|
| 0–8 | Set-up | Keys created before class if possible (aistudio.google.com/apikey → Colab secret `GEMINI_API_KEY`). Step 0 while you explain what a vision-language model is: image and text in, text out, no class list, no box head; the prompt is the program. |
| 8–22 | Part 1 | Step 1b: a free question. Step 1c: the same prompt run three times without and with a schema; count the valid replies, the fences, the label drift. The point: a person can read prose, a program cannot. |
| 22–40 | Part 2 | Step 2a with the three prompts: read the confusion table, look at the mistakes with the model's reasons. Step 2b: change one thing in the prompt (a rule, a description) and rerun live. Discuss overfitting a prompt to 14 photos. |
| 40–58 | Part 3 | Step 3a on one photo, Gemini next to YOLO. Step 3b: recall and precision. Step 3c: ask for the number vs count the boxes; when they disagree, which one is wrong? |
| 58–75 | Part 4 | Step 4a: polygons (on these plans the model's "mask" is the box's corners — show the raw reply; the JSON only breaks without a schema on the homework's scans). Step 4b: the boxes to SAM 3, or, in the chat notebooks, every plan at once. Step 4c: room by room, three ways, usda_5544 first (the box route does well) then usda_5540 (the boxes are off and SAM 3 measures the wrong rectangles). |
| 75–90 | Part 5 + wrap-up | The prompt lab (opened from the printed link) with a phone photo. Step 6: the table. Report template: `docs/MP5_Workshop_Report_Template.md`. |

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

- **"Gemini is busy ... waiting" on a built-in example** means a cache miss, which should not happen: the cache is keyed on the
  example file's bytes, so it hits on any machine. If it does, the cache folder was not pulled (Step 0 refreshes the clone).
- **"Gemini is busy ... waiting"** on a live step for a minute or more: the free tier's per-minute limit. It resolves itself; a
  whole class on one key would not, hence one key per student.
- **"request failed: You exceeded your current quota"**: the daily cap of that model. Pick another model in Step 0
  (the precomputed answers are per model, so switching only affects live requests) or come back tomorrow.
- **No key, live step**: a clear message, nothing else happens. The precomputed steps still work.
- **Invalid JSON in Step 4a without a schema**: expected and part of the lesson; the schema version is one tick away.
- **SAM 3 not loaded** (no GPU, or Step 0 asked for a restart after installing transformers): Step 4b falls back to
  the box areas and says so.
- **The prompt lab prints no link**: it is opened from a link rather than embedded (the embedded frame is unreliable in Colab). Run the cell again; it needs a live key.
- **Licences**: the photos and plans keep the licences of the earlier labs (see the README); the model's replies are
  the students' to use.

## 7. Rebuilding or changing the material

See `README.md`. Prompts live in `aec_llm/config.py` (`PROMPTS`, filled with each set's `intro`, class list, `hints`,
`rules`). After changing a prompt, rerun `build/precompute.py` with a key in `GEMINI_API_KEY` (only the missing replies
are requested; delete the stale files in `data/cache/<variant>` if you want the folder tidy) and
`build/make_notebooks.py`. To change the examples, edit the specs in `config.py` and rerun `build/prepare_data.py`.

**JSON in the students' eyes (2026-09-23).** The prompts sent to Gemini and the cache are unchanged, but every prompt the
students see (printed in a cell, or in the copy box of a chat step) goes through `ui.student_prompt`, which rewrites
"Reply with JSON only, no other text, in this form:" to "Reply in this form:" and "Output a JSON list ..." to "Output
in the form of a list ...". In the chat notebooks that rewritten text is what the students paste into hokie.ai; the
reply parser is tolerant of fenced or prose JSON, but a chat model may answer a JSON-free prompt with a bullet list,
which the Score button then cannot read. Watch for that in class.
