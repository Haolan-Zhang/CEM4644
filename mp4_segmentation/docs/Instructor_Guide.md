# MP4 · Instructor guide

## 1. What the students get

Two Colab notebooks generated from `build/make_notebooks.py`. Students never write code: every cell is a form
with a play button; the outputs are three-panel views (drawing, mask, overlay), hit / miss / extra overlays, a
box-drawing tool, tables of square feet next to the drawing's own numbers, and a small upload app.

| | Workshop | Homework |
|---|---|---|
| Drawings | three 1940 USDA farmhouse floor plans: `usda_5544` (easy), `usda_5540` (medium), `usda_5539` (L-shaped) | seven sheets: `usda_5542` + `va_floor` (floor plans), `test_fp`, `test_fp_2`, `uscg_motorpool`, `uscg_pile` (structural), `va_ceiling` (electrical) |
| Teaches SAM 3 | yes: photo steps, ask-by-name, wording, inspector | no: straight into the take-off |
| Take-off | one cell per drawing (Step 3a/3b/3c) | one cell per discipline (Steps 2a, 3a, 4a), a dropdown of that discipline's sheets, and after each a phrase cell (Steps 2b, 3b, 4b: `lab.ask`, the same things asked for by name and scored against the key) |
| What is taken off | the scale (two readings) and the area of every room | areas (rooms, footings, pits) **and** counts |
| Counts | not in the take-off cells (rooms only); Step 3d counts the doors from one `example: door` box, scored against the key | one box labelled `example: <thing>` per counting category; SAM 3 finds the rest and the key checks it |
| Own drawings | 1 | 3, from at least two disciplines |
| Report questions | 7 | 5 |
| Time | about 90 min | about 2.5 h |

Everything is in feet and square feet. Every homework sheet asks for one scale box, the first of the two it
originally had (the 10'-0" dimension on `va_floor`, the 32'-0" dimension on `usda_5542`, the 14'-0" bay on `test_fp_2`, the
15'-0" bay on `uscg_motorpool`, the 16'-0" bay on `uscg_pile`, ten grid cells on `va_ceiling`; on `test_fp`, which prints no dimension, the first footing box is the scale). The box
is used for every area but the SCALE CHECK lines are not printed (`SheetSet.scale_check=False`, 2026-09-23). The
workshop keeps the full scale exercise with both references. On a
foundation plan one `footing` box is the whole task: it is the example SAM 3 counts from and the footing whose area
is measured. The VA sheets' wrong 0–16 ft graphic bar was dropped from the tasks on 2026-09-22 at the instructor's
request.

**A count is always a model result.** There is no task anywhere in the lab where the student boxes every instance of
something and the boxes are then tallied against the answer key: a row of hand-drawn boxes ticked off against a key
measures the student's eyesight, not SAM 3, and the student learns nothing from it. Where a sheet has something to
count, the student draws **one** box labelled `example: <thing>` and `segment_like` finds the rest; the answer key's
list of instances exists to check *that* count (found / missed / extra). This is why the three workshop floor plans
no longer ask for windows and doors: on a floor plan the only take-off left is the rooms.

Part 1 of the workshop opens on two ordinary site photos (`data/intro`) so that students see the three prompt types
(phrase, box, click) on a photograph before they meet a drawing; the box and the click use SAM 3's tracker head
(the classic SAM prompt encoder, same checkpoint, loaded on first use).

Everything the students see in Part 2 and Step 4a of the workshop is precomputed (`data/masks/workshop`) and
instant. The live model is needed for Steps 1a/1b, every take-off cell, Step 4c and the upload app.

## 2. What SAM 3 actually does on these drawings (measured on this material)

Everything in this section was measured through the notebook's own code (`aec_seg.ui.takeoff_compute`), on the
files that ship here, with **student-like boxes**: every answer-key box with each side moved by 0.5–2 % of the box
size, 70 % of the time outward, fixed seed; the scale boxes moved by up to 4 px per side, which is what a student
who zooms in and aims at the arrowheads achieves.

### 2.1 Asking by name (workshop, Part 2) — regions at confidence ≥ 0.3, (best confidence), found / key, extras

| thing | `usda_5544` | `usda_5540` | `usda_5539` |
|---|---|---|---|
| room (any) | 6 (0.79), 6/16, 0 extra | 6 (0.72), 5/12, 0 extra | 9 (0.72), 8/14, 0 extra |
| bedroom | 4 (0.59), 3/3, 1 extra | 4 (0.55), 2/2, 2 extra | 5 (0.48), 2/2, 3 extra |
| bathroom | 2 (0.79), 1/1, 1 extra | 1 (0.53), 1/1, 0 extra | 1 (0.67), 1/1, 0 extra |
| living room | 1 (0.31), 1/1 | 0 (peak 0.20), 0/1 | 0 (peak 0.29), 0/1 |
| kitchen | **0** (peak 0.15), 0/1 | **0**, 0/1 | **0** (peak 0.13), 0/1 |
| porch / closet / door / window | **0** | **0** | **0** |
| *curved line* (= door swings) | 30 regions → **17/17** doors, 2 extra | 25 → **13/14**, 9 extra | **0** (this sheet's arcs are drawn lighter) |
| *thick black line* (= walls; typed live in Step 4c) | 31 regions | 28 | 32 |

The door and window counts in that row are from the feasibility measurements; the shipped keys of these three sheets
measure rooms only, so Step 2b draws its hits and misses for the room words and says plainly, for *door* and
*window*, that the key does not list them one by one.

The vocabulary (`THINGS` in `aec_seg/config.py`) is nine entries: *room (any)*, *bedroom*, *kitchen*, *living room*,
*bathroom*, *porch*, *closet*, *door*, *window*, each with its alternative wordings. **`wall`, `stairs` and
`fireplace` were removed**: no answer key lists any of them, so Steps 2a/2b and 4a could only show a picture with
nothing to check it against. Their shape words are still there to be typed live in Step 4c (*thick black line*,
*hatched square*, *steps*), where a phrase with no key is the point of the step.

Read to the class: **enclosed spaces work** (*room* finds the large rooms with 0.7–0.8 confidence and no extras;
what it misses are the closets, which are 3–10 sq ft); **functions do not** (*kitchen*, *living room*, *porch*,
*closet* — the model sees no object, only a word); **drawing conventions do not, but shapes do** (*door* → nothing,
*curved line* → every swing; *wall* → nothing, *thick black line* → the wall runs). *bedroom* returns rooms in
general, not bedrooms: the model is not reading the lettering. **No wording finds a window on any of these sheets**
(*window*, *window opening*, *short parallel lines*, *gap in the wall* return nothing at any confidence) — and that
is where the lab leaves windows: boxing them by hand and ticking them off against a key would test the student, not
the model.

### 2.2 The take-off (workshop Part 3, homework Parts 2-4)

Scale error is the student's reading against the answer key's own box for the same reference. The count column is
what SAM 3 returns from ONE example box through `segment_like` at the confidence the key suggests: *matched of the
key's total, extras*. There is no hand-tally column any more — a count a student makes by hand is not a measurement
of the model, so the lab no longer asks for one.

| sheet | scale error (main ref / cross-check) | count from one `example:` box (confidence: matched / key, extras) | areas: n, median, worst |
|---|---|---|---|
| `usda_5544` | −0.6 % / +0.6 % | Step 3d, door (0.4): median **11/17** over all 17 possible examples (1–17), 0 extra; only 1 example finds all 17 | room n=9 (closets are not in the take-off), **median 3.9 %**, worst 21 % (the hall) |
| `usda_5540` | −0.3 % / +0.5 % | Step 3d, door (0.4): median **14/14** (1–14), 2 extra; 9 of 14 examples find all | room n=7, **median 0.6 %**, worst 26 % (the hall) |
| `usda_5539` | −0.3 % / +0.4 % | Step 3d, door (0.4): median **11/11** (1–11), 4 extra; 6 of 11 examples find all | room n=9, **median 2.6 %**, worst 20 % (the hall) |
| `usda_5542` | −0.4 % (boxed, not printed) | – (rooms only) | room n=10 (closets are not in the take-off), **median 2.6 %**, worst 23 % (the hall) |
| `va_floor` | −1.4 % (boxed, not printed) | – (rooms only) | room n=19 (22 spaces in the key, 3 open alcoves left out), **median 2.0 %**, worst 7 % |
| `va_ceiling` | −0.7 % (boxed, not printed) | 2×4 (0.4) **39/41, 0 ex**; 2×2 (0.4) 8/8, 1 ex; cans (0.5) **14/14, 0 ex** | – (nothing to measure) |
| `test_fp` | −3.2 % (the footing box, not printed) | footing (0.3) **10/10**, 3–5 ex | footing n=10, **median 4.3 %**, worst 8.5 % |
| `test_fp_2` | −1.4 % (boxed, not printed) | footing (0.4) **19/19**, 1 ex | footing n=19, median 6.2 %, worst 19 %; elevator shaft n=1, 5.2 % |
| `uscg_motorpool` | −1.7 % (boxed, not printed) | footing (0.3) 19/20, 2–5 ex; grid bubble (0.5) **32/32**, 2 ex | footing n=20, median 6.0 %, worst 32 %; **equipment pit n=2, median 92 %** |
| `uscg_pile` | −1.8 % (boxed, not printed) | pile footing (0.3) **29/29**, 2 ex | – (the caps are too small to measure) |

The counting categories are named after the same object as the area category, so a sheet that has both reads
`footing`, whose first box is also the counting example (no `example: footing` label, and no `spread footing` beside `footing`); `uscg_motorpool` keeps a second category,
`grid bubble`, because it is a different object.

Three of those numbers are **designed failures** and are the teaching points of their sheets:

- `va_floor` had two fixture counts (water closets, lavatories) in an earlier version; they were dropped so that the floor plans are room areas and nothing else.
- `uscg_motorpool` **pits +92 %**: a 2'-0" × 2'-6" pit is 36 × 39 px on a 2270 px sheet. The mask leaks into the wall
  footing around it and the cell prints "your box is only 36 px across… the number is your box, not the pit."

Room totals against the answer key, same run: `usda_5544` −3.5 %, `usda_5540` −1.5 %, `usda_5539` −3.9 %,
`usda_5542` −2.6 %, `va_floor` −7.3 %. Footing totals: `test_fp` −3 %, `test_fp_2` +2 %, `uscg_motorpool` +3 %.

### 2.3 Asking by name on the homework sheets (Steps 2b, 3b, 4b) — regions at confidence ≥ 0.3 (≥ 0.5), found / key, extras, median area error of the regions found

The phrase cell sends the words straight to SAM 3 with no box, draws the answer key's hits (green), misses (red) and
extra regions (blue), and for an area category measures every region found as it comes (no clip, no hull: the raw
mask, which on a furnished room runs a few percent low). Measured 2026-09-19 with the live model:

| sheet | works | finds nothing |
|---|---|---|
| `usda_5542` | *room* **10/14**, 0 extra, areas median 3 % (worst 45 %, the L-shaped hall); at 0.5 only 5/14. *bedroom* 6/14 at 0.3. *curved line* 31 regions = the door swings (no key) | *door*, *window* |
| `va_floor` | *room* **13/22**, 0 extra, median 4 %; at 0.5 nothing survives. *curved line* 12 regions | *door*, *window* |
| `test_fp` | *square* → footings **10/10**, 5 extra, areas median 2 % (at 0.5: 7/10, 2 extra, median 1 %). *rectangle* 10/10 but 9 extra | *footing*, *foundation*, *column*, *hatched square* |
| `test_fp_2` | *square* **19/19**, 4 extra, areas median 10 % (the E-footings are drawn smaller than their mark; at 0.5 still 19/19, 1 extra). *rectangle* 19/19, 6 extra | *footing*, *hatched square* |
| `uscg_motorpool` | *circle* → grid bubbles **32/32**, 6 extra (3 at 0.5). *square* → footings 12/20, 5 extra, median 3 %; *rectangle* 7/20 | *footing*, *hatched square* |
| `uscg_pile` | *rectangle* → pile caps **29/29**, 8 extra; at 0.5 28/29, 2 extra | *footing*, *square*, *hatched square* |
| `va_ceiling` | **nothing.** *rectangle* 7/41 fixtures with 24 extras; *small circle* 45 regions, 0 of the 14 recessed lights | *light fixture*, *light*, *diffuser*, *sprinkler*, *rectangle with a diagonal line*, *circle with a cross*, *square* |

So the phrase route gives the students two real findings: a shape word gets the plain symbols of a foundation plan
(and their areas, within a few percent), and it gets nothing at all on the ceiling plan, where the example box in
Step 4a counts 39/41. The trade words fail everywhere. The confidence slider is the other lesson here: 0.3 keeps the
extras, 0.5 loses real ones (`va_floor` *room* goes from 13/22 to 0).

### 2.4 How a room mask becomes an area, and why (the "fill the bites" rule)

A room mask stops correctly at the walls, but the door swings, counters, bathtubs, stairs and fireplaces drawn
inside the room are cut out of it, so the raw pixel count reads low. Three read-outs were measured on all 69 rooms
of the five sheets that have rooms, with the same student-like boxes (median and worst |error| per sheet):

| sheet | raw pixels | **convex hull** | bounding box |
|---|---|---|---|
| `usda_5544` | 6.1 % / 28.2 % | **3.5 % / 20.7 %** | 2.3 % / 18.4 % |
| `usda_5540` | 7.7 % / 13.8 % | **3.5 % / 24.6 %** | 2.1 % / **49.3 %** |
| `usda_5539` | 6.5 % / 13.4 % | **3.8 % / 11.0 %** | 1.5 % / 23.9 % |
| `usda_5542` | 7.3 % / 30.3 % | **4.0 % / 28.9 %** | 3.0 % / 23.0 % |
| `va_floor` | 8.9 % / 16.6 % | **2.0 % / 7 %** (19 rooms, key completed 2026-09-22) | 3.4 % / 7.2 % |

Split by shape (the answer key's polygon tells which rooms are not rectangles):

| | rooms | raw | **convex hull** | bounding box |
|---|---|---|---|---|
| rectangular rooms | 62 | median 7.0 % | **median 3.5 %** | median 1.9 % |
| non-rectangular rooms (L-shaped halls, living rooms, porches) | 7 | median 8.0 %, worst 13.4 % | **median 4.0 %, worst 24.6 %** | median 5.0 %, worst **49.3 %** |

**The lab ships the convex hull** (`AREA_RULE = "hull"` in `aec_seg/ui.py`). The bounding box is better on a plain
rectangle — it gives back the whole bite — but on an L-shaped room it adds the *entire* missing corner: the
six-sided hall of `usda_5540` reads **+49 %** with the bounding box against +25 % with the hull, and the eight-sided
hall of `usda_5539` +24 % against +11 %. The hull can add at most half of a missing corner, it halves the raw error
on non-rectangular rooms as well as on rectangles, and its worst case is better than the bounding box's on four of
the five sheets. The difference in the medians (3.5 % against 1.9 % on rectangles) is not worth a 49 % outlier that
a student would have no way of spotting. When the hull has to fill back more than 15 % of a room, the cell says so
and tells the student to look at the picture.

Areas of single objects use `segment_visual` (the tracker head), never `segment_box`: with a loose student box the
concept head returns the box itself (measured on this material at up to +95 %) while the tracker head still returns
the object (+4 %).

## 3. Suggested workshop timing (90 min)

| Min | Part | What to do and say |
|---|---|---|
| 0–8 | Step 0 | Everyone switches the runtime to *T4 GPU* and runs Step 0 (clone + SAM 3 load, 2–3 min) while you explain detection (a box) versus segmentation (pixels), and why pixels can be turned into square feet. Choose *Run anyway* on Colab's author warning. |
| 8–18 | Part 1 | Step 1a on the concrete-pour photo: *person*, *helmet*, *wet concrete*, then a word that is not in the photo; watch the confidences and the slider. Step 1b: a box around a worker, a click on a helmet — one object each, no words. Step 1c: browse the three drawings; point out that there is **no scale bar**, that the overall dimension lines are where the scale comes from, and that the notebook knows every room's drawn area. |
| 18–32 | Part 2 | *room (any)* on `usda_5544`, then *bedroom*, then *kitchen*, *door*, *window*. The hit/miss overlay of Step 2b makes the difference visible: the room words light up green and red on the key, while *kitchen*, *door* and *window* return nothing to draw at all. Message: the number is only as good as the word, and the answer key is how you find out. |
| 32–62 | Part 3 | Do Step 3a together, slowly: the scale box from arrowhead to arrowhead (zoom in), then the second dimension down the side, then every room; Submit; read the two scale lines and the room table. Then let them do 3b and 3c on their own (about 12 min each). Walk around: the two things that go wrong are a loose room box and boxing the *dimension line* instead of the span between the arrowheads. |
| 62–70 | Step 3d | One box round one door on `usda_5540` (arc included): 14/14. Then the same on `usda_5544`: 11 or so of 17, the double door and the closet doors missed. Pick a poor example on purpose (a closet door in the corner) and watch the count collapse: the example is the prompt. Windows were left out on purpose: a median 2/9 on `usda_5544` (faint, varying lengths). |
| 70–80 | Part 4 | Step 4a with *door*: *door* finds nothing, *curved line* finds the swings. Step 4b: weak regions and what they sit on. Step 4c: *thick black line*, *circle*, *hatched square*. |
| 80–90 | Part 5 + wrap-up | The upload app from the link (a photo of any drawing works): a box is two clicks, the scale comes from a box along a printed dimension plus its length in feet, and the three ways of asking are the notebook's (phrase, box for an area, one example box for a count). Point at `lab.report_summary()` for the numbers and at `docs/MP4_Workshop_Report_Template.md`. |

If you are short of time, drop Step 3c (`usda_5539`) and set it as homework; it is the drawing where the L-shaped
footprint and the porches make the take-off hardest.

## 4. Answer key for the report questions

Points are suggestions out of 100. Accept any well-argued answer; the numbers below are what a correct run
produces, with a few percent of variation because the boxes are drawn by hand.

### Workshop

1. **Words that work and words that do not (12).** *room (any)* finds most enclosed spaces with high confidence
   and few extras; *bedroom* returns rooms in general, not bedrooms (the model does not read the lettering);
   *kitchen*, *living room*, *porch*, *closet*, *door*, *window* find nothing at all. The
   found/missed/extra numbers are in section 2. What the misses have in common: the model was trained on
   photographs, so it finds *things that look like things*; a room is an enclosed space it can see, a "kitchen" is
   a function it cannot, and a window on a plan is a drawing convention, not a picture of a window.
2. **The take-off numbers (18).** Per drawing: both scale readings within about 1 % of the answer key when each box
   is drawn arrowhead to arrowhead, and the 1.6–2.1 % disagreement between the two dimensions of a 1940 scan noticed
   (the cell prints both estimates in px/ft and how far apart they are); the room table with most rooms inside ±8 %. The total of the
   indoor rooms should land within a few percent of the drawing's total. Full marks need all three drawings and the
   total-versus-key comparison.
3. **Why the worst rooms are worst (16).** Expected answers: the hall is a set of doorways, not a room, so the mask
   is a cross and the printed area is a rectangle; a kitchen loses the counter and the range; a bathroom loses the
   tub; a porch with a screen rail instead of a wall has nothing to stop the mask; a box drawn loosely pushes the
   number up. Full marks mention that a 1 % error in the scale is a 2 % error in every area.
4. **One example, all the doors (10).** Best counts around 14/14 on `usda_5540`, 11/11 on `usda_5539`, 11–13 of 17
   on `usda_5544` (the double door and the closet doors go missing), extras mostly the bathtub and a porch corner.
   A different example changes the count: a poor one (a closet door, a door boxed without its arc) can drop it to a
   handful, because the box *is* the prompt. Hardest: `usda_5544`, where the swings are small and several doors
   are double. Why an example beats the word: SAM 3 matches what it *sees* in the box (an arc and a gap) against the
   rest of the sheet, whereas the word *door* means a photographed door, which a drawing never contains.
5. **Wording and weak regions (14).** *door* → nothing, *curved line* → the swings (with the bathtub and a few
   corners as extras); *window* → nothing, and none of its three alternatives does better. (In Step 4c, typed live:
   *wall* → nothing, *thick black line* → most wall runs.) A documented weak region (what it
   sits on, its confidence) and one plain mistake. The advice to a colleague should be: check every number against
   the drawing, use boxes rather than phrases for a take-off, and never hand on a phrase count unchecked.
6. **Own words (10).** A shape word works because the model was trained on photographs of *things*; on a drawing
   a door is an arc, a wall is a thick black stripe and a window is a gap with thin lines in it. Those are shapes,
   and shapes are what the model can be asked for.
7. **Own drawing and reflection (20).** Their own sheet, the phrase, the result, a judgement. Useful in practice: a
   first pass over many sheets, counting repeated symbols, checking a room schedule, sanity-checking someone else's
   take-off. Misleading: open-plan spaces, anything without a known dimension, small symbols, any drawing where the
   convention matters more than the picture. Needed before it goes into an estimate: a printed dimension or a
   schedule to set and check the scale, clean drawings, and a person who signs off.

### Homework

1. **Floor plans (22).** The scale from the printed 10'-0" dimension (about 21.6 px/ft) and the room table with the
   worst rooms named. From Step 2b: *room* finds 13 of the 22
   spaces at 0.3 with areas a few percent under the ones from the student's boxes (raw masks, furniture bitten out;
   the hall on `usda_5542` is the worst); *door* and *window* return nothing; *curved line* returns the swings.
2. **Structural plans (22).** Per sheet the scale route and the footing areas against the sizes the marks give
   (`F12.0` = 12 ft, `E4'-6"` = 4 ft 6 in, and the `FOOTING SCHEDULE` on `uscg_motorpool`), plus the count SAM 3 made
   from the first `footing` box, which is the example. On `uscg_pile` the caps are 2'-6" × 5'-0" and only about 27 × 53 px on the sheet:
   below roughly 60 px the mask stops tracing the symbol and becomes a rounded copy of the drawn box, so the
   "area" is the student's box. That is why the sheet is a counting exercise only. From Step 3b: *footing* finds
   nothing on any sheet; *square* finds 10/10 and 19/19 footings on the two example plans (areas within 2 % and 10 %)
   but only 12/20 on the motor pool sheet; *rectangle* finds all 29 pile caps and *circle* all 32 grid bubbles, each
   with a handful of extras that go at 0.5. The phrase areas are close to the box areas where the symbol is a clean
   square; the box route still wins on the schedule-sized footings of `uscg_motorpool`.
3. **MEP (22).** The counts of 2×4 fixtures, 2×2 fixtures and recessed lights from one example each, the confidence
   used, and what the extras were. A rotated example loses roughly 40 % of the count and doubles the false
   positives. The trade words return nothing because SAM 3 has no notion of an MEP legend: a "light fixture" on
   this sheet is a rectangle with a diagonal and a circle. From Step 4b: on this sheet not even the shape words work
   (*rectangle* 7/41 with 24 extras, *small circle* 0/14), so the example box of Step 4a (39/41) is the only route —
   which is the answer to the "does the best phrase get near the count" part of the question.
4. **Own drawings (14).** Three sheets from at least two disciplines; what failed and why (hand-drawn sheets,
   photographs at an angle, colour fills, symbols too small, no known dimension).
5. **Reflection (20).** What decides reliability: how large the thing is on the sheet (about 60 px is the floor),
   how regularly it repeats, and whether it is drawn as a simple closed outline. Counting repeated symbols is the
   most reliable task in the whole lab; measuring small symbols is the least. Boxes or phrases: boxes for anything
   that goes into an estimate (the scale, the areas, the counts); a phrase for a first look at a floor plan (*room*)
   or a foundation plan (*square*), never on an MEP sheet, and never a trade word.

## 5. Failure modes the students will meet

- **A small symbol measures as the box that was drawn round it.** Below about **60 px across on the sheet** SAM 3
  gives back a rounded copy of the box instead of the thing inside it (`uscg_pile`'s pile caps at 27 × 53 px, the
  pits and post footings on `uscg_motorpool` at 36 px). The take-off cell prints three different warnings, in this
  order: *your box is only N px across… the number is your box*; *the mask is N % of your box: it has spilled
  outside the box*; and, at over 90 % fill, *look at the outline: if it is a rounded copy of your box…*. The third
  one fires on correct readings too (a tight box round a footing is 92–101 % filled), so tell the students it is a
  "check the picture" prompt, not a verdict.
- **The wrong scale bar.** `va_floor` and `va_ceiling` carry a 0–16 ft graphic bar that is exactly 2× too long
  (about 43 px/ft against the printed dimension's 21.6). It is no longer a task; a student who boxes it anyway as the
  scale gets every area four times too small, which is worth a word in class.
- **A rotated example box.** `segment_like` generalises to the symbol *family*, but a rotated example costs about
  40 % of the count on the ceiling plan and starts returning the stipple-hatched toilet ceilings as extras. Tell
  the students to pick a clean example lying the same way as most of the others.
- **Open-plan spaces.** A porch with a screen rail, a hall that is a set of doorways, a dining corner with no wall on one
  side: the mask has nothing to stop it, so the number is set by where the box was drawn. The cell no longer prints a
  warning for this (it fired on too many correct readings); the hall is the worst room on all three workshop plans
  (20–26 %), and question 3 asks the students to find the reason in the picture.
- **A phrase that finds nothing.** Expected for *kitchen*, *living room*, *porch*, *closet*, *door*, *window*, and
  for every trade word on a structural or MEP sheet (*footing*, *pile*, *grid bubble*, *light fixture*, *diffuser*,
  *sprinkler*). Send the students to the shape words (Step 4a and 4c) and, on a sheet with something to count, to
  the `example:` box.
- **No `example:` box drawn.** A take-off on a sheet with a counting category prints, for each one that has no
  example box, *"not counted. Draw ONE clean box labelled 'example: …' and submit again"*. Two or more boxes with
  the same `example:` label are not an error either: the first is used and the cell says so.
- **Boxing the dimension line instead of the dimension.** The scale box must go from arrowhead tip to arrowhead tip.
  Boxing the whole line including the extension lines makes the scale a few percent too large, and every area
  twice that much too small.
- **Step 0 asks for a restart.** Colab has shipped an older `transformers` without SAM 3; Step 0 installs a newer
  one and asks for *Runtime → Restart session*. Run Step 0 again after the restart; the install is kept.
- **The box tool does not appear** (Steps 1b and every take-off). It is a third-party widget; Step 0 enables
  Colab's custom widget manager. Re-run Step 0, then the step.
- **The app link could not be created.** Part 5 prints a link instead of embedding the app; if the tunnel fails,
  run the cell again.

## 6. Rebuilding or changing the material

See `README.md`. In short: drop `<id>.png` and `<id>.key.json` into `data/sheets/<set>/`, add the id to that
set's `credits.json`, then

```bash
python -m aec_seg.data data/sheets/workshop data/sheets/homework   # validate every answer key
python build/precompute_masks.py                                   # workshop phrases (GPU, a few minutes)
python build/make_notebooks.py
```

The notebooks are generated from the answer keys: the sheet list, the labels of the box-drawing tool, the task
text under each take-off cell and the credits all come out of the key files, so a new sheet needs no code change.
`aec_seg/data.py` validates every key and the notebook refuses to open a set with a broken one — pixel boxes
outside the image, a room type outside the vocabulary, two scale references marked `use`, a missing legend image
and so on are reported in plain words with the file and the field.

The labels of the box-drawing tool are built in `Sheet.box_labels()` (`aec_seg/data.py`): `scale: <ref label>` per
scale reference, the plain area category per area, and `example: <category>` per counting category. Adding a
counting category to a key therefore adds one `example:` label and one model-made count to that sheet's report;
nothing in the lab produces a count any other way.

The vocabulary of Part 2 and Step 4a is `THINGS` in `aec_seg/config.py`; each entry says which wordings to
precompute and what to look up in the answer key. Removing an entry leaves its precomputed masks behind:
`build/precompute_masks.py` only ever adds, so delete the stale `data/masks/<set>/<sheet>/<phrase>.png` files and
their entries in that folder's `index.json` by hand. `AREA_RULE` in `aec_seg/ui.py` selects how a room mask is turned
into an area (see section 2); `MIN_SYMBOL_PX` is the size below which the cell warns that the answer is the box.
