# MP4 · Instructor guide

## 1. What the students get

Two Colab notebooks built from one template. Students never write code: each cell is a form with a
▶ button; outputs are three-panel views (plan, mask, overlay), hit/miss/extra overlays, tables of
square metres next to the drawing's own numbers, a box-drawing tool and a small upload app.

| | Workshop | Homework |
|---|---|---|
| Plans | set A: 6 homes drawn cleanly from CubiCasa5K's vector data (1293, 2536, 2090, 6457, 207, 7696) | set B: 7 scanned CubiCasa5K plans (14341, 5018, 1217, 8138, 9136, 11615, 10715), two with both storeys on one sheet, one Swedish |
| Steps start with | room (any), window, door (wording), room (any) | bedroom, toilet, stairs, living room |
| Take-off targets | two rooms, one window, one door | every bedroom of a plan, the stairs, a toilet, a bathtub; the four rooms of plan 5018 |
| Own plans | 1 | 3 |
| Time | about 90 min | about 2 h |

The homework still uses the scanned plans of the first version; it is due to be redesigned around
counting components on structural plans (see section 6).

**The plans.** All plans are from **CubiCasa5K** (CC BY-NC-SA 4.0). For set A the notebook shows a
*rendering of the dataset's own vector drawing* (`model.svg`: uniform black walls, light-blue windows,
door swings, fixture symbols, no dimension strings), not the scanned image; SAM 3 behaves much better on
these clean drawings (section 2). The vector frame is drawn at exactly 1 px = 1 cm; `build/prepare_plans.py`
resamples each plan by its own factor (so students cannot guess the scale), adds a 5 m scale bar and writes
the answer key: every room's polygon and real area, every door, window and fixture (the SVG's nested
transforms are composed, so the symbols land where they are drawn). That key is what every "the drawing
says..." line in the notebook comes from.

Part 1 opens on two ordinary site photos (Wikimedia Commons, public domain / CC BY-SA; `data/intro`) so that
students see the three prompts (phrase, box, click) on a photo before the drawings; the click and the box use
SAM 3's tracker head (the classic SAM prompt encoder, same checkpoint, loaded on first use).

Everything the students see for the built-in plans and phrases is precomputed and instant. Only the
live steps run SAM 3 (Steps 1a and 1b, Step 3a and 3b boxes, Step 4d, 4e, Part 5); on a T4 each request takes well
under a second, on CPU about a minute.

## 2. What SAM 3 actually does on these plans (measured)

Instances at confidence ≥ 0.3 on the workshop plans: count (best score), then how many of the answer
key's items they cover and how many regions are extra.

| plan | room | bedroom | bathroom | toilet | sink | bathtub | curved line (= doors) | thick black line (= walls) |
|---|---|---|---|---|---|---|---|---|
| 1293 | 5 (0.78), 6/6, 0 extra | 3 (0.53), 1/1, 2 extra | 2 (0.65), 1/1, 1 extra | 1 (0.76), 1/1 | 0 of 2 | – | 12 (0.90), 5/6, 6 extra | 32 (0.73), 12/15, 14 extra |
| 2536 | 12 (0.84), 9/10, 1 extra | 6 (0.52), 3/3, 3 extra | 3 (0.81), 2/2, 1 extra | 1 (0.33), 1/1 | 4 (0.56), 3/3 | 2 (0.81), 1/1, 1 extra | 15 (0.88), 9/9, 2 extra | 33 (0.77), 17/22, 13 extra |
| 2090 | 5 (0.66), 5/6, 0 extra | 4 (0.43), 1/1, 3 extra | 1 (0.78), 1/1 | 1 (0.90), 1/1 | 3 (0.68), 2/2, 1 extra | – | 9 (0.92), 3/5, 6 extra | 25 (0.78), 11/12, 11 extra |
| 6457 | 8 (0.79), 7/12, 1 extra | 5 (0.49), no bedrooms in the key (rooms are labelled H) | 3 (0.82), 2/2 | 3 (0.83), not in the key | 2 (0.34), 1/3 | – | 17 (0.82), 9/9, 3 extra | 42 (0.71), 20/21, 14 extra |
| 207 | 15 (0.87), 14/19, 0 extra | 9 (0.64), 5/5, 4 extra | 2 (0.89), 2/4 | 2 (0.74), 2/2 | 5 (0.61), 5/5 | – | 24 (0.78), 17/17, 4 extra | 62 (0.70), 26/31, 29 extra |
| 7696 | 1 (0.35), 1/5 | 0 | 1 (0.69), 1/1 | 1 (0.82), not in the key | 1 (0.76), 1/2 | – | 7 (0.91), 2/3, 5 extra | 15 (0.68), 6/6, 6 extra |

*kitchen*, *living room*, *balcony*, *stairs*, *door*, *window* and *wall* find nothing on any plan at any
confidence. Read the table to the class like this:

- **Enclosed spaces work.** *room* finds most rooms with high scores (0.66–0.87) and almost no extras;
  what it misses are the smallest rooms (closets, WC) and it merges open-plan spaces. *bedroom* and
  *bathroom* find the right rooms plus a few look-alikes. Functions without walls (*kitchen*, *living
  room*) find nothing: the model sees no object, only a word.
- **Symbols work when the symbol is a picture of the thing**: *toilet* (a bowl seen from above), *sink*,
  *bathtub*.
- **Drawing conventions are not in the vocabulary, but shapes are.** *door* finds nothing; *curved line*
  finds every door swing (9/9, 17/17) with a few extras (the bathtub, the toilet, a corner). *wall*
  finds nothing; *thick black line* finds most wall segments. *window* finds nothing by any word, so
  windows are counted from a box example (Step 3b). Ask the students why: the model was trained on
  photographs, in which a door is a slab in a frame and a wall is a surface; on a plan they are an arc
  and a thick line. Describe what is *drawn*, not what it *means*.
- **The confidence slider matters**: the useful range on drawings is 0.2–0.5. The notebooks default to 0.3.

**Room take-off (Step 3b).** A bare box on a clean drawing makes SAM 3 cut out the furniture symbols
inside it (the kitchen counter, the wardrobe), not the empty floor: 26 of 64 test boxes covered less than
60 % of the box that way. The notebook therefore asks for *empty room* **with** the student's box as the
example, keeps the region that fits the box best, clips it to the box, fills the holes left by symbols and
removes the thick black walls. Measured through the notebook's own code with boxes moved by −5 to +20 cm
per side (a student's box), on the five largest rooms of every plan:

| rooms | median error | mean error | above 25 % | bias |
|---|---|---|---|---|
| 29 | 4.5 % | 8.5 % | 2 | −4.9 % (masks stop a little short of the wall) |

The two failures are the honest cases: the hall of 2536 (open to the living room, +42 %, the notebook
prints "the mask covers only 57 % of your box: no wall on one side (open plan)?") and the kitchenette of
7696 (no walls at all, −54 %, same note). Living rooms that continue into a dining area come out 10–15 %
low (207, 6457).

**Find-all (Step 3b, box example).** One box around the biggest window / door / toilet / sink, confidence
0.3, found / missed / extra (instances between 0.2× and 5× the example's size are kept):

| plan | windows | doors | toilets | sinks |
|---|---|---|---|---|
| 1293 | 5/0/6 | 6/0/6 | 1/0/0 | 1/1/1 |
| 2536 | 5/0/27 | 9/0/41 | 1/0/1 | 3/0/3 |
| 2090 | 3/0/1 | 5/0/9 | 1/0/0 | 2/0/4 |
| 6457 | 6/0/8 | 9/0/27 | – | 3/0/8 |
| 207 | 19/6/10 | 17/0/14 | 2/0/4 | 5/0/3 |
| 7696 | 1/0/3 | 3/0/1 | – | 2/0/3 |

Recall is almost perfect; the extras are the lesson: a window example also matches cabinets and other
light bars, a door example matches other arcs and rectangles of the same size. Raising the confidence
to 0.5 removes most extras and starts to lose real ones. A take-off from this needs a person to strike
the blue boxes, which takes a minute per plan, against half an hour of counting by hand.

## 3. Suggested workshop timing (90 min)

| Min | Part | What to say / do |
|---|---|---|
| 0–5 | Step 0 | Everyone switches the runtime to T4 GPU and runs Step 0 (clone, SAM 3 load: 2–3 min) while you explain detection (boxes) vs. segmentation (pixels) and why pixels can be counted. Choose *Run anyway* on Colab's author warning. |
| 5–15 | Part 1 | Step 1a on the concrete-pour photo: *person*, *helmet*, *wet concrete*, *hose*, then a word that is not there; the confidences and the slider. Step 1b: a box around a worker, a click on a helmet: one object each, no words. Then Step 1c: browse the plans; point out the scale bar, the Finnish labels (legend in the notebook), and that the notebook knows every room's real area from the dataset's annotations. |
| 15–35 | Part 2 | *room* on 1293: 5 regions, 6/6 rooms found, 0 extra. Then *toilet*, *sink*. Then *kitchen*, *window* and *door*: nothing, and the hit/miss overlay shows the red windows. Then *curved line* (Step 4b wording, or 4e): every door swing. Step 2c: the m² table per room type against the drawing. Play the estimation game; ask for scores. Key message: the number is only as good as the word, and the drawing's answer key is how you find out. |
| 35–60 | Part 3 | Step 3a: box the scale bar (zoom in); a 1 % scale error is a 2 % area error. Step 3b: tight boxes on two rooms, read the "−5 %" / "+42 %" lines and the open-plan note, then one box on a window with *find_all*: count the green, red and blue boxes on the plan. Then a door: found, with extras. Discuss what a take-off needs that the model does not have (the convention, and a person). |
| 60–78 | Part 4 | Compare 1293 and 2536. Wording (Step 4b, *door*): *door* vs *door arc* vs *curved line* vs *arc*. Inspector: weak regions with 20–40 % confidence and the room they sit on. Negative box on a merged region (2090: living room + kitchen). Own words: *thick black line*, *circle*, *small rectangle*. |
| 78–90 | Part 5 + wrap-up | Own plan through the upload app if time allows (a photo of any plan works). Report template: `docs/MP4_Workshop_Report_Template.md`. |

## 4. Answer key and marking notes (100 points)

Points are suggestions. Accept any well-argued answer; the numbers below are what a correct run
produces, with small variation because boxes are drawn by hand.

1. **Words that work and words that do not (10).** Rooms, bedrooms, bathrooms, toilets, sinks, bathtubs: found (few misses); kitchen, living room, door, window, wall: nothing; *curved line* finds the doors, *thick black line* the walls. Found/missed/extra for two things at 0.3, e.g. 2536 *room* 9 of 10 found with 1 extra, *window* 0 of 5. Misses have in common: the smallest rooms, symbols that are not pictures of the thing, anything drawn as a convention.
2. **Square metres per room type and the game (12).** The Step 2c table; bedrooms and bathrooms within about 10 % (or one look-alike room added), kitchen and living room 0 vs the drawing, room (any) close to the floor area minus the smallest rooms. At 0.2 a few more regions; at 0.7 only the best rooms survive. Any game score.
3. **Room take-off (14, workshop) / plan 5018 (14, homework).** Two rooms with SAM 3's m², the drawing's m² and the error; typical errors 1–10 %, up to 40–50 % on open-plan spaces, with the reason named (no wall on one side; the box itself is the better number there). Homework: on 5018 the ground-floor living room (OH) and kitchen (K) and two upstairs bedrooms (MH); the sheet prints KERROSALA 121 / 74 m² (gross floor area per storey) and HUONEISTOALA 98 / 68 m² (net apartment area): the sum of the students' rooms is a part of the net area, not the gross.
4. **Counting with find-all (14).** Workshop: windows on their plan, found/missed/extra at the best confidence (section 2: all found, 1–27 extra), extras are cabinets, short light bars and text. Doors: all found from one example with many extras. Homework: toilets and bathtubs (fixtures work well: most found, few extras), windows on 1217, and the observation that the easiest symbol is the one that looks like the object.
5. **Compare plans and wording (12).** Step 4a: SAM 3's bedroom and bathroom m² for the two plans against the drawing's totals; the drawing decides. Step 4b (*door*): *door* and *arc* find nothing, *door arc* nothing or little, *curved line* every swing; the lesson that a shape word beats a concept word on a drawing.
6. **A mistake and the negative box (10).** Any documented region: what was included or missed, at which confidence; whether the negative box fixed it; the advice to a colleague (check every number against the drawing, use boxes for the take-off, never trust a phrase count).
7. **Own plans (10; 16 in the homework).** Phrase, count, area and a judgement per plan; what failed (hand-drawn plans, photos at an angle, plans with colour fills, drawings where the scale is unknown).
8. **Reflection (18; 12 in the homework).** Useful: a first pass over many plans, counting repeated symbols, checking a room schedule; misleading: open-plan spaces, drawing conventions, anything without a scale. Needs: clean drawings, a scale bar or a known dimension, a room schedule to check against, a person who signs off.

## 5. Known failure modes

- **Step 0 is slow.** SAM 3 is 3.4 GB; on Colab it downloads in one to two minutes. If the runtime has no GPU the precomputed steps still work; live steps take about a minute each on CPU.
- **Step 0 asks for a restart.** Colab shipped an older `transformers` without SAM 3: Step 0 installs a newer one and asks for *Runtime → Restart session*; after the restart, run Step 0 again (the install is kept).
- **Box-drawing tool missing** (Steps 3a, 3b, 4d). It is a third-party widget; Step 0 enables Colab's custom widget manager. Re-run Step 0, then the step.
- **"The mask covers only N % of your box."** An open-plan space: the model has no wall to stop at. The box area printed with the note is the better estimate; that is the point of the note.
- **"The mask fills the whole box: draw a tighter box"** (windows, doors, fixtures). A loose box returns roughly the box contents. Zoom in and redraw.
- **A phrase finds nothing at all.** Expected for kitchen, living room, balcony, stairs, door, window, wall. Send the students to *curved line* / *thick black line* (Step 4b, 4e) and to the box example (Step 3b).
- **A cell looks stuck.** *Runtime → Interrupt execution*, run the cell again. Nothing is lost.
- **Share link in the upload app fails.** The app still works inside the notebook; the link is only needed for phones.
- **Licence.** CubiCasa5K is CC BY-NC-SA 4.0: teaching use is fine, commercial use is not, and derived overlays fall under the same licence.

## 6. Rebuilding or changing the material

See `README.md`. The plans are chosen in `aec_seg/config.py` (`plans` of each set: CubiCasa sample id,
resampling factor, title; `render=True` for the clean vector rendering, `False` for the scanned image);
after a change run `build/prepare_plans.py --dir <extracted cubicasa5k>` (or `--zip cubicasa5k.zip`),
`build/precompute_masks.py` (only new plans and phrases are computed) and `build/make_notebooks.py`.
The vocabulary is `THINGS` in `aec_seg/config.py`; each entry says what to look up in the answer key.
Check a candidate plan with `Sam3Engine.segment(img, "room")` before adding it: on some plans (7696 here)
*room* finds almost nothing.

**Planned homework redesign.** Counting components on structural drawings: the two foundation plans of
the original MP4 notebook (`test_fp.png`, `test_fp_2.png` in the SAM_Example_Image repository) plus, if
wanted, the four public-domain foundation sheets used in an earlier version (US Army standard drawings
21-01-78 and 21-01-30, HABS Whittier hospital S1, HAER Shenandoah-Dives Mill; in git history before commit
b802e88). Measured on those sheets: *small square* finds the spread footings (score 0.8–0.85), *circle*
the grid bubbles, *hexagon* the footing tags; one box on a footing finds the rest with extras (about twice
the true count at 0.3, close to it at 0.5); a tight box on a footing gives its area.
