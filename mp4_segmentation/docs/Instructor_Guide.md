# MP4 · Instructor guide

## 1. What the students get

Two Colab notebooks built from one template. Students never write code: each cell is a form with a
▶ button; outputs are three-panel views (plan, mask, overlay), hit/miss/extra overlays, tables of
square metres next to the drawing's own numbers, a box-drawing tool and a small upload app.

| | Workshop | Homework |
|---|---|---|
| Plans | set A: 3 homes drawn cleanly from CubiCasa5K's vector data (1293, 2536, 207) | set B: 7 scanned CubiCasa5K plans (14341, 5018, 1217, 8138, 9136, 11615, 10715), two with both storeys on one sheet, one Swedish |
| Steps start with | room (any), window, door (wording), room (any) | bedroom, toilet, stairs, living room |
| Take-off | every window and every room of each of the 3 plans, one cell per plan | the same on three plans of their choice, one of them a two-storey sheet |
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
live steps run SAM 3 (Steps 1a and 1b, the Step 3a rooms, Step 4c, Part 5); on a T4 each request takes well
under a second, on CPU about a minute.

## 2. What SAM 3 actually does on these plans (measured)

Instances at confidence ≥ 0.3 on the workshop plans: count (best score), then how many of the answer
key's items they cover and how many regions are extra.

| plan | room | bedroom | bathroom | toilet | sink | bathtub | curved line (= doors) | thick black line (= walls) |
|---|---|---|---|---|---|---|---|---|
| 1293 | 5 (0.78), 6/6, 0 extra | 3 (0.53), 1/1, 2 extra | 2 (0.65), 1/1, 1 extra | 1 (0.76), 1/1 | 0 of 2 | – | 12 (0.90), 5/6, 6 extra | 32 (0.73), 12/15, 14 extra |
| 2536 | 12 (0.84), 9/10, 1 extra | 6 (0.52), 3/3, 3 extra | 3 (0.81), 2/2, 1 extra | 1 (0.33), 1/1 | 4 (0.56), 3/3 | 2 (0.81), 1/1, 1 extra | 15 (0.88), 9/9, 2 extra | 33 (0.77), 17/22, 13 extra |
| 207 | 15 (0.87), 14/19, 0 extra | 9 (0.64), 5/5, 4 extra | 2 (0.89), 2/4 | 2 (0.74), 2/2 | 5 (0.61), 5/5 | – | 24 (0.78), 17/17, 4 extra | 62 (0.70), 26/31, 29 extra |

*kitchen*, *living room*, *balcony*, *stairs*, *door*, *window* and *wall* find nothing on any plan at any
confidence (the table also held three plans that were dropped in favour of a shorter workshop; their numbers were alike). Read the table to the class like this:

- **Enclosed spaces work.** *room* finds most rooms with high scores (0.66–0.87) and almost no extras;
  what it misses are the smallest rooms (closets, WC) and it merges open-plan spaces. *bedroom* and
  *bathroom* find the right rooms plus a few look-alikes. Functions without walls (*kitchen*, *living
  room*) find nothing: the model sees no object, only a word.
- **Symbols work when the symbol is a picture of the thing**: *toilet* (a bowl seen from above), *sink*,
  *bathtub*.
- **Drawing conventions are not in the vocabulary, but shapes are.** *door* finds nothing; *curved line*
  finds every door swing (9/9, 17/17) with a few extras (the bathtub, the toilet, a corner). *wall*
  finds nothing; *thick black line* finds most wall segments. *window* finds nothing by any word, so
  windows are boxed by hand in Step 3a. Ask the students why: the model was trained on
  photographs, in which a door is a slab in a frame and a wall is a surface; on a plan they are an arc
  and a thick line. Describe what is *drawn*, not what it *means*.
- **The confidence slider matters**: the useful range on drawings is 0.2–0.5. The notebooks default to 0.3.

**The take-off (Step 3a, one cell per plan).** Three kinds of box in one drawing tool, in this order: the 5 m scale
bar, every window, every room (small things first, so later boxes do not overlap them). The scale is read from the
student's bar box and used for the areas (a 1 % scale error is a 2 % area error; the drawing's own scale is printed
next to it). Windows are counted from the boxes against the answer key (found / missed / extra). Rooms: a bare box
on a clean drawing makes SAM 3 cut out the furniture symbols, not the floor, so the notebook asks for *empty room*
**with** the student's box as the example, keeps the region that fits the box best, clips it to the box, fills the
holes left by symbols and removes the thick black walls. Measured through the notebook's own code with a bar box
within ±3 px, window boxes within ±4 px and room boxes moved by −5 to +20 cm per side:

| plan | rooms matched | median error | above 25 % | total of the rooms vs the drawing | windows |
|---|---|---|---|---|---|
| 1293 | 5/5 | 3.9 % | 0 | −5 % | 5/5 |
| 2536 | 9/9 | 1.4 % | 1 (the open hall ET, +27 %, with the open-plan note) | +1 % | 5/5 |
| 207 | 19/19 | 1.7 % | 1 (the small entrance ET, −25 %) | −6 % | 25/25 |

Living rooms that continue into a dining area come out 10–15 % low (207, OH); the open-plan hall of 2536 comes out
high because the model has no wall to stop at. Both are printed with a note and are the cases to discuss.

## 3. Suggested workshop timing (90 min)

| Min | Part | What to say / do |
|---|---|---|
| 0–5 | Step 0 | Everyone switches the runtime to T4 GPU and runs Step 0 (clone, SAM 3 load: 2–3 min) while you explain detection (boxes) vs. segmentation (pixels) and why pixels can be counted. Choose *Run anyway* on Colab's author warning. |
| 5–15 | Part 1 | Step 1a on the concrete-pour photo: *person*, *helmet*, *wet concrete*, *hose*, then a word that is not there; the confidences and the slider. Step 1b: a box around a worker, a click on a helmet: one object each, no words. Then Step 1c: browse the plans; point out the scale bar, the Finnish labels (legend in the notebook), and that the notebook knows every room's real area from the dataset's annotations. |
| 15–30 | Part 2 | *room* on 1293: 5 regions, 5/5 rooms found, 0 extra. Then *toilet*, *sink*. Then *kitchen*, *window* and *door*: nothing, and the hit/miss overlay shows the red windows. Key message: the number is only as good as the word, and the drawing's answer key is how you find out. |
| 30–60 | Part 3 | Step 3a on 1293 together: the bar box (zoom in), the five window boxes, the five room boxes, Submit; read the scale line, the window line, the room table and the total against the floor area. Then 2536 and 207 on their own (207 has 19 rooms and 25 windows: ten minutes). Discuss the open-plan note on 2536's hall. |
| 60–78 | Part 4 | Wording (Step 4a, *door*): *door* vs *door arc* vs *curved line* vs *arc*. Inspector: weak regions with 20–40 % confidence and the room they sit on. Own words: *thick black line*, *circle*, *small rectangle*. |
| 78–90 | Part 5 + wrap-up | Own plan through the app link if time allows (a photo of any plan works). Report template: `docs/MP4_Workshop_Report_Template.md`. |

## 4. Answer key and marking notes (100 points)

Points are suggestions. Accept any well-argued answer; the numbers below are what a correct run
produces, with small variation because boxes are drawn by hand.

1. **Words that work and words that do not (12).** Rooms, bedrooms, bathrooms, toilets, sinks, bathtubs: found (few misses); kitchen, living room, door, window, wall: nothing. Found/missed/extra for two things at 0.3, e.g. 2536 *room* 9 of 10 found with 1 extra, *window* 0 of 5. Misses have in common: the smallest rooms, symbols that are not pictures of the thing, anything drawn as a convention.
2. **The take-off, three plans (30, workshop) / three plans of their choice (30, homework).** Per plan: the scale reading within about 1 % of the drawing's; all windows found (a missed one is usually a box that did not cover the light-blue bar); every room with an error under 10 % except the open-plan hall of 2536 and the entrance of 207; the total of the rooms within a few percent of the floor area. Full marks need the reason for the worst rooms (open plan, a loose box, the mask stopping at a door opening) and the scale-squared remark. Homework: on 5018 the sheet prints KERROSALA 121 / 74 m² (gross floor area per storey) and HUONEISTOALA 98 / 68 m² (net): the sum of the rooms is a part of the net area, not the gross.
3. **Wording and weak regions (12).** Step 4a (*door*): *door* and *arc* find nothing, *door arc* nothing or little, *curved line* every swing. Step 4b: any documented weak region and mistake; the advice to a colleague (check every number against the drawing, use boxes for the take-off, never trust a phrase count).
4. **Own words (10).** *curved line* / *thick black line* / *circle* / *small rectangle* find swings, walls, toilet bowls and cabinets that *door*, *wall*, *toilet*, *cabinet* miss; a shape word works because the model was trained on photographs of things, not on the conventions that stand for them.
5. **Own plans (12; 16 in the homework).** Phrase, count, area and a judgement per plan; what failed (hand-drawn plans, photos at an angle, plans with colour fills, drawings where the scale is unknown).
6. **Reflection (24; 20 in the homework).** Useful: a first pass over many plans, counting repeated symbols, checking a room schedule; misleading: open-plan spaces, drawing conventions, anything without a scale. Needs: clean drawings, a scale bar or a known dimension, a room schedule to check against, a person who signs off.

## 5. Known failure modes

- **Step 0 is slow.** SAM 3 is 3.4 GB; on Colab it downloads in one to two minutes. If the runtime has no GPU the precomputed steps still work; live steps take about a minute each on CPU.
- **Step 0 asks for a restart.** Colab shipped an older `transformers` without SAM 3: Step 0 installs a newer one and asks for *Runtime → Restart session*; after the restart, run Step 0 again (the install is kept).
- **Box-drawing tool missing** (Steps 1b, 3a). It is a third-party widget; Step 0 enables Colab's custom widget manager. Re-run Step 0, then the step.
- **"The mask covers only N % of your box."** An open-plan space: the model has no wall to stop at. The box area printed with the note is the better estimate; that is the point of the note.
- **A phrase finds nothing at all.** Expected for kitchen, living room, balcony, stairs, door, window, wall. Send the students to *curved line* / *thick black line* (Step 4a, 4c).
- **A cell looks stuck.** *Runtime → Interrupt execution*, run the cell again. Nothing is lost.
- **The app link could not be created.** Part 5 prints a link instead of embedding the app; if the tunnel fails, run the cell again.
- **Licence.** CubiCasa5K is CC BY-NC-SA 4.0: teaching use is fine, commercial use is not, and derived overlays fall under the same licence.

## 6. Rebuilding or changing the material

See `README.md`. The plans are chosen in `aec_seg/config.py` (`plans` of each set: CubiCasa sample id,
resampling factor, title; `render=True` for the clean vector rendering, `False` for the scanned image);
after a change run `build/prepare_plans.py --dir <extracted cubicasa5k>` (or `--zip cubicasa5k.zip`),
`build/precompute_masks.py` (only new plans and phrases are computed) and `build/make_notebooks.py`.
The vocabulary is `THINGS` in `aec_seg/config.py`; each entry says what to look up in the answer key.
Check a candidate plan with `Sam3Engine.segment(img, "room")` before adding it: on some plans (CubiCasa 7696, dropped)
*room* finds almost nothing.

**Planned homework redesign.** Counting components on structural drawings: the two foundation plans of
the original MP4 notebook (`test_fp.png`, `test_fp_2.png` in the SAM_Example_Image repository) plus, if
wanted, the four public-domain foundation sheets used in an earlier version (US Army standard drawings
21-01-78 and 21-01-30, HABS Whittier hospital S1, HAER Shenandoah-Dives Mill; in git history before commit
b802e88). Measured on those sheets: *small square* finds the spread footings (score 0.8–0.85), *circle*
the grid bubbles, *hexagon* the footing tags; one box on a footing finds the rest with extras (about twice
the true count at 0.3, close to it at 0.5); a tight box on a footing gives its area.
