# MP4 · Instructor guide

## 1. What the students get

Two Colab notebooks built from one template. Students never write code: each cell is a form with a
▶ button; outputs are three-panel views (plan, mask, overlay), hit/miss/extra overlays, tables of
square metres next to the drawing's own numbers, a box-drawing tool and a small upload app.

| | Workshop | Homework |
|---|---|---|
| Plans | set A: 6 homes (13828, 9493, 14466, 11032, 548, 1902) | set B: 7 homes (14341, 5018, 1217, 8138, 9136, 11615, 10715), two of them with both storeys on one sheet, one Swedish |
| Steps start with | room (any), window, bathroom, kitchen | bedroom, toilet, stairs, living room |
| Take-off targets | two rooms, one window, one door | every bedroom of a plan, the stairs, a toilet, a bathtub; the four rooms of plan 5018 |
| Own plans | 1 | 3 |
| Time | about 90 min | about 2 h |

The plans are from **CubiCasa5K** (CC BY-NC-SA 4.0). Its `F1_scaled.png` images are drawn at exactly
1 px = 1 cm and its vector annotations give every room's polygon and real size, every door, window
and fixture. `build/prepare_plans.py` resamples each plan by its own factor (so students cannot guess
the scale), adds a 5 m scale bar and writes the answer key. That key is what every "the drawing
says..." line in the notebook comes from.

Everything the students see for the built-in plans and phrases is precomputed and instant. Only the
live steps run SAM 3 (Step 3a and 3b boxes, Step 4d, 4e, Part 5); on a T4 each request takes well
under a second, on CPU about a minute.

## 2. What SAM 3 actually does on these plans (measured)

Instances at confidence ≥ 0.3, best score in brackets, on the workshop plans:

| plan | room | bedroom | bathroom | kitchen | living room | toilet | sink | stairs | door / window / wall |
|---|---|---|---|---|---|---|---|---|---|
| 13828 | 13 (0.57) | 2 (0.41) | 1 (0.72) | 0 | 0 | 2 (0.52) | 2 (0.57) | 1 (0.30) | nothing |
| 9493 | 8 (0.61) | 5 (0.38) | 0 | 0 | 0 | 0 | 1 (0.34) | 3 (0.65) | nothing |
| 14466 | 7 (0.79) | 6 (0.48) | 3 (0.85) | 0 | 0 | 2 (0.68) | 5 (0.44) | 1 (0.30) | nothing |
| 11032 | 13 (0.63) | 5 (0.56) | 5 (0.60) | 0 | 0 | 2 (0.55) | 4 (0.52) | 0 | nothing |
| 548 | 6 (0.65) | 1 (0.42) | 2 (0.57) | 0 | 0 | 0 | 0 | 6 (0.80) | nothing |
| 1902 | 1 (0.31) | 0 | 2 (0.57) | 0 | 0 | 1 (0.69) | 1 (0.40) | 0 | nothing |

The same pattern holds on set B. Read it to the class like this:

- **Words for enclosed spaces work**: *room* finds half to four fifths of the rooms and merges open-plan
  spaces; *bedroom* and *bathroom* find some. Words for functions without walls do not: *kitchen*,
  *living room* and *balcony* find nothing on any plan, whatever the confidence.
- **Symbol words work when the symbol is a picture of the thing**: *toilet* (a toilet bowl seen from
  above), *sink*, *stairs*. *Bathtub* rarely.
- **Drawing conventions are not in the model's vocabulary**: *door* (an arc), *window* (a gap with a
  line) and *wall* (a thick black line) find nothing by name. That is why Step 3b counts them from a
  box example instead.
- **The confidence slider matters**: the useful range on drawings is 0.2–0.5, much lower than on photos.
  The notebooks default to 0.3.

**Boxes (Step 3b).** A tight box around a room: SAM 3 cuts out the space, the notebook clips the mask to
the box and fills the holes left by furniture symbols. On 65 rooms (the five largest indoor rooms of
every plan, boxes drawn on the answer key's outline) the error is 8.5 % (median) and 12.9 % (mean).
Errors above 25 % come from three cases the students should recognise: an **open-plan kitchen** with no
wall on one side (the model returns the counter; the notebook prints "the mask covers only N % of your
box... the box itself is X m²"), an **L-shaped room** (a rectangle cannot fit it: the box includes the
neighbour), and a living room that **continues into a dining area**.

**Find-all (Step 3b, box example).** One box around a window finds windows with good recall but many
extras (workshop plans: 15/19 found with 22 extra, 8/8 with 16 extra, 6/7 with 4 extra, 6/17 with 56
extra, 3/3 with 4, 1/1 with 4); the extras are other short line segments and text. One box around a
door finds almost nothing (1–3 of 8–20): the door arc is too thin and varied. One box around a
fixture (a toilet or sink) finds most of the same symbols with few extras (11/16 with 1 extra, 6/6,
4/6, 9/15 with 0 extra). Doors are the honest failure case; ask students why a symbol that every
architect reads instantly is invisible to the model (it was trained on photographs of doors, not on the
convention that stands for one).

## 3. Suggested workshop timing (90 min)

| Min | Part | What to say / do |
|---|---|---|
| 0–5 | Step 0 | Everyone switches the runtime to T4 GPU and runs Step 0 (clone, SAM 3 load: 2–3 min) while you explain detection (boxes) vs. segmentation (pixels) and why pixels can be counted. Choose *Run anyway* on Colab's author warning. |
| 5–12 | Part 1 | Browse the plans. Point out the scale bar, the Finnish labels (legend in the notebook), the printed m² on some plans, and that the notebook knows every room's real area from the dataset's annotations. |
| 12–35 | Part 2 | *room* on 13828: 13 regions, merged living/dining, missed small rooms. Then *toilet*, *sink*. Then *kitchen* and *window*: nothing, and the hit/miss overlay shows 19 red windows. Step 2c: the m² table per room type against the drawing. Play the estimation game; ask for scores. Key message: the number is only as good as the word, and the drawing's answer key is how you find out. |
| 35–60 | Part 3 | Step 3a: box the scale bar (zoom in); a 1 % scale error is a 2 % area error. Step 3b: tight boxes on two rooms, read the "+4 %" / "−29 %" lines, then one box on a window with *find_all*: count the green, red and blue boxes on the plan. Then a door: nothing. Discuss what a take-off needs that the model does not have (the convention). |
| 60–78 | Part 4 | Compare 13828 and 9493. Wording: *bathroom* vs *shower room* vs *wc*. Inspector: weak regions with 20–40 % confidence and the room they sit on. Negative box on a merged region. Own words: *bed*, *small square*, *thick line*. |
| 78–90 | Part 5 + wrap-up | Own plan through the upload app if time allows (a photo of any plan works). Report template: `docs/MP4_Workshop_Report_Template.md`. |

## 4. Answer key and marking notes (100 points)

Points are suggestions. Accept any well-argued answer; the numbers below are what a correct run
produces, with small variation because boxes are drawn by hand.

1. **Words that work and words that do not (10).** Rooms, toilets, sinks and stairs: found (with misses); kitchen, living room, door, window, wall: nothing. Found/missed/extra for two things at 0.3, e.g. 13828 *toilet* 2 found of 2, *window* 0 of 19. Misses have in common: symbols drawn thin, small, or merged with furniture; a room with no closed wall.
2. **Square metres per room type and the game (12).** The Step 2c table; bedrooms usually within 10–30 % (merged pairs of bedrooms or a missed one), bathrooms often under (one of two found), kitchen and living room 0 vs the drawing. At 0.2 more regions and merged areas; at 0.7 almost nothing survives. Any game score.
3. **Room take-off (14, workshop) / plan 5018 (14, homework).** Two rooms with SAM 3's m², the drawing's m² and the error; typical errors 2–15 %, up to 50 % on open-plan or L-shaped rooms, with the reason named. Homework: on 5018 the ground-floor living room (OH) and kitchen (K) and two upstairs bedrooms (MH); the sheet prints KERROSALA 121 / 74 m² (gross floor area per storey) and HUONEISTOALA 98 / 68 m² (net apartment area): the sum of the students' rooms is a part of the net area, not the gross.
4. **Counting with find-all (14).** Workshop: windows on their plan, found/missed/extra at the best confidence (see section 2), extras are short line segments and text. Homework: toilets and bathtubs (fixtures work well: most found, few extras), windows on 1217 (30 windows: about 11 found at 0.3 with many extras), and the observation that the easiest symbol is the one that looks like the object.
5. **Compare plans and wording (12).** Step 4a: SAM 3's bedroom and bathroom m² for the two plans against the drawing's totals; the drawing decides. Step 4b: the wording that gave the most sensible mask and how far apart the areas were (e.g. *bathroom* vs *wc* vs *shower room*).
6. **A mistake and the negative box (10).** Any documented region: what was included or missed, at which confidence; whether the negative box fixed it; the advice to a colleague (check every number against the drawing, use boxes for the take-off, never trust a phrase count).
7. **Own plans (10; 16 in the homework).** Phrase, count, area and a judgement per plan; what failed (hand-drawn plans, photos at an angle, plans with colour fills, drawings where the scale is unknown).
8. **Reflection (18; 12 in the homework).** Useful: a first pass over many plans, counting repeated symbols, checking a room schedule; misleading: open-plan spaces, drawing conventions, anything without a scale. Needs: clean drawings, a scale bar or a known dimension, a room schedule to check against, a person who signs off.

## 5. Known failure modes

- **Step 0 is slow.** SAM 3 is 3.4 GB; on Colab it downloads in one to two minutes. If the runtime has no GPU the precomputed steps still work; live steps take about a minute each on CPU.
- **Step 0 asks for a restart.** Colab shipped an older `transformers` without SAM 3: Step 0 installs a newer one and asks for *Runtime → Restart session*; after the restart, run Step 0 again (the install is kept).
- **Box-drawing tool missing** (Steps 3a, 3b, 4d). It is a third-party widget; Step 0 enables Colab's custom widget manager. Re-run Step 0, then the step.
- **"The mask covers only N % of your box."** An open-plan space: the model has no wall to stop at. The box area printed with the note is the better estimate; that is the point of the note.
- **"The mask fills the whole box: draw a tighter box"** (windows, doors, fixtures). A loose box returns roughly the box contents. Zoom in and redraw.
- **A phrase finds nothing at all.** Expected for kitchen, living room, balcony, door, window, wall. Send the students to Step 3b.
- **A cell looks stuck.** *Runtime → Interrupt execution*, run the cell again. Nothing is lost.
- **Share link in the upload app fails.** The app still works inside the notebook; the link is only needed for phones.
- **Licence.** CubiCasa5K is CC BY-NC-SA 4.0: teaching use is fine, commercial use is not, and derived overlays fall under the same licence.

## 6. Rebuilding or changing the material

See `README.md`. The plans are chosen in `aec_seg/config.py` (`plans` of each set: CubiCasa sample id,
resampling factor, title); after a change run `build/prepare_plans.py --zip cubicasa5k.zip`,
`build/precompute_masks.py` (only new plans and phrases are computed) and `build/make_notebooks.py`.
The vocabulary is `THINGS` in `aec_seg/config.py`; each entry says what to look up in the answer key.
Plans where nothing works (for instance CubiCasa 11653, 20102) exist; check a candidate with
`Sam3Engine.segment(img, "room")` before adding it.
