# MP4 · Instructor guide

## 1. What the students get

Two Colab notebooks built from one template. Students never write code: each cell is a form with a
▶ button; outputs are three-panel views (photo, mask, overlay), material-mix charts, comparison and
time-series charts, a region inspector, a box-drawing tool and a small upload app.

| | Workshop | Homework |
|---|---|---|
| Photos | structural work on site (31 photos) + 2 foundation plans | interior finishing (19 photos) + 2 foundation plans |
| Materials | concrete, rebar, formwork, scaffolding, structural steel, brick, soil, timber, worker, machine, sky | drywall, studs, insulation, pipes, tiles, concrete, ceiling, floor, windows, worker |
| Time series | A: house footings, 3 photos over 10 days (2021); B: one building 1937–1939, 10 photos | C: three 1938 interiors |
| Take-off (Part 5) | US Army 1956 dormitory (grid of square footings F1–F6, footing schedule, 19'-4" bays) and mess hall (14 column footings of 6'-0" on 24'-0" bays) | HABS Whittier hospital (36 columns on octagonal footings, dimension strings) and HAER Shenandoah-Dives Mill (CAD plan with a 50-ft scale bar and two circular tanks) |
| Own photos | 1 (optional) | 5 (required) |
| Time | about 90 min | about 2 h |

The model is SAM 3 used as is: no training. Everything the students see for the built-in photos is
precomputed and instant. Only five steps run the model live (own words, negative-box correction, words on a
drawing, plan take-off, own photo); on a T4 each request takes about a second, on CPU about a minute.

## 2. Suggested workshop timing (90 min)

| Min | Part | What to say / do |
|---|---|---|
| 0–5 | Step 0 | Everyone runs Step 0 (clones the repo, loads SAM 3: 2–3 min) while you explain detection (boxes) vs. segmentation (pixels) and why pixels can be counted. Choose *Run anyway* on Colab's author warning. |
| 5–12 | Part 1 | Browse the photos. Point out the 1937 series: one building from excavation to completion. |
| 12–30 | Part 2 | Original → mask → overlay. Move the confidence slider and watch the share change. Material mix of one photo. Play the estimation game; ask for scores. Key message: the number is *share of the photo*, and the model's confidence threshold is a choice. |
| 30–45 | Part 3 | Compare the day-1 and day-11 footing photos (rebar and formwork down, concrete and blockwork up). Then series B: brick and sky over two years. Ask what changed for reasons that are not progress (camera position, black-and-white film, weather). |
| 45–65 | Part 4 | Wording: *rebar* finds nothing, *steel reinforcement bars* finds the cage; *steel beam* vs *steel frame*. Region inspector: weak regions with 20–40 % confidence. Negative box: draw over the wrong region and see the share drop. Then let them type their own phrases. |
| 65–83 | Part 5 | Real foundation plans. Step 5a: *footing* finds nothing; *small square* finds the footings on the dormitory and the *columns* (16" squares) on the mess hall; *circle* finds the grid bubbles. Say it out loud: the model sees shapes, the engineer supplies the meaning. Step 5b on the mess hall: reference box across the 24'-0" bay E–F, a tight box on one column footing (6'-0" square = 36 sq ft; they get 37–39), then *find_all*: about 22–24 similar regions for 14 real footings. Have them count on the drawing and name the extras (boiler-room squares, detail boxes). |
| 83–90 | Part 6 + wrap-up | Own photos on the phone through the public link if time allows. Report template: `docs/MP4_Workshop_Report_Template.md`. |

## 3. Answer key and marking notes (100 points)

1. **Estimation game and threshold (12).** Any score. The material mix at 0.5 must be quoted. Lowering the threshold to 0.3 adds weak regions and raises the share; 0.8 keeps only the model's surest regions. Good answers say that neither is "the truth": the threshold is a choice that trades misses against false regions.
2. **Progress over a series (14).** Series A: rebar and formwork shares fall, concrete (oversite) and brick/blockwork rise; series B: soil and formwork give way to brick and finished building, sky share changes with the camera position. Full marks for one clear non-progress cause (viewpoint, zoom, black-and-white photo, weather, people in front).
3. **Wording (12).** *steel reinforcement bars* finds the rebar cage (site_01: 33 %); *rebar*, *steel bars*, *reinforcing steel* find nothing; *metal rods* works. The model learned everyday language, not trade jargon; a descriptive phrase beats a technical term. Good answers also mention that different wordings can give different region counts and shares.
4. **Errors and correction (14).** Expect: a rebar cage labelled *scaffolding* (site_04), sheetrock on a table missed as *drywall* (int_21), concrete pours where wet concrete is not recognised as *concrete*, weak regions at 20–40 %. The negative box removes the wrong region and the share drops. Advice for the colleague: state the phrase and threshold used, check the overlay, do not compare shares across photos taken from different positions.
5. **Plan take-off A (12).** *Workshop, plan_dormitory:* *footing* finds nothing, *small square* finds most footings (about 45 at confidence 0.4, with extras), *circle* finds the 9 grid bubbles. Scale from the 19'-4" bay between bubbles 1 and 2: about 126 px, so 0.15 ft per pixel (6.5 px per foot). The footings are small on this drawing (an F2 of 4'-6" is only about 30 px), and on a line drawing the mask follows the box more than the outline, so a box drawn exactly on the outline gives 20–22 sq ft for F2 and a loose box gives 40–50 sq ft with the ⚠ note. Full marks for reading the mark, looking it up in the schedule, and naming the box as the main error source. *Homework, plan_whittier:* scale from the two 12'-0" bays on the bottom dimension line (about 14.3 px per foot at this size). The octagonal footing of columns 7/11/15/19 is 5'-4" across, i.e. about 23.6 sq ft; a tight box gives 25–27 sq ft. *find_all* from one octagon finds about 14 similar regions (the interior octagons) and misses the wall footings, which are drawn cut by the wall lines.
6. **Plan take-off B (14).** *Workshop, plan_mess_hall:* reference across the 24'-0" bay E–F (about 225 px, 0.107 ft per pixel); one column footing (dashed 6'-0" square) measures 37–39 sq ft against 36; *find_all* at confidence 0.3 returns about 22–24 regions of similar size for 14 real column footings (two rows of seven in the mess hall basement): the extras are the boiler-room and transformer-vault squares and detail boxes; at 0.4 it drops to about 17. Expect the count, the total (14 × 36 = 504 sq ft) and a list of extras. *Homework, plan_mill:* *tank* finds nothing, *circle* finds the two tanks and the hatched circle at confidence 0.9. Reference along the 0–50 FEET scale bar (about 158 px, 0.32 ft per pixel); one tank is about 52 ft across, so a circle of 2,100 sq ft; the mask gives about 1,880 sq ft because it stops at the internal lines. Checking the scale: measure the bar twice, or measure the METERS bar (15 m = 49.2 ft) and compare.
7. **Own photos (10; 6 in the workshop).** Phrase, share and a judgement per photo; which surfaces or wordings failed (reflective floors, cluttered walls, jargon).
8. **Reflection (12; 18 in the workshop).** Useful: progress photos from a fixed camera, PPE or housekeeping checks, quick material inventories; misleading: any comparison across different viewpoints, a share mistaken for a quantity. Turning it into a quantity needs a fixed camera or drawings with a scale, a reference length in the photo, and several photos per area.

## 4. Known failure modes

- **Step 0 is slow.** SAM 3 is 3.4 GB; on Colab it downloads in one to two minutes. If the runtime has no GPU the precomputed steps still work; live steps take about a minute each on CPU.
- **Step 0 asks for a restart.** Colab shipped an older `transformers` without SAM 3: Step 0 installs a newer one and asks for *Runtime → Restart session*; after the restart, run Step 0 again (the install is kept).
- **Box-drawing tool missing** (Steps 4c and 5b). It is a third-party widget; Step 0 enables Colab's custom widget manager. Re-run Step 0, then the step.
- **A cell looks stuck.** *Runtime → Interrupt execution*, run the cell again. Nothing is lost.
- **Share link in the upload app fails.** The app still works inside the notebook; the link is only needed for phones.
- **Words on a drawing find nothing.** Expected: construction words (*footing*, *column*, *pile*, *tank*) find nothing on a line drawing; shape words (*small square*, *circle*, *rectangle*) do. On the mess hall *small square* finds the 16" columns, not the 6-ft footings around them.
- **The take-off area follows the box.** On a drawing SAM 3 snaps to a clear closed outline (dashed footing square, octagon, tank circle) but on small or faint symbols it returns roughly the box contents; the ⚠ note fires when the mask fills more than 85 % of the box. Tighter box, or measure a bigger element.
- **find_all over-counts.** It returns every region that looks like the example; the size filter keeps the same size class, but schedule boxes, detail drawings and room outlines of similar size get in. That is the error analysis the question asks for. Full sheets (title block, sections, schedules) confuse it completely, which is why only plan-view regions are used.
- **Wikimedia photos.** All are open-licence and credited; CC BY-SA photos require the same licence for derived overlays, which is fine for teaching material.

## 5. Rebuilding or changing the material

See `README.md`. Materials and their wordings live in `aec_seg/config.py`; after a change run
`build/precompute_masks.py` (only new phrases are computed) and `build/make_notebooks.py`. The four plans
and their crop regions are defined in `build/curate_plans.py`; their references, targets and expected values in
the `plans` set of `aec_seg/config.py`; which plans each notebook gets, in `VARIANTS` of `build/make_notebooks.py`.
