# MP4 · Instructor guide

## 1. What the students get

Two Colab notebooks built from one template. Students never write code: each cell is a form with a
▶ button; outputs are three-panel views (photo, mask, overlay), material-mix charts, comparison and
time-series charts, a region inspector, a box-drawing tool and a small upload app.

| | Workshop | Homework |
|---|---|---|
| Photos | structural work on site (31 photos) | interior finishing (19 photos) + 2 structural plans |
| Materials | concrete, rebar, formwork, scaffolding, structural steel, brick, soil, timber, worker, machine, sky | drywall, studs, insulation, pipes, tiles, concrete, ceiling, floor, windows, worker |
| Time series | A: house footings, 3 photos over 10 days (2021); B: one building 1937–1939, 10 photos | C: three 1938 interiors |
| Extra | — | quantity take-off of footings with box prompts and a known width |
| Own photos | 1 (optional) | 5 (required) |
| Time | about 80 min | about 2 h |

The model is SAM 3 used as is: no training. Everything the students see for the built-in photos is
precomputed and instant. Only four steps run the model live (own words, negative-box correction, plan
take-off, own photo); on a T4 each request takes about a second, on CPU about a minute.

## 2. Suggested workshop timing (80 min)

| Min | Part | What to say / do |
|---|---|---|
| 0–5 | Step 0 | Everyone runs Step 0 (clones the repo, loads SAM 3: 2–3 min) while you explain detection (boxes) vs. segmentation (pixels) and why pixels can be counted. Choose *Run anyway* on Colab's author warning. |
| 5–12 | Part 1 | Browse the photos. Point out the 1937 series: one building from excavation to completion. |
| 12–30 | Part 2 | Original → mask → overlay. Move the confidence slider and watch the share change. Material mix of one photo. Play the estimation game; ask for scores. Key message: the number is *share of the photo*, and the model's confidence threshold is a choice. |
| 30–45 | Part 3 | Compare the day-1 and day-11 footing photos (rebar and formwork down, concrete and blockwork up). Then series B: brick and sky over two years. Ask what changed for reasons that are not progress (camera position, black-and-white film, weather). |
| 45–65 | Part 4 | Wording: *rebar* finds nothing, *steel reinforcement bars* finds the cage; *steel beam* vs *steel frame*. Region inspector: weak regions with 20–40 % confidence. Negative box: draw over the wrong region and see the share drop. Then let them type their own phrases. |
| 65–75 | Part 5 | Own photos on the phone through the public link: a wall, the floor, the street outside. |
| 75–80 | Wrap-up | Report template: `docs/MP4_Workshop_Report_Template.md`. |

## 3. Answer key and marking notes (100 points)

1. **Estimation game and threshold (12).** Any score. The material mix at 0.5 must be quoted. Lowering the threshold to 0.3 adds weak regions and raises the share; 0.8 keeps only the model's surest regions. Good answers say that neither is "the truth": the threshold is a choice that trades misses against false regions.
2. **Progress over a series (14).** Series A: rebar and formwork shares fall, concrete (oversite) and brick/blockwork rise; series B: soil and formwork give way to brick and finished building, sky share changes with the camera position. Full marks for one clear non-progress cause (viewpoint, zoom, black-and-white photo, weather, people in front).
3. **Wording (12).** *steel reinforcement bars* finds the rebar cage (site_01: 33 %); *rebar*, *steel bars*, *reinforcing steel* find nothing; *metal rods* works. The model learned everyday language, not trade jargon; a descriptive phrase beats a technical term. Good answers also mention that different wordings can give different region counts and shares.
4. **Errors and correction (14).** Expect: a rebar cage labelled *scaffolding* (site_04), sheetrock on a table missed as *drywall* (int_21), concrete pours where wet concrete is not recognised as *concrete*, weak regions at 20–40 %. The negative box removes the wrong region and the share drops. Advice for the colleague: state the phrase and threshold used, check the overlay, do not compare shares across photos taken from different positions.
5. **Plan 1 take-off, homework (12).** Scale from the F12 footing: about 0.14 ft per pixel at the 1280-px width used here (any value consistent with their box). The measured area should be within about 10 % of 144 sq ft; errors come from the box edge, the mask edge on the drawing's line, and the drawing resolution.
6. **Plan 2 take-off, homework (14).** Scale from F4.0; areas of F4.0 (about 16 sq ft), E4-6, E4-10, E5-0 and the elevator shaft opening with the overlay screenshot. The opening or an irregular footing is usually hardest (the mask follows the drawing's lines, not the intended outline). Checking: measure the reference twice, compare with the schedule on the drawing, sanity-check aspect ratios.
7. **Own photos (10; 6 in the workshop).** Phrase, share and a judgement per photo; which surfaces or wordings failed (reflective floors, cluttered walls, jargon).
8. **Reflection (12; 18 in the workshop).** Useful: progress photos from a fixed camera, PPE or housekeeping checks, quick material inventories; misleading: any comparison across different viewpoints, a share mistaken for a quantity. Turning it into a quantity needs a fixed camera or drawings with a scale, a reference length in the photo, and several photos per area.

## 4. Known failure modes

- **Step 0 is slow.** SAM 3 is 3.4 GB; on Colab it downloads in one to two minutes. If the runtime has no GPU the precomputed steps still work; live steps take about a minute each on CPU.
- **Step 0 asks for a restart.** Colab shipped an older `transformers` without SAM 3: Step 0 installs a newer one and asks for *Runtime → Restart session*; after the restart, run Step 0 again (the install is kept).
- **Box-drawing tool missing** (Steps 4c and 5a). It is a third-party widget; Step 0 enables Colab's custom widget manager. Re-run Step 0, then the step.
- **A cell looks stuck.** *Runtime → Interrupt execution*, run the cell again. Nothing is lost.
- **Share link in the upload app fails.** The app still works inside the notebook; the link is only needed for phones.
- **Wikimedia photos.** All are open-licence and credited; CC BY-SA photos require the same licence for derived overlays, which is fine for teaching material.

## 5. Rebuilding or changing the material

See `README.md`. Materials and their wordings live in `aec_seg/config.py`; after a change run
`build/precompute_masks.py` (only new phrases are computed) and `build/make_notebooks.py`.
