# MP4 · Segmentation for a quantity take-off (no-code lab)

Teaching material for **CEM4644**: promptable segmentation with **SAM 3** on real American construction drawings.
Students set the scale from a printed dimension, box rooms, footings and symbols, and get areas in **square feet**
and counts, every one of them checked against the drawing's own answer key. Students need **no programming**: every
notebook cell is a Colab form and the code is hidden in `aec_seg/`.

| Notebook | Open | Drawings |
|---|---|---|
| `MP4_Workshop_Segmentation.ipynb` | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Haolan-Zhang/CEM4644/blob/master/mp4_segmentation/MP4_Workshop_Segmentation.ipynb) | three 1940 USDA farmhouse floor plans (easy, medium, L-shaped) |
| `MP4_Homework_Segmentation.ipynb` | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Haolan-Zhang/CEM4644/blob/master/mp4_segmentation/MP4_Homework_Segmentation.ipynb) | seven sheets from three disciplines: 2 floor plans, 4 structural foundation plans, 1 reflected ceiling plan |

Everything is in feet and square feet. In the workshop **the scale is set by the student** from a printed dimension;
the homework sheets use the scale stored with each drawing, so a homework take-off is boxes and nothing else. There is no training exercise
in MP4: SAM 3 is used as it comes.

## What the students do

**Workshop (90 min).**
1. **Meet SAM 3** on an ordinary site photo: a phrase, a box, a click. Then browse the three drawings.
2. **Ask by name** on a drawing: original → mask → overlay for *room*, *bedroom*, *door*, *window*…, with the count,
   the area and, for the room words, the answer key's hits / misses / extras.
3. **The take-off**, one cell per drawing: box the printed overall dimension (the scale) and the second dimension as a
   check, then box every room. The notebook prints your scale next to the key's and every room as
   *your sq ft | the drawing's sq ft | error % | what the sheet prints*. Nothing is counted in these cells; Step 3d
   then shows SAM 3's fourth prompt: **one box around one door as an example**, and the model finds every other
   door on the sheet (the key marks found / missed / extra). Doors, not windows: from any example box the model
   finds a median 11/17, 14/14 and 11/11 doors, but only 2/9 windows on the first sheet.
4. **Where it goes wrong**: wording, the region inspector, words of their own.
5. **Their own drawing** in a small Gradio app, opened from a link: the scale from a box along a printed dimension, then a phrase, a box for an area, or one example box for a count.

**Homework (about 2.5 h).** No SAM 3 teaching: straight into a take-off across disciplines, two ways on each.
1. **The drawings**: all seven, what each one is, and the MEP symbol legend.
2. **Floor plans, structural plans, MEP plans** (Parts 2, 3, 4), two cells each. *Step Na, your boxes*: the take-off
   cell. Room areas on the floor plans (plus two fixture counts on the clinic sheet); footing areas and a footing
   count on the structural sheets; light-fixture counts on the ceiling plan. **Every count is made by the model from
   one box labelled `example: <thing>`** — students never tally their own boxes against the key. *Step Nb, a phrase*:
   the same things asked for by name (`lab.ask`), no box at all; the regions are scored against the answer key
   (hits / misses / extras, and the area of every room or footing found). Measured: *room* finds most rooms, *square* the footings, *rectangle* the pile caps, *circle* the grid bubbles; *footing*,
   *light fixture* and every other trade word find nothing, and nothing at all works by phrase on the ceiling plan.
3. **Their own drawing**.

## What is in this folder

```
MP4_Workshop_Segmentation.ipynb   student notebook (generated)
MP4_Homework_Segmentation.ipynb   student notebook (generated)
aec_seg/          all the code the notebooks call (hidden from students)
  config.py       the two drawing sets and the vocabulary of phrases
  data.py         reading a sheet + its answer key (and VALIDATING the key), the mask store
  engine.py       the SAM 3 wrapper: segment (phrase), segment_box, segment_like, segment_room, segment_visual
  ui.py           every notebook step; takeoff_compute() is the whole take-off without a widget
  lab.py          the `lab` object the notebooks call
  app.py          the upload app (link only)
  viz.py          overlays and panels
data/intro/       two site photos (Wikimedia Commons) for the opening steps + credits.json
data/sheets/      workshop/ and homework/: <id>.png + <id>.key.json + credits.json (+ a legend image)
data/masks/       SAM 3 precomputed for every workshop drawing x phrase, so Part 2 and Step 4a are instant
docs/             report templates, instructor guide, Homework_Answer_Keys.md (every homework key drawn on its sheet,
                  with what SAM 3 measures on it; built by build/answer_key_report.py), a copy of the SAM License
build/            instructor-side scripts: precompute_masks.py, make_notebooks.py, answer_key_report.py
```

### The answer key (`data/sheets/<set>/<id>.key.json`)

```json
{"id": "usda_5544", "file": "usda_5544.png", "title": "Five-room farmhouse, 27 ft x 34 ft",
 "discipline": "floor plan", "units": "ft", "size": [W, H],
 "credit": {"title": "...", "author": "...", "license": "...", "source": "https://..."},
 "scale": {"px_per_ft": 20.3, "how": "the 27'-0\" dimension measures 548 px ..."},
 "scale_refs": [{"label": "27'-0\" dimension", "feet": 27.0, "box": [x1,y1,x2,y2], "axis": "x", "use": true, "note": ""}],
 "areas": [{"label": "BEDROOM", "type": "bedroom", "category": "room", "indoor": true,
            "printed": "12' x 14'", "printed_sqft": 168.0, "box": [x1,y1,x2,y2], "poly": [[x,y], ...],
            "true_sqft": 164.0, "note": ""}],
 "counts": {"footing": [[x1,y1,x2,y2], ...]},
 "count_hints": {"footing": {"threshold": 0.4, "size_range": [0.2, 5.0], "tip": "..."}},
 "legend": null, "tasks": ["Box the 27'-0\" dimension ...", "..."]}
```

Pixel coordinates in the shipped image. For a room, `box` is the inside faces of the walls, `poly` the outline, and
`true_sqft` the **drawn** area (polygon area ÷ px_per_ft²) — that is the answer; `printed`/`printed_sqft` is what the
sheet prints and is shown as information. For a footing or a pit, `true_sqft` is the size its mark or the schedule
gives. `scale_refs[].use` marks the one the areas are computed with; a second ref with `use: false` is a cross-check.
`counts` keys are plain student words, and they use the **same word as the area category** when they are the same
object (`footing`, not `spread footing` next to an area called `footing`). A `counts` entry lists every instance on
the sheet, and it exists to check the model's count: the student boxes **one** example (`example: grid bubble`; for a thing that is also measured, the first `footing` box is the example), SAM 3
finds the rest, and the key says what should have been found. A sheet with nothing to count carries `"counts": {}`.
`aec_seg.data.check_key` validates all of this and the notebook refuses to load a set with a broken key:
`python -m aec_seg.data data/sheets/workshop` prints the report.

**Box labels** (what the drawing tool offers, straight from the key): `scale: <ref label>` for each scale reference,
the plain area category (`room`, `footing`, `pit`) for something to measure, and `example: <category>` for something
to count. There is no label for "one of the things I am counting": a count a student tallies by hand against an
answer key is not a model result, so the lab does not ask for one.

## The drawings, their sources and licences

| id | set | drawing | source | licence |
|---|---|---|---|---|
| `usda_5544` | workshop | USDA design 710-5544, five-room farmhouse (Misc. Pub. 360 p. 17, 1940) | [archive.org/details/plansoffarmbuild360unit](https://archive.org/details/plansoffarmbuild360unit) | Public domain, work of the U.S. Government (17 U.S.C. 105) |
| `usda_5540` | workshop | USDA design 710-5540, four-room and attic farmhouse (p. 13, 1940); re-cropped from the source PDF so the side porch is complete | same | same |
| `usda_5539` | workshop | USDA design 710-5539, four-room farmhouse, L-shaped footprint (p. 12, 1940); stray letters cleaned from the left margin | same | same |
| `usda_5542` | homework | USDA design 710-5542, five-room farmhouse (p. 15, 1940) | same | same |
| `va_floor` | homework | VA outpatient / PACT clinic lease module, floor plan (2018) | [cfm.va.gov/til/rTemplate](https://www.cfm.va.gov/til/rTemplate/) | Public domain, US Government work ([VA copyright policy](https://department.va.gov/copyright-policy/)) |
| `va_ceiling` | homework | the same clinic, reflected ceiling plan; the symbol legend ships with it | same | same |
| `test_fp` | homework | foundation plan, the course's example drawing 1 | [github.com/Haolan-Zhang/SAM_Example_Image](https://github.com/Haolan-Zhang/SAM_Example_Image) | provided for this course |
| `test_fp_2` | homework | foundation plan, the course's example drawing 2 | same | provided for this course |
| `uscg_motorpool` | homework | US Coast Guard Motor Pool Facility, Support Center New York, shop building foundation plan S-1 (1984); cropped to keep the footing schedule | [Wikimedia Commons / DPLA](https://commons.wikimedia.org/wiki/File:Building_928_Structural_Foundation_Plan,_Details_and_Notes_Shop_Building,_June_20,_1984_-_DPLA_-_2938a6cfb0663504470f4b44fe7ae27e.tiff) | Public domain, US Government work |
| `uscg_pile` | homework | US Coast Guard Building 785 bowling facility, pile footing plan (1983) | [Wikimedia Commons / DPLA](https://commons.wikimedia.org/wiki/File:Building_785_Sixteen_Lane_Bowling_Facility_Foundation_Plan,_January_19,_1983_-_DPLA_-_ec587aa2a393d5917cb73be19a2da195.tiff) | Public domain, US Government work |

The USDA sheets were rendered from the scanned publication, cropped to the plan and cleaned of page artefacts; the
VA sheets were rendered from the VA's own template PDF. Every sheet keeps its own pixel size; nothing is rescaled to
a round number, so the scale has to be measured. Each `credits.json` repeats the credit of every sheet in the set.

**Model.** SAM 3 (Meta, Nov 2025). The official checkpoint `facebook/sam3` is behind a manual approval form, so the
notebook loads the public mirror `jetjodh/sam3` (identical weights and config, no account needed); the env var
`AEC_SEG_MODEL_ID` points it at a local copy. The SAM License allows redistribution with a copy of the licence,
which is in `docs/SAM_LICENSE.txt`. Precomputed masks were produced with the same model at score threshold 0.1 and
the notebook applies the student's threshold when reading them.

## What SAM 3 can and cannot do on a construction drawing

Measured on these sheets (numbers per drawing are in `docs/Instructor_Guide.md`):

- **Enclosed spaces work.** A box plus the phrase *empty room* (`segment_room`) traces a room to its walls. The raw
  mask runs about 10 % low because door swings, counters, bathtubs and fireplaces are cut out of it, so the take-off
  gives the bites back on rooms that are rectangles and leaves odd-shaped spaces alone.
- **Trade words return nothing.** *footing*, *pile*, *grid bubble*, *light fixture*, *diffuser*, *sprinkler*,
  *window*, *door*, *wall* find nothing at any confidence. **Shape words do**: *curved line* finds door swings,
  *thick black line* the walls, *square* / *small circle* the footings and the fixture symbols.
- **Repeated symbols are counted from one example box** (`segment_like`), not from a phrase: 29/29 pile footings,
  39/41 light fixtures, 19/19 footings on `test_fp_2`. The example must be unrotated and uncluttered. This is the
  only way a count is produced in the lab: a student's own boxes are never tallied against the answer key, because a
  tally of hand-drawn boxes says nothing about the model.
- **A symbol under about 60 px on the sheet cannot be measured**: the mask becomes a rounded copy of the drawn box.
  The take-off cell says so instead of printing a number with no warning.
- **Use `segment_visual`, not `segment_box`, for the area of one object**: with a student's loose box the concept
  head returns the box itself (+95 %), the tracker head still returns the object (+4 %).

## Rebuilding (instructors only)

```bash
pip install "transformers>=4.57.2" torch pillow numpy scipy matplotlib ipywidgets nbformat jupyter-bbox-widget
python -m aec_seg.data data/sheets/workshop data/sheets/homework   # check the answer keys
python build/precompute_masks.py                                   # SAM 3 on every workshop drawing x phrase (GPU)
python build/make_notebooks.py                                     # both notebooks + both report templates
```

To change the drawings: put `<id>.png` and `<id>.key.json` in `data/sheets/<set>/`, add the id to that set's
`credits.json`, then rerun `precompute_masks.py` (workshop only) and `make_notebooks.py`. The notebooks read the
sheet list, the labels of the box tool and the task text straight from the answer keys, so nothing else has to
change. To change the vocabulary of Part 2 and Step 4a: edit `THINGS` in `aec_seg/config.py` and rerun
`precompute_masks.py` (only new phrases are computed).
