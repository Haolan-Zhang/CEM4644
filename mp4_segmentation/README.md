# MP4 · Segmentation for a quantity take-off (no-code lab)

Teaching material for **CEM4644**: promptable segmentation with **SAM 3** on real American construction drawings.
Students set the scale from a printed dimension, box rooms, footings and symbols, and get areas in **square feet**
and counts, every one of them checked against the drawing's own answer key. Students need **no programming**: every
notebook cell is a Colab form and the code is hidden in `aec_seg/`.

| Notebook | Open | Drawings |
|---|---|---|
| `MP4_Workshop_Segmentation.ipynb` | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Haolan-Zhang/CEM4644/blob/master/mp4_segmentation/MP4_Workshop_Segmentation.ipynb) | three 1940 USDA farmhouse floor plans (easy, medium, L-shaped) |
| `MP4_Homework_Segmentation.ipynb` | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Haolan-Zhang/CEM4644/blob/master/mp4_segmentation/MP4_Homework_Segmentation.ipynb) | seven sheets from three disciplines: 2 floor plans, 4 structural foundation plans, 1 reflected ceiling plan |

Everything is in feet and square feet. **The scale is always set by the student**, from a printed dimension or from
something whose real size the sheet states — never trusted from a scale bar (one homework sheet carries a graphic
bar that is wrong by a factor of two, and finding that out is one of the exercises). There is no training exercise
in MP4: SAM 3 is used as it comes.

## What the students do

**Workshop (90 min).**
1. **Meet SAM 3** on an ordinary site photo: a phrase, a box, a click. Then browse the three drawings.
2. **Ask by name** on a drawing: original → mask → overlay for *room*, *bedroom*, *door*, *window*…, with the count,
   the area and the answer key's hits / misses / extras.
3. **The take-off**, one cell per drawing: box the printed overall dimension (the scale), box every window and every
   door, box every room. The notebook prints your scale next to the key's, your counts as found / missed / extra, and
   every room as *your sq ft | the drawing's sq ft | error % | what the sheet prints*.
4. **Where it goes wrong**: wording, the region inspector, words of their own.
5. **Their own drawing** in a small Gradio app, opened from a link.

**Homework (about 2 h).** No SAM 3 teaching: straight into a take-off across disciplines.
1. **The drawings**: all seven, what each one is, and the MEP symbol legend.
2. **The take-off**, the same cell three times: 2a floor plans, 2b structural plans, 2c MEP plans. Rooms and fixtures
   on the floor plans; footing areas and footing counts on the structural sheets; light-fixture counts from one
   example box on the ceiling plan.
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
docs/             report templates, instructor guide, a copy of the SAM License
build/            instructor-side scripts: precompute_masks.py, make_notebooks.py
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
 "counts": {"window": [[x1,y1,x2,y2], ...]},
 "count_hints": {"window": {"threshold": 0.4, "size_range": [0.2, 5.0], "tip": "..."}},
 "legend": null, "tasks": ["Box the 27'-0\" dimension ...", "..."]}
```

Pixel coordinates in the shipped image. For a room, `box` is the inside faces of the walls, `poly` the outline, and
`true_sqft` the **drawn** area (polygon area ÷ px_per_ft²) — that is the answer; `printed`/`printed_sqft` is what the
sheet prints and is shown as information. For a footing or a pit, `true_sqft` is the size its mark or the schedule
gives. `scale_refs[].use` marks the one the areas are computed with; a second ref with `use: false` is a cross-check.
`counts` keys are plain student words. `aec_seg.data.check_key` validates all of this and the notebook refuses to
load a set with a broken key: `python -m aec_seg.data data/sheets/workshop` prints the report.

## The drawings, their sources and licences

| id | set | drawing | source | licence |
|---|---|---|---|---|
| `usda_5544` | workshop | USDA design 710-5544, five-room farmhouse (Misc. Pub. 360 p. 17, 1940) | [archive.org/details/plansoffarmbuild360unit](https://archive.org/details/plansoffarmbuild360unit) | Public domain, work of the U.S. Government (17 U.S.C. 105) |
| `usda_5540` | workshop | USDA design 710-5540, four-room and attic farmhouse (p. 13, 1940); re-cropped from the source PDF so the side porch is complete | same | same |
| `usda_5539` | workshop | USDA design 710-5539, four-room farmhouse, L-shaped footprint (p. 12, 1940); stray letters cleaned from the left margin | same | same |
| `usda_5542` | homework | USDA design 710-5542, five-room farmhouse (p. 15, 1940) | same | same |
| `va_floor` | homework | VA outpatient / PACT clinic lease module, floor plan (2018) | [cfm.va.gov/til/rTemplate](https://www.cfm.va.gov/til/rTemplate/) | Public domain, US Government work ([VA copyright policy](https://department.va.gov/copyright-policy/)) |
| `va_ceiling` | homework | the same clinic, reflected ceiling plan; the symbol legend ships with it | same | same |
| `test_fp` | homework | foundation plan, instructor's example image 1 | [github.com/Haolan-Zhang/SAM_Example_Image](https://github.com/Haolan-Zhang/SAM_Example_Image) | provided by the course instructor |
| `test_fp_2` | homework | foundation plan, instructor's example image 2 | same | provided by the course instructor |
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
  39/41 light fixtures, 19/19 square footings. The example must be unrotated and uncluttered.
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
