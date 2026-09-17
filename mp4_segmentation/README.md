# MP4 · Segmentation for quantity take-off on floor plans (no-code lab)

Teaching material for **CEM4644**: promptable segmentation with **SAM 3** on real architectural floor plans.
Students ask for rooms, fixtures and openings by name, measure rooms in square metres with boxes, count windows
and doors, compare plans, and examine the model's errors, always against the drawing's own answer key.
Students need **no programming**: every notebook cell is a Colab form and the code is hidden.

| Notebook | Open | Plans |
|---|---|---|
| `MP4_Workshop_Segmentation.ipynb` | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Haolan-Zhang/CEM4644/blob/master/mp4_segmentation/MP4_Workshop_Segmentation.ipynb) | set A: three homes drawn cleanly from the dataset's vector data (two flats, a large house) |
| `MP4_Homework_Segmentation.ipynb` | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Haolan-Zhang/CEM4644/blob/master/mp4_segmentation/MP4_Homework_Segmentation.ipynb) | set B: seven scanned plans, including two-storey sheets and a Swedish plan; other things to segment (bedrooms, toilets, bathtubs, stairs). To be redesigned around counting components on structural plans. |

Both notebooks are generated from one template and differ in the plans and in what the steps start with.
There is no training exercise in MP4 (SAM 3 is used as is).

## What the students do

1. **Meet SAM 3**: on an ordinary site photo, ask by phrase (a dropdown or their own words), draw a box, tap an object; then browse the six plans with their facts (rooms, floor area, doors, windows) and a legend of the Finnish room labels.
2. **Ask by name**: original → mask → overlay for *room*, *bedroom*, *toilet*, *window*...; count, pixels, square metres; hits, misses and extras drawn against the answer key.
3. **Take-off with boxes**, one cell per plan: a box on the 5 m bar gives the scale, a box on every window gives the count (checked against the answer key), a tight box on every room gives its area (SAM 3 asked for *empty room* with the box as the example, symbol holes filled, walls removed, the student's scale applied), each next to the drawing's own area and the total against the floor area.
4. **Where it goes wrong**: wording; region inspector; own words.
5. **Own plan**: upload any floor plan in a small Gradio app, opened from a link.

## What is in this folder

```
MP4_Workshop_Segmentation.ipynb   student notebook (generated)
MP4_Homework_Segmentation.ipynb   student notebook (generated)
aec_seg/          all the code the notebooks call (hidden from students)
data/intro/       two site photos (Wikimedia Commons) for the opening steps + credits.json
data/plans/       the plans (PNG with a scale bar; set A rendered from the vector drawing, set B the scanned image)
                  and their answer keys (<id>.key.json: rooms with real areas and polygons, doors, windows,
                  fixtures, walls, scale) + credits.json
data/masks/       SAM 3 results precomputed for every plan x phrase (and alternative wordings), so the
                  notebook is instant and works without a GPU for everything except the live steps
models/           nothing: SAM 3 is downloaded from the Hugging Face Hub in Step 0 (about 3.4 GB)
docs/             report templates, instructor guide, a copy of the SAM License
build/            instructor-side scripts: prepare_plans.py, precompute_masks.py, make_notebooks.py
```

**Plans.** Ten plans from **CubiCasa5K** (Kalervo et al. 2019, CC BY-NC-SA 4.0, https://zenodo.org/records/2613548),
the `high_quality_architectural` subset. Every sample has a vector drawing (`model.svg`: walls, windows, door swings,
fixture symbols, room polygons with real dimensions) drawn at exactly 1 unit = 1 cm, and a scanned image aligned with it.
Set A (workshop) shows a clean rendering of the vector drawing (cairosvg; the "UNDEFINED" labels of unnamed rooms
dropped), on which SAM 3 works far better than on the scans; set B (homework) still shows the scanned image.
`build/prepare_plans.py` resamples each plan to its own scale (so the scale is not a round number), draws a 5 m scale
bar, and writes the answer key, composing the SVG's nested transforms so that every symbol lands where it is drawn. Every
check the notebook prints ("the drawing says 12.9 m²", "found 5 of 5 windows") comes from those annotations.

**Model.** SAM 3 (Meta, Nov 2025). The official checkpoint `facebook/sam3` is behind a manual approval form, so the
notebook loads the public mirror `jetjodh/sam3` (identical weights and config, no account needed). The SAM License
allows redistribution with a copy of the licence, which is in `docs/SAM_LICENSE.txt`. Precomputed masks were produced
with the same model at score threshold 0.1; the notebook applies the student's threshold when reading them.

**What SAM 3 can and cannot do on a clean drawing** (measured, see the instructor guide): *room*, *bedroom*, *bathroom*
find the rooms (scores 0.5–0.9, few extras) but merge open-plan spaces and miss the smallest rooms; *kitchen* and
*living room* find nothing; symbol words work for *toilet*, *sink*, *bathtub*; *door* and *wall* find nothing but the
shape words *curved line* and *thick black line* find the door swings and the walls; *window* finds nothing by any word,
so windows are counted from a box example (`Sam3Engine.segment_like`). A bare box on a room returns the furniture symbols,
so rooms are measured with the phrase *empty room* plus the box (`Sam3Engine.segment_room`): median error 4.5 % on 29
rooms with student-like boxes.

## Rebuilding (instructors only)

```bash
pip install "transformers>=4.57.2" torch pillow numpy scipy matplotlib ipywidgets nbformat jupyter-bbox-widget cairosvg
python build/prepare_plans.py --zip cubicasa5k.zip     # from Zenodo (5.5 GB), or --dir <extracted folder>; writes data/plans/*
python build/precompute_masks.py                        # SAM 3 on every plan x phrase (needs a GPU, a few minutes)
python build/make_notebooks.py
```

To change the plans: edit the `plans` dictionaries in `aec_seg/config.py` (CubiCasa sample id, resampling factor,
title; `render=True` on the set for the clean vector rendering), rerun the three scripts. To change the vocabulary: edit `THINGS` in `aec_seg/config.py` and rerun
`precompute_masks.py` (only new phrases are computed).

## Sources and licences

| Item | Source | Licence |
|---|---|---|
| Floor plans and answer keys | CubiCasa5K, https://zenodo.org/records/2613548 (CubiCasa Oy; Kalervo, Ylioinas, Häikiö, Karhu, Kannala, "CubiCasa5K: A Dataset and an Improved Multi-Task Model for Floorplan Image Analysis", SCIA 2019) | CC BY-NC-SA 4.0 (non-commercial teaching use; derived overlays under the same licence) |
| SAM 3 | https://huggingface.co/facebook/sam3 via https://huggingface.co/jetjodh/sam3 | SAM License (Meta), copy in `docs/SAM_LICENSE.txt` |
