# MP4 · Segmentation for quantity take-off on floor plans (no-code lab)

Teaching material for **CEM4644**: promptable segmentation with **SAM 3** on real architectural floor plans.
Students ask for rooms, fixtures and openings by name, measure rooms in square metres with boxes, count windows
and doors, compare plans, and examine the model's errors, always against the drawing's own answer key.
Students need **no programming**: every notebook cell is a Colab form and the code is hidden.

| Notebook | Open | Plans |
|---|---|---|
| `MP4_Workshop_Segmentation.ipynb` | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Haolan-Zhang/CEM4644/blob/master/mp4_segmentation/MP4_Workshop_Segmentation.ipynb) | set A: six homes (houses with furniture, apartments, a studio) |
| `MP4_Homework_Segmentation.ipynb` | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Haolan-Zhang/CEM4644/blob/master/mp4_segmentation/MP4_Homework_Segmentation.ipynb) | set B: six other homes, including two-storey sheets and a Swedish plan; other things to segment (bedrooms, toilets, bathtubs, stairs) |

Both notebooks are generated from one template and differ in the plans and in what the steps start with.
There is no training exercise in MP4 (SAM 3 is used as is).

## What the students do

1. **Meet the plans**: browse six plans with their facts (rooms, floor area, doors, windows) and a legend of the Finnish room labels.
2. **Ask by name**: original → mask → overlay for *room*, *bedroom*, *toilet*, *window*...; count, pixels, square metres; hits, misses and extras drawn against the answer key; square metres per room type next to the drawing's; an area-estimation game.
3. **Take-off with boxes**: check the scale on the 5 m bar; draw tight boxes around rooms (area in m², checked against the drawing) and around a window or door, then *find_all* counts every look-alike (found / missed / extra).
4. **Compare and examine errors**: two plans side by side; wording; region inspector; correction with a negative box; own words.
5. **Own plan**: upload any floor plan in a small Gradio app.

## What is in this folder

```
MP4_Workshop_Segmentation.ipynb   student notebook (generated)
MP4_Homework_Segmentation.ipynb   student notebook (generated)
aec_seg/          all the code the notebooks call (hidden from students)
data/plans/       the plans (PNG with a scale bar) and their answer keys (<id>.key.json: rooms with real areas
                  and polygons, doors, windows, fixtures, walls, scale) + credits.json
data/masks/       SAM 3 results precomputed for every plan x phrase (and alternative wordings), so the
                  notebook is instant and works without a GPU for everything except the live steps
models/           nothing: SAM 3 is downloaded from the Hugging Face Hub in Step 0 (about 3.4 GB)
docs/             report templates, instructor guide, a copy of the SAM License
build/            instructor-side scripts: prepare_plans.py, precompute_masks.py, make_notebooks.py
```

**Plans.** Twelve plans from **CubiCasa5K** (Kalervo et al. 2019, CC BY-NC-SA 4.0, https://zenodo.org/records/2613548),
the `high_quality_architectural` subset. The dataset's `F1_scaled.png` images are drawn at exactly 1 pixel = 1 cm and
come with vector annotations (room polygons with real dimensions, doors, windows, fixtures). `build/prepare_plans.py`
resamples each plan to its own scale (so the scale is not a round number), draws a 5 m scale bar, and writes the answer
key. Every check the notebook prints ("the drawing says 12.9 m²", "found 7 of 11 windows") comes from those annotations.

**Model.** SAM 3 (Meta, Nov 2025). The official checkpoint `facebook/sam3` is behind a manual approval form, so the
notebook loads the public mirror `jetjodh/sam3` (identical weights and config, no account needed). The SAM License
allows redistribution with a copy of the licence, which is in `docs/SAM_LICENSE.txt`. Precomputed masks were produced
with the same model at score threshold 0.1; the notebook applies the student's threshold when reading them.

**What SAM 3 can and cannot do on a drawing** (measured, see the instructor guide): room words work (*room*, *bedroom*,
*bathroom*, *kitchen*) but merge open-plan spaces and miss small rooms; symbol words work for *toilet*, *sink*, *stairs*;
*door* and *window* find nothing by name, so they are counted with a box example (`Sam3Engine.segment_like`); *wall* is
not a concept the model has. A tight box on a room snaps to the walls and gives the area within a few percent.

## Rebuilding (instructors only)

```bash
pip install "transformers>=4.57.2" torch pillow numpy matplotlib ipywidgets nbformat jupyter-bbox-widget
python build/prepare_plans.py --zip cubicasa5k.zip     # from Zenodo (5.5 GB); writes data/plans/*
python build/precompute_masks.py                        # SAM 3 on every plan x phrase (needs a GPU, a few minutes)
python build/make_notebooks.py
```

To change the plans: edit the `plans` dictionaries in `aec_seg/config.py` (CubiCasa sample id, resampling factor,
title), rerun the three scripts. To change the vocabulary: edit `THINGS` in `aec_seg/config.py` and rerun
`precompute_masks.py` (only new phrases are computed).

## Sources and licences

| Item | Source | Licence |
|---|---|---|
| Floor plans and answer keys | CubiCasa5K, https://zenodo.org/records/2613548 (CubiCasa Oy; Kalervo, Ylioinas, Häikiö, Karhu, Kannala, "CubiCasa5K: A Dataset and an Improved Multi-Task Model for Floorplan Image Analysis", SCIA 2019) | CC BY-NC-SA 4.0 (non-commercial teaching use; derived overlays under the same licence) |
| SAM 3 | https://huggingface.co/facebook/sam3 via https://huggingface.co/jetjodh/sam3 | SAM License (Meta), copy in `docs/SAM_LICENSE.txt` |
