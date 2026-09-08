# MP4 · Segmentation for progress and quantity measurement (no-code lab)

Teaching material for **CEM4644**: promptable segmentation with **SAM 3**, materials measured as a share of
a site photo, comparison across photos and through time, error analysis by wording and confidence,
and quantity take-off of footings on structural plans. Students need **no programming**: every notebook
cell is a Colab form and the code is hidden.

| Notebook | Open | Photos | Extra |
|---|---|---|---|
| `MP4_Workshop_Segmentation.ipynb` | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Haolan-Zhang/CEM4644/blob/master/mp4_segmentation/MP4_Workshop_Segmentation.ipynb) | structural work on site: concrete, rebar, formwork, steel, scaffolding, brick; two time series | — |
| `MP4_Homework_Segmentation.ipynb` | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Haolan-Zhang/CEM4644/blob/master/mp4_segmentation/MP4_Homework_Segmentation.ipynb) | interior finishing: drywall, studs, insulation, pipes, tiles | footing take-off on two structural plans (box prompts + scale) |

Both notebooks are generated from one template. There is no training exercise in MP4 (SAM 3 is used as is).

## What is in this folder

```
MP4_Workshop_Segmentation.ipynb   student notebook (generated)
MP4_Homework_Segmentation.ipynb   student notebook (generated)
aec_seg/          all the code the notebooks call (hidden from students)
data/photos/      the photos (Wikimedia Commons, open licences; credits.json in each folder) and the two plans
data/masks/       SAM 3 results precomputed for every photo x material (and alternative wordings), so the
                  notebook is instant and works without a GPU for everything except the live steps
models/           nothing: SAM 3 is downloaded from the Hugging Face Hub in Step 0 (about 3.4 GB)
docs/             report templates, instructor guide, a copy of the SAM License
build/            instructor-side scripts: curate_photos.py, postprocess_photos.py, precompute_masks.py, make_notebooks.py
```

**Model.** SAM 3 (Meta, Nov 2025). The official checkpoint `facebook/sam3` is behind a manual approval form,
so the notebook loads the public mirror `jetjodh/sam3` (identical weights and config, no account needed).
The SAM License allows redistribution with a copy of the licence, which is in `docs/SAM_LICENSE.txt`.
Precomputed masks were produced with the same model at score threshold 0.2; the notebook applies the
student's threshold when reading them.

## Rebuilding (instructors only)

```bash
pip install "transformers>=4.57.2" torch pillow numpy matplotlib ipywidgets nbformat jupyter-bbox-widget
python build/curate_photos.py --raw <cache>          # downloads from Wikimedia Commons (thumbnail CDN), writes credits.json
python build/postprocess_photos.py                   # drops weak photos, crops the 1937 album scans
python build/precompute_masks.py --sets site interior
python build/make_notebooks.py
```

To add photos: append a line to `PHOTOS` in `build/curate_photos.py` (id, title, author, licence, URL), rerun the
three scripts. To change the material vocabulary: edit `aec_seg/config.py` and rerun `precompute_masks.py`
(only new phrases are computed).

## Sources and licences

| Item | Source | Licence |
|---|---|---|
| Photos | Wikimedia Commons, individual credits in `data/photos/*/credits.json` and at the bottom of each notebook | Public domain, CC0, CC BY 2.0/3.0/4.0, CC BY-SA 2.0/3.0/4.0 (as credited) |
| Plan drawings | course material (Haolan-Zhang/SAM_Example_Image) | course material |
| SAM 3 | https://huggingface.co/facebook/sam3 via https://huggingface.co/jetjodh/sam3 | SAM License (Meta), copy in `docs/SAM_LICENSE.txt` |
