# MP2 · Image classification for construction (no-code lab)

Teaching material for **CEM4644**: binary defect / no-defect classification, multi-class
classification, confidence and error analysis, and a small training exercise. Students need
**no programming**: every notebook cell is a Colab form (drop-downs, sliders, buttons) and the
code is hidden.

| Notebook | Open | Binary task | Multi-class task |
|---|---|---|---|
| `MP2_Workshop_Image_Classification.ipynb` | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Haolan-Zhang/CEM4644/blob/master/mp2_image_classification/MP2_Workshop_Image_Classification.ipynb) | building façade: defect vs. no defect | façade defect type (7 classes) |
| `MP2_Homework_Image_Classification.ipynb` | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Haolan-Zhang/CEM4644/blob/master/mp2_image_classification/MP2_Homework_Image_Classification.ipynb) | concrete surface: crack vs. no crack | architectural style (10 classes) |

Both notebooks are generated from the same template; they differ only in the datasets, the
task names and two report questions.

### Alternative copies with a Gradio step

Same notebooks, same questions, but Step 3b (*Break it yourself*) is a small Gradio app rendered
inside the cell instead of slider widgets. It gains a drawing pad (paint a crack or a stain with a
brush) and an adversarial-attack tab (invisible pixel noise computed from the model's own gradient).
Everything else is identical, so the two versions can be swapped without changing the report.

| Notebook | Open |
|---|---|
| `MP2_Workshop_Image_Classification_gradio.ipynb` | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Haolan-Zhang/CEM4644/blob/master/mp2_image_classification/MP2_Workshop_Image_Classification_gradio.ipynb) |
| `MP2_Homework_Image_Classification_gradio.ipynb` | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Haolan-Zhang/CEM4644/blob/master/mp2_image_classification/MP2_Homework_Image_Classification_gradio.ipynb) |

## What is in this folder

```
MP2_Workshop_Image_Classification.ipynb   student notebook (generated)
MP2_Homework_Image_Classification.ipynb   student notebook (generated)
MP2_*_Image_Classification_gradio.ipynb   the same notebooks with a Gradio app in Step 3b (generated)
aec_lab/          all the code the notebooks call (hidden from students)
data/             small image sets shipped as zips + the "tricky photos" galleries
models/           course models (ConvNeXt V2 femto, fp16) + the base model for the training exercise
docs/             report templates, instructor guide with answer key
build/            instructor-side scripts to rebuild data, models, galleries and notebooks
```

The student notebook clones this repository into the Colab session (about 100 MB) and imports
`aec_lab`. Nothing is hosted anywhere else: no Hugging Face account, no Google Drive.
The only runtime download is `pip install gradio` (used by the Step 3d upload app, and by Step 3b
in the Gradio copies). The notebooks ask students to switch the runtime to a T4 GPU first.

## Rebuilding (instructors only)

```bash
pip install torch torchvision transformers datasets pyarrow pillow scikit-image ipywidgets nbformat pandas matplotlib
python build/prepare_datasets.py --hf-cache <hf_cache> --work <work> --styles-plates <plates> --styles-manifest <manifest>
python build/train_course_models.py --work <work>          # a few minutes on a GPU
python build/make_tricky.py --work <work>
python build/make_notebooks.py                              # regenerates both notebooks + report templates
python build/make_notebooks.py --gradio                     # regenerates the two *_gradio copies
```

To swap a dataset: add a `DatasetSpec` in `aec_lab/config.py`, produce `data/<key>.zip` with the
layout `<key>/{train,test}/<class>/*.jpg` (+ `about.json`), train a model for it, build its
tricky gallery, and point a variant in `build/make_notebooks.py` at it.

## Data and model sources

| Set | Source | Licence |
|---|---|---|
| Façade defects (BD3) | https://github.com/Praveenkottari/BD3-Dataset via https://huggingface.co/datasets/chandrabhuma/building_defect_vqa | CC-BY-4.0 |
| Concrete cracks | https://huggingface.co/datasets/mohammadnajeeb/concrete_crack_images (Özgenel 2019) | CC-BY-4.0 |
| Façade styles (synthetic) | https://huggingface.co/datasets/Jonathandav/facade-styles | MIT |
| Base model | https://huggingface.co/facebook/convnextv2-femto-1k-224 | Apache-2.0 |
| Out-of-scope sample images | scikit-image `data` module | public domain / CC0 |
