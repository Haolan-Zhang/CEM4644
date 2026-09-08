# MP3 · Object detection for construction (no-code lab)

Teaching material for **CEM4644**: object detection with YOLO, worker / PPE monitoring, confidence
thresholds, missed and false detections, a boxes-to-decisions dashboard and a small training
exercise. Students need **no programming**: every notebook cell is a Colab form and the code is hidden.

| Notebook | Open | Dataset |
|---|---|---|
| `MP3_Workshop_Object_Detection.ipynb` | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Haolan-Zhang/CEM4644/blob/master/mp3_object_detection/MP3_Workshop_Object_Detection.ipynb) | workers and PPE: person, helmet / NO helmet, vest / NO vest |
| `MP3_Homework_Object_Detection.ipynb` | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Haolan-Zhang/CEM4644/blob/master/mp3_object_detection/MP3_Homework_Object_Detection.ipynb) | construction machinery: excavator, dump truck, wheel loader |

Both notebooks are generated from one template and differ only in the dataset, the counting
question, the dashboard and two report questions.

## What is in this folder

```
MP3_Workshop_Object_Detection.ipynb   student notebook (generated)
MP3_Homework_Object_Detection.ipynb   student notebook (generated)
aec_det/          all the code the notebooks call (hidden from students)
data/             YOLO-format image sets shipped as zips + the "tricky photos" galleries
models/           yolo11n.pt (COCO-pretrained base) and the two fine-tuned course detectors
docs/             report templates, instructor guide with answer key
build/            instructor-side scripts to rebuild data, models, galleries and notebooks
```

The notebook clones this repository into the Colab session and installs `ultralytics` (pinned).
The only other runtime download is the Gradio package for the upload app.

## Rebuilding (instructors only)

```bash
pip install ultralytics==8.4.143 torch torchvision pillow scikit-image ipywidgets nbformat pandas matplotlib
python build/prepare_datasets.py --key construction_safety --src <yolo folder> --work <work> --pool 300 --test 209 --val 100
python build/prepare_datasets.py --key excavators --src <yolo folder> --work <work> --pool 300 --test 250 --val 100 --test-from-train 80
python build/train_course_models.py --key construction_safety --work <work> --epochs 30
python build/train_course_models.py --key excavators --work <work> --epochs 25
python build/make_tricky.py --work <work>
python build/make_notebooks.py
```

## Data and model sources

| Set | Source | Licence |
|---|---|---|
| Workers and PPE | Roboflow 100 "construction-safety" via https://huggingface.co/datasets/LibreYOLO/construction-safety-gsnvb | CC-BY-4.0 |
| Construction machinery | Roboflow 100 "excavators" via https://huggingface.co/datasets/LibreYOLO/excavators-czvg9 | CC-BY-4.0 |
| Detector | YOLO11n, https://github.com/ultralytics/ultralytics | AGPL-3.0 (library and weights; fine for teaching, check before commercial use) |
| Out-of-scope sample images | scikit-image `data` module | public domain / CC0 |
