# MP2 · Instructor guide

## 1. What the students get

Two Colab notebooks, one for the workshop and one for the homework, built from the same template.
Students never write or read code: each cell is a form with a ▶ button, and the outputs are
galleries, confidence bars, confusion matrices, interactive sliders and a small upload app.

| | Workshop | Homework |
|---|---|---|
| Binary task | façade wall: defect vs. no defect (BD3) | concrete surface: crack vs. no crack |
| Multi-class task | façade defect type, 7 classes (BD3) | architectural style, 10 classes (synthetic façades) |
| Training exercise | yes (either task) | yes (either task) |
| Own photos | 1 (optional in class) | 5 (required, question 9) |
| Time | about 80 min | about 2 h |

Course model accuracies on the unseen test sets (`models/*/training_log.json`):

| Model | Test photos | Accuracy |
|---|---|---|
| façade binary | 793 | 99.7 % |
| façade defect type (7 classes) | 793 | 94.7 % |
| concrete crack binary | 1,000 | 100 % |
| architectural style (10 classes) | 120 | 99.2 % |

**Gradio copies.** `MP2_Workshop_Image_Classification_gradio.ipynb` and
`MP2_Homework_Image_Classification_gradio.ipynb` are the same notebooks with one step replaced by
a small Gradio app that renders inside the cell:

- *Step 3b · Break it yourself*: the sliders now update the photo and the confidence bars live, and
  a second tab is a drawing pad where students paint a crack, a stain or a shadow with a brush (or
  upload a wall of their own and draw on it). "Load the photo with my slider changes" copies the
  current slider result into the pad.
- *Step 3b, third tab · Invisible noise*: a white-box adversarial attack on the course model
  (FGSM for one step, PGD with a random start for more; `aec_lab/adversarial.py`). The student
  picks the strength (maximum change per pixel, out of 255), the number of steps and optionally a
  target class, and gets before / after / amplified-noise panels, the confidences, and a JPEG
  re-test. Measured on 20 façade test photos with the course model:

  | strength | steps | verdict flipped | still flipped after JPEG 75 |
  |---|---|---|---|
  | 2 | 10 | 14 / 20 | 0 / 20 |
  | 4 | 1 (FGSM) | 3 / 20 | 1 / 20 |
  | 4 | 10 | 18 / 20 | 3 / 20 |
  | 8 | 10 | 20 / 20 | 10 / 20 |

  Talking points: no training and no data are needed, only the model's gradient (a *white-box*
  attack); the change is invisible while the sliders tab needed large visible changes; one step is
  much weaker than ten; and saving the file as JPEG usually destroys the attack, so lab attacks and
  attacks on a real inspection pipeline are different problems. Each attack takes well under a
  second on a GPU and a few seconds on CPU.
The report questions are unchanged, so the two versions can be mixed in one class. The copies pin
`gradio==6.26.0` (the version they were tested with) and install it without touching torch, numpy
or pillow. The app prints a temporary public link (the same mechanism as Step 3d) and is embedded
through it; on Colab that is required, because the alternative kernel-proxy embedding loses its
connection to the server. Use the originals if Gradio cannot be installed or the link is blocked.

The binary models are almost perfect on their own test sets. That is deliberate and is itself a
teaching point (Part 3): the test photos come from the same buildings and cameras as the training
photos, and the model falls apart on photos that look different.

## 2. Suggested workshop timing (80 min)

| Min | Part | What to say / do |
|---|---|---|
| 0–5 | Step 0 | Everyone switches the runtime to **T4 GPU** (*Runtime → Change runtime type*) and clicks ▶ on Step 0 while you introduce the topic. Tell them to choose *Run anyway* if Colab warns about the author. |
| 5–15 | Part 1 | Labels, classes, training vs. test split. Let them play the guessing game; ask for scores. Point out that *minor crack* vs *major crack* and *stain* vs *algae* are hard even for people: labels are human decisions. |
| 15–30 | Part 2 | Confidence bars, accuracy, confusion matrix. Step 2d: the threshold slider is where "AI in construction" becomes an engineering decision: missed defect vs. false alarm. Ask who would set 0.1 and who 0.9, and why. |
| 30–45 | Part 3 | Tricky gallery first (5 min), then sliders (5 min), then the domain-shift step, then phones out for Step 3d. The public Gradio link can be opened on a phone to use its camera. The app takes about a minute to install. |
| 45–55 | Part 4 | Seven classes. Confusion pairs: minor vs major crack, stain vs peeling/plain. Discuss whether accuracy is a fair summary when classes overlap. |
| 55–73 | Part 5 | Training. Suggested ladder: 20 photos / 1 pass, 100 / 2, 300 / 3, then random start. On a T4 each run takes under a minute. Compare runs on the leaderboard; with the extra time, let them try a run of their own design. |
| 73–80 | Wrap-up | Step 6 prints the numbers for the report. Report template: `docs/MP2_Workshop_Report_Template.md`. |

## 3. Answer key and marking notes

Points are suggestions (total 100). Accept any well-argued answer; the numbers below are what a
correct run produces, with small variation because some steps sample photos at random.

1. **Guessing game (8 pts).** Any score. Good answers name the pairs *minor/major crack*, *stain/algae*, *peeling/spalling* (workshop) or the historic styles (homework) and explain why: the boundary is a judgement call, image scale matters, classes overlap.
2. **Accuracy (10 pts).** Workshop: about 99.7 % of 793 photos. Homework: 100 % of 1,000 photos. Full marks require the number *and* a reason not to trust it as the only check: test photos resemble training photos; unusual materials, lighting or scales were never seen; the cost of a missed defect.
3. **Threshold (12 pts).** Lowering the threshold catches more defects but raises false alarms; raising it does the opposite. For inspection a missed defect is usually the worse error, so a low threshold plus human review of flagged photos is the usual answer. Any threshold is acceptable if justified with both error counts.
4. **Three fooling photos (12 pts).** Expect: drawn line read as a crack (edge detector behaviour), shadows or stains read as defects, blur or zoom removing the texture the model relies on, other-domain photos, out-of-scope images. Marks for a plausible *mechanism*, not only a description.
5. **Not-a-wall photos (10 pts).** The model must pick one of its classes because its outputs are confidences that add up to 100 %; there is no "none of the above". Remedies: add an *other* class with examples, require a minimum confidence, detect unusual inputs, keep a person in the loop, control how photos are taken.
6. **Multi-class confusion (10 pts).** Workshop: minor vs major crack dominate; stain vs plain/peeling next. Homework: the synthetic style images are so consistent that the model makes almost no mistakes on the test set (typically Georgian vs Victorian Terrace); the interesting errors show up in the tricky gallery and on the students' own photos of real buildings, which is the point to bring out. A person would make many of the same mistakes. Accuracy hides *which* mistakes are made; the confusion matrix or per-class accuracy is fairer.
7. **Training runs (18 pts).** More photos and more passes raise accuracy with diminishing returns; a random start is far worse because the network first has to learn what edges and textures are. Leaderboard must be included. Typical results on a GPU (CPU gives the same numbers, only slower):

   | photos / passes | façade binary | façade 7-class | concrete binary |
   |---|---|---|---|
   | 20 / 1, pretrained | about 50 % (chance) | about 20 % | 50–80 % |
   | 100 / 2, pretrained | 85–90 % | 45–55 % | 95–100 % |
   | 300 / 3, pretrained | about 95 % | about 80 % | 100 % |
   | all / 3, pretrained | 100 % | about 90 % | 100 % |
   | 100 / 2, random start | about 60 % | about 20 % | 60–80 % |
   | all / 3, random start | about 60 % | about 30 % | 80–95 % |
8. **Own photos, homework only (8 pts).** Screenshots of five photos with verdicts; sensible explanation for wrong ones (scale, lighting, material never seen, not a surface at all).
9. **Reflection (12 pts, or 20 in the workshop).** Any concrete use case (site progress photos, delivery checks, PPE, defects in handover inspections) with a realistic data plan (who takes the photos, who labels, how many, edge cases, drift over time, privacy).

## 4. Known failure modes

- **No GPU available in Colab** (quota used up, or the runtime type was not changed). Everything still works on CPU. Evaluation of 793 photos takes about 30 s; a training run with 300 photos and 3 passes takes about 3 minutes. Tell students to use fewer photos.
- **"Run anyway" warning.** Colab shows it for notebooks loaded from GitHub. Harmless.
- **Widgets do not react.** Re-run the cell. Sliders only update when released (not while dragging).
- **A cell looks stuck** (spinning for more than two minutes with no output). Use *Runtime → Interrupt execution*, then run the same cell again. Nothing is lost: the models and photos stay loaded.
- **Step 3d share link fails.** The app still works inside the notebook (upload from the computer). The public link is only needed for phones.
- **Step 0 fails to clone.** GitHub is unreachable from the network; try again or download the repository zip and upload the folder to the Colab file browser.
- **Gradio copies: the app area stays blank or says "Connection to the server was lost".** Give it 10–20 seconds first. On Colab the app is served through Gradio's temporary public link (printed above the app, and usable on a phone), so if the link cannot be created the app will not work; re-run the cell, and if it still fails use the non-Gradio version of the notebook. Step 3b is the only step affected.
- **Gradio copies: the drawing pad shows no photo.** Click *Another photo*; the pad is reloaded with the new photo.
- **Session reset.** Colab discards everything after about 90 minutes of inactivity. Students re-run Step 0 and continue; the leaderboard starts empty again.

## 5. Rebuilding or changing the material

See `README.md` in this folder for the four build scripts. Common changes:

- **Different questions or wording**: edit `build/make_notebooks.py`, run it (and once more with `--gradio` for the copies), commit the notebooks and the report templates.
- **A new dataset**: add a `DatasetSpec` in `aec_lab/config.py`, extend `build/prepare_datasets.py` to produce `data/<key>.zip`, add a job in `build/train_course_models.py`, run `build/make_tricky.py`, then point a variant in `build/make_notebooks.py` at it.
- **A bigger model**: change `base_convnextv2_femto` to another Hugging Face image-classification checkpoint (saved with `save_pretrained`). Femto was chosen because four fine-tuned copies fit in the repository at 10 MB each and train in a minute on CPU.

All datasets are used under their licences (CC-BY-4.0 and MIT); attributions are listed at the end of each notebook.
