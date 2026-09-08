# MP3 · Instructor guide

## 1. What the students get

Two Colab notebooks built from one template. Students never write code: each cell is a form with a
▶ button; outputs are photos with boxes, tables, curves, sliders, a dashboard and a small upload app.

| | Workshop | Homework |
|---|---|---|
| Dataset | workers and PPE (person, helmet, NO helmet, vest, NO vest) | machinery (excavator, dump truck, wheel loader) |
| Counting game | how many workers without a helmet? | how many machines? |
| Dashboard | PPE compliance and photos to flag | machine counts per photo |
| Training exercise | yes | yes |
| Own photos | 1 (optional in class) | 5 (required, question 8) |
| Time | about 90 min | about 2 h |

Course detector scores on the unseen test photos are in `models/*_training_log.json` and are
printed by Step 2b. Expect a quality score (mAP50) around 80 for the PPE set and around 90 for
the machinery set, with **NO helmet** the weakest class (it is rare in the training data), which
is exactly the point to bring out in question 2.

## 2. Suggested workshop timing (90 min)

| Min | Part | What to say / do |
|---|---|---|
| 0–7 | Step 0 + opener | Everyone runs Step 0 (about 2 min) and meanwhile opens the Ultralytics browser demo (ultralytics.com/yolo) on the webcam: it finds *person* and *chair* but knows no helmet. Choose *Run anyway* on Colab's author warning. |
| 7–15 | Part 1a–1b | Labels are boxes drawn by people. Play the counting game; ask for scores. Point at small, distant and half-hidden workers: if people disagree, the model will too. |
| 15–25 | Step 1c | Everyone labels three photos with the drawing tool. Collect a few speeds (boxes per minute) and the hours the full training set would take. This is the moment to talk about who labels data, how much it costs, and why label quality limits the model. |
| 25–42 | Part 2 | Confidence and the threshold (Step 2a), then recall / precision / mAP50 (Step 2b, 2c). Step 2d: look at missed *NO helmet* heads. Ask which mistake matters on a real site. |
| 42–57 | Part 3 | Tricky photos, sliders (*move away* is the most instructive), then Step 3c: the general-purpose YOLO sees *person* but no helmets; fine-tuning adds the site classes. Step 3d: a model only knows its own classes. Phones out for Step 3e. |
| 57–65 | Part 4 | The dashboard: run it at 0.3, 0.5, 0.7 and compare AI vs. labels. Alarm vs. statistic. |
| 65–83 | Part 5 | Training ladder: 60 photos / 10 passes → 120 / 15 → all / 12 → random start. On a T4 each run takes 1–2 minutes; on CPU tell them to stay at 60 photos / 10 passes (about 5 minutes). |
| 83–90 | Wrap-up | Step 6 prints the numbers. Report template: `docs/MP3_Workshop_Report_Template.md`. |

## 3. Answer key and marking notes (100 points)

1. **Counting game (6).** Any score; good answers mention small/distant workers, occlusion, helmets that are hard to see, and that labellers make judgement calls too.
2. **Labelling yourself (8).** Typical: 1–2 minutes per photo, 5–15 boxes per minute, about half to two thirds of the boxes agreeing with the dataset at the 50 % overlap rule. Good answers name the hard cases (partly hidden workers, NO vest vs. vest under a jacket, tiny heads) and conclude that the full training set is days of work for one person, so labelling is the real cost of a custom detector and label consistency matters.
3. **Reliable vs. missed classes (12).** Workshop: *person* and *helmet* are found most reliably; *NO helmet* has the lowest recall (rare class, small boxes). Homework: *dump truck* and *wheel loader* are strong, *excavator* weaker in some photos (rarer in the test set, many shapes). Full marks need the recall numbers and a plausible reason.
4. **Threshold (12).** Low threshold: fewer misses, more false alarms; high threshold: the opposite. Instant alarm: rather high threshold (false alarms erode trust) unless the cost of a miss is severe; weekly statistic: a middle threshold with a person checking flagged photos. Any threshold is fine if justified with both numbers.
5. **Fooling photos (12).** Expect: tiny objects after *move away*, dark or blurred photos, rotation, the astronaut counted as a person with a helmet, out-of-scope images producing confident boxes. Marks for the mechanism (scale, texture, colour, nothing similar in training).
6. **Pretrained vs. fine-tuned, domain shift (12).** COCO YOLO finds *person* / *truck* but has no helmet or vest classes; fine-tuning adds them from a few hundred photos. On other-domain photos each model can only answer with its own classes and invents them. Lesson: an "AI camera" must be trained (or checked) on the classes and photos of *your* site.
7. **Dashboard (10).** AI compliance is close to the labelled compliance but not identical; count the missed and false flags. Instant alarm: higher threshold; monthly statistic: middle threshold, errors average out. Homework: total machine counts, photos exactly right, over/under-counting.
8. **Training runs (14).** More photos and passes raise mAP50 with diminishing returns; a random start stays near zero with this little data. A detector must learn *where* as well as *what*, needs many boxes per class, and is judged by a stricter metric than accuracy. Typical test scores (PPE set, GPU; the backbone is kept frozen in the students' runs):

   | photos / passes | mAP50 on the test photos |
   |---|---|
   | 60 / 10, pretrained | about 35 |
   | 120 / 15, pretrained | about 55 |
   | all (240) / 12, pretrained | about 60 |
   | course model (897 photos / 30 passes, full network) | 82 |
   | all / 12, random start | about 0 |
9. **Own photos, homework only (8).** Screenshots of five photos with verdicts and sensible explanations.
10. **Deployment reflection (8, or 16 in the workshop).** Camera placement (height, coverage, lighting), what happens on an alarm, failure modes (weather, night, occlusion, drift), privacy and consent of workers, data collection and labelling plan for the site's own classes.

## 4. Known failure modes

- **Step 0 takes long or fails.** `pip install ultralytics` needs the network; retry. Colab's author warning: *Run anyway*.
- **No GPU.** Everything works on CPU; detection of 200 photos takes about a minute, training 30 photos × 2 passes about 4 minutes. Keep training small.
- **A cell looks stuck** (spinner for more than two minutes with no output). *Runtime → Interrupt execution*, run the cell again. Nothing is lost.
- **Step 1c drawing tool does not appear.** It is a third-party widget; Step 0 switches on Colab's custom widget manager. Run Step 0 again, then Step 1c. Without the tool, skip 1c and discuss question 2 from the counting game instead.
- **Share link in Step 3e fails.** The app still works inside the notebook (upload from the computer). The link is only needed for phones.
- **Live camera tab is slow or blank.** One or two frames per second through the Colab tunnel is normal. If the camera does not start (browser permission), use the *Photo* tab and take a snapshot instead.
- **Session reset.** After about 90 minutes idle Colab discards everything. Re-run Step 0 and continue; the leaderboard starts empty again.

## 5. Rebuilding or changing the material

See `README.md` in this folder for the build scripts. To swap a dataset, add a `DetSpec` in
`aec_det/config.py` (classes, colours, counting question, dashboard type), package it with
`build/prepare_datasets.py` from a YOLO-format folder, train with `build/train_course_models.py`,
build its tricky gallery, and point a variant in `build/make_notebooks.py` at it.
