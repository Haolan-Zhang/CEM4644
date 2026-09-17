# MP5 · One generalist for everything: a vision-language model by prompt (no-code lab)

Teaching material for **CEM4644**: the classification, detection and floor-plan take-off tasks of MP2, MP3 and MP4,
this time all done by one **multimodal language model** (Gemini through its free API) steered only by prompts, with
**structured output** shown both ways (JSON asked for in the prompt vs. a JSON schema enforced by the API), scored
against the same answer keys as before and compared with the earlier labs' trained specialists. For the plans, the
model's boxes can also be handed to SAM 3 (from MP4). Students need **no programming**: every notebook cell is a
Colab form and the code is hidden.

| Notebook | Open | Examples |
|---|---|---|
| `MP5_Workshop_LLM_Vision.ipynb` | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Haolan-Zhang/CEM4644/blob/master/mp5_llm_vision/MP5_Workshop_LLM_Vision.ipynb) | façade defects (14 photos, 7 classes), site safety (6 photos, 52 boxes), 3 clean floor plans |
| `MP5_Homework_LLM_Vision.ipynb` | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Haolan-Zhang/CEM4644/blob/master/mp5_llm_vision/MP5_Homework_LLM_Vision.ipynb) | architectural styles (20 images, 10 classes), machinery (6 photos), 3 scanned floor plans |

**Chat-window variants.** `MP5_Workshop_LLM_Vision_chat.ipynb` and `MP5_Homework_LLM_Vision_chat.ipynb` route the
single-example steps (describe a photo, JSON asked nicely, boxes on one photo, the count, rooms on one plan) through a
plain chat window, **hokie.ai** (https://hokie.ai.vt.edu/, Virginia Tech's free access to GPT models): the student
downloads the example image, pastes the prompt, attaches the image, and pastes the reply back into a cell that parses,
draws and scores it with the same code as the API path (`aec_llm/chat.py`; coordinate order and scale can be switched
when the chat used another convention). The batch steps and the schema-enforced steps stay on the Gemini API. Generated
with `python build/make_notebooks.py --chat` (`--all` for both kinds); the plain notebooks are untouched.

All notebooks are generated from one template. Students need a **free Gemini API key** (https://aistudio.google.com/apikey)
stored as the Colab secret `GEMINI_API_KEY`; without one, the precomputed answers of the built-in examples still work
and only live requests (own prompts, own images) are unavailable. A GPU is only needed for the LLM + SAM 3 step.

## What the students do

1. **Talk to the model**: a free question about a photo; then the same classification asked as JSON in the prompt (run three times: fences, invalid JSON, drifting labels) and with a JSON schema enforced by the API.
2. **Classification by prompt**: the MP2 photos with three prompts (class names / with descriptions / with rules), accuracy and confusion table against the answer key and the MP2 model; then their own prompt.
3. **Detection and counting**: boxes as `[ymin, xmin, ymax, xmax]` on a 0–1000 grid, converted to pixels and scored like MP3 (right label and IoU ≥ 0.5) next to the MP3 YOLO model; a count asked directly vs. counted from the model's own boxes.
4. **Rooms on a plan**: the model's own polygons vs. its boxes handed to SAM 3 (`Sam3Engine.segment_room` from MP4), both scored room by room against the drawing, with MP4's phrase result for comparison.
5. **Prompt lab**: a small Gradio app for their own image and words (raw reply next to what it draws).
6. **Wrap-up**: one table, generalist vs. the three specialists, with time and tokens.

## What is in this folder

```
MP5_Workshop_LLM_Vision.ipynb     student notebook (generated)
MP5_Homework_LLM_Vision.ipynb     student notebook (generated)
aec_llm/          all the code the notebooks call (hidden from students)
  config.py       example sets, model ids, the prompt templates and the JSON schemas
  client.py       the Gemini call: image + prompt (+ schema), disk cache, rate-limit retries, tolerant JSON parsing
  tasks.py        scoring: classification, detection (IoU matching), counting, rooms (polygons or boxes -> SAM 3)
  ui.py, viz.py   the notebook steps and their pictures;  chat.py  the paste-back steps of the chat variants;  app.py  the prompt-lab app;
  lab.py          the object the notebooks talk to
data/<variant>/   photos (from MP2's test split), sites (from MP3's test split, every box), plans (from MP4), each with
                  index.json: the answer key and what the earlier lab's specialist model said
data/cache/       Gemini's replies for every built-in example x prompt (so the notebook is instant and works without a key)
docs/             report templates, instructor guide
build/            instructor-side scripts: prepare_data.py, precompute.py, make_notebooks.py
```

**Model.** `gemini-3.5-flash-lite` by default (Step 0 also offers `gemini-3.5-flash`, `gemini-3.8-flash`, `gemini-3.6-flash`),
thinking level *low*. Boxes follow Google's convention (`box_2d` = `[ymin, xmin, ymax, xmax]`, 0–1000); polygons are `[x, y]`
points on the same grid. Structured output uses `response_json_schema`. The free tier caps every `gemini-3.x-flash` model
at 20 requests per day (measured), while the lite model allows a class-sized number; the client waits and retries on
per-minute limits and 503s and says so, and every reply is cached, so students hit the service only with their own prompts.

**Answer keys and specialists.** The photos, site photos and plans are copies of the earlier labs' test material with
their labels; `build/prepare_data.py` also runs the MP2 ConvNeXt, the MP3 YOLO11n and MP4's SAM 3 (*room* by phrase) on
them and stores the results in the index files, so the comparison in the notebooks needs no extra model.

## Rebuilding (instructors only)

```bash
pip install google-genai gradio nbformat scipy pillow matplotlib      # plus torch/transformers/ultralytics for prepare_data.py
python build/prepare_data.py                    # examples + specialist answers from ../mp2_*, ../mp3_*, ../mp4_*
export GEMINI_API_KEY=...                       # never commit it
python build/precompute.py                      # Gemini on every example x prompt (rate-limited: 20-40 minutes)
python build/make_notebooks.py
```

To change the prompts: `PROMPTS` in `aec_llm/config.py`, then `precompute.py` (only missing replies are requested; replies
of old prompts can be deleted from `data/cache`). To change the examples: the specs in `aec_llm/config.py` and
`prepare_data.py`.

## Sources and licences

| Item | Source | Licence |
|---|---|---|
| Façade defect photos | BD3 Building Defect Dataset (via MP2) | CC-BY-4.0 |
| Façade style images | Jonathandav/facade-styles (computer-generated, via MP2) | MIT |
| Site safety and machinery photos | Roboflow 100 `construction-safety`, `excavators` (via MP3) | CC-BY-4.0 |
| Floor plans and answer keys | CubiCasa5K (via MP4) | CC BY-NC-SA 4.0 |
| Gemini | Google, Gemini API (free tier) | Google's API terms; students use their own key |
| SAM 3 | Meta, loaded from the MP4 folder | SAM License (copy in `mp4_segmentation/docs`) |
