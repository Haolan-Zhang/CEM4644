# CEM4644 course material: rules for building notebooks

No-code Colab labs for an AI-for-construction course. Students cannot code; every step is a form cell over a hidden
helper package. These rules apply to every notebook and every change.

## 1. All text is editable in the notebook

The instructor edits wording in Colab, not in the Python package. Everything a student reads must live in the notebook:

- **Descriptions and explanations:** markdown cells and each form cell's `#@markdown` notes. The hidden code never
  prints an explanation, a hint or a "how to read this" sentence.
- **Results:** every result line, table caption, plot title, axis label, button label, placeholder and prompt is a
  wording written in its cell as `name_text = """..."""` and passed to the lab method (`lab.step(..., name_text=name_text)`).
  Numbers are filled in through `{placeholders}`; `**bold**` works; each line is shown as its own line.
- **Apps (Gradio):** all titles, labels and result lines come from one `labels_text` block of `key: wording` lines in the cell.
- **Code holds only fallbacks:** default wordings live in one module (e.g. `texts.py`); the notebook's copy is what is shown.
- **Rebuilds keep edits:** the notebook generator keeps every markdown cell, `#@markdown` note and `*_text` wording it
  finds in the existing notebook; `--fresh-text` is the only way to reset to the defaults.
- Error messages for broken input may stay in the code.

## 2. Writing style (reference: `In_Class_Activity_Predicting_Concrete_Strength.ipynb`)

Text must be **precise, consistent and concise**. Students are engineering students: they can handle technical terms.

- **Use the field's technical terms, defined once in passing:** AI model, input variables, output, training data, test
  data, model output, prediction, threshold, binary classification task, confusion matrix, false pass (false positive),
  false fail (false negative), MAE, R², importance, predicted vs. actual, prompt, few-shot prompting.
- **No made-up or childish terms.** "Model output", not "answers"; "Predict a New Recording", not "blind test" (not a
  machine-learning term); "Make Your Own Predictions", not a "game"; "ground truth", not "answer key". If a term is not
  used in the field, do not invent it.
- **One name per thing, everywhere.** The dropdown label, the note, the result line and the report question use the
  same name ("decision tree model", not "trees" in one cell and "gradient boosting" in another).
- **Titles:** `# In-Class Activity: <Task>` or `# Homework: <Task>`, then one sentence: "In this activity, we will use
  AI to <task> from <data>."
- **Parts:** `## Part N · <Title Case Topic>`, then at most two sentences on what the part explores ("In this part, we
  will explore ...").
- **Steps:** `▶ Step 1b · Classification: Predict Strength Level` (Title Case; "Type: Action" where it helps).
- **Notes, in this order:** what the step does or what to do, in the imperative ("Choose a strength level for each mix,
  click Submit, and compare ..."; "Run the cell to see ..."); the terms it needs, defined inline ("a binary
  classification task"); each model or option on its own line as `Name: what it does` with a short "for example, ..."
  in parentheses; a practical tip last ("If the sliders are not showing, run the cell again.").
- **Form controls:** only real choices are `#@param`; fixed settings are plain code lines (`task = "grades"`).
- **Results:** bold labels and numbers, no commentary (`**MAE:** {mae} {unit}`). Interpreting results is the student's
  job in the report questions.
- **Report questions:** `### 📝 Report question N`, then one `> From Step X: ...` paragraph per step. Ask for specific
  numbers, name the figure or table to include, ask for a comparison, and end with one judgment question tied to
  practice ("Which would be more concerning on a real project, and why?"). Step references must match the notebook.
- **Generative AI:** call it "multimodal/generative AI (GPT/HokieAI)"; HokieAI steps give numbered instructions (copy
  the prompt, open https://hokie.ai.vt.edu/, paste the output back, click Score).
- **Sources:** end with `### Data and model sources`: one bullet per item (title, author or agency, year, licence, URL).
- **No filler:** no motivation paragraphs, rhetorical questions, jokes or exclamation marks; emoji only in the
  conventions (▶, ✅, 📝).
- **American English:** American spelling and usage everywhere students look (color, gray, meter, labeled, center,
  license, -ize), including notebook text, result wordings, prompts, drawings and data files.
- **Imperial units only:** every quantity in US customary units (ft, in, ft², yd³, lb, lb/yd³, psi, kip, °F, gal,
  kWh). Convert source data that comes in metric before students see it; never mix units or show metric equivalents.

## 3. Notebook conventions

- Step 0 downloads only its lab folder (sparse clone) and prints only `✅ Ready in N s.`
- Every scored step calls a model; no step is a tally the student does by hand.
- Gradio only in one or two steps, opened from a link. Variants are copies; originals stay untouched.
- Never credit the user as the instructor.
- Text-only changes need no end-to-end run; code changes are run headlessly before committing.

## 4. Data and secrets

- Use only public-domain data (US government: OSHA, FAR, UFGS, VA, USACE, USDA, HABS) or material made for this
  course. Never use another course's materials, student work, or copyrighted handouts (e.g. RSMeans), not even cropped
  or redrawn. Nothing marked CUI.
- Never read, print or commit the files `secret`, `gemini_free` and `hokieai` (API keys). Read a key only inside code,
  scrub it from any error text, and check staged changes for it before committing.
- `_candidates/` folders are local working space (gitignored).
