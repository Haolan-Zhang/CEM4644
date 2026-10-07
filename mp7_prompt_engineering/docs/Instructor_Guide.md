# MP7 · Instructor guide

## 0. Two notebooks

- `MP7_Workshop_Construction_Documents.ipynb` (first class): Parts 1–3 only, one given prompt each and no prompt
  comparison. Students send the prompt, score HokieAI's output against the ground truth, and explain its errors. The
  prompts are deliberately not the best versions, so the outputs have mistakes to discuss: Part 1 uses v2 (role and
  definitions; without the definitions the categories are undefined), Part 2 uses v1 (task only), Part 3 uses v2
  (answers with quotes, so the answers can be checked against the clauses, but no NOT IN TEXT instruction).
- `MP7_Workshop_Prompt_Engineering.ipynb` (the class on prompting): all five parts with prompt versions v1–v3 and the
  student's own, compared in a chart per part. Sections 1–4 below describe it.

## 1. The workshop

| Part | Examples | Prompt versions | Ground truth |
|---|---|---|---|
| 1 · Incident reports | 30 OSHA construction reports (2023–2025), 6 per class | v1 task only · v2 role and definitions · v3 definitions and worked examples · own | OSHA's OIICS event title of each record, mapped to the Focus Four |
| 2 · Submittal review | 12 practice submittals: 6 concrete mix designs, 6 mortar and grout mixes; each states its intended use, not the specification's portion of the structure; 0–3 planted noncompliances each (two comply fully) | v1 task only · v2 requirements first, then compare · v3 comparison table · own | the planted noncompliances; six requirements checked per submittal, each one decision |
| 3 · Contract questions | 27 questions on 29 FAR construction clauses: 10 project situations to apply a clause to, 10 factual, 7 not answerable | v1 questions only · v2 answers with quotes · v3 only from the text, applied to the facts, with quotes · own | short answers from the clauses; quotes checked word for word |
| 4 · Takeoff | 5 foundation plans (3 quantities each), 5 gable roofs (7 each) | v1 task only · v2 step by step · v3 estimating conventions · own | computed from the generated dimensions; tolerance 1–2 % |
| 5 · Door schedules | 5 sheets, one unit type each (5–9 doors) | v1 whole sheet · v2 cropped · v3 cropped + column names · own | the generated schedule; a door counts only with all six fields exact |

Five parts is more than 75 minutes of HokieAI round trips: assign Parts 4–5 (or 2–3) as homework, or let students
choose two of the three prompt versions per part.

In every step the student sends the prompt (and, for Parts 4–5, the downloaded drawing) to HokieAI, pastes the output
back, and scores it. Editing the prompt in the box labels the output "my own prompt (edited vN)". The comparison steps
(1c, 2b, 3b, 4c, 5b) chart every scored output of the session by prompt version.

## 2. Measured with gpt-5 through the HokieAI API (2026-10-02)

| Part | v1 | v2 | v3 |
|---|---|---|---|
| Incident reports (30) | 73 % (repeat 73 %) | 77 % | 77 % (repeat 77 %) |
| Foundation plans (15 quantities) | 60 % | 93 % | 100 % |
| Gable roofs (35 quantities) | 66 % | 66 % | 100 % |
| Door schedules (35 doors) | 0 % (declined: "can't read at this resolution") | 100 % | 100 % |

HokieAI's web chat may use a different model or settings, so the students' numbers can differ.

Parts 2 and 3, first version (10 submittals, 21 questions), run in the HokieAI web chat on 2026-10-03 (the API key's
quota was used up):

| Part | v1 | v2 | v3 |
|---|---|---|---|
| Submittal review (concrete 3, masonry 3, masonry 4: 18 requirements) | 18/18 | 18/18 | 18/18 |
| Contract questions (21) | 21/21, no quotes | 21/21, 20/20 quotes found | 21/21, 17/17 quotes found |

Earlier API runs on concrete submittals 1 and 2: v1 5/5 and 3/5, v2 and v3 5/5 and 4/5 (the extra flag "cementitious content
not adequate for the strength" counts as a false flag). With the current HokieAI model these two parts barely separate
the prompt versions: every version found every planted noncompliance and refused all four unanswerable questions. What
does change is the evidence: v1 gives answers with nothing to check, v2 and v3 give quotes that can be verified.

Both parts were therefore made harder on 2026-10-03 (the current version): submittals that describe their use
indirectly, values to compute or exactly at a limit, and requirements that do not apply (see the marking notes); contract
questions that apply a clause to dates and amounts, and unanswerable questions with a well-known outside answer. The
harder version has not been measured yet.

## 3. Marking notes

1. Incident reports: the gains from definitions and examples are small; the remaining disagreements are ambiguous
   reports that OSHA's coding rules decide (a shock that causes a ladder fall is a *Fall*; a lock cutting a thumb is
   *Struck-by*). A good answer names the rule that would fix the case. In a first trial sample, a bare v1 prompt was
   blocked by Azure's content filter while the same prompt with a role line went through; students may meet this.
2. Submittal review: each submittal names its use (basement walls, warehouse floor slab, office partitions, ...), so
   the first step is matching it to a row of the specification. The planted noncompliances and the traps:

   | Submittal | Noncompliant | Complies, although it looks wrong |
   |---|---|---|
   | Concrete 1 · basement walls | w/cm 342 / 600 = 0.57 > 0.55 (not stated: "see batch weights"); air 7.0 % > 6.5 % | |
   | Concrete 2 · warehouse floor slab | air-entraining admixture in a hard-troweled slab; cementitious 470 + 60 = 530 < 540 lb | air 2.5 % (≤ 3 %) |
   | Concrete 3 · entrance stoop and sidewalks | 4500 psi at 56 days instead of 28 | w/cm 0.45 and air 4.5 % exactly at the F2 limits |
   | Concrete 4 · spread footings | slump 7 in > 6 in footing limit | w/cm 0.58 (no limit for F0) |
   | Concrete 5 · stair-tower foundation walls | none | w/cm 330 / 600 = 0.55 and air 6.5 % exactly at the limits |
   | Concrete 6 · loading-dock apron | 1 in aggregate (3/4 in required); w/cm 263 / 560 = 0.47 > 0.45 | f'c 5000 psi (above the minimum) |
   | Mortar and grout 1 · maintenance shop walls | Type M mortar (Type S specified) | grout slump 11 in and 2000 psi at the limits |
   | Mortar and grout 2 · office partitions | self-consolidating grout slump flow 23 in < 24 in | mortar cement (only masonry cement is prohibited) |
   | Mortar and grout 3 · elevator pit walls | masonry cement; air-entraining admixture; fine grout | |
   | Mortar and grout 4 · brick veneer backup walls | none | slump flow 30 in, VSI 1, 2000 psi at the limits |
   | Mortar and grout 5 · storage room partitions | Type O mortar; grout 1800 psi | |
   | Mortar and grout 6 · loading-dock retaining wall | Type N mortar (below grade needs Type S); grout slump 7 in < 8 in | |

   The 7-day strength, f'cr, unit weight and chloride content are distractors with no requirement. Every concrete
   submittal is scored on all six concrete requirements, so a flag where the specification sets no limit (w/cm for a
   footing) is a false flag. A good answer to the "why" question names decomposition: listing the applicable
   requirements first stops the model from checking the submittal against the wrong portion of the structure.
3. Contract questions: the situations test whether the model applies the clause rather than recites it: the strike
   notice on March 16 is late (due within 10 days of March 3); costs under an oral change are recoverable only from
   April 6 (20 days before the April 26 notice); the gymnasium's warranty ran from possession and ended June 1, 2028; the
   suspension adjustment is $18,000 (profit excluded); the rock notice came after the conditions were disturbed; the
   quantity adjustment covers the 100 cubic yards above 115 percent. Seven questions have no answer in the text: three
   are blanks the Contracting Officer fills in (days to commence, liquidated damages, the self-performance percentage),
   and four belong to clauses not included (insurance limits, prompt-payment days, overhead and profit markup, the
   Disputes appeal deadline), each with a common answer the model may bring from outside. An answer such as "12 percent"
   or "90 days" is a made-up value; a quote that is not found word for word is a misquote.
4. Takeoff: without the conventions the model uses the wall centerline (deducting at corners), counts bar locations
   instead of the share of grouted cells, takes a full 1 ft strip as the starter course, and uses 2 rake edges instead
   of 4. Step-by-step reasoning fixes slips, not conventions, which is why v2 helps the plans and not the roofs.
   One plan per pair leaves a dimension to be derived; misreading it is a vision error, not a method error.
5. Door schedules: at full-sheet resolution the text is a few pixels high; cropping is what makes the task possible.
   A good answer to "what would you check" names counting the rows against the sheet, and spot-checking sizes.

## 4. Rebuilding

`build/generate_practice.py --seed N` makes a new set (same families, new dimensions and schedules); a per-student
set is possible. `build/prepare_incidents.py` needs OSHA's download (a browser User-Agent for scripted access).
The trial runs and prompts that produced section 2 are kept outside the repository.
