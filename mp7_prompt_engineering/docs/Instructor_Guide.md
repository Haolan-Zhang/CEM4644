# MP7 · Instructor guide

## 1. The workshop

| Part | Examples | Prompt versions | Answer key |
|---|---|---|---|
| 1 · Incident reports | 30 OSHA construction reports (2023–2025), 6 per class | v1 task only · v2 role and definitions · v3 definitions and worked examples · own | OSHA's OIICS event title of each record, mapped to the Focus Four |
| 2 · Takeoff | 5 foundation plans (3 quantities each), 5 gable roofs (7 each) | v1 task only · v2 step by step · v3 estimating conventions · own | computed from the generated dimensions; tolerance 1–2 % |
| 3 · Door schedules | 5 sheets, one unit type each (5–9 doors) | v1 whole sheet · v2 cropped · v3 cropped + column names · own | the generated schedule; a door counts only with all six fields exact |

In every step the student sends the prompt (and, for Parts 2–3, the downloaded drawing) to HokieAI, pastes the output
back, and scores it. Editing the prompt in the box labels the output "my own prompt (edited vN)". The comparison steps
(1c, 2c, 3b) chart every scored output of the session by prompt version.

## 2. Measured with gpt-5 through the HokieAI API (2026-10-02)

| Part | v1 | v2 | v3 |
|---|---|---|---|
| Incident reports (30) | 73 % (repeat 73 %) | 77 % | 77 % (repeat 77 %) |
| Foundation plans (15 quantities) | 60 % | 93 % | 100 % |
| Gable roofs (35 quantities) | 66 % | 66 % | 100 % |
| Door schedules (35 doors) | 0 % (declined: "can't read at this resolution") | 100 % | 100 % |

HokieAI's web chat may use a different model or settings, so the students' numbers can differ.

## 3. Marking notes

1. Incident reports: the gains from definitions and examples are small; the remaining disagreements are ambiguous
   reports that OSHA's coding rules decide (a shock that causes a ladder fall is a *Fall*; a lock cutting a thumb is
   *Struck-by*). A good answer names the rule that would fix the case. In a first trial sample, a bare v1 prompt was
   blocked by Azure's content filter while the same prompt with a role line went through; students may meet this.
2. Takeoff: without the conventions the model uses the wall centerline (deducting at corners), counts bar locations
   instead of the share of grouted cells, takes a full 1 ft strip as the starter course, and uses 2 rake edges instead
   of 4. Step-by-step reasoning fixes slips, not conventions, which is why v2 helps the plans and not the roofs.
   One plan per pair leaves a dimension to be derived; misreading it is a vision error, not a method error.
3. Door schedules: at full-sheet resolution the text is a few pixels high; cropping is what makes the task possible.
   A good answer to "what would you check" names counting the rows against the sheet, and spot-checking sizes.

## 4. Rebuilding

`build/generate_practice.py --seed N` makes a new set (same families, new dimensions and schedules); a per-student
set is possible. `build/prepare_incidents.py` needs OSHA's download (a browser User-Agent for scripted access).
The trial runs and prompts that produced section 2 are kept outside the repository.
