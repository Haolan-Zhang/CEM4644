# CEM4644 · MP4 report — Workshop (in class)

Name: ______________________    Date: ____________

Notebook: `MP4_Workshop_Segmentation.ipynb` — photos: *Structural work on site: concrete, rebar, formwork, steel, scaffolding*.

Answer every question in a few sentences. Paste screenshots where the question asks for photos or overlays. Numbers must come from **your** run of the notebook.

## Question 1

What was your score in the estimation game? Pick one photo and give its material mix at confidence 0.5 (the numbers from Step 2b). Then move the threshold to 0.3 and 0.8 for one material: how much does the share change, and why?

*Your answer:*



## Question 2

From Step 3b: which materials rise and which fall over the series, and does that match what a site manager would expect? Give one example where the number changes for a reason that has nothing to do with progress (camera position, sky, an old black-and-white photo...).

*Your answer:*



## Question 3

From Step 4a: which wording gave the most sensible mask for the material you chose, and how far apart were the shares? Why would *rebar* and *steel reinforcement bars* give different answers?

*Your answer:*



## Question 4

Describe one mistake you found in Step 4b or 4c (what was included or missed, at which confidence). Did the negative box fix it? What would you tell a colleague who wants to use these percentages in a progress report?

*Your answer:*



## Question 5

Plan *plan_dormitory* (Step 5a and 5b): which words found the footings and which found nothing? Give the scale you got from the 19'-4" bay (feet per pixel), the measured area of one F2 and one F3 footing, and compare with the footing schedule on the drawing (4'-6" and 5'-0" square). Where does the error come from: your box, the mask edge, or the drawing?

*Your answer:*



## Question 6

Plan *plan_mess_hall*: with the 24'-0" bay as reference, measure one column footing (the detail says 6'-0" square) and then use *find_all* to count the column footings. How many did SAM 3 find, what total area, and what should the answer be (14 footings of 36 sq ft)? List what it missed and what it added that is not a footing, and say which confidence worked best.

*Your answer:*



## Question 7

Test 1 photo(s) of your own (walls, floors, a site, a street). For each: the phrase you used, the share measured, and whether the mask is right. What kind of surface or wording failed?

*Your answer:*



## Question 8

Where on a project would a measurement like *share of the photo covered by X* be useful, and where would it mislead? What would you need (camera position, reference lengths, drawings, several photos) to turn it into a real quantity?

*Your answer:*


