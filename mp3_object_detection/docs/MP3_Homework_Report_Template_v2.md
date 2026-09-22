# CEM4644 · MP3 report — Homework (individual)

Name: ______________________    Date: ____________

Notebook: `MP3_Homework_Object_Detection_v2.ipynb` — dataset: *Construction machinery: excavators, dump trucks, wheel loaders*.

Answer every question in a few sentences. Paste screenshots where the question asks for photos or tables. Numbers must come from **your** run of the notebook (Step 7 prints them).

## Question 1

Which machine does the model find most reliably and which one does it miss most often (give the recall numbers)? Look at the missed and the false ones in Step 2b: what do they have in common? Then compare with the numbers you got in the workshop for the PPE model (your Step 6 there): which of the two datasets is the harder one for a detector, and what makes it harder - the size of the objects, how alike the classes look, how many training photos there were?

*Your answer:*



## Question 2

What does the machinery model make of the PPE photos, and what does the PPE model make of them? Does either model stay silent, or does it label what it sees with the wrong names? What does this tell you about buying a detector that was trained on someone else's photos?

*Your answer:*



## Question 3

From the dashboard: how many machines does the AI count in total and how many do the labels contain? On how many photos is the count exactly right? Run it at threshold 0.3 and 0.7 and explain which one you would use for (a) an automatic equipment log that nobody checks and (b) a weekly report that a person reads.

*Your answer:*



## Question 4

Copy your leaderboard. How does the quality score grow with the number of training photos, and where does it start to flatten? How far is your best run from the course model, and what would it take to close the gap? What did the random start do, and what does that tell you about the 120,000 everyday photos the pretrained model had already seen?

*Your answer:*



## Question 5

Test 5 photos of your own in Step 6a. Include the screenshots. For each photo: what the model found, what it should have found, and what made the wrong ones hard (distance, angle, a machine the classes do not cover, a photo unlike the training photos). Which two of your photos would you add to the training set, and why those?

*Your answer:*


