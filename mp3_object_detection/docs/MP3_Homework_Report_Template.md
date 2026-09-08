# CEM4644 · MP3 report — Homework (individual)

Name: ______________________    Date: ____________

Notebook: `MP3_Homework_Object_Detection.ipynb` — dataset: *Construction machinery: excavators, dump trucks, wheel loaders*.

Answer every question in a few sentences. Paste screenshots where the question asks for photos or tables. Numbers must come from **your** run of the notebook (Step 6 prints them).

## Question 1

What was your score in the counting game? Which photos were hard for **you** (small or distant objects, objects partly hidden, unclear cases), and would you have drawn the boxes the same way as the labellers?

*Your answer:*



## Question 2

Labelling: how long did you need per photo, and how many of your boxes agreed with the dataset labels? Which objects or classes were hard to decide? At your speed, how many hours would the whole training set take, and what does that mean for anyone who wants a detector for their own site?

*Your answer:*



## Question 3

Which class does the model find most reliably and which one does it miss most often (give the recall numbers)? Look at the missed objects in Step 2d: what do they have in common (size, distance, lighting, overlap, rarity in the training data)?

*Your answer:*



## Question 4

Set the threshold to 0.2 and to 0.8. What happens to the number of missed objects and to the number of false alarms? Which threshold would you choose for an automatic site alarm, and which for a weekly report? Explain the difference.

*Your answer:*



## Question 5

List three photos (tricky gallery, sliders, or your own) where the detector missed something or invented something. For each, say what it found, what it should have found, and what you think confused it.

*Your answer:*



## Question 6

In Step 3c, what does the general-purpose YOLO see in a site photo, and what does the fine-tuned course model add? In Step 3d, what happens when a model gets photos from the other dataset? What does this tell you about buying an 'AI camera' for your own site?

*Your answer:*



## Question 7

From the dashboard: how many machines does the AI count in total and how many do the labels contain? On how many photos is the count exactly right? Run it at threshold 0.3 and 0.7 and explain which one you would use for (a) an automatic equipment log and (b) a quick check by a person.

*Your answer:*



## Question 8

Copy your leaderboard. How did the quality score change with more photos and more passes? What happened with a random start? Why does a detector need far more training than the classifier in MP2 to reach a useful score?

*Your answer:*



## Question 9

Test 5 photos of your own in Step 3e (photos of construction machinery (from a site you can access, from the street, or photos you find online)). Include screenshots. Which detections were right, which were wrong, and what made the wrong ones hard?

*Your answer:*



## Question 10

Imagine this detector running on a site camera. Where would you place the camera, what would you do with each alarm, and what could go wrong (technically and for the people being filmed)? What data would you need to collect to make it work on your own site?

*Your answer:*


