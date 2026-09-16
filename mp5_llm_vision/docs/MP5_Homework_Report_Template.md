# CEM4644 · MP5 report — Homework (individual)

Name: ______________________    Date: ____________

Notebook: `MP5_Homework_LLM_Vision.ipynb` — examples: *Architectural styles, machinery, scanned floor plans*.

Answer every question in a few sentences. Paste screenshots where the question asks for overlays, tables or raw replies. Numbers must come from **your** run of the notebook.

## Question 1

From Step 1c: how many of the plain replies (A) were valid JSON, and did the label stay the same across the runs? What did the schema (B) change, and what did it not change? Why does a program that has to read the reply (to fill a table, to count, to draw a box) need B rather than A?

*Your answer:*



## Question 2

From Step 2a: the accuracy of the three prompts and of the MP2 model on the 20 style images. The images are computer-generated and the MP2 model was trained on the same kind of images: is that a fair comparison? Which styles does Gemini confuse with each other, and does the rules prompt help?

*Your answer:*



## Question 3

From Step 2b: what did you change in the prompt and what accuracy did you get? If it went up, what is the risk of tuning a prompt on the same photos you score it on (think of MP2's training / test split)?

*Your answer:*



## Question 4

From Step 3b: Gemini's recall and precision against the MP3 YOLO model's on the machinery photos. Which machine is hardest and why? From Step 3c: does the model's count of excavators agree with its own boxes and with the answer key?

*Your answer:*



## Question 5

From Step 3c on two photos: the model's count, the count of its own boxes and the answer key. When they disagree, which one is wrong and how would you know on a site where there is no answer key? Which of the two ways of counting would you trust on a site camera, and why?

*Your answer:*



## Question 6

From Step 4c on plan 8138 and on plan 11615 (scanned drawings with furniture and dimension strings, unlike the clean drawings of the workshop): copy the per-room tables. Which way holds up better on a scan, and what does the scan's clutter do to the polygons? Compare with the MP4 numbers for the same plans.

*Your answer:*



## Question 7

Run at least 3 experiments of your own in Step 5 (a photo from a site or from the internet, a plan, a changed prompt, the schema on and off). For each: the image, the prompt, the raw reply, and whether it was right. What kind of request broke the model, and how did it break (wrong answer, invented objects, unreadable reply)?

*Your answer:*



## Question 8

From Step 6: for each task, would you use the generalist, the specialist, or both together (as in Step 4b)? Argue with the numbers you got and with what each needs: labelled data, training, a GPU, a network connection, money per request, and someone who checks. What does structured output guarantee about a reply, and what does it not guarantee?

*Your answer:*


