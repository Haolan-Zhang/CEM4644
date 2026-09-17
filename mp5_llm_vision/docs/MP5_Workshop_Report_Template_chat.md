# CEM4644 · MP5 report — Workshop (in class)

Name: ______________________    Date: ____________

Notebook: `MP5_Workshop_LLM_Vision.ipynb` — examples: *Façade defects, site safety, floor plans*.

Answer every question in a few sentences. Paste screenshots where the question asks for overlays, tables or raw replies. Numbers must come from **your** run of the notebook.

## Question 1

From Step 1c: across your tries, how many of the chat replies were valid JSON, was the label always one of the categories, and did it stay the same? In Step 2a, run the *basic* prompt with the schema off and on: what does the schema change in the replies, and what does it not change (the label can still be wrong)? Why does a program that has to read the reply (to fill a table, to count, to draw a box) need the schema rather than a polite request?

*Your answer:*



## Question 2

From Step 2a: the accuracy of the three prompts (basic / with descriptions / with descriptions and rules) and of the MP2 model on the same 14 photos. Which classes does Gemini confuse (use the confusion table), and what did the descriptions and the rules change?

*Your answer:*



## Question 3

From Step 2b: what did you change in the prompt and what accuracy did you get? If it went up, what is the risk of tuning a prompt on the same photos you score it on (think of MP2's training / test split)?

*Your answer:*



## Question 4

From Step 3b: Gemini's recall and precision against the MP3 YOLO model's. Which label is hardest for Gemini (helmet, NO helmet, vest, NO vest, person) and why might that be? Paste one overlay from Step 3a and explain the extras (thick boxes marked '?'). Also: on the photo you gave the chat in Step 3a, how did its boxes compare with the MP3 model's on the same photo, and what coordinate convention did the chat use?

*Your answer:*



## Question 5

From Step 3c on two photos: the chat's count, the number of boxes it gave you in Step 3a, and the answer key. When they disagree, which one is wrong and how would you know on a site where there is no answer key? Which of the two ways of counting would you trust on a site camera, and why?

*Your answer:*



## Question 6

From Step 4a on two plans: copy the per-room tables from your pasted replies. Which rooms did the chat model outline well and which not (missing polygons, merged rooms, shapes in the wrong place)? Then from Step 4b: the table for all three plans from the API, with and without the schema. How do the two routes compare on the plan you did by hand, how does either compare with MP4's SAM 3 by phrase, and what does the batch table tell you that one plan could not?

*Your answer:*



## Question 7

Run at least 2 experiments of your own in Step 5 (a photo from a site or from the internet, a plan, a changed prompt, the schema on and off). For each: the image, the prompt, the raw reply, and whether it was right. What kind of request broke the model, and how did it break (wrong answer, invented objects, unreadable reply)?

*Your answer:*



## Question 8

From Step 6: for each task, would you use the generalist, the specialist, or both together? Argue with the numbers you got and with what each needs: labelled data, training, a GPU, a network connection, money per request, and someone who checks. What does structured output guarantee about a reply, and what does it not guarantee?

*Your answer:*


