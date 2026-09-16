# CEM4644

Course material for CEM4644 (AI for construction).

| Folder | Content |
|---|---|
| [`mp2_image_classification/`](mp2_image_classification/) | MP2 · Image classification (defect / no-defect, multi-class, training exercise). No-code Colab notebooks for the workshop and the homework (plus copies with two Gradio steps), the hidden helper code, data, models and instructor guide. |
| [`mp3_object_detection/`](mp3_object_detection/) | MP3 · Object detection (worker / PPE monitoring, construction machinery). No-code Colab notebooks for the workshop and the homework, YOLO course models, tricky galleries, dashboard and training exercise, instructor guide. |
| [`mp4_segmentation/`](mp4_segmentation/) | MP4 · Segmentation for quantity take-off on floor plans (SAM 3 on real CubiCasa5K plans drawn cleanly from the dataset's vector data: rooms, fixtures and openings by name; scale, areas and counts with boxes, checked against each plan's answer key). No-code Colab notebooks for the workshop and the homework, plus the hidden helper code, plans, precomputed masks and instructor guide. |
| [`mp5_llm_vision/`](mp5_llm_vision/) | MP5 · One generalist for everything: a vision-language model (Gemini) does the classification, detection and floor-plan take-off of MP2–MP4 by prompt, with structured JSON output shown both ways (asked for in the prompt vs. enforced by a schema), scored against the same answer keys and compared with the earlier labs' specialists; the model's boxes can be handed to SAM 3. No-code Colab notebooks for the workshop and the homework, hidden helper code, examples with precomputed replies, a prompt-lab app and an instructor guide. |
