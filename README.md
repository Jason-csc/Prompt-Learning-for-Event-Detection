## Prompt Learning for Event Detection


A sentence-level event detection system based on prompt learning.

### Dataset
Given the sentence, the NLP system needs to detect all event types included in that sentence. In this project, we apply [**MAVEN**](https://uofi.app.box.com/s/eo14g4aklh6sji2yw7la8p4a2dsurnc2) which is the largest event detection dataset. MAVEN has 168 event types in total.

For the `data/{train/valid/test}.json`, top 10 frequent event types are included while `data/{train/valid/test}_full.json` contain the complete event types.


### Environment Setup & Run

All setup codes are included in `event_detection.ipynb` along with `helper.py`, which runs on **Google Colab**. For the local running, ignore the ***Google Colab Setup*** section.


