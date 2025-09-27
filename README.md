# Probing Hallucinations in Small-Scale Language Models (LMEnt-1B)

This repository contains the implementation details and scripts for the **Methodology** section of the paper "Probing Hallucinations in Small-Scale Language Models Trained with the Knowledge Analysis Suite (KAS)."

This study systematically investigates the influence of **topic frequency**, **temporal recency**, and **prompt quality** on the hallucination rates of a 1B-parameter language model.

---

## ⚙️ Methodology Implementation

The core methodology is structured around three stages: **Automated Dataset Creation**, **Correctness Evaluation**, and **Model Evaluation**.

### 1. Automated Dataset Creation

This stage generates a controlled, fill-in-the-blank evaluation dataset directly from the training corpus, leveraging the KAS infrastructure to enrich questions with essential metadata.

* **Script:** `create_df_for_model_eval.py`
* **Process:**
    * The `handle_chunk` function processes random data chunks sampled from the training set.
    * It identifies target **topics** (entities) using KAS. A critical filtering step **ensures** each mention has a **single, unambiguous candidate** for the topic. This step is necessary to accurately calculate and maintain the integrity of the **topic frequency** count.
    * The `generate_completion_questions` function from `generate_question.py` is then used to construct the fill-in-the-blank questions.
    * A sentence's suffix (verifiable facts like **DATE**, **PERSON**, **GPE**, **ORG**, or **CARDINAL** numbers) is removed to become the **expected answer**.
    * Sentences are filtered to exclude those with ambiguous pronouns or multiple occurrences of the answer within the question.
* **Metadata:** Each question is enriched with:
    * **Topic Frequency:** The count of distinct chunks in which the topic appeared in the training corpus.
    * **Temporal Exposure:** The training step number when the model encountered the chunk (for both the fifth and sixth epochs).
    * **Previous Sentence:** The sentence immediately preceding the question, which serves as the **high-quality prompt** for evaluation.

---

### 2. Correctness Evaluation

To accurately determine if the model's output is a hallucination (a confident but factually incorrect statement), we employ a robust, hybrid evaluation framework that goes beyond simple string matching.

* **Script:** `hallucination_checker.py`
* **Function:** `check_example(gold_answer, model_output)`
* **Evaluation Logic:**
    1.  **Exact Match:** The model's output must contain the normalized expected answer as a substring.
    2.  **Value-Based Match (for Dates):** For `DATE` answers, standard parsing is used to compare the normalized date (year-month-day) regardless of the original format (e.g., "January 1, 2009" vs. "1 January 2009").
    3.  **Semantic Match (for Textual Answers):** For entity types like **PERSON**, **ORG**, and **GPE**, a pre-trained sentence transformer model (`all-MiniLM-L6-v2`) computes the **cosine similarity** between the gold answer and candidates extracted from the model's output. A result is marked correct if the similarity score meets a defined threshold ($0.80$ in the current configuration).

---

### 3. Model Evaluation

This phase executes the questions against the model at specific checkpoints to measure how hallucination rates change with additional training and different prompting strategies.

* **Script:** `inference.py`
* **Model:** A **1B-parameter language model** (specifically `dhgottesman/LMEnt-1B-6E`).
* **Checkpoints:** Evaluation is performed at two stages of training to track the evolution of factual recall:
    1.  After the **fifth training epoch** (checkpoint `step540000`).
    2.  After the **sixth training epoch** (checkpoint `step658032`).
* **Prompting:** The model is tested with two prompt types for each question:
    1.  **Base Prompt:** Only the fill-in-the-blank question sentence.
    2.  **High-Quality Prompt:** The **Previous Sentence** plus the question sentence (providing context).