"""
This file's purpose is to generate completion sentences dataset to evaluate the model.
"""

import json
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

from generate_question import generate_completion_questions


def handle_chunk(batch, chunk):
    """
    Given a chunk from the training dataset, generates some completion sentences about it and enhances Dataframe's rows with some metadata
    :return: A list of Dataframe rows with questions and expected answers.
    """
    df_rows = []
    step_number = batch['step_number']
    chunk_text = chunk["chunk_text"]

    mention_to_candidates = defaultdict(dict)
    for ent in chunk["entitities"]:
        for cand in ent["candidates"]:
            mention_to_candidates[ent["text_mention"]][cand["name"]] = cand['count']

    # Step 2: keep only mentions with exactly one unique candidate overall and keep it's count
    valid_mentions = {m: list(cands.values())[0] for m, cands in mention_to_candidates.items() if len(cands) == 1}

    paragraph, possible_answers = chunk_text, valid_mentions

    for qa in generate_completion_questions(paragraph, possible_answers):
        df_rows.append(
            {
                "step_number": step_number,
                "question": qa["question"].strip('.'),
                "answer": qa["answer"],
                "entity_type": qa["entity_type"],
                "prev_sentence": qa["prev_sentence"],
                "count": sum((valid_mentions[topic] for topic in qa["topics"])),
                "topics": qa["topics"]
            }
        )
    return df_rows


def main():
    with open('chunk_sample.json', encoding='utf-8') as f:
        data = json.load(f)

    df_rows = []

    for batch in tqdm(data):
        for chunk in batch["chunks"]:
            df_rows.extend(handle_chunk(batch, chunk))
        break

    df = pd.DataFrame(df_rows)
    df.to_pickle("QA_df_with_metadata.pkl")


if __name__ == '__main__':
    main()
