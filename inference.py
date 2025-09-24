import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, Olmo2ForCausalLM

tqdm.pandas()

MAX_ANSWER_LENGTH_BY_WORDS = 4

model_id = "dhgottesman/LMEnt-1B-6E"  # the *repo id only*, no /tree/main
checkpoint_sub = "step540000"  # "step658032"     # the folder inside the repo

tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder=checkpoint_sub, device_map='cuda')
model = Olmo2ForCausalLM.from_pretrained(model_id, subfolder=checkpoint_sub, device_map='cuda')


def generate_model_answer(question):
    """
    "Ask" The model the question in order to get its answer
    :param question: The completion sentence to complete using the model.
    :return: The answer of the model.
    """
    try:
        inputs = tokenizer(question, return_tensors="pt").to(model.device)
        out_ids = model.generate(**inputs, max_new_tokens=7, do_sample=True, top_p=0.9, temperature=0.9)
        return tokenizer.decode(out_ids[0], skip_special_tokens=True)[len(question):]
    except Exception as e:
        print("An error occured when generating model answer:", e)
        return "ERROR"


def main():
    """
    Goes over the Question-Answer dataframe and ask the model the questions and the questions with extra context from the previous sentence.
    """
    df = pd.read_csv('QA_df_with_metadata.csv', encoding='utf-8')
    df["prev_sentence"] = df["prev_sentence"].fillna('')

    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Define chunk size
    chunk_size = 10000

    # Iterate in chunks
    for i in range(0, len(df_shuffled), chunk_size):
        chunk = df_shuffled.iloc[i:i + chunk_size]
        print(f"Starting to evaluate chunk {i}-{i + chunk_size}")

        chunk = chunk[chunk["answer"].str.split().str.len() <= MAX_ANSWER_LENGTH_BY_WORDS]

        chunk['model_answer'] = chunk['question'].apply(generate_model_answer)  # progress_apply
        chunk['model_answer_wt_context'] = chunk.apply(lambda row: generate_model_answer(
            row["prev_sentence"] + " " + row["question"] if type(row["prev_sentence"]) == type(" ") == type(
                row["question"]) else "ERROR"), axis=1)  # progress_apply

        chunk.to_csv(f'/Path/to/directory/results/df_with_answers_{i}.csv', encoding='utf-8')


if __name__ == '__main__':
    main()
