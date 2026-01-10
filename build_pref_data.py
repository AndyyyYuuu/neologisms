# Scripts to build model's own preference data

from transformers import pipeline
import pandas as pd
import os
from tqdm import tqdm
import re 


tqdm.pandas()

pipe = pipeline("text-generation", model="meta-llama/Llama-3.2-1B-Instruct")

def build_responses(path_from: str, path_to: str, n_instructions: int, n_samples: int) -> None: 
    def get_response(question: str): 
        message = [
            #{
            #    "role": "system",
            #    "content": [{"type": "text", "text": "You are a helpful assistant"},]
            #},
            {
                "role": "user",
                "content": [{"type": "text", "text": question},]
            }
        ]
        #print(message)
        return pipe(message, pad_token_id=pipe.tokenizer.eos_token_id, max_new_tokens=1024)[0]['generated_text'][-1]['content'] # type: ignore
    df = pd.read_json(path_from, lines=True).conversations.iloc[:n_instructions].map(lambda x: x[0]).to_frame()
    print(df.conversations[0])
    df = df.loc[df.index.repeat(n_samples)].reset_index(drop=True)
    df["response"] = df.conversations.progress_apply(get_response)
    df.rename(columns={"conversations": "instruction"}, inplace=True)
    df.to_csv(path_to)
    print(df.head())

#build_responses("data/lima_train.jsonl", "data/responses.csv", 50, 7)


def get_evaluation(instruction: str, response: str) -> float: 
    PROMPT = open("prompts/eval_prompt.txt", "r").read().replace(r"{instruction}", instruction).replace(r"{response}", response)
    message = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant"},]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": PROMPT},]
        },
    ]
    ev = pipe(message, pad_token_id=pipe.tokenizer.eos_token_id, max_new_tokens=1024)[0]['generated_text'][-1]['content']
    found = re.search(r'"score":([^,}\n]+)', ev) # claude.ai
    if not found:
        found = re.search(r'core:([^,}\n]+)', ev)
        if not found: 
            print("retry: not found")
            return get_evaluation(instruction, response)
    found_num = found.group().strip()[-1]
    if not (found_num.isnumeric() and 1 <= float(found_num) <= 5):
        print("retry: not numeric: ", found_num)
        return get_evaluation(instruction, response)
    return float(found_num)


# AI use: The below script was written, in part, by Composer 1.

def build_minmax(path_from: str, path_to_min: str, path_to_max: str) -> None: 
    df = pd.read_csv(path_from)

    df["score"] = df.progress_apply(lambda x: get_evaluation(x.instruction, x.response), axis=1)
    min_df = df.groupby("instruction").min().reset_index()
    min_df.to_csv(path_to_min)
    print(min_df.head())
    max_df = df.groupby("instruction").max().reset_index()
    max_df.to_csv(path_to_max)
    print(max_df.head())

build_minmax("data/responses.csv", "data/worst_responses.csv", "data/best_responses.csv")

