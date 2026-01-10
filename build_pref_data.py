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
    df.to_csv(path_to)
    print(df.head())

build_responses("data/lima_train.jsonl", "data/responses.csv", 50, 7)
