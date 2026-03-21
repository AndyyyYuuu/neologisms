import requests
from dotenv import load_dotenv
import os
from tqdm import tqdm
import pandas as pd
import json
import random
from pathlib import Path

DIR = Path(__file__).parent

random.seed(8128)

load_dotenv()

OPEN_ROUTER_API_KEY = os.getenv("OPEN_ROUTER_API_KEY")

def get_response(messages: list[dict], max_tokens: int = 1024) -> list[str]:
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPEN_ROUTER_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "google/gemini-2.5-flash",
            "messages": messages,
            "max_tokens": max_tokens
        }
    )
    try: 
        return response.json()["choices"][0]["message"]["content"]
    except KeyError:
        raise KeyError(f"KeyError: failed to get response, got {response.json()}")

# Rather expensive on tokens; will keep this here but opt for cheaper method
def iter_length_response(question: str, target_length: int, error_margin: int = 0.1, max_iters: int = 20) -> str:
    this_response = get_response([{"role": "user", "content": f"{question}\nGive me a response that is {target_length} words in length."}])
    this_length = len(this_response.split())
    #best_response = this_response
    prev_response = this_response
    #best_length = this_length
    chat_history = [{"role": "assistant", "content": this_response}]
    for i in tqdm(range(max_iters)):
        tqdm.write(f"Deviation: {this_length - target_length}")
        if target_length * (1 - error_margin) <= this_length <= target_length * (1 + error_margin):
            return this_response
        if this_length > target_length:
            change_string = f"{this_length - target_length} words shorter."
        else:
            change_string = f"{target_length - this_length} words longer."

        prompt = [{"role": "assistant", "content": prev_response}, {"role": "user", "content": f"Give me a response that is {change_string}."}]
        
        this_response = get_response(prompt)
        this_length = len(this_response.split())
    return this_response

def get_length_response(question: str, target_length: int) -> (str, int):
    chat_history = [{"role": "user", "content": f"{question}\nGive me a response that is {target_length} words in length."}]
    response = get_response(chat_history)
    return response, len(response.split())

def sample_length_responses(start: int, stop: int, n_buckets: int, question: str) -> pd.DataFrame:
    bucket_size = (stop - start) // n_buckets
    buckets = [range(start + i * bucket_size, start + (i + 1) * bucket_size) for i in range(n_buckets)]
    responses = []
    for bucket in buckets:
        target = random.choice(bucket)
        response, length = get_length_response(question, target)
        responses.append({"question": question, "response": response, "length": length})
    return pd.DataFrame(responses)

def build_length_dataset(questions_csv_path: str, dest_csv_path: str, start: int, stop: int, n_buckets: int):
    src_df = pd.read_csv(questions_csv_path)
    questions = src_df.iloc[:, 0].tolist()
    df = pd.DataFrame(columns=["question", "response", "length"])
    for question in tqdm(questions, desc=f"Sampling {len(questions)} questions"):
        new_df = sample_length_responses(start, stop, n_buckets, question)
        df = pd.concat([df, new_df])
        df.to_csv(dest_csv_path, index=False)

if __name__ == "__main__":
    #df = pd.read_json("data/lima_train.jsonl", lines=True).conversations
    build_length_dataset("data/lima_train_subset.csv", DIR / "data/lima_train_subset_lengths.csv", start=5, stop=300, n_buckets=10)
    #response = iter_get_length_response("How does a neural network work?", 400, error_margin=0.05, max_iters=10)
    #df = sample_length_responses(range(100, 1000, 100), "How does a neural network work?")
    #print(len(response.split()))