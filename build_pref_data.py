# Scripts to build model's own preference data

from transformers import pipeline
import pandas as pd
import os
from tqdm import tqdm
import re 


tqdm.pandas()

pipe = pipeline("text-generation", model="meta-llama/Llama-3.2-3B-Instruct")

def build_responses(path_from: str, path_to: str, n_instructions: int, n_samples: int, batch_size: int = 1) -> None: 
    df = pd.read_json(path_from, lines=True).conversations.iloc[:n_instructions].map(lambda x: x[0]).to_frame()
    print(df.conversations[0])
    df = df.loc[df.index.repeat(n_samples)].reset_index(drop=True)
    
    # Prepare all messages for batch processing
    messages = []
    for question in df.conversations:
        message = [
            {
                "role": "user",
                "content": [{"type": "text", "text": question},]
            }
        ]
        messages.append(message)
    
    # Process in batches
    all_responses = []
    for i in tqdm(range(0, len(messages), batch_size), desc="Generating responses"):
        batch_messages = messages[i:i+batch_size]
        batch_results = pipe(
            batch_messages, 
            pad_token_id=pipe.tokenizer.eos_token_id, 
            max_new_tokens=1024,
            batch_size=batch_size
        )
        for result in batch_results:
            all_responses.append(result[0]['generated_text'][-1]['content'])
    
    df["response"] = all_responses
    df.rename(columns={"conversations": "instruction"}, inplace=True)
    df.to_csv(path_to)
    print(df.head())

build_responses("data/lima_train.jsonl", "data/responses.csv", 50, 7)


def get_evaluation(instruction: str, response: str, deterministic: bool = True) -> float: 
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
    ev = pipe(message, pad_token_id=pipe.tokenizer.eos_token_id, max_new_tokens=1024, do_sample=not deterministic)[0]['generated_text'][-1]['content']
    found = re.search(r'"score":([^,}\n]+)', ev) # Written using claude.ai
    if not found:
        found = re.search(r'core:([^,}\n]+)', ev)
        if not found: 
            print("retry: not found")
            return get_evaluation(instruction, response, True)
    found_num = found.group().strip()[-1]
    if not (found_num.isnumeric() and 1 <= float(found_num) <= 5):
        print("retry: not numeric: ", found_num)
        return get_evaluation(instruction, response, True)
    return float(found_num)

# AI use: The below scripts were written, in part, by Composer 1.
#         Especially the fancy regex extractions.
#         Couldn't do that myself. 

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
    
    max_retries = 3
    for attempt in range(max_retries):
        ev = pipe(message, pad_token_id=pipe.tokenizer.eos_token_id, max_new_tokens=1024, do_sample=attempt > 0)[0]['generated_text'][-1]['content']
        
        # Try multiple extraction patterns, ordered from most specific to least
        patterns = [
            r'"score"\s*:\s*(\d+)',  # "score": 3 or "score":3
            r'"score"\s*:\s*(\d+\.\d+)',  # Handle decimals if needed
            r'score\s*:\s*(\d+)',  # score: 3 (no quotes)
            r'score\s*[=:]\s*(\d+)',  # score = 3 or score: 3
            r'(\d+)\s*(?:out of|/)\s*5',  # "3 out of 5" or "3/5"
            r'\b([1-5])\b',  # Fallback: any single digit 1-5 in the text
        ]
        
        for pattern in patterns:
            found = re.search(pattern, ev, re.IGNORECASE)
            if found:
                try:
                    score = float(found.group(1))
                    if 1 <= score <= 5:
                        return score
                except (ValueError, IndexError):
                    continue
        
        if attempt < max_retries - 1:
            print(f"retry {attempt + 1}/{max_retries}: extraction failed")
    
    print(f"Failed after {max_retries} attempts. Storing full response for manual review.")
    return ev  # Return full response for manual review


def get_evaluations_batch(instructions: list[str], responses: list[str], batch_size: int = 8) -> list[float]:
    """Process evaluations in batches for efficiency."""
    eval_prompt_template = open("prompts/eval_prompt.txt", "r").read()
    
    # Prepare all messages
    messages = []
    for instruction, response in zip(instructions, responses):
        PROMPT = eval_prompt_template.replace(r"{instruction}", instruction).replace(r"{response}", response)
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
        messages.append(message)
    
    # Process in batches
    all_scores = []
    patterns = [
        r'"score"\s*:\s*(\d+)',
        r'"score"\s*:\s*(\d+\.\d+)',
        r'score\s*:\s*(\d+)',
        r'score\s*[=:]\s*(\d+)',
        r'(\d+)\s*(?:out of|/)\s*5',
        r'\b([1-5])\b',
    ]
    
    for i in tqdm(range(0, len(messages), batch_size), desc="Processing batches"):
        batch_messages = messages[i:i+batch_size]
        batch_results = pipe(
            batch_messages, 
            pad_token_id=pipe.tokenizer.eos_token_id, 
            max_new_tokens=1024,
            batch_size=batch_size
        )
        
        for result in batch_results:
            ev = result['generated_text'][-1]['content']
            score_found = False
            
            for pattern in patterns:
                found = re.search(pattern, ev, re.IGNORECASE)
                if found:
                    try:
                        score = float(found.group(1))
                        if 1 <= score <= 5:
                            all_scores.append(score)
                            score_found = True
                            break
                    except (ValueError, IndexError):
                        continue
            
            if not score_found:
                print(f"Warning: Could not extract score from evaluation. Response: {ev[:100]}...")
                all_scores.append(3.0)  # Default to middle score if extraction fails
    
    return all_scores


def build_minmax(path_from: str, path_to: str) -> None: 
    df = pd.read_csv(path_from)[:10]

    # Use batch processing instead of sequential apply
    df["score"] = get_evaluations_batch(df.instruction.tolist(), df.response.tolist())
    
    min_df = df.loc[df.groupby("instruction").score.idxmin()]
    max_df = df.loc[df.groupby("instruction").score.idxmax()]
    min_df = min_df.rename(columns={"response": "rejected"})
    max_df = max_df.rename(columns={"response": "chosen"})
    pref_df = pd.merge(min_df[["instruction", "rejected"]], max_df[["instruction", "chosen"]], on="instruction", how="outer").reset_index(drop=True)
    pref_df.to_csv(path_to)
    print(pref_df.head())

build_minmax("data/responses.csv", "data/pref_responses.csv")

#print(get_evaluation("Can brain cells move? By movement I mean long distance migration (preferably within the brain only).",
#                     "No, brain cells (neurons) are not capable of moving long distances through the blood or any other medium. They are specialized cells that reside in a highly organized and protected environment called the brain.  The brain is surrounded by a protective covering called the blood-brain barrier (BBB) that prevents substances and cells from entering or leaving the brain.\n\nIn addition, the BBB is a complex network of blood vessels that is highly selective and tight, making it difficult for substances to pass through.  This barrier is designed to protect the brain from damage and toxins, but it also limits the ability of brain cells to move and interact with their environment.\n\nAs a result, brain cells are not capable of long-distance migration, and any movement of brain cells is typically limited to short distances within the brain itself.  They can move in response to specific signals and stimuli, but they do not have the ability to migrate or travel to other parts of the body.\n\nIt's worth noting that there are some exceptions to this rule, such as the migratory behavior of certain neurons in the retina and the olfactory epithelium. However, these movements are generally limited to specific locations within the brain and do not involve long-distance migration."))
#print(get_response("Can brain cells move? By movement I mean long distance migration (preferably within the brain only)."))
#df = df.progress_apply(lambda x: )

#df = pd.read_json(os.path.dirname(os.path.abspath(__file__)) + 'data/lima_train.jsonl', lines=True)
#df = pd.read_json(open(os.path.dirname(os.path.abspath(__file__)) + "/data/lima_train.jsonl", "r", encoding="utf8"))
