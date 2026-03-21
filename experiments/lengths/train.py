from dataset import LengthDataset
import neologisms
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime

MIN_LENGTH = 5
MAX_LENGTH = 50

DIR = Path(__file__).parent

run_id = f"run_{datetime.now().strftime("%m%d_%H%M")}_{MIN_LENGTH}-{MAX_LENGTH}"

dataset = LengthDataset(DIR / "data/lima_train_subset_lengths.csv", min_length=MIN_LENGTH, max_length=MAX_LENGTH)


CONFIG = neologisms.TrainConfig(
    INITIAL_TOKEN = " the",
    NEO_PROMPT_PATH = "prompts/llama_instruct_train_prompt.txt",
    DATASET = dataset,
    N_EPOCHS = 64,
    SAVE_PATH = DIR / f"saves/{run_id}",
    BETA = 0.2,
    PROBS_CACHE_PATH = DIR / f"saves/{run_id}/length_neo_ref_lp_llama-3.2-1b-instruct.pt",
    ON_THE_FLY_REF_PROBS = True,
    MODEL_BACKEND = neologisms.HFTransformerBackend("meta-llama/Llama-3.2-1B-Instruct", "/Volumes/backrooms/huggingface", dtype=torch.bfloat16),
    EPOCH_SIZE = 128,
    DO_WANDB = False,
    NEO_DTYPE = torch.float32,
    SPECIAL_DATA_PROCESS_FN = None,
    
)

if __name__ == "__main__":
    print(f"# pairs: {len(dataset)}")
    print(f"Run ID: {run_id}")
    neologisms.train(CONFIG)
