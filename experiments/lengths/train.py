import argparse
from dataset import LengthDataset
import neologisms
import pandas as pd
import torch
from pathlib import Path
from dotenv import load_dotenv
import os
from experiments.prep_run import prep_run

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
parser.add_argument("--job-id", default="local")
args = parser.parse_args()

MIN_LENGTH = 5
MAX_LENGTH = 50

DIR = Path(__file__).parent

MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", None)

dataset = LengthDataset(DIR / "data/lima_train_subset_lengths.csv", min_length=MIN_LENGTH, max_length=MAX_LENGTH, one_per_question=True)

run_dir = prep_run(
    {
        "job_id": args.job_id,
        "run_name": f"{MIN_LENGTH}-{MAX_LENGTH}",
        "model": args.model,
        "min_length": MIN_LENGTH,
        "max_length": MAX_LENGTH,
        "n_epochs": 64,
        "beta": 0.2,
        "learning_rate": 1e2,
        "epoch_size": 128,
    },
    output_root=DIR / "saves",
)

CONFIG = neologisms.TrainConfig(
    INITIAL_TOKEN = " the",
    NEO_PROMPT_PATH = "prompts/llama_instruct_train_prompt.txt",
    DATASET = dataset,
    N_EPOCHS = 64,
    SAVE_PATH = run_dir,
    BETA = 0.2,
    LEARNING_RATE = 1e-4,
    PROBS_CACHE_PATH = run_dir / f"ref_lp_{args.model.split('/')[-1].lower()}.pt",
    ON_THE_FLY_REF_PROBS = True,
    MODEL_BACKEND = neologisms.HFTransformerBackend(args.model, MODEL_CACHE_DIR, dtype=torch.float32),
    EPOCH_SIZE = 128,
    DO_WANDB = bool(os.getenv("WANDB_API_KEY")),
    NEO_DTYPE = torch.float32,
    SPECIAL_DATA_PROCESS_FN = None,
)

if __name__ == "__main__":
    print(f"# pairs: {len(dataset)}")
    neologisms.train(CONFIG)
