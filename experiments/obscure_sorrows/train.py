from dict_dataset import DictData
import neologisms
import pandas as pd
import torch
from pathlib import Path

DIR = Path(__file__).parent

WORD = "foreclearing"
DEFINITION = (df := pd.read_csv("data/obscure_sorrows.csv", skipinitialspace=True).set_axis(['word', 'definition'], axis=1))[(df["word"].str.strip('"') == WORD)]["definition"].iloc[0]
print(DEFINITION)

CONFIG = neologisms.TrainConfig(
    INITIAL_TOKEN = " the",
    NEO_PROMPT_PATH = "prompts/llama_instruct_train_prompt.txt",
    DATASET = DictData("data/en_dict.csv", DEFINITION),
    N_EPOCHS = 64,
    SAVE_PATH = DIR / "saves/epochs",
    BETA = 0.2,
    PROBS_CACHE_PATH = DIR / "saves/ref_lp_llama-3.2-1b-instruct.pt",
    ON_THE_FLY_REF_PROBS = True,
    MODEL_BACKEND = neologisms.HFTransformerBackend("meta-llama/Llama-3.2-1B-Instruct", "/Volumes/backrooms/huggingface", dtype=torch.bfloat16),
    EPOCH_SIZE = 128,
    DO_WANDB = False,
    NEO_DTYPE = torch.bfloat16,
    SPECIAL_DATA_PROCESS_FN = None,
    
)

if __name__ == "__main__":
    neologisms.train(CONFIG)