import argparse
from pathlib import Path

DIR = Path(__file__).parent

DEFAULT_WORD = "foreclearing"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--word",
        default=DEFAULT_WORD,
        help=f"Obscure Sorrows word to train on (default: {DEFAULT_WORD})",
    )
    return parser.parse_args()


def get_definition(word: str) -> str:
    import pandas as pd

    df = pd.read_csv(DIR / "data/obscure_sorrows.csv", skipinitialspace=True).set_axis(
        ["word", "definition"], axis=1
    )
    matches = df[df["word"].str.strip('"') == word]["definition"]
    if matches.empty:
        raise ValueError(f"Unknown obscure sorrow word: {word}")
    return matches.iloc[0]


def build_config(definition: str):
    from dict_dataset import DictData
    import neologisms
    import torch

    return neologisms.TrainConfig(
        INITIAL_TOKEN = " the",
        NEO_PROMPT_PATH = "prompts/llama_instruct_train_prompt.txt",
        DATASET = DictData("data/en_dict.csv", definition),
        N_EPOCHS = 64,
        SAVE_PATH = DIR / "saves/epochs",
        BETA = 0.2,
        PROBS_CACHE_PATH = DIR / "saves/ref_lp_llama-3.2-3b-instruct.pt",
        ON_THE_FLY_REF_PROBS = True,
        MODEL_BACKEND = neologisms.HFTransformerBackend(
            "meta-llama/Llama-3.2-3B-Instruct",
            "/Volumes/backrooms/huggingface",
            dtype=torch.float32,
        ),
        EPOCH_SIZE = 128,
        DO_WANDB = False,
        NEO_DTYPE = torch.float32,
        SPECIAL_DATA_PROCESS_FN = None,
    )


def main() -> None:
    args = parse_args()
    definition = get_definition(args.word)
    print(definition)
    import neologisms

    neologisms.train(build_config(definition))

if __name__ == "__main__":
    main()
