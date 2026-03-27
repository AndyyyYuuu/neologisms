from torch.utils.data import Dataset
import pandas as pd
import random
from neologisms.dataset import DPODataset

class LengthDataset(DPODataset):
    def __init__(self, csv_path: str, min_length: int, max_length: int, one_per_question: bool = False, seed: int = 42):
        df = pd.read_csv(csv_path)
        data = []
        if one_per_question:
            rng = random.Random(seed)
            for question, group in df.groupby("question"):
                chosen = group[(group["length"] >= min_length) & (group["length"] < max_length)]
                rejected = group[(group["length"] < min_length) | (group["length"] >= max_length)]
                if len(chosen) == 0 or len(rejected) == 0:
                    continue
                data.append({
                    "prompt": question,
                    "chosen": rng.choice(chosen["response"].tolist()),
                    "rejected": rng.choice(rejected["response"].tolist()),
                })
        else:
            for i, c in df[(df["length"] >= min_length) & (df["length"] < max_length)].iterrows():
                for j, r in df[(df["length"] < min_length) | (df["length"] >= max_length)].iterrows():
                    if c["question"] == r["question"]:
                        data.append({
                            "prompt": c["question"],
                            "chosen": c["response"],
                            "rejected": r["response"]
                        })
        super().__init__(data)