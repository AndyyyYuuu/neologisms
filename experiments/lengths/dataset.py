from torch.utils.data import Dataset
import pandas as pd
from neologisms.dataset import DPODataset

class LengthDataset(DPODataset):
    def __init__(self, csv_path: str, min_length: int, max_length: int):
        df = pd.read_csv(csv_path)
        chosen = df[(df["length"] >= min_length) & (df["length"] < max_length)]
        rejected = df[(df["length"] < min_length) | (df["length"] >= max_length)]
        data = []
        for i, c in chosen.iterrows():
            for j, r in rejected.iterrows():
                if c["question"] == r["question"]:
                    data.append({
                        "prompt": c["question"],
                        "chosen": c["response"],
                        "rejected": r["response"]
                    })
        super().__init__(data)