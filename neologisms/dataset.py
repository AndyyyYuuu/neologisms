import torch
from torch.utils.data import Dataset
import pandas as pd


class DPODataset(Dataset):
    def __init__(self, data: list[dict]):
        self.data = data
        assert all(key in data[0] for key in ["prompt", "chosen", "rejected"])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return (self.data[idx]["prompt"], 
                self.data[idx]["chosen"], 
                self.data[idx]["rejected"])


class CSVData(DPODataset): 
    def __init__(self, path:str):
        df = pd.read_csv(path)
        df = df.to_dict(orient="records")
        super().__init__(df)

