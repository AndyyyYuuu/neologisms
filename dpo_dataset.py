import torch
from torch.utils.data import Dataset


class DPODataset(Dataset):
    def __init__(self, data: list[dict]):
        self.data = data
        assert all(key in data[0] for key in ["instruction", "chosen", "rejected"])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return (self.data[idx]["instruction"], 
                self.data[idx]["chosen"], 
                self.data[idx]["rejected"])


