import torch
from torch.utils.data import Dataset
import pandas as pd
#from transformers import pipeline
from dpo_dataset import DPODataset

#pipe = pipeline("text-generation", model="google/gemma-2b")

#print(pipe("hello"))

class PrefData(DPODataset): 
    def __init__(self, path:str):
        df = pd.read_csv(path)
        df = df.to_dict(orient="records")
        super().__init__(df)

