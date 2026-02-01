
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from dpo_dataset import DPODataset
from collections.abc import Callable

#tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
#def tokenize(string: str) -> torch.Tensor:
#    return tokenizer(string, return_tensors='pt')["input_ids"][0]

# Context: """
#   <|bos|> Definition: \n
#   <word> \n
# """
# Predict: "<pos> <definition>"

class DictData(DPODataset): 
    def __init__(self, path: str, target_definition: str):

        df = pd.read_csv(path)
        
        df = df[df.pos.notnull()].reset_index(drop=True)

        df.definition = df.apply(lambda x: x.pos + " " + str(x.definition), axis=1) # add part of speech

        # Convert to format: list of dicts: prompt, chosen, rejected
        
        data = []
        for _, row in df.iterrows():
            data.append({
                "prompt": "The definition is: \n",
                "chosen": target_definition,
                "rejected": row.definition
            })
        
        super().__init__(data)

    
    def __getitem__(self, idx: int) -> dict:
        return super().__getitem__(idx)

#dataset = DictData("data/en_dict.csv", "n. The act of deliberately refusing to learn the scientific explanations of things out of fear that it will ruin the magic")
#print(dataset.df.head(10))

#for i in range(10):
#    print(dataset[i])