
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
def tokenize(string: str) -> torch.Tensor:
    return tokenizer(string, return_tensors='pt')["input_ids"][0]

# Context: """
#   <|bos|> Definition: \n
#   <word> \n
# """
# Predict: "<pos> <definition>"

class DictData(Dataset): 
    def __init__(self, path: str, tokenize, template: str):
        df = pd.read_csv(path)
        df = df[df.pos.notnull()].reset_index(drop=True)
        def tokenize_one(word: str) -> None | int: 
            if not (type(word) is str):
                return None
            word_ids = tokenize(word)[1:]
            if len(word_ids) == 1:
                return int(word_ids[0].item())
            return None

        print(df.head())

        df.word = df.word.map(tokenize_one)
        df = df[df.word.notnull()].reset_index(drop=True)
        df.word = df.word.astype(int)
        print(df.head())
        df.definition = df.apply(lambda x: x.pos + " " + x.definition, axis=1)
        df.definition = df.definition.map(lambda x: list(i.item() for i in tokenize(x)[1:]))
        self.df = df

        template_split = template.split("{}")
        self.template_1 = tokenize(template_split[0])
        self.template_2 = tokenize(template_split[1])[1:]

    
    def __len__(self): 
        return len(self.df)
    
    def __getitem__(self, idx: int) -> tuple:
        row = self.df.iloc[idx]
        return (torch.cat((self.template_1,
                           torch.tensor(row.word, dtype=torch.long).unsqueeze(0),
                           self.template_2)),
                torch.tensor(row.definition, dtype=torch.long))

dataset = DictData("data/en_dict.csv", tokenize, "Definition: \n{}\n")
#print(dataset.df.head(10))

pos_counts = dataset.df.pos.groupby(dataset.df.pos).count()
pos_counts = pos_counts.reset_index(name='count').sort_values(['count'], ascending=False)
#print(pos_counts.tail(20))

for i in range(10):
    print(dataset[i])