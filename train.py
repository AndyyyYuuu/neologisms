import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import pandas as pd
from torch.utils.data import Dataset
from dataset import DictData

#print("Fetching dataset ...")

#dataset = pd.read_json(hf_hub_download(repo_id="GAIR/lima", filename="train.jsonl", repo_type="dataset"), lines=True)
#print(type(dataset))
#print(dataset.head(10))


print("Preparing ...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

print(model.model)

def token_to_embed(token_id: int) -> torch.Tensor: 
    return model.model.embed_tokens.weight[token_id].to(device)

def tokenize(string: str) -> torch.Tensor:
    return tokenizer(string, return_tensors='pt').to(device)["input_ids"][0]

def str_to_embed(string: str) -> torch.Tensor:
    return torch.stack([token_to_embed(int(tok_id.item())) for tok_id in tokenize(string)])


dataset = DictData("data/en_dict.csv", tokenize)
print(dataset.df)
exit()
train_loader = DataLoader(dataset, shuffle=True)

neo_id = tokenize("Ensure")[-1].item()
neo_embed = token_to_embed(int(neo_id))
neo_param = nn.Parameter(neo_embed)


def APOLoss(beta: float): 
    # Use log probs
    def apo(prob_policy_y_c, prob_policy_y_r, prob_ref_y_c, prob_ref_y_r): 
        return -torch.log(torch.sigmoid((prob_policy_y_c / prob_policy_y_r + prob_ref_y_c / prob_ref_y_r) * beta)) - torch.log(torch.sigmoid(prob_policy_y_c - prob_policy_y_r))
        #       ^^^ modified DPO (Rafailov et al., 2024)                                                             ^^^ APO-up from D’Oosterlinck et al. (2025)
    return apo

loss = APOLoss(beta=0.2)

optim = torch.optim.Adafactor([neo_param])

train_iter = iter(train_loader)

N_EPOCHS = 10

for epoch in range(N_EPOCHS):
    pass