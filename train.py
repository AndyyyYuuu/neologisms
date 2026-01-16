import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import pandas as pd
from torch.utils.data import Dataset
from csv_dataset import CSVData
from tqdm import tqdm
from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    INITIAL_TOKEN: str
    NEO_PROMPT: str
    DATA_SOURCE: str
    MODEL_NAME: str
    N_EPOCHS: int
    SAVE_PATH: str
    MODEL_CACHE_DIR: str
    BETA: float


CONFIG = Config(
    INITIAL_TOKEN = " good",
    NEO_PROMPT = r"Give me a response you think is {}. ",
    DATA_SOURCE = "data/pref_responses.csv",
    MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct",
    N_EPOCHS = 10,
    SAVE_PATH = "saves/neo_param",
    MODEL_CACHE_DIR = "/Volumes/backrooms/huggingface", # Don't mind me, that's the name of my hard drive
    BETA = 0.2
)

#print("Fetching dataset ...")

#dataset = pd.read_json(hf_hub_download(repo_id="GAIR/lima", filename="train.jsonl", repo_type="dataset"), lines=True)
#print(type(dataset))
#print(dataset.head(10))


print("Loading resources ...")

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

def clear_cache():
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()

model = AutoModelForCausalLM.from_pretrained(
    CONFIG.MODEL_NAME, 
    cache_dir=CONFIG.MODEL_CACHE_DIR,    
    local_files_only=False,
    dtype=torch.bfloat16
).to(device)

# Train *only* the neologism
for param in model.parameters():
    param.requires_grad = False

tokenizer = AutoTokenizer.from_pretrained(
    CONFIG.MODEL_NAME, 
    cache_dir=CONFIG.MODEL_CACHE_DIR,
    resume_download=True,
    local_files_only=False
)

print(model.model)


dataset = CSVData(CONFIG.DATA_SOURCE)
train_loader = DataLoader(dataset, shuffle=True)


def token_to_embed(token_id: int) -> torch.Tensor: 
    return model.model.embed_tokens.weight[token_id].to(device)

def tokenize(string: str) -> torch.Tensor:
    return tokenizer(string, return_tensors='pt')["input_ids"][0].to(device)

def str_to_embed(string: str) -> torch.Tensor:
    token_ids = tokenize(string)
    return model.model.embed_tokens.weight[token_ids]  # Vectorized embedding lookup

def ids_to_embed(ids: torch.Tensor) -> torch.Tensor:
    ids = ids.to(device)
    return model.model.embed_tokens.weight[ids]  # Vectorized embedding lookup


neo_template = CONFIG.NEO_PROMPT.split("{}")
template_1 = str_to_embed(neo_template[0])
template_2 = str_to_embed(neo_template[1])

def get_neo_prompt(neo: torch.Tensor) -> torch.Tensor:
    return torch.cat((template_1, neo.unsqueeze(0).to(device), template_2), dim=0)


# Remember: this right here is what we're *actually* optimizing
neo_id = tokenize(CONFIG.INITIAL_TOKEN)[-1].item()
neo_embed = token_to_embed(int(neo_id))
neo_param = nn.Parameter(neo_embed.to(device))
ref_neo_param = neo_embed.to(device)



def APOLoss(beta: float): 
    # Use log probs
    def apo(prob_policy_y_c, prob_policy_y_r, prob_ref_y_c, prob_ref_y_r): 
        return -torch.log(torch.sigmoid(beta * (prob_policy_y_c - prob_policy_y_r + prob_ref_y_c - prob_ref_y_r))) - torch.log(torch.sigmoid(prob_policy_y_c - prob_policy_y_r))
        #       ^^^ modified DPO (Rafailov et al., 2024)                                                             ^^^ APO-up from D’Oosterlinck et al. (2025)
    return apo

loss_fn = APOLoss(beta=CONFIG.BETA)

optim = torch.optim.Adafactor([neo_param])

# Gets the log probability of the response given the prompt
def get_log_probs(model: AutoModelForCausalLM, prompt_embed: torch.Tensor, response_ids: torch.Tensor, grad: bool) -> torch.Tensor:
    prompt_len = prompt_embed.shape[0]
    response_embed = ids_to_embed(response_ids)
    full_embed = torch.cat((prompt_embed, response_embed), dim=0)  # (T, E)
    with torch.set_grad_enabled(grad):
        logits = model(inputs_embeds=full_embed.unsqueeze(0), return_dict=True).logits # (B, T, V)

    logits = logits.squeeze(0) # (T, V)
    response_logits = logits[prompt_len-1:-1] # (T_response, V)
    log_probs = torch.nn.functional.log_softmax(response_logits, dim=-1) # (T_response, V)
    token_log_probs = log_probs.gather(dim=-1, index=response_ids.unsqueeze(-1)).squeeze(-1) # (T_response)
    return token_log_probs.sum()  # (scalar)

# Precompute log probs for reference embedding
ref_log_probs_cache = []

for prompt, chosen, rejected in tqdm(train_loader, desc="Computing reference log probs"):
    chosen_ids = tokenize(chosen)
    rejected_ids = tokenize(rejected)
    pre_prompt_embed = str_to_embed(prompt)
    ref_full_prompt_embed = torch.cat((pre_prompt_embed, get_neo_prompt(ref_neo_param)), dim=0)
    with torch.no_grad():
        log_prob_ref_c = get_log_probs(model, ref_full_prompt_embed, chosen_ids, grad=False)
        log_prob_ref_r = get_log_probs(model, ref_full_prompt_embed, rejected_ids, grad=False)
    ref_log_probs_cache.append((log_prob_ref_c.item(), log_prob_ref_r.item()))
    clear_cache()


for epoch in tqdm(range(CONFIG.N_EPOCHS), desc="Training"):
    epoch_losses = []
    for batch_idx, (prompt, chosen, rejected) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG.N_EPOCHS}")):
        chosen_ids = tokenize(chosen)
        rejected_ids = tokenize(rejected)
        pre_prompt_embed = str_to_embed(prompt)
        full_prompt_embed = torch.cat((pre_prompt_embed, get_neo_prompt(neo_param)), dim=0)

        log_prob_c = get_log_probs(model, full_prompt_embed, chosen_ids, grad=True)
        log_prob_r = get_log_probs(model, full_prompt_embed, rejected_ids, grad=True)
        log_prob_ref_c = torch.tensor(ref_log_probs_cache[batch_idx][0], device=device)
        log_prob_ref_r = torch.tensor(ref_log_probs_cache[batch_idx][1], device=device)

        clear_cache()

        loss = loss_fn(log_prob_c, log_prob_r, log_prob_ref_c, log_prob_ref_r)
        optim.zero_grad()
        loss.backward()
        optim.step()

        epoch_losses.append(loss.item())

    avg_loss = sum(epoch_losses) / len(epoch_losses)
    
    tqdm.write(f"Epoch {epoch+1}/{CONFIG.N_EPOCHS} mean loss: {avg_loss:.4f} | saving to {CONFIG.SAVE_PATH}/epoch_{epoch+1}.pt")
    torch.save(neo_param.data, f"{CONFIG.SAVE_PATH}/epoch_{epoch+1}.pt")
    