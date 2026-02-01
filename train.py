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
import os
from dict_dataset import DictData


@dataclass(frozen=True)
class Config:
    INITIAL_TOKEN: str
    NEO_PROMPT: str
    DATASET: Dataset
    MODEL_NAME: str
    N_EPOCHS: int
    SAVE_PATH: str
    MODEL_CACHE_DIR: str
    PROBS_CACHE_DIR: str
    BETA: float
    REFERENCE_LOG_PROBS_PATH: str
    ON_THE_FLY_REF_PROBS: bool
    EPOCH_SIZE: int | None = None
    DO_WANDB: bool = False

CONFIG = Config(
    INITIAL_TOKEN = "good",
    NEO_PROMPT = r"Provide a dictionary definition for the word \"{}\": ",
    DATASET = DictData("data/en_dict.csv", "n. The act of deliberately refusing to learn the scientific explanations of things out of fear that it will ruin the magic"),
    MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct",
    N_EPOCHS = 64,
    SAVE_PATH = "saves/dict_neo",
    MODEL_CACHE_DIR = "/Volumes/backrooms/huggingface", # Don't mind me, that's the name of my hard drive
    BETA = 0.2,
    PROBS_CACHE_DIR = "saves",
    REFERENCE_LOG_PROBS_PATH = "dict_neo_ref_lp_llama-3.2-1b-instruct.pt",
    ON_THE_FLY_REF_PROBS = True,
    EPOCH_SIZE = 128,
    DO_WANDB = True,
)

if CONFIG.DO_WANDB:
    import wandb
    wandb.init(
        project="neologisms",
        config={
            "model": CONFIG.MODEL_NAME,
            "initial_token": CONFIG.INITIAL_TOKEN,
            "beta": CONFIG.BETA,
            "epoch_size": CONFIG.EPOCH_SIZE,
        }
    )




if not os.path.exists(CONFIG.SAVE_PATH):
    os.makedirs(CONFIG.SAVE_PATH)

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


dataset = CONFIG.DATASET
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
    return torch.cat((template_1, neo.unsqueeze(0), template_2), dim=0)


# Remember: this right here is what we're *actually* optimizing
if CONFIG.INITIAL_TOKEN is not None:
    neo_id = tokenize(CONFIG.INITIAL_TOKEN)[-1].item()
    neo_embed = token_to_embed(int(neo_id))
else:
    embed_shape = token_to_embed(0).shape
    neo_embed = torch.zeros(embed_shape, dtype=torch.bfloat16, device=device)
neo_param = nn.Parameter(neo_embed.to(torch.float32).to(device))
ref_neo_param = neo_embed.clone().detach().to(torch.float32)
#ref_neo_param.requires_grad = False


def APOLoss(beta: float): 
    # Use log probs
    def apo(prob_policy_y_c, prob_policy_y_r, prob_ref_y_c, prob_ref_y_r): 
        return -nn.functional.logsigmoid(beta * (prob_policy_y_c - prob_policy_y_r + prob_ref_y_c - prob_ref_y_r)) - nn.functional.logsigmoid(prob_policy_y_c - prob_policy_y_r)
        #return -torch.log(torch.sigmoid((prob_policy_y_c / prob_policy_y_r + prob_ref_y_c / prob_ref_y_r) * beta)) - torch.log(torch.sigmoid(prob_policy_y_c - prob_policy_y_r))

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
    token_log_probs = torch.clamp(token_log_probs, min=-50.0, max=0.0)
    return token_log_probs.sum()  # (scalar)

def stability_check(t: torch.Tensor) -> None:
    if torch.isnan(t).any():
        print("Sound the alarm! NaN found!")
        return 1
    if torch.isinf(t).any():
        print("Sound the alarm! Inf found!")
        return 1
    return 0

def compute_ref_log_probs(pre_prompt_embed: torch.Tensor, chosen_ids: torch.Tensor, rejected_ids: torch.Tensor) -> tuple[float, float]:
    with torch.no_grad():
        ref_full_prompt_embed = torch.cat((pre_prompt_embed, get_neo_prompt(ref_neo_param)), dim=0)
        log_prob_ref_c = get_log_probs(model, ref_full_prompt_embed, chosen_ids, grad=False)
        log_prob_ref_r = get_log_probs(model, ref_full_prompt_embed, rejected_ids, grad=False)
    return (log_prob_ref_c.item(), log_prob_ref_r.item())
    

if not CONFIG.ON_THE_FLY_REF_PROBS:
    # Precompute log probs for reference embedding
    ref_log_probs_cache = []

    ref_log_probs_path = os.path.join(CONFIG.PROBS_CACHE_DIR, CONFIG.REFERENCE_LOG_PROBS_PATH)
    if os.path.exists(ref_log_probs_path):
        ref_log_probs_cache = torch.load(ref_log_probs_path)
        print(f"Loaded reference log probs from {ref_log_probs_path}!")
    else:
        for prompt, chosen, rejected in tqdm(train_loader, desc="Computing reference log probs"):
            chosen_ids = tokenize(chosen)
            rejected_ids = tokenize(rejected)
            pre_prompt_embed = str_to_embed(prompt)
            ref_log_probs_cache.append(compute_ref_log_probs(pre_prompt_embed, chosen_ids, rejected_ids))
        torch.save(ref_log_probs_cache, ref_log_probs_path)
        print(f"Saved reference log probs to {ref_log_probs_path}")

for epoch in tqdm(range(CONFIG.N_EPOCHS), desc="Training"):
    epoch_losses = []
    epoch_grad_norms = []
    for batch_idx, (prompt, chosen, rejected) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG.N_EPOCHS}", total=CONFIG.EPOCH_SIZE if CONFIG.EPOCH_SIZE is not None else len(train_loader))):
        if CONFIG.EPOCH_SIZE is not None and batch_idx >= CONFIG.EPOCH_SIZE:
            break
        #param_before = neo_param.data.clone()
        #print(f"neo_param norm before: {param_before.norm().item():.6f}")
        #print(f"neo_param first 5 values before: {param_before[:5].tolist()}")
        chosen_ids = tokenize(chosen)
        rejected_ids = tokenize(rejected)
        pre_prompt_embed = str_to_embed(prompt)
        full_prompt_embed = torch.cat((pre_prompt_embed, get_neo_prompt(neo_param)), dim=0)
        if CONFIG.ON_THE_FLY_REF_PROBS:
            log_prob_ref_c, log_prob_ref_r = compute_ref_log_probs(pre_prompt_embed, chosen_ids, rejected_ids)
        else:
            log_prob_ref_c = torch.tensor(ref_log_probs_cache[batch_idx][0], device=device)
            log_prob_ref_r = torch.tensor(ref_log_probs_cache[batch_idx][1], device=device)

        log_prob_c = get_log_probs(model, full_prompt_embed, chosen_ids, grad=True)
        log_prob_r = get_log_probs(model, full_prompt_embed, rejected_ids, grad=True)

        clear_cache()

        loss = loss_fn(log_prob_c, log_prob_r, log_prob_ref_c, log_prob_ref_r)
        if stability_check(loss): continue
        optim.zero_grad()
        loss.backward()
        # Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_norm_([neo_param], max_norm=1.0)
        optim.step()
        #print(neo_param.grad.norm() if neo_param.grad is not None else "None")
        #param_after = neo_param.data.clone()
        #print(f"neo_param norm after: {param_after.norm().item():.6f}")
        #print(f"neo_param first 5 values after: {param_after[:5].tolist()}")
        epoch_losses.append(loss.item())
        epoch_grad_norms.append(neo_param.grad.norm())
        

    avg_loss = sum(epoch_losses) / len(epoch_losses)
    if CONFIG.DO_WANDB:
        wandb.log({
            "loss": avg_loss,
            "neo_param_grad_norm": sum(epoch_grad_norms) / len(epoch_grad_norms),
        })
    
    tqdm.write(f"Epoch {epoch+1}/{CONFIG.N_EPOCHS} mean loss: {avg_loss:.4f} | saving to {CONFIG.SAVE_PATH}/epoch_{epoch+1}.pt")
    torch.save(neo_param.data, f"{CONFIG.SAVE_PATH}/epoch_{epoch+1}.pt")
    