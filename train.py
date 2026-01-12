import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import pandas as pd
from torch.utils.data import Dataset
from pref_dataset import PrefData
from tqdm import tqdm

SAVE_PATH = "saves/neo_param.pt"

#print("Fetching dataset ...")

#dataset = pd.read_json(hf_hub_download(repo_id="GAIR/lima", filename="train.jsonl", repo_type="dataset"), lines=True)
#print(type(dataset))
#print(dataset.head(10))


print("Loading resources ...")

device = torch.device("cuda" if torch.cuda.is_available() else "mps")

cache_dir = "/Volumes/The Hole/huggingface"

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct", 
    cache_dir=cache_dir,
    local_files_only=False,
    torch_dtype=torch.float16
).to(device)

ref_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct", 
    cache_dir=cache_dir,
    local_files_only=False,
    torch_dtype=torch.float16
).to(device)

ref_model.eval()
for param in ref_model.parameters():
    param.requires_grad = False

# Train *only* the neologism
for param in model.parameters():
    param.requires_grad = True

tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct", 
    cache_dir=cache_dir,
    resume_download=True,
    local_files_only=False
)

print(model.model)


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

def get_neo_prompt(template: str, neo: nn.Parameter) -> torch.Tensor:
    template = template.split("{}")
    return torch.cat((str_to_embed(template[0]), neo.unsqueeze(0).to(device), str_to_embed(template[1])), dim=0)

print("Preparing prompts ...")

INITIAL_TOKEN = " good"  # Token whose embedding we start with
NEO_PROMPT = r"Give me a response you think is {}. "  # Prompt template for the neologism

dataset = PrefData("data/pref_responses.csv")
train_loader = DataLoader(dataset, shuffle=True)

# Remember: this right here is what we're *actually* optimizing
neo_id = tokenize(INITIAL_TOKEN)[-1].item()
neo_embed = token_to_embed(int(neo_id))
neo_param = nn.Parameter(neo_embed.to(device))


def APOLoss(beta: float): 
    # Use log probs
    def apo(prob_policy_y_c, prob_policy_y_r, prob_ref_y_c, prob_ref_y_r): 
        return -torch.log(torch.sigmoid((prob_policy_y_c / prob_policy_y_r + prob_ref_y_c / prob_ref_y_r) * beta)) - torch.log(torch.sigmoid(prob_policy_y_c - prob_policy_y_r))
        #       ^^^ modified DPO (Rafailov et al., 2024)                                                             ^^^ APO-up from D’Oosterlinck et al. (2025)
    return apo

loss_fn = APOLoss(beta=0.2)

optim = torch.optim.Adafactor([neo_param])

N_EPOCHS = 10

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


for epoch in tqdm(range(N_EPOCHS), desc="Training"):
    epoch_losses = []
    for batch_idx, (prompt, chosen, rejected) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{N_EPOCHS}")):
        chosen_ids = tokenize(chosen)
        rejected_ids = tokenize(rejected)
        pre_prompt_embed = str_to_embed(prompt)
        full_prompt_embed = torch.cat((pre_prompt_embed, get_neo_prompt(NEO_PROMPT, neo_param)), dim=0)
        log_prob_c = get_log_probs(model, full_prompt_embed, chosen_ids, grad=True)
        log_prob_r = get_log_probs(model, full_prompt_embed, rejected_ids, grad=True)
        log_prob_ref_c = get_log_probs(ref_model, full_prompt_embed, chosen_ids, grad=False)
        log_prob_ref_r = get_log_probs(ref_model, full_prompt_embed, rejected_ids, grad=False)
        torch.mps.empty_cache()
        loss = loss_fn(log_prob_c, log_prob_r, log_prob_ref_c, log_prob_ref_r)
        optim.zero_grad()
        loss.backward()
        optim.step()
        epoch_losses.append(loss.item())
    avg_loss = sum(epoch_losses) / len(epoch_losses)
    tqdm.write(f"Epoch {epoch+1}/{N_EPOCHS} mean loss: {avg_loss:.4f}")

torch.save(neo_param.data, SAVE_PATH)
print("Saved to ", SAVE_PATH)