import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import hf_hub_download
import pandas as pd
from torch.utils.data import Dataset
#from .csv_dataset import CSVData
from tqdm import tqdm
from dataclasses import dataclass
import os
import csv
from .template import EmbeddingTemplate
from .lm import LMBackend, HFTransformerBackend
from typing import Callable



@dataclass(frozen=True)
class TrainConfig:
    INITIAL_TOKEN: str
    NEO_PROMPT_PATH: str
    DATASET: Dataset
    N_EPOCHS: int
    SAVE_PATH: str
    PROBS_CACHE_PATH: str
    BETA: float
    MODEL_BACKEND: LMBackend
    LEARNING_RATE: float = 1e-3
    ON_THE_FLY_REF_PROBS: bool = True
    EPOCH_SIZE: int | None = None
    DO_WANDB: bool = False
    NEO_DTYPE: torch.dtype = torch.float32
    
    SPECIAL_DATA_PROCESS_FN: Callable | None = None
   


#print("Fetching dataset ...")

#dataset = pd.read_json(hf_hub_download(repo_id="GAIR/lima", filename="train.jsonl", repo_type="dataset"), lines=True)
#print(type(dataset))
#print(dataset.head(10))

def clear_cache(device: torch.device):
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def APOLoss(beta: float): 
    # Use log probs
    def apo(prob_policy_y_c, prob_policy_y_r, prob_ref_y_c, prob_ref_y_r): 
        return -nn.functional.logsigmoid(beta * (prob_policy_y_c - prob_policy_y_r + prob_ref_y_c - prob_ref_y_r)) - nn.functional.logsigmoid(prob_policy_y_c - prob_policy_y_r)
        #return -torch.log(torch.sigmoid((prob_policy_y_c / prob_policy_y_r + prob_ref_y_c / prob_ref_y_r) * beta)) - torch.log(torch.sigmoid(prob_policy_y_c - prob_policy_y_r))

        #       ^^^ modified DPO (Rafailov et al., 2024)                                                             ^^^ APO-up from D’Oosterlinck et al. (2025)
    return apo




def stability_check(t: torch.Tensor) -> None:
    if torch.isnan(t).any():
        print("Sound the alarm! NaN found!")
        return 1
    if torch.isinf(t).any():
        print("Sound the alarm! Inf found!")
        return 1
    return 0

def run_train(CONFIG: TrainConfig) -> None: 
    """
    Train the model for a given configuration
    Args:
        CONFIG (TrainConfig): Configuration for the training
    """
    do_wandb = CONFIG.DO_WANDB
    if do_wandb:
        import wandb
        from dotenv import load_dotenv
        load_dotenv()
        wandb_key = os.getenv("WANDB_API_KEY")
        if wandb_key is not None:
            wandb.login(key=wandb_key)
            wandb.init(
                project="neologisms",
                config={
                    "model": CONFIG.MODEL_BACKEND.name,
                    "initial_token": CONFIG.INITIAL_TOKEN,
                    "beta": CONFIG.BETA,
                    "epoch_size": CONFIG.EPOCH_SIZE,
                }
            )
        else:
            print("WANDB_API_KEY is not set, skipping wandb logging")
            do_wandb = False
    
    if not os.path.exists(CONFIG.SAVE_PATH):
        os.makedirs(CONFIG.SAVE_PATH)

    
    print("Loading resources ...")

    model_backend = CONFIG.MODEL_BACKEND
    model = model_backend.model # Aliases for readability
    device = model_backend.device
    print(model_backend)

    dataset = CONFIG.DATASET
    train_loader = DataLoader(dataset, shuffle=True)
    ref_loader = DataLoader(dataset, shuffle=False)
    # Train *only* the neologism
    for param in model.parameters():
        param.requires_grad = False

    # Remember: this right here is what we're *actually* optimizing
    if CONFIG.INITIAL_TOKEN is not None:
        neo_id = model_backend.tokenize(CONFIG.INITIAL_TOKEN)[-1].item()
        neo_embed = model_backend.token_to_embed(int(neo_id))
    else:
        embed_shape = model_backend.token_to_embed(0).shape
        neo_embed = torch.zeros(embed_shape, dtype=CONFIG.NEO_DTYPE, device=device)
    neo_param = nn.Parameter(neo_embed.to(CONFIG.NEO_DTYPE).to(device))
    ref_neo_param = neo_embed.clone().detach().to(CONFIG.NEO_DTYPE).to(device)
    
    #ref_neo_param.requires_grad = False

    prompt_template = EmbeddingTemplate(CONFIG.NEO_PROMPT_PATH, model_backend.str_to_embed, model_backend.token_to_embed)

    loss_fn = APOLoss(beta=CONFIG.BETA)
    optim = torch.optim.Adafactor([neo_param], lr=CONFIG.LEARNING_RATE)

    # Gets the log probability of the response given the prompt
    def get_log_probs(model_backend: LMBackend, prompt_embed: torch.Tensor, response_ids: torch.Tensor, grad: bool) -> torch.Tensor:
        prompt_len = prompt_embed.shape[0]
        response_embed = model_backend.ids_to_embed(response_ids)
        full_embed = torch.cat((prompt_embed, response_embed), dim=0)  # (T, E)
        with torch.set_grad_enabled(grad):
            logits = model_backend.embeds_forward(full_embed.unsqueeze(0)).logits # (B, T, V)

        logits = logits.squeeze(0) # (T, V)
        response_logits = logits[prompt_len-1:-1] # (T_response, V)
        log_probs = torch.nn.functional.log_softmax(response_logits, dim=-1) # (T_response, V)
        token_log_probs = log_probs.gather(dim=-1, index=response_ids.unsqueeze(-1)).squeeze(-1) # (T_response)
        token_log_probs = torch.clamp(token_log_probs, min=-50.0, max=0.0)
        return token_log_probs.sum()  # (scalar)

    def compute_ref_log_probs(pre_prompt_embed: torch.Tensor, chosen_ids: torch.Tensor, rejected_ids: torch.Tensor) -> tuple[float, float]:
        with torch.no_grad():
            ref_full_prompt_embed = torch.cat((pre_prompt_embed, prompt_template.format(ref_neo_param)), dim=0)
            log_prob_ref_c = get_log_probs(model_backend, ref_full_prompt_embed, chosen_ids, grad=False)
            log_prob_ref_r = get_log_probs(model_backend, ref_full_prompt_embed, rejected_ids, grad=False)
        return (log_prob_ref_c.item(), log_prob_ref_r.item())


    if not CONFIG.ON_THE_FLY_REF_PROBS:
        # Precompute log probs for reference embedding, keyed by dataset index
        ref_log_probs_cache = {}

        ref_log_probs_path = CONFIG.PROBS_CACHE_PATH
        if os.path.exists(ref_log_probs_path):
            ref_log_probs_cache = torch.load(ref_log_probs_path)
            print(f"Loaded reference log probs from {ref_log_probs_path}!")
        else:
            for idx, (prompt, chosen, rejected) in enumerate(tqdm(ref_loader, desc="Computing reference log probs")):
                chosen_ids = model_backend.tokenize(chosen)
                rejected_ids = model_backend.tokenize(rejected)
                pre_prompt_embed = model_backend.str_to_embed(prompt)
                ref_log_probs_cache[idx] = compute_ref_log_probs(pre_prompt_embed, chosen_ids, rejected_ids)
            torch.save(ref_log_probs_cache, ref_log_probs_path)
            print(f"Saved reference log probs to {ref_log_probs_path}")
    


    

    for epoch in tqdm(range(CONFIG.N_EPOCHS), desc="Training"):
        epoch_losses = []
        epoch_grad_norms = []
        epoch_loader = ref_loader if not CONFIG.ON_THE_FLY_REF_PROBS else train_loader
        for batch_idx, (prompt, chosen, rejected) in enumerate(tqdm(epoch_loader, desc=f"Epoch {epoch+1}/{CONFIG.N_EPOCHS}", total=CONFIG.EPOCH_SIZE if CONFIG.EPOCH_SIZE is not None else len(epoch_loader))):
            if CONFIG.EPOCH_SIZE is not None and batch_idx >= CONFIG.EPOCH_SIZE:
                break
            #param_before = neo_param.data.clone()
            #print(f"neo_param norm before: {param_before.norm().item():.6f}")
            #print(f"neo_param first 5 values before: {param_before[:5].tolist()}")
            chosen_ids = model_backend.tokenize(chosen)
            rejected_ids = model_backend.tokenize(rejected)
            pre_prompt_embed = model_backend.str_to_embed(prompt)
            full_prompt_embed = torch.cat((pre_prompt_embed, prompt_template.format(neo_param)), dim=0)
            if CONFIG.ON_THE_FLY_REF_PROBS:
                log_prob_ref_c, log_prob_ref_r = compute_ref_log_probs(pre_prompt_embed, chosen_ids, rejected_ids)
            else:
                log_prob_ref_c = torch.tensor(ref_log_probs_cache[batch_idx][0], device=device)
                log_prob_ref_r = torch.tensor(ref_log_probs_cache[batch_idx][1], device=device)

            log_prob_c = get_log_probs(model_backend, full_prompt_embed, chosen_ids, grad=True)
            log_prob_r = get_log_probs(model_backend, full_prompt_embed, rejected_ids, grad=True)

            clear_cache(device)

            loss = loss_fn(log_prob_c, log_prob_r, log_prob_ref_c, log_prob_ref_r)
            if stability_check(loss):
                tqdm.write("Warning: Loss is NaN or Inf, skipping this batch")
                continue
            optim.zero_grad()
            loss.backward()
            # Clip gradients
            # torch.nn.utils.clip_grad_norm_([neo_param], max_norm=1.0)
            optim.step()
            #print(neo_param.grad.norm() if neo_param.grad is not None else "None")
            #param_after = neo_param.data.clone()
            #print(f"neo_param norm after: {param_after.norm().item():.6f}")
            #print(f"neo_param first 5 values after: {param_after[:5].tolist()}")
            epoch_losses.append(loss.item())
            epoch_grad_norms.append(neo_param.grad.norm())
            

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        if do_wandb:
            wandb.log({
                "loss": avg_loss,
                "neo_param_grad_norm": sum(epoch_grad_norms) / len(epoch_grad_norms),
            })
        
        tqdm.write(f"Epoch {epoch+1}/{CONFIG.N_EPOCHS} mean loss: {avg_loss:.4f} | saving to {CONFIG.SAVE_PATH}/epoch_{epoch+1}.pt")
        torch.save(neo_param.data, f"{CONFIG.SAVE_PATH}/epoch_{epoch+1}.pt")
