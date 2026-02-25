import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import re
from template import EmbeddingTemplate

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
MODEL_CACHE_DIR = "/Volumes/backrooms/huggingface"
NEO_PARAM_PATH = "saves/foreclearing_1.pt"
ZERO_CONTROL = False  # Set to True to zero out all values in the neo_param tensor
DEFAULT_CONTROL = False  # Set to True to use the default strings specified in each prompt template file
PROMPT_TEMPLATE_PATH = "prompts/llama_instruct_train_prompt.txt"

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    cache_dir=MODEL_CACHE_DIR,
    local_files_only=False,
    dtype=torch.bfloat16
).to(device)



tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(string: str) -> torch.Tensor:
    return tokenizer(string, return_tensors='pt')["input_ids"][0].to(device)

def str_to_embed(string: str) -> torch.Tensor:
    token_ids = tokenize(string)
    return model.model.embed_tokens.weight[token_ids]

def token_to_embed(token_id: int) -> torch.Tensor: 
    return model.model.embed_tokens.weight[token_id].to(device)

neo_param = torch.load(NEO_PARAM_PATH, map_location='cpu')
print(neo_param)
print(neo_param.shape)
if ZERO_CONTROL:
    neo_param = torch.zeros(neo_param.shape)
neo_param = neo_param.to(device)

prompt_template = EmbeddingTemplate(PROMPT_TEMPLATE_PATH, str_to_embed, token_to_embed)

def generate_response(max_new_tokens: int = 32, temperature: float = 0.0) -> str:

    generated_tokens = []
    if DEFAULT_CONTROL:
        prompt = prompt_template.default().to(torch.bfloat16)
    else:
        prompt = prompt_template.format(neo_param).to(torch.bfloat16)
    assert not torch.isnan(prompt).any()
    assert not torch.isinf(prompt).any()

    past_key_values = None
    
    for _ in tqdm(range(max_new_tokens), desc="Generating response"):
        output = model(inputs_embeds=prompt.unsqueeze(0), return_dict=True, past_key_values=past_key_values, use_cache=bool(past_key_values))
        past_key_values = output.past_key_values
        logits = output.logits
        next_logits = logits[0, -1, :]
        
        if temperature > 0.0:
            probs = torch.softmax(next_logits / temperature, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()
        else:
            next_id = torch.argmax(next_logits).item()
        if next_id == tokenizer.eos_token_id:
            break
    
        generated_tokens.append(next_id)
        prompt = torch.cat((prompt, token_to_embed(next_id).unsqueeze(0)), dim=0).to(torch.bfloat16)
    
    return tokenizer.decode(generated_tokens, skip_special_tokens=True)

for i in tqdm(range(10), desc="Repeating samples"):
    tqdm.write(generate_response(max_new_tokens=32, temperature=0.5))